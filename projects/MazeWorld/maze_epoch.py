import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

# import torch.multiprocessing as mp
import multiprocessing as mp
from multiprocessing import Queue

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager, GeneratorBase
from airsoul.utils import weighted_loss, img_pro, img_post
from airsoul.dataloader import MazeDataSet, PrefetchDataLoader, MazeSplitShortDataSet

def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

@EpochManager
class MazeEpochVAE:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "noise",
                        "kl_weight",
                        "reconstruction_error",
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys[3:])
            self.lr = self.config.lr_vae
            self.lr_decay_interval = self.config.lr_vae_decay_interval
            self.lr_start_step = self.config.lr_vae_start_step
        else:
            self.logger_keys = ["reconstruction_error", 
                        "kl_divergence"]
            self.stat = DistStatistics(*self.logger_keys)
        self.max_maze = None

    def preprocess(self):
        if(self.is_training):
            self.sigma_scheduler = LinearScheduler(self.config.sigma_scheduler, 
                                                   self.config.sigma_value)
            self.lambda_scheduler = LinearScheduler(self.config.lambda_scheduler, 
                                                    self.config.lambda_value)
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_vae, verbose=self.main, max_maze=self.max_maze, folder_verbose=False), # TODO
            batch_size=self.config.batch_size_vae,
            rank=self.rank,
            world_size=self.world_size,
            num_workers = 1,
            )
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_vae_stop')):
            if(epoch_id >= self.config.epoch_vae_stop):
                return False
        return True

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # folder_name,# bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        seq_len = self.config.seq_len_vae
        for sub_idx, seg_obs in segment_iterator(
                            self.config.seq_len_vae, self.config.seg_len_vae,
                            self.device, obs_arr):
            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()

            if(self.is_training):
                sigma = self.sigma_scheduler()
            else:
                sigma = 0
            loss = self.model.module.vae_loss(
                    seg_obs,
                    _sigma=sigma,
                    seq_len=seq_len)
            losses.append(loss)
            if(self.is_training):
                syn_loss = (loss["Reconstruction-Error"] + self.lambda_scheduler() * loss["KL-Divergence"]) / loss["count"]
                # print(syn_loss)
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    reconstruction_error = loss["Reconstruction-Error"] / loss["count"],
                    kl_divergence = loss["KL-Divergence"] / loss["count"],
                    count = loss["count"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            self.sigma_scheduler(), 
                            self.lambda_scheduler(), 
                            stat_res["reconstruction_error"]["mean"], 
                            stat_res["kl_divergence"]["mean"],
                            epoch=epoch_id,
                            iteration=batch_id)
            # update the scheduler
            self.sigma_scheduler.step()
            self.lambda_scheduler.step()
        else:
            self.stat.gather(self.device,
                    reconstruction_error=loss["Reconstruction-Error"] / loss["count"], 
                    kl_divergence=loss["KL-Divergence"] / loss["count"], 
                    count=loss["count"], 
                    )
            
        
    def epoch_end(self, epoch_id, batch_id):
        if(not self.is_training):
            stat_res = self.stat()
            
            if(self.logger is not None):
                self.logger(stat_res["reconstruction_error"]["mean"], 
                        stat_res["kl_divergence"]["mean"], 
                        epoch=epoch_id)


@EpochManager
class MazeEpochCausal: # the computer
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if (self.config.has_attr("is_visualize")):
            self.is_visualize = self.config.is_visualize  
        else:
            self.is_visualize = False
        print(f"is_visualize: {self.is_visualize}") 
        
        if (self.config.has_attr("max_maze")):
            self.max_maze = self.config.max_maze
            print(f"max_maze: {self.max_maze}")
        else:
            self.max_maze = None
            
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
            
        else:
            if self.config.output.endswith("txt"):
                output_folder = os.path.dirname(self.config.output)

            else:
                output_folder = self.config.output
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)


            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 1
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_start')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze=self.max_maze), # TODO
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size, 
            num_workers = 1,
            prefetch_batches=1,
            )
        

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # bev_arr,
                local_batch_id=-1,
                global_batch_id=-1,
                global_epoch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        current_prediction_observations = []
        for sub_idx, seg_cmd, seg_obs, seg_behavior_act, seg_label_act in segment_iterator(
                                seqL, self.config.seg_len_causal, self.device, 
                                cmd_arr, (obs_arr, 1), behavior_actid_arr, label_actid_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()
            # seg_bev = seg_bev.permute(0, 1, 4, 2, 3)
            # seg_bev = seg_bev.contiguous()

            loss, obs_pred, a_pred, __ = self.model.module.sequential_loss(
                                    prompts = seg_cmd,
                                    observations = seg_obs,
                                    tags = None, 
                                    behavior_actions = seg_behavior_act,
                                    rewards = None,
                                    label_actions = seg_label_act, 
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    is_training=self.is_training,
                                    reduce_dim=self.reduce_dim,) 
                                
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_latent * loss["wm-latent"]
                        + self.config.lossweight_worldmodel_raw * loss["wm-raw"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                                loss_worldmodel_raw = loss["wm-raw"] / loss["count_wm"],
                                loss_worldmodel_latent = loss["wm-latent"] / loss["count_wm"],
                                loss_policymodel = loss["pm"] / loss["count_pm"])
        
        
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            stat_res["loss_worldmodel_raw"]["mean"], 
                            stat_res["loss_worldmodel_latent"]["mean"],
                            stat_res["loss_policymodel"]["mean"],
                            epoch=global_epoch_id,
                            iteration=local_batch_id)
        else:
            loss_wm_r = []
            loss_wm_l = []
            loss_pm = []
            counts = []

            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            # print(losses[0]["pm"].shape) 
            # print(losses[0]["count_pm"].shape)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1) # 当mask掉黑屏动作时，count_pm会出现0，导致此处出现NaN
            counts = torch.cat([loss["count_pm"] for loss in losses], dim=1)


            bsz = loss_wm_r.shape[0]
            seg_num = loss_wm_l.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_worldmodel_raw=loss_wm_r[i], 
                        validate_worldmodel_latent=loss_wm_l[i], 
                        validate_policymodel=loss_pm[i],
                        count=counts[i])

                    
    def epoch_end(self, epoch_id, batch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
                        epoch=epoch_id)
            print(f"logger end epoch: {epoch_id}")
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    print(f"Saving the validation results to {self.config.output}")
                    for key_name in stat_res:
                        print(f"key_name: {key_name}")
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)


@EpochManager
class MazeEpochCausalSplit: # the computer
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if (self.config.has_attr("is_visualize")):
            self.is_visualize = self.config.is_visualize  
        else:
            self.is_visualize = False
        print(f"is_visualize: {self.is_visualize}") 
        
        if (self.config.has_attr("max_maze")):
            self.max_maze = self.config.max_maze
            print(f"max_maze: {self.max_maze}")
        else:
            self.max_maze = None
            
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
            
        else:
            if self.config.output.endswith("txt"):
                output_folder = os.path.dirname(self.config.output)

            else:
                output_folder = self.config.output
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)


            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 1
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_start')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeSplitShortDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze=self.max_maze), # TODO
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size, 
            num_workers = 1,
            prefetch_batches=1,
            )
        

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # bev_arr,
                local_batch_id=-1,
                global_batch_id=-1,
                global_epoch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        current_prediction_observations = []
        seqL = self.config.seq_len_causal if isinstance(self.config.seq_len_causal, int) else self.config.seq_len_causal[1] - self.config.seq_len_causal[0]
        for sub_idx, seg_cmd, seg_obs, seg_behavior_act, seg_label_act in segment_iterator(
                                seqL, self.config.seg_len_causal, self.device, 
                                cmd_arr, (obs_arr, 1), behavior_actid_arr, label_actid_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_obs = seg_obs.contiguous()
            # seg_bev = seg_bev.permute(0, 1, 4, 2, 3)
            # seg_bev = seg_bev.contiguous()

            loss, obs_pred, a_pred, __ = self.model.module.sequential_loss(
                                    prompts = seg_cmd,
                                    observations = seg_obs,
                                    tags = None, 
                                    behavior_actions = seg_behavior_act,
                                    rewards = None,
                                    label_actions = seg_label_act, 
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
                                    is_training=self.is_training,
                                    reduce_dim=self.reduce_dim,) 
                                
            losses.append(loss)
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_latent * loss["wm-latent"]
                        + self.config.lossweight_worldmodel_raw * loss["wm-raw"]
                        + self.config.lossweight_policymodel * loss["pm"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                                loss_worldmodel_raw = loss["wm-raw"] / loss["count_wm"],
                                loss_worldmodel_latent = loss["wm-latent"] / loss["count_wm"],
                                loss_policymodel = loss["pm"] / loss["count_pm"])
        
        
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                            stat_res["loss_worldmodel_raw"]["mean"], 
                            stat_res["loss_worldmodel_latent"]["mean"],
                            stat_res["loss_policymodel"]["mean"],
                            epoch=global_epoch_id,
                            iteration=local_batch_id)
        else:
            loss_wm_r = []
            loss_wm_l = []
            loss_pm = []
            counts = []

            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            # print(losses[0]["pm"].shape) 
            # print(losses[0]["count_pm"].shape)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1) # 当mask掉黑屏动作时，count_pm会出现0，导致此处出现NaN
            counts = torch.cat([loss["count_pm"] for loss in losses], dim=1)

            bsz = loss_wm_r.shape[0]
            seg_num = loss_wm_l.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            loss_wm_r = torch.mean(loss_wm_r[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_wm_l = torch.mean(loss_wm_l[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            loss_pm = torch.mean(loss_pm[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_worldmodel_raw=loss_wm_r[i], 
                        validate_worldmodel_latent=loss_wm_l[i], 
                        validate_policymodel=loss_pm[i],
                        count=counts[i])

                    
    def epoch_end(self, epoch_id, batch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
                        epoch=epoch_id, iteration=batch_id)
            print(f"logger end epoch: {epoch_id}")
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(self.config.output):
                        os.makedirs(self.config.output)
                    print(f"Saving the validation results to {self.config.output}")
                    for key_name in stat_res:
                        print(f"key_name: {key_name}")
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{self.config.output}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)




@EpochManager
class MazeEpochCausalShort: # the computer
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
        if (self.config.has_attr("is_visualize")):
            self.is_visualize = self.config.is_visualize  
        else:
            self.is_visualize = False
        print(f"is_visualize: {self.is_visualize}") 
        
        if (self.config.has_attr("max_maze")):
            self.max_maze = self.config.max_maze
            print(f"max_maze: {self.max_maze}")
        else:
            self.max_maze = None
            
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_raw",
                        "loss_worldmodel_latent",
                        "loss_policymodel"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.lr = self.config.lr_causal
            self.lr_decay_interval = self.config.lr_causal_decay_interval
            self.lr_start_step = self.config.lr_causal_start_step
            self.reduce_dim = 1
            
        else:
            if self.config.output.endswith("txt"):
                output_folder = os.path.dirname(self.config.output)

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        self.init_logger()
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
    def init_logger(self):
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is None):
            # self.logger_keys = self.get('logger_keys')
            if(self.logger_keys is not None and len(self.logger_keys)!=0):
                assert type(self.logger_keys) == list, \
                    f"The logger_keys must be a list of string."
                process_name = f"Generation-{self.__class__.__name__}"
                max_iter = -1
                log_file = self.log_config.log_file
                self.logger = Logger(
                        *self.logger_keys,
                        on=self.main, 
                        max_iter=max_iter,
                        use_tensorboard=self.log_config.use_tensorboard,
                        log_file=log_file,
                        prefix=f"{self.run_name}-{process_name}",
                        field=f"{self.log_config.tensorboard_log}/{self.run_name}-{process_name}")

    def __call__(self, epoch_id, rank):
        import cv2
        # nohup python -m projects.MazeWorld.generator_test ./generator-configs/blockTest.yaml > static_cache.log 2>&1 &
        batch_size = 1 # TODO
        pred_len = self.pred_len
        loss_batch = []
        cache_generate = False
        o_generate = False
        video_generate = True
        # history_cache = None
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))

            output_folder_path = os.path.join(self.output_root, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, behavior_act_arr, label_act_arr, rew_arr = batch_data
            obs_arr = obs_arr.permute(0, 1, 4, 2, 3) # (B, T, H, W, C) to (B, T, C, H, W)
            states = obs_arr.contiguous()
            commands = cmd_arr.contiguous()
            actions = behavior_actid_arr.contiguous()

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape of {states.shape}")
            assert states.shape[1] == actions.shape[1] + 1, f"states shape: {states.shape}, actions shape: {actions.shape}"
            history_cache = None
            self.model.module.reset()
            loss_records = []
            pred_records = []
            real_records = []

            for cl_step in range(self.start_position, self.end_position):
                # if cl_step in self.start_points:
                #     print(f"cl_step: {cl_step} start_points")
                #     history_cache = None
                #     self.model.module.reset()
                if cl_step in self.record_points:
                    pred_len = self.pred_len 
                else:
                    pred_len = 1
                end = min(cl_step, states.shape[1] - 1)
                pred_obs_list, history_cache = self.model.module.generate_states_only(
                        prompts=commands[:, end:end+pred_len],
                        current_observation=states[:, end:end+1], 
                        action_trajectory=actions[:, end:end+pred_len],
                        history_observation=None, #states[start:end],
                        history_action=None, #actions[start:end],
                        history_update_memory=False, 
                        autoregression_update_memory=False, # TOTEST
                        cache=history_cache,
                        single_batch=True,
                        history_single_step=False,
                        future_single_step=False,
                        raw_images=True,
                        need_numpy=False)

                real = states[:, end+1:end+1+pred_len]
                mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                        loss_type="mse",
                                        gt=real, 
                                        need_cnt=True,
                                        )
                mse_loss = mse_loss/255/255
                loss_records.append(mse_loss.detach().numpy()/cnt)  
                import copy
                if cl_step in self.record_points:
                    if cache_generate == True:
                        np.save(os.path.join(output_folder_path, f"cache_{cl_step}.npy"), history_cache)
                        print(f"Saved cache to {os.path.join(output_folder_path, f'cache_{cl_step}.npy')}")
                    pred_records.append(pred_obs_list.cpu().detach().numpy())
                    real = real.clone().cpu().detach().numpy()
                    real_records.append(real)
                    print(pred_obs_list.cpu().detach().numpy().shape, real.shape)
                    plotLongDemo(pred_obs_list.cpu().detach().numpy(), 
                        real, 
                        os.path.join(output_folder_path, f"demo_{cl_step}.png"))
            loss_records = np.array(loss_records)

            # save the loss record to npy 
            np.save(os.path.join(output_folder_path, f"losses.npy"), loss_records)
            print(f"Saved losses to {os.path.join(output_folder_path, f'losses.npy')}")
            
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)
        loss_batch = np.array(loss_batch)
        bsz = loss_batch.shape[0]
        seg_num = loss_batch.shape[1] // self.downsample_length
        valid_seq_len = seg_num * self.downsample_length
        loss_batch = np.mean(loss_batch[:, :valid_seq_len].reshape(bsz, seg_num, -1), axis=-1)
        self.stat.gather(self.device,
                validate_worldmodel_raw=loss_batch[0], 
                count=cnt)

    def epoch_end(self, epoch_id, batch_id):
        stat_res = self.stat()
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is not None):
            self.logger(stat_res["validate_worldmodel_raw"]["mean"],
                    epoch=epoch_id)
        if(self.extra_info is not None):
            if(self.extra_info.lower() == 'validate' and self.main):
                if not os.path.exists(self.config.output):
                    os.makedirs(self.config.output)
                for key_name in stat_res:
                    res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                    file_path = f'{self.config.output}/result_{key_name}.txt'
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    with open(file_path, 'w') as f_model:
                        f_model.write(res_text)
            
