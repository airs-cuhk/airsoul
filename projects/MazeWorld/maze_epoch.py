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
            MazeRandomShortDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze=self.max_maze), # TODO
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size, 
            num_workers = 1,
            prefetch_batches=1,
            )
        

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        current_prediction_observations = []
        seqL = self.config.seq_len_causal if isinstance(self.config.seq_len_causal, int) else self.config.seq_len_causal[1] - self.config.seq_len_causal[0]
        self.model.module.decision_model.causal_model.position = 0
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
                            epoch=epoch_id,
                            iteration=batch_id)
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
class MazeEpochDinoV2Causal: # the computer
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
        print(f"Finish init MazeEpochDinoV2Causal")
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_start')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            ProcthorPaddingDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze=self.max_maze, use_feat=True), # TODO
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size, 
            num_workers = 1,
            prefetch_batches=1,
            )
        

    def compute(self, cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, # bev_arr,
                epoch_id=-1, 
                batch_id=-1):
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

            seg_obs = seg_obs.contiguous()
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
                            epoch=epoch_id,
                            iteration=batch_id)
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
            
            # for all position that all these 3 loss is 0, set the count to 0
            zero_mask = (loss_wm_r == 0) & (loss_wm_l == 0) & (loss_pm == 0)
            counts[zero_mask] = 0

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





class MAZEGenerator(GeneratorBase):

    def __call__(self, epoch_id, rank):
    
        folder_count = 0

        for folder in os.listdir(self.config.data_root):
            folder_path = os.path.join(self.config.data_root, folder)
            
            if os.path.isdir(folder_path):
                states = np.load(os.path.join(folder_path, 'observations.npy'))
                actions = np.load(os.path.join(folder_path, 'actions_behavior_id.npy'))

                in_context_len = self.config.in_context_len
                pred_len = self.config.pred_len
                start = self.config.start_position
                temp = self.config.temp
                drop_out = self.config.drop_out
                len_causal = self.config.seg_len_causal
                output_folder = self.config.output
                
                end = min(start + in_context_len, len(states))

                pred_obs_list = self.model.module.generate_step_by_step(
                    observations=states[start:end+1],
                    actions=actions[start:end],
                    actions_gt=actions[end:end+pred_len],
                    temp=temp,
                    drop_out = drop_out,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    in_context_len = in_context_len,
                    len_causal = len_causal,
                    n_step=pred_len
                )

                real = [states[i] for i in range(end+1, end + 1 + pred_len)] 

                pred_obs_list_with_initial = pred_obs_list
                
                
                video_folder = os.path.join(output_folder, f'video_{folder_count}')
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                video_filename = os.path.join(video_folder, f"pred_obs_video_{folder_count}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                frame_height, frame_width = pred_obs_list_with_initial[0].shape[:2]
                video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))

                for real_frame, pred_frame in zip(real, pred_obs_list_with_initial):
                    rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    concatenated_img = np.hstack((rotated_real, rotated_pred))

                    img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                    video_writer.write(img)

                video_writer.release() 

                print(f"Saved video with {len(real)} frames to {video_filename}")

                
                updated_cache = None
                print(f"Cache cleared after generating {len(real)} frames.")

                folder_count += 1  

                if folder_count >= 16:
                    print("Processed 16 folders. Stopping.")
                    break 




class compound_error_generator(GeneratorBase): #TODO   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = np.array([1, 10, 100, 1000, 9000])
        # [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        if (self.config.has_attr("max_maze")):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

        if self.end_position > self.config.seq_len_causal:
            assert False, f"end_position should be smaller than seq_len_causal, got {self.end_position} vs {self.config.seq_len_causal}"    
        
        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

    def preprocess(self):
        self.K_step_list = [1, 2, 4, 8]
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
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
        K_step_list = self.K_step_list
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1: # to deal with the trajectory folder...
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape")

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
            for checkpoint_id in range(0, self.end_position):
                end = min(checkpoint_id, states.shape[1] - 1)
                if end in self.record_points:
                    last_history_cache = history_cache.copy()
                    for pred_len in K_step_list:
                        history_cache = last_history_cache
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
                                need_numpy=False
                                )
                        real = states[:, end+1:end+1+pred_len]
                        mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                                loss_type="mse",
                                                gt=real, 
                                                need_cnt=True,
                                                )
                        mse_loss = mse_loss/255/255
                        print(f"check_point {checkpoint_id} with mse_loss: {mse_loss/cnt}, cnt: {cnt}")
                        loss_records.append(mse_loss.detach().numpy()/cnt)  
                        K_folder = os.path.join(output_folder_path, f"K_{pred_len}")
                        if not os.path.exists(K_folder):
                            os.makedirs(K_folder)
                        np.save(os.path.join(K_folder, f"loss_{checkpoint_id}.npy"), mse_loss.detach().numpy()/cnt)
                        print(f"Saved loss to {os.path.join(K_folder, f'loss_{checkpoint_id}.npy')}")
                        np.save(os.path.join(K_folder, f"pred_{checkpoint_id}.npy"), pred_obs_list.cpu().detach().numpy())
                        np.save(os.path.join(K_folder, f"real_{checkpoint_id}.npy"), real.cpu().detach().numpy())
                        print(f"Saved pred and real to {os.path.join(K_folder, f'pred_{checkpoint_id}.npy')} and {os.path.join(K_folder, f'real_{checkpoint_id}.npy')}")

                else:
                    pred_len = 1
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
                            need_numpy=False
                            )


    def epoch_end(self, epoch_id, batch_id):
        pass
            

class MPinteractive_trajectory(GeneratorBase):
    

    def epoch_end(self, epoch_id, batch_id):
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        # if self.output_root is not None:
        #     if not os.path.exists(self.output_root):
        #         os.makedirs(self.output_root)
        #         print(f"Created output folder {self.output_root}")
        # if self.output_root is None:
        #     assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"
        


    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")

    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent




        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):


            # TODO learning the fixed length of context with segment mode
            for sub_idx, seg_cmd, seg_obs, seg_behavior_act, seg_label_act in segment_iterator(
                            self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                            cmd_arr, (obs_arr, 1), behavior_actid_arr, label_actid_arr):
                pass
            max_steps = 10000
            n_range = (15,16)
            maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
            
            # origin_task = MazeTaskSampler(n_range=n_range, allow_loops=True, 
            #             landmarks_number_range=(6, 10),
            #             commands_sequence = 10000,
            #             verbose=False)
            # new_task = Resampler(origin_task)
            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))

            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            maze_env.set_task(new_task)

            done = False
            observation_list = []
            reward_list = []
            bev_list = []
            cmd_list = []
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            # (H, W, C) to (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None # observation
            last_action = None # np.zeros_like(maze_env.action_space.sample())
            inference_record = []
            action_record = []
            loss_record = []
            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)
            print(f"output folder: {output_folder}")
            print("-----------------------------")
            import tqdm
            K_step = 1
            cache = None
            self.model.module.reset()
            for step in range(max_steps):
                if done:
                    print(f"done at step {step}")
                    break
                cmd_string = information["command"]
                # action = self.model.module.step(observation, last_observation, last_action) # Replace it with your own policy function
                pred_obs_list, action, cache = self.model.module.generate_states_and_action(
                                command,
                                observation, 
                                future_steps=K_step,
                                history_observation=None, #last_observation,
                                history_action=None, #last_action,
                                history_update_memory=True, 
                                autoregression_update_memory=True,
                                cache=cache,
                                single_batch=True,
                                history_single_step=True,
                                raw_images=True,
                                need_predict_states=True,
                                need_numpy=True)
                action = action[0,0]
                inference_record.append(pred_obs_list)
                action_record.append(action)
                # print(f"action: {action}")
                # print(f"action: {np.info(action)}")
                # print("-----------------------------")  
                obs, reward, done, information = maze_env.step(action)
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                # print(f"maze_task: {maze_env.maze_core._instant_rewards}")
                maze_env.render()
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                observation_list.append(obs)
                mse_loss = np.mean((obs - pred_obs_list[0])**2/(255*255))
                loss_record.append(mse_loss)
                # print(f"mse loss: {mse_loss}")
                last_observation = observation
                last_action = action
                observation = obs
                reward_list.append(reward)
                bev_list.append(maze_env.get_local_map()[1])
                cmd_list.append(information["command"])
                sum_reward += reward

            inference_record = np.array(inference_record)
            observation_list = np.array(observation_list)
            reward_list = np.array(reward_list)
            # save reward record to npy
            np.save(os.path.join(output_folder, "reward.npy"), reward_list)
            print(f"Saved reward to {os.path.join(output_folder, 'reward.npy')}")
            print("------------------------------")
            print(f"sum reward: {sum_reward}")
            print("------------------------------")
            import matplotlib.pyplot as plt
            # plt.plot(loss_record, label="mse loss", alpha=0.5)
            mean_loss_record = []
            downsample_length = 50
            loss_record = np.array(loss_record)
            # save loss to npy
            np.save(os.path.join(output_folder, "loss.npy"), loss_record)
            print(f"Saved loss to {os.path.join(output_folder, 'loss.npy')}")
            for i in range(0, len(loss_record)):
                mean_loss_record.append(np.mean(loss_record[max(i - downsample_length, 0):min(i + downsample_length, len(loss_record))]))
            plt.plot(range(0, len(loss_record)), mean_loss_record, label="mse loss")
            plt.legend()
            plt.savefig(os.path.join(output_folder, "mse_loss.png"))
            plt.close()
            print(f"Saved mse loss plot to {os.path.join(output_folder, 'mse_loss.png')}")
            maze_env.save_trajectory(os.path.join(output_folder, "trajectory.png"))
            print(f"Saved trajectory to {os.path.join(output_folder, 'trajectory.png')}")
            maze_env.save_trajectory_npy(os.path.join(output_folder, f"trajectory.npy"))
            pickle.dump(new_task, open(os.path.join(output_folder, "task.pkl"), "wb"))
            
            video_folder = os.path.join(output_folder, f'video')
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video_filename = os.path.join(video_folder, f"pred_obs_video{0}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            frame_height, frame_width = inference_record[0].shape[-2:]
            video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
            frame_count = 0
            T = 1
            for real_frames, pred_frames in zip(observation_list, inference_record):
                # (B, T, C, H, W) to (H, W, C) just pick up the first frame of T, and we default B=1
                real_frame = real_frames.transpose(1, 2, 0)
                pred_frame = pred_frames[0, 0].transpose(1, 2, 0)
                rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                concatenated_img = np.hstack((rotated_real, rotated_pred))
                # save the concatenated image

                img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                if frame_count % 100 == 0:
                    cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
                    if T > 1:
                        ARimageFolder = os.path.join(video_folder, f"ARimage_{frame_count}")
                        if not os.path.exists(ARimageFolder):
                            os.makedirs(ARimageFolder)
                        whole_ARimage = None
                        for i in range(T):
                            ARimage = pred_frames[0,i].transpose(1, 2, 0)
                            rotated_ARimage = cv2.rotate(ARimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            ARimage = np.clip(rotated_ARimage, 0, 255).astype(np.uint8)
                            ARreal = real_frames[0,i].transpose(1, 2, 0)
                            rotated_ARreal = cv2.rotate(ARreal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            ARimage = np.clip(rotated_ARreal, 0, 255).astype(np.uint8)
                            # concatenate the ARimage and ARreal up and down
                            ARconcatenated_img = np.vstack((rotated_ARreal, rotated_ARimage))
                            if i == 0:
                                whole_ARimage = ARconcatenated_img
                            else:
                                whole_ARimage = np.hstack((whole_ARimage, ARconcatenated_img))
                            ARimage = np.clip(ARconcatenated_img, 0, 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(ARimageFolder, f"ARframe_{i}.png"), ARimage)
                        cv2.imwrite(os.path.join(ARimageFolder, f"whole_ARimage.png"), whole_ARimage)
                        
                    cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
                frame_count += 1
                video_writer.write(img)
            video_writer.release() 
            print(f"Saved video with {len(observation_list)} frames to {video_filename}")




class interactive_trajectory(GeneratorBase):

    def epoch_end(self, epoch_id, batch_id):
        # MazeTaskDataSet
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        # if self.output_root is not None:
        #     if not os.path.exists(self.output_root):
        #         os.makedirs(self.output_root)
        #         print(f"Created output folder {self.output_root}")
        # if self.output_root is None:
        #     assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"
        


    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")

    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent
        record_points = [8500,9500]
        print(f"record points: {record_points}")
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            max_steps = 10000
            n_range = (15,16)
            maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
            
            # origin_task = MazeTaskSampler(n_range=n_range, allow_loops=True, 
            #             landmarks_number_range=(6, 10),
            #             commands_sequence = 10000,
            #             verbose=False)
            # new_task = Resampler(origin_task)
            
            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))
            # new_task = Resampler(new_task, resample_landmarks=False, resample_landmarks_color=False)
            # print(f"----------Resampled--------")
            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            maze_env.set_task(new_task)

            done = False
            observation_list = []
            reward_list = []
            bev_list = []
            cmd_list = []
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            # (H, W, C) to (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None # observation
            last_action = None # np.zeros_like(maze_env.action_space.sample())
            inference_record = []
            action_record = []
            loss_record = []
            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)
            print(f"output folder: {output_folder}")
            print("-----------------------------")
            import tqdm
            K_step = 1
            cache = None
            self.model.module.reset()
            L_task = []
            commands = []
            start_step = -1
            for step in range(max_steps):
                if step in record_points:
                    last_command = information["command"]
                    maze_env.refresh_command()
                    current_command = maze_env.maze_core.get_command()
                    print(f"current command is {current_command}, and last is : {last_command}")
                    start_step = step
                if done:
                    print(f"done at step {step}")
                    break
                # cmd_string = information["command"]
                # action = self.model.module.step(observation, last_observation, last_action) # Replace it with your own policy function
                pred_obs_list, action, cache = self.model.module.generate_states_and_action(
                                command,
                                observation, 
                                future_steps=K_step,
                                history_observation=None, #last_observation,
                                history_action=None, #last_action,
                                history_update_memory=True, 
                                autoregression_update_memory=True,
                                cache=cache,
                                single_batch=True,
                                history_single_step=True,
                                raw_images=True,
                                need_predict_states=True,
                                need_numpy=True)
                action = action[0,0]
                inference_record.append(pred_obs_list)
                action_record.append(action)
                # print(f"action: {action}")
                # print(f"action: {np.info(action)}")
                # print("-----------------------------")  
                obs, reward, done, information = maze_env.step(action)
                if reward != 0 and start_step != -1:
                    l_task = step - start_step
                    L_task.append(l_task)
                    print(f"The length of {start_step} to finish is: {l_task}")
                    start_step = -1
                if step - start_step > 500 and start_step != -1:
                    print(f"The {start_step} fails")
                    start_step = -1
                    L_task.append(-1)

                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                # print(f"maze_task: {maze_env.maze_core._instant_rewards}")
                commands.append(command)
                maze_env.render()
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                observation_list.append(obs)
                mse_loss = np.mean((obs - pred_obs_list[0])**2/(255*255))
                loss_record.append(mse_loss)
                # print(f"mse loss: {mse_loss}")
                last_observation = observation
                last_action = action
                observation = obs
                reward_list.append(reward)
                bev_list.append(maze_env.get_local_map()[1])
                cmd_list.append(information["command"])
                sum_reward += reward

            inference_record = np.array(inference_record)
            observation_list = np.array(observation_list)
            reward_list = np.array(reward_list)
            commands = np.array(commands)
            cmd_list = np.array(cmd_list)
            L_task = np.array(L_task)
            # save reward record to npy
            np.save(os.path.join(output_folder, "reward.npy"), reward_list)
            print(f"Saved reward to {os.path.join(output_folder, 'reward.npy')}")
            np.save(os.path.join(output_folder, "cmd.npy"), cmd_list)
            print(f"Saved cmd to {os.path.join(output_folder, 'cmd.npy')}")
            np.save(os.path.join(output_folder, "Ltask.npy"), L_task)
            print(f"Saved Ltask to {os.path.join(output_folder, 'Ltask.npy')} with shape of {L_task.shape}")
            # L_task
            print("------------------------------")
            print(f"sum reward: {sum_reward}")
            print("------------------------------")
            import matplotlib.pyplot as plt
            # plt.plot(loss_record, label="mse loss", alpha=0.5)
            mean_loss_record = []
            downsample_length = 50
            loss_record = np.array(loss_record)
            # save loss to npy
            np.save(os.path.join(output_folder, "loss.npy"), loss_record)
            print(f"Saved loss to {os.path.join(output_folder, 'loss.npy')}")
            for i in range(0, len(loss_record)):
                mean_loss_record.append(np.mean(loss_record[max(i - downsample_length, 0):min(i + downsample_length, len(loss_record))]))
            plt.plot(range(0, len(loss_record)), mean_loss_record, label="mse loss")
            plt.legend()
            plt.savefig(os.path.join(output_folder, "mse_loss.png"))
            plt.close()
            print(f"Saved mse loss plot to {os.path.join(output_folder, 'mse_loss.png')}")
            maze_env.save_trajectory(os.path.join(output_folder, "trajectory.png"))
            print(f"Saved trajectory to {os.path.join(output_folder, 'trajectory.png')}")
            maze_env.save_trajectory_npy(os.path.join(output_folder, f"trajectory.npy"))
            pickle.dump(new_task, open(os.path.join(output_folder, "task.pkl"), "wb"))
            
            video_folder = os.path.join(output_folder, f'video')
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            video_filename = os.path.join(video_folder, f"pred_obs_video{0}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            frame_height, frame_width = inference_record[0].shape[-2:]
            video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
            frame_count = 0
            T = 1
            for real_frames, pred_frames in zip(observation_list, inference_record):
                # (B, T, C, H, W) to (H, W, C) just pick up the first frame of T, and we default B=1
                real_frame = real_frames.transpose(1, 2, 0)
                pred_frame = pred_frames[0, 0].transpose(1, 2, 0)
                rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                concatenated_img = np.hstack((rotated_real, rotated_pred))
                # save the concatenated image

                img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
                if frame_count % 100 == 0:
                    cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
                    if T > 1:
                        ARimageFolder = os.path.join(video_folder, f"ARimage_{frame_count}")
                        if not os.path.exists(ARimageFolder):
                            os.makedirs(ARimageFolder)
                        whole_ARimage = None
                        for i in range(T):
                            ARimage = pred_frames[0,i].transpose(1, 2, 0)
                            rotated_ARimage = cv2.rotate(ARimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            ARimage = np.clip(rotated_ARimage, 0, 255).astype(np.uint8)
                            ARreal = real_frames[0,i].transpose(1, 2, 0)
                            rotated_ARreal = cv2.rotate(ARreal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            ARimage = np.clip(rotated_ARreal, 0, 255).astype(np.uint8)
                            # concatenate the ARimage and ARreal up and down
                            ARconcatenated_img = np.vstack((rotated_ARreal, rotated_ARimage))
                            if i == 0:
                                whole_ARimage = ARconcatenated_img
                            else:
                                whole_ARimage = np.hstack((whole_ARimage, ARconcatenated_img))
                            ARimage = np.clip(ARconcatenated_img, 0, 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(ARimageFolder, f"ARframe_{i}.png"), ARimage)
                        cv2.imwrite(os.path.join(ARimageFolder, f"whole_ARimage.png"), whole_ARimage)
                        
                    cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
                frame_count += 1
                video_writer.write(img)
            video_writer.release() 
            print(f"Saved video with {len(observation_list)} frames to {video_filename}")


class learned_from_label_interactive_trajectory_without_interupt(GeneratorBase):

    def epoch_end(self, epoch_id, batch_id):
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.learning_steps = self.config.learning_steps 
        self.test_steps = self.config.test_steps
        self.test_points = self.config.test_points


        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        
        
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
            
        if self.config.has_attr("task_resample"):
            self.task_resample = self.config.task_resample
        else:
            self.task_resample = False
        
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, world_size=self.world_size, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        # print(f"saving in {self.output_folder_path}")
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")

    def exploration(self, env, max_steps, model):
        pass

    class ExploreHistory: # TODO put it to other place
        def __init__(self, agent_name, env_info, env_type, keys = ["obs", "action", "reward", "command"]):
            self.history = {}
            self.time = 0
            
            self.env_info = env_info
            self.env_type = env_type
            self.agent_name = agent_name
            for key in keys:
                self.history[key] = []
        def update(self, values):
            keys = self.history.keys()
            for key in keys:
                if key in values.keys():
                    self.history[key].append(values[key])
                else:
                    self.history[key].append(None)
            self.time += 1
        def get(self, key):
            return self.history[key]
        def get_all(self):
            return self.history
        def add_key(self, key):
            if key in self.history.keys():
                return False
            self.history[key] = []
            for i in range(self.time):
                self.history[key].append(None)

            return True
        def clear(self):
            keys = self.history.keys
            self.history = {}
            self.time = 0
            for key in keys:
                self.history[key] = []
        def __len__(self):
            return self.time
        def __str__(self):
            return f"ExploreHistory of {self.env_name} with {self.env_type}, totally {self.time} steps"    


    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent

        max_steps = 11000
        learning_steps = self.learning_steps
        test_steps = self.test_steps
        n_range = (15,16)
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        print(f"------start with learning steps {learning_steps}------------")
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):

            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))

            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            
            if self.task_resample == True:
                print(f"resampling task: ")
                new_task = Resampler(new_task) # TODO:
            maze_env.set_task(new_task)

            done = False
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None 
            last_action = None 
            last_cmd = information["command"]

            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)
            print(f"output folder: {output_folder}")
            print("-----------------------------")

            # learning from the Oracle agent
            start_step = -1
            reward = 0
            cache = None

            self.model.module.reset()
            maze_history = self.ExploreHistory("OracleLeadsDivLong", new_task, "maze", keys = ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss", "prediction", "oracle_length"])
            label_agent = OracleAgent(maze_env=maze_env, render=False)
            

            for step in range(learning_steps):
                if done:
                    print(f"done at step {step}")
                    break
                action = label_agent.step(observation, reward)
                pred_obs_list, pred_act_list, cache = self.model.module.generate_states_only(
                                prompts=command,
                                current_observation=np.transpose(observation, (2, 0, 1)), 
                                action_trajectory=np.array([action]),
                                history_observation=None, #states[start:end],
                                history_action=None, #actions[start:end],
                                history_update_memory=False, 
                                autoregression_update_memory=False, # TOTEST
                                cache=cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=True, 
                                need_action=True)
                
                last_cmd = information["command"]
                obs, reward, done, information = maze_env.step(action)
                mse_loss = np.mean((np.transpose(obs, (2, 0, 1)) - pred_obs_list[0,0])**2/(255*255))

                last_command = command
                last_observation = observation
                last_action = action

                observation = obs
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                sum_reward += reward
                # ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss"]
                to_update = {
                    "obs": last_observation,
                    "oracle_action": action,
                    "agent_action": pred_act_list[0, 0], 
                    "reward": reward, 
                    "command": last_cmd, 
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs_list[0, 0],
                }
                maze_history.update(to_update)
            print(f"sum reward during learning from oracle: {sum_reward}")

            maze_env.refresh_command() # To start a new command to record

            current_command = maze_env.maze_core.get_command()
            information["command"] = current_command
            command = np.repeat(current_command, 256, axis=0)
            last_cmd = information["command"]
            
            import tqdm
            K_step = 1
            start_step = -1
            # (H, W, C) to (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
            sum_reward = 0
            
            test_points = self.test_points #[100, 1000, 9000]
            print(f"test points: {test_points}")

            self.temp_scheduler = LinearScheduler(self.config.temp_scheduler, 
                                self.config.temp_value)
            last_cmd_idx = maze_env.get_commands_sequence_idx()
            for step in range(test_steps):
                if done:
                    print(f"done at step {step}")
                    break
                pred_obs, action, cache = self.model.module.policy(command, observation, cache=cache, temperature=self.temp_scheduler())
                # print(self.temp_scheduler())
                self.temp_scheduler.step()
                action = action[0, 0]
                if action == 16:
                    action = 0
                
                obs, reward, done, information = maze_env.step(action)
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                
                # # TODO: for SEL
                check_SEL = self.config.check_SEL
                if check_SEL:
                    cmd_idx = maze_env.get_commands_sequence_idx()
                    if (information["command"] != last_cmd).any() and cmd_idx != last_cmd_idx:
                        print(f"command changed at step {step}")
                        oracle_traj_actions = maze_env.get_oracle_trajectory()
                        len_oracle_traj_actions = len(oracle_traj_actions)
                        after_sel_mse_loss = np.mean((obs - maze_env.maze_core.get_observation())**2/(255*255))
                        assert after_sel_mse_loss < 10e-6, f"after_sel_mse_loss: {after_sel_mse_loss}"
                    else:
                        len_oracle_traj_actions = -1

                else:
                    len_oracle_traj_actions = -1
                last_cmd_idx = maze_env.get_commands_sequence_idx()
                
                
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                mse_loss = np.mean((obs - pred_obs[0, 0])**2/(255*255))

                observation = obs
                sum_reward += reward
                to_update = {
                    # "obs": last_observation, # (C, H, W)
                    "agent_action": action, 
                    "reward": reward, 
                    "command": last_cmd, 
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs[0, 0],
                    "oracle_length": len_oracle_traj_actions,
                }
                last_cmd = information["command"]
                last_observation = observation
                last_action = action
                maze_history.update(to_update)
            
            
            print(f"Model total Reward: {sum_reward} with total steps {maze_history.__len__()}")
            # save maze_history to pkl
            pickle.dump(maze_history.get_all(), open(os.path.join(maze_output_folder, "maze_history.pkl"), "wb"))
            print(f"Saved maze_history to {os.path.join(maze_output_folder, 'maze_history.pkl')}")
            maze_env.save_trajectory(os.path.join(maze_output_folder, f"trajectory.png"))
            print(f"Saved trajectory to", os.path.join(maze_output_folder, f"trajectory.png"))





class learned_from_label_interactive_trajectory(GeneratorBase):

    def epoch_end(self, epoch_id, batch_id):
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.learning_steps = self.config.learning_steps 
        self.test_steps = self.config.test_steps
        self.test_points = self.config.test_points


        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        
        
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
            
        if self.config.has_attr("task_resample"):
            self.task_resample = self.config.task_resample
        else:
            self.task_resample = False
        
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, world_size=self.world_size, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        # print(f"saving in {self.output_folder_path}")
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")

    def exploration(self, env, max_steps, model):
        pass

    class ExploreHistory: # TODO put it to other place
        def __init__(self, agent_name, env_info, env_type, keys = ["obs", "action", "reward", "command"]):
            self.history = {}
            self.time = 0
            
            self.env_info = env_info
            self.env_type = env_type
            self.agent_name = agent_name
            for key in keys:
                self.history[key] = []
        def update(self, values):
            keys = self.history.keys()
            for key in keys:
                if key in values.keys():
                    self.history[key].append(values[key])
                else:
                    self.history[key].append(None)
            self.time += 1
        def get(self, key):
            return self.history[key]
        def get_all(self):
            return self.history
        def add_key(self, key):
            if key in self.history.keys():
                return False
            self.history[key] = []
            for i in range(self.time):
                self.history[key].append(None)

            return True
        def clear(self):
            keys = self.history.keys
            self.history = {}
            self.time = 0
            for key in keys:
                self.history[key] = []
        def __len__(self):
            return self.time
        def __str__(self):
            return f"ExploreHistory of {self.env_name} with {self.env_type}, totally {self.time} steps"    


    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent

        max_steps = 11000
        learning_steps = self.learning_steps
        test_steps = self.test_steps
        n_range = (15,16)
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        print(f"------start with learning steps {learning_steps}------------")
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):

            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))

            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            
            if self.task_resample == True:
                print(f"resampling task: ")
                new_task = Resampler(new_task) # TODO:
            maze_env.set_task(new_task)

            done = False
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None 
            last_action = None 
            last_cmd = information["command"]

            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)
            print(f"output folder: {output_folder}")
            print("-----------------------------")

            # learning from the Oracle agent
            start_step = -1
            reward = 0
            cache = None

            self.model.module.reset()
            maze_history = self.ExploreHistory("OracleLeadsDivLong", new_task, "maze", keys = ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss", "prediction", "oracle_length", "ce_loss"])
            label_agent = OracleAgent(maze_env=maze_env, render=False)
            

            for step in range(learning_steps):
                if done:
                    print(f"done at step {step}")
                    break
                action = label_agent.step(observation, reward)
                pred_obs_list, pred_act_list, cache = self.model.module.generate_states_only(
                                prompts=command,
                                current_observation=np.transpose(observation, (2, 0, 1)), 
                                action_trajectory=np.array([action]),
                                history_observation=None, #states[start:end],
                                history_action=None, #actions[start:end],
                                history_update_memory=False, 
                                autoregression_update_memory=False, # TOTEST
                                cache=cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=True, 
                                need_action=True)
                
                last_cmd = information["command"]
                obs, reward, done, information = maze_env.step(action)
                mse_loss = np.mean((np.transpose(obs, (2, 0, 1)) - pred_obs_list[0,0])**2/(255*255))

                last_command = command
                last_observation = observation
                last_action = action

                observation = obs
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                sum_reward += reward
                # ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss"]
                to_update = {
                    "obs": last_observation,
                    "oracle_action": action,
                    "agent_action": pred_act_list[0, 0], 
                    "reward": reward, 
                    "command": last_cmd, 
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs_list[0, 0],
                }
                maze_history.update(to_update)
            print(f"sum reward during learning from oracle: {sum_reward}")

            maze_env.refresh_command() # To start a new command to record

            current_command = maze_env.maze_core.get_command()
            information["command"] = current_command
            command = np.repeat(current_command, 256, axis=0)
            last_cmd = information["command"]
            
            import tqdm
            K_step = 1
            start_step = -1
            # (H, W, C) to (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
            sum_reward = 0
            
            test_points = self.test_points #[100, 1000, 9000]
            print(f"test points: {test_points}")

            self.temp_scheduler = LinearScheduler(self.config.temp_scheduler, 
                                self.config.temp_value)
            last_cmd_idx = maze_env.get_commands_sequence_idx()
            for step in range(test_steps):
                if done:
                    print(f"done at step {step}")
                    break
                if step in test_points:
                    maze_env.refresh_command() # To start a new command to record
                    current_command = maze_env.maze_core.get_command()
                    command = np.repeat(current_command, 256, axis=0)
                    information["command"] = current_command
                
                label_action = label_agent.step(observation, reward)
                pred_obs, action, cache, act_distribution = self.model.module.policy(command, observation, cache=cache, temperature=self.temp_scheduler(), need_distribution=True)
                # cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
                # print(f"act_distribution: {act_distribution[0].shape}")
                # ce_loss = cross_entropy_loss(torch.tensor(act_distribution[0]), torch.tensor([label_action]))
                # print(f"ce_loss: {ce_loss}")
                
                # print(self.temp_scheduler())
                self.temp_scheduler.step()
                action = action[0, 0]
                if action == 16:
                    action = 0
                
                obs, reward, done, information = maze_env.step(action)
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                
                # for SEL
                check_SEL = self.config.check_SEL
                if check_SEL:
                    cmd_idx = maze_env.get_commands_sequence_idx()
                    if (information["command"] != last_cmd).any() and cmd_idx != last_cmd_idx:
                        print(f"command changed at step {step}")
                        oracle_traj_actions = maze_env.get_oracle_trajectory()
                        len_oracle_traj_actions = len(oracle_traj_actions)
                        # # save the 2 obs to png
                        # cv2.imwrite(os.path.join(maze_output_folder, f"{step}_after_oracle_obs.png"), maze_env.maze_core.get_observation())
                        # cv2.imwrite(os.path.join(maze_output_folder, f"{step}_oracle_obs.png"), obs)
                        # print(f"Saved 2 obs to {maze_output_folder}")
                        after_sel_mse_loss = np.mean((obs - maze_env.maze_core.get_observation())**2/(255*255))
                        assert after_sel_mse_loss < 10e-6, f"after_sel_mse_loss: {after_sel_mse_loss}"
                    else:
                        len_oracle_traj_actions = -1

                else:
                    len_oracle_traj_actions = -1
                last_cmd_idx = maze_env.get_commands_sequence_idx()
                
                
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                mse_loss = np.mean((obs - pred_obs[0, 0])**2/(255*255))

                observation = obs
                sum_reward += reward
                to_update = {
                    # "obs": last_observation, # (C, H, W)
                    "agent_action": action, 
                    "reward": reward, 
                    "command": last_cmd, 
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs[0, 0],
                    "oracle_length": len_oracle_traj_actions,
                    # "ce_loss": ce_loss,
                }
                last_cmd = information["command"]
                last_observation = observation
                last_action = action
                maze_history.update(to_update)
            
            
            print(f"Model total Reward: {sum_reward} with total steps {maze_history.__len__()}")
            # save maze_history to pkl
            pickle.dump(maze_history.get_all(), open(os.path.join(maze_output_folder, "maze_history.pkl"), "wb"))
            print(f"Saved maze_history to {os.path.join(maze_output_folder, 'maze_history.pkl')}")
            maze_env.save_trajectory(os.path.join(maze_output_folder, f"trajectory.png"))
            print(f"Saved trajectory to", os.path.join(maze_output_folder, f"trajectory.png"))

class rand_cmd_interactive_trajectory(GeneratorBase):

    def epoch_end(self, epoch_id, batch_id):
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.learning_steps = self.config.learning_steps 
        self.test_steps = self.config.test_steps
        self.test_points = self.config.test_points


        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        
        
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
            
        if self.config.has_attr("task_resample"):
            self.task_resample = self.config.task_resample
        else:
            self.task_resample = False
        
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, world_size=self.world_size, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        # print(f"saving in {self.output_folder_path}")
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")

    def exploration(self, env, max_steps, model):
        pass

    class ExploreHistory: # TODO put it to other place
        def __init__(self, agent_name, env_info, env_type, keys = ["obs", "action", "reward", "command"]):
            self.history = {}
            self.time = 0
            
            self.env_info = env_info
            self.env_type = env_type
            self.agent_name = agent_name
            for key in keys:
                self.history[key] = []
        def update(self, values):
            keys = self.history.keys()
            for key in keys:
                if key in values.keys():
                    self.history[key].append(values[key])
                else:
                    self.history[key].append(None)
            self.time += 1
        def get(self, key):
            return self.history[key]
        def get_all(self):
            return self.history
        def add_key(self, key):
            if key in self.history.keys():
                return False
            self.history[key] = []
            for i in range(self.time):
                self.history[key].append(None)

            return True
        def clear(self):
            keys = self.history.keys
            self.history = {}
            self.time = 0
            for key in keys:
                self.history[key] = []
        def __len__(self):
            return self.time
        def __str__(self):
            return f"ExploreHistory of {self.env_name} with {self.env_type}, totally {self.time} steps"    


    def add_gaussian_noise(self, image_array, mean=0, std=25):
        """
        为图像数组添加高斯噪声。
        
        参数:
            image_array: 输入的图像NumPy数组。
            mean: 高斯噪声的均值。
            std: 高斯噪声的标准差，控制噪声强度。
            
        返回:
            添加高斯噪声后的图像数组。
        """
        # 生成与图像尺寸相同的高斯噪声
        noise = np.random.normal(mean, std, image_array.shape)
        # 将噪声叠加到原图，并限制像素值范围在0-255之间
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.float32) # 确保数据类型为uint8
        return noisy_image
    
    def add_salt_pepper_noise(self, image_array, prob=0.05):
        """
        为图像数组添加椒盐噪声。
        
        参数:
            image_array: 输入的图像NumPy数组。
            prob: 噪声比例，即图像中像素被噪声点替代的概率。
            
        返回:
            添加椒盐噪声后的图像数组。
        """
        # import pdb
        # pdb.set_trace()
        original_shape = image_array.shape
        assert original_shape == (768, )
        image_array = image_array.reshape(3, 16, 16)
        noisy_image = np.copy(image_array)
        # 计算需要添加噪声的像素总数
        num_noise = int(prob * image_array.size)
        
        # 添加盐噪声（白点）
        # 随机生成num_noise个像素的坐标
        coords = [np.random.randint(0, i, num_noise) for i in image_array.shape]
        noisy_image[coords[0], coords[1]] = 255
        
        # 添加椒噪声（黑点）
        coords = [np.random.randint(0, i, num_noise) for i in image_array.shape]
        noisy_image[coords[0], coords[1]] = 0
        noisy_image = noisy_image.reshape((768, ))
        return noisy_image.astype(np.float32)

    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent

        max_steps = 11000
        learning_steps = self.learning_steps
        test_steps = self.test_steps
        n_range = (15,16)
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        print(f"------start with learning steps {learning_steps}------------")
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):

            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))

            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            
            if self.task_resample == True:
                print(f"resampling task: ")
                new_task = Resampler(new_task) # TODO:
            maze_env.set_task(new_task)

            done = False
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None 
            last_action = None 
            last_cmd = information["command"]

            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok = True)
            print(f"output folder: {output_folder}")
            print("-----------------------------")

            # learning from the Oracle agent
            start_step = -1
            reward = 0
            cache = None

            self.model.module.reset()
            maze_history = self.ExploreHistory("OracleLeadsDivLong", new_task, "maze", keys = ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss", "prediction", "oracle_length", "image_command"])
            label_agent = OracleAgent(maze_env=maze_env, render=False)
            
            for step in range(learning_steps):
                if done:
                    print(f"done at step {step}")
                    break
                action = label_agent.step(observation, reward)
                pred_obs_list, pred_act_list, cache = self.model.module.generate_states_only(
                                prompts=command,
                                current_observation=np.transpose(observation, (2, 0, 1)), 
                                action_trajectory=np.array([action]),
                                history_observation=None, #states[start:end],
                                history_action=None, #actions[start:end],
                                history_update_memory=False, 
                                autoregression_update_memory=False, # TOTEST
                                cache=cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=True, 
                                need_action=True)
                
                last_cmd = information["command"]
                obs, reward, done, information = maze_env.step(action)
                mse_loss = np.mean((np.transpose(obs, (2, 0, 1)) - pred_obs_list[0,0])**2/(255*255))

                last_command = command
                last_observation = observation
                last_action = action

                observation = obs
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                sum_reward += reward
                # ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss"]
                to_update = {
                    "obs": last_observation,
                    "oracle_action": action,
                    "agent_action": pred_act_list[0, 0], 
                    "reward": reward, 
                    "command": last_cmd, 
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs_list[0, 0],
                }
                maze_history.update(to_update)
            print(f"sum reward during learning from oracle: {sum_reward}")

            maze_env.refresh_command() # To start a new command to record

            current_command = maze_env.maze_core.get_command()
            information["command"] = current_command
            command = np.repeat(current_command, 256, axis=0)
            last_cmd = information["command"]
            
            import tqdm
            K_step = 1
            start_step = -1
            
            P_rnd = 0.3
            # def add_noise(command):
                
            # (H, W, C) to (C, H, W)
            observation = np.transpose(observation, (2, 0, 1))
            sum_reward = 0
            
            test_points = self.test_points #[100, 1000, 9000]
            print(f"test points: {test_points}")

            self.temp_scheduler = LinearScheduler(self.config.temp_scheduler, 
                                self.config.temp_value)
            last_cmd_idx = maze_env.get_commands_sequence_idx()
            for step in range(test_steps):
                if done:
                    print(f"done at step {step}")
                    break
                if step in test_points:
                    maze_env.refresh_command() # To start a new command to record
                    current_command = maze_env.maze_core.get_command()
                    command = np.repeat(current_command, 256, axis=0)
                    information["command"] = current_command
                import random
                # if random.random() < P_rnd:
                #     print(f"randomly change command to all black at step {step}")
                #     command = add_noise(command)
                command = self.add_gaussian_noise(command)
                # command = self.add_salt_pepper_noise(command)
                cmd_image = command.reshape(3, 16, 16)
                # save to local
                cmd_image_save_folder = os.path.join(maze_output_folder, f"cmd_image_save_folder")
                if not os.path.exists(cmd_image_save_folder):
                    os.makedirs(cmd_image_save_folder)
                    
                tensor_hwc = cmd_image.transpose(1, 2, 0)
                
                # 2. 转换颜色通道: RGB -> BGR (如果原张量是RGB顺序)
                image_bgr = cv2.cvtColor(tensor_hwc, cv2.COLOR_RGB2BGR)
                
                # 3. 确保数据类型和范围正确 (如果是浮点数且范围在0-1之间)
                if image_bgr.dtype != np.uint8:
                    # 假设浮点数范围是0-1，缩放到0-255
                    if image_bgr.max() <= 1.0:
                        image_bgr = (image_bgr * 255).astype(np.uint8)
                    else:
                        image_bgr = image_bgr.astype(np.uint8)
                
                # 4. 使用cv2.imwrite保存图像
                success = cv2.imwrite(os.path.join(cmd_image_save_folder, f"{step}_cmd_image.png"), image_bgr)
                
                
                pred_obs, action, cache = self.model.module.policy(command, observation, cache=cache, temperature=self.temp_scheduler())
                # print(self.temp_scheduler())
                self.temp_scheduler.step()
                action = action[0, 0]
                if action == 16:
                    action = 0
                this_command = command.copy()
                
                obs, reward, done, information = maze_env.step(action)
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                
                # for SEL
                check_SEL = self.config.check_SEL
                if check_SEL:
                    cmd_idx = maze_env.get_commands_sequence_idx()
                    if (information["command"] != last_cmd).any() and cmd_idx != last_cmd_idx:
                        print(f"command changed at step {step}")
                        oracle_traj_actions = maze_env.get_oracle_trajectory()
                        len_oracle_traj_actions = len(oracle_traj_actions)
                        # # save the 2 obs to png
                        # cv2.imwrite(os.path.join(maze_output_folder, f"{step}_after_oracle_obs.png"), maze_env.maze_core.get_observation())
                        # cv2.imwrite(os.path.join(maze_output_folder, f"{step}_oracle_obs.png"), obs)
                        # print(f"Saved 2 obs to {maze_output_folder}")
                        after_sel_mse_loss = np.mean((obs - maze_env.maze_core.get_observation())**2/(255*255))
                        assert after_sel_mse_loss < 10e-6, f"after_sel_mse_loss: {after_sel_mse_loss}"
                    else:
                        len_oracle_traj_actions = -1

                else:
                    len_oracle_traj_actions = -1
                last_cmd_idx = maze_env.get_commands_sequence_idx()
                
                
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                mse_loss = np.mean((obs - pred_obs[0, 0])**2/(255*255))

                observation = obs
                sum_reward += reward
                to_update = {
                    # "obs": last_observation, # (C, H, W)
                    "agent_action": action, 
                    "reward": reward, 
                    "command": last_cmd, 
                    "image_command": this_command,
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs[0, 0],
                    "oracle_length": len_oracle_traj_actions,
                }
                last_cmd = information["command"]
                last_observation = observation
                last_action = action
                maze_history.update(to_update)
            
            
            print(f"Model total Reward: {sum_reward} with total steps {maze_history.__len__()}")
            # save maze_history to pkl
            pickle.dump(maze_history.get_all(), open(os.path.join(maze_output_folder, "maze_history.pkl"), "wb"))
            print(f"Saved maze_history to {os.path.join(maze_output_folder, 'maze_history.pkl')}")
            maze_env.save_trajectory(os.path.join(maze_output_folder, f"trajectory.png"))
            print(f"Saved trajectory to", os.path.join(maze_output_folder, f"trajectory.png"))




class DaggerTempGenerator(GeneratorBase):

    def epoch_end(self, epoch_id, batch_id):
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.learning_steps = self.config.learning_steps 
        self.test_steps = self.config.test_steps
        if self.config.has_attr("test_points"):
            self.test_points = self.config.test_points
        else:
            self.test_points = None # [100, 1000, 9000]
        
        
        
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, world_size=self.world_size, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        # print(f"saving in {self.output_folder_path}")
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")


    def exploration(self, env, max_steps, model):
        pass

    class ExploreHistory: # TODO put it to other place
        def __init__(self, agent_name, env_info, env_type, keys = ["obs", "action", "reward", "command"]):
            self.history = {}
            self.time = 0
            
            self.env_info = env_info
            self.env_type = env_type
            self.agent_name = agent_name
            for key in keys:
                self.history[key] = []
        def update(self, values):
            keys = self.history.keys()
            for key in keys:
                if key in values.keys():
                    self.history[key].append(values[key])
                else:
                    self.history[key].append(None)
            self.time += 1
        def get(self, key):
            return self.history[key]
        def get_all(self):
            return self.history
        def add_key(self, key):
            if key in self.history.keys():
                return False
            self.history[key] = []
            for i in range(self.time):
                self.history[key].append(None)

            return True
        def clear(self):
            keys = self.history.keys
            self.history = {}
            self.time = 0
            for key in keys:
                self.history[key] = []
        def __len__(self):
            return self.time
        def __str__(self):
            return f"ExploreHistory of {self.env_name} with {self.env_type}, totally {self.time} steps"    


    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent

        max_steps = 11000
        learning_steps = self.learning_steps
        test_steps = self.test_steps
        n_range = (15,16)
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        print(f"------start with learning steps {learning_steps}------------")
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):

            folder_name = folder_name[0] # batch size is 1
            new_task_path = batch_data[0]
            new_task = pickle.load(open(new_task_path, 'rb'))

            print(f"task: {new_task}")
            print("-----------------------------\n\n")  
            new_task = Resampler(new_task) # TODO:
            maze_env.set_task(new_task)

            done = False
            sum_reward = 0
            
            observation, information = maze_env.reset()
            observation = np.array(observation, dtype=np.uint8)
            command = information["command"]
            command = np.repeat(command, 256, axis=0)
            last_observation = None 
            last_action = None 
            last_cmd = information["command"]

            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)

            print(f"output folder: {maze_output_folder}")
            print("-----------------------------")

            # learning from the Oracle agent
            start_step = -1
            reward = 0
            cache = None

            self.model.module.reset()
            # maze_history = self.ExploreHistory("OracleLeadsDivLong", new_task, "maze", keys = ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss", "prediction", "oracle_length"])
            label_agent = OracleAgent(maze_env=maze_env, render=False)
            maze_env.refresh_command() # To start a new command to record

            current_command = maze_env.maze_core.get_command()
            information["command"] = current_command
            command = np.repeat(current_command, 256, axis=0)
            last_cmd = information["command"]
            
            import tqdm
            K_step = 1
            start_step = -1
            # (H, W, C) to (C, H, W)
            sum_reward = 0
            
            test_points = [] # self.test_points #[100, 1000, 9000]
            print(f"test points: {test_points}")

            self.temp_scheduler = LinearScheduler(self.config.temp_scheduler, 
                                self.config.temp_value)
            observation_list = [observation.copy()]
            cmd_list = [information["command"]]
            bact_id_list = []
            lact_id_list = []
            bact_val_list = []
            lact_val_list = []
            bact_type_list = []
            reward_list = []
            
            observation = np.transpose(observation, (2, 0, 1))
            for step in range(test_steps):
                if done:
                    print(f"done at step {step}")
                    break
                
                pred_obs, action, cache = self.model.module.policy(command, observation, cache=cache, temperature=self.temp_scheduler())

                self.temp_scheduler.step()
                action = action[0, 0]
                if action == 16:
                    action = 0
                oracle_action = label_agent.step(observation, reward)
                obs, reward, done, information = maze_env.step(action)
                
                bact_id_list.append(action)
                lact_id_list.append(oracle_action)
                bact_val_list.append(maze_env.list_actions[action])
                lact_val_list.append(maze_env.list_actions[oracle_action])
                bact_type_list.append("Dagger")
                observation_list.append(obs)
                reward_list.append(reward)
                cmd_list.append(information["command"])
                
                command = information["command"]
                command = np.repeat(command, 256, axis=0)
                
                obs = np.array(obs, dtype=np.uint8)
                obs = np.transpose(obs, (2, 0, 1))
                mse_loss = np.mean((obs - pred_obs[0, 0])**2/(255*255))

                observation = obs
                sum_reward += reward
                last_cmd = information["command"]
                last_observation = observation
                last_action = action
            
            # Save the data
            # observation_list
            np.save(os.path.join(maze_output_folder, "observations.npy"), np.array(observation_list))
            # cmd_list
            np.save(os.path.join(maze_output_folder, "commands.npy"), np.array(cmd_list))
            # bact_id_list
            np.save(os.path.join(maze_output_folder, "actions_behavior_id.npy"), np.array(bact_id_list))
            # lact_id_list 
            np.save(os.path.join(maze_output_folder, "actions_label_id.npy"), np.array(lact_id_list))
            # bact_val_list 
            np.save(os.path.join(maze_output_folder, "actions_behavior_val.npy"), np.array(bact_val_list))
            # lact_val_list 
            np.save(os.path.join(maze_output_folder, "actions_label_val.npy"), np.array(lact_val_list))
            # bact_type_list 
            np.save(os.path.join(maze_output_folder, "actions_behavior_prior.npy"), np.array(bact_type_list))
            # reward_list 
            np.save(os.path.join(maze_output_folder, "rewards.npy"), np.array(reward_list))
            
            # Save the task.pkl
            pickle.dump(new_task, open(os.path.join(maze_output_folder, "task.pkl"), "wb"))
            
            # print(f"Model total Reward: {sum_reward} with total steps {maze_history.__len__()}")
            # # save maze_history to pkl
            # pickle.dump(maze_history.get_all(), open(os.path.join(maze_output_folder, "maze_history.pkl"), "wb"))
            print(f"Saved maze_history to {os.path.join(maze_output_folder, 'maze_history.pkl')}")
            maze_env.save_trajectory(os.path.join(maze_output_folder, f"trajectory.png"))
            print(f"Saved trajectory to", os.path.join(maze_output_folder, f"trajectory.png"))




import cloudpickle
@staticmethod
def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        elif msg == 'get_task_id':
          task_id = env.get_task_id()
          pipe.send(('result', task_id))
        elif msg == 'refresh_command':
          assert len(args) == 0
          tmp = env.refresh_command()
          pipe.send(('result', tmp))
        elif msg == "get_oracle_action":
          assert len(args) == 0
          action = env.get_oracle_action()
          pipe.send(('result', action))
        elif msg == "get_original_observation":
          assert len(args) == 0
          ori_obs = env.get_original_observation()
          pipe.send(('result', ori_obs))
        elif msg == 'get_folder_path':
          assert len(args) == 0
          folder_path = env.get_folder_path()
          pipe.send(('result', folder_path))
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()



class DaggerGenerator(GeneratorBase):

    def epoch_end(self, epoch_id, batch_id):
        pass
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.N_envs = self.config.N_envs # TODO:
        print(f"N_envs is: {self.N_envs}")
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.learning_steps = self.config.learning_steps 
        self.test_steps = self.config.test_steps


        
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
            
            
    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeTaskDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, world_size=self.world_size, folder_verbose=True),
            batch_size=self.N_envs, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
        self.output_folder_path = output_folder_path
        # print(f"saving in {self.output_folder_path}")
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")


    def muti_process_batch_worker(self, action_batch, envs):
        assert len(action_batch) == len(envs)
        for i in range(len(action_batch)):
            action = action_batch[i]
            env = envs[i]
            obs, reward, done, information = env.step(action)
        return 
    def sequential_batch_worker(self, action_batch, envs):
        assert len(action_batch) == len(envs)
        obs_batch = []
        reward_batch = []
        done_batch = []
        information_batch = []
        for i in range(len(action_batch)):
            action = action_batch[i]
            env = envs[i]
            obs, reward, done, information = env.step(action)
            
            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(done)
            information_batch.append(information)
        return obs_batch, reward_batch, done_batch, information_batch

    def parallel_batch_worker(self, action_batch, envs):
        """
        使用多进程并行执行环境步进
        """
        assert len(action_batch) == len(envs)
        num_envs = len(action_batch)
        
        # 创建队列用于进程间通信
        result_queue = mp.Queue()
        processes = []
        
        def worker_process(env, action, index, queue):
            """单个环境的工作进程函数"""
            try:
                obs, reward, done, information = env.step(action)
                queue.put((index, obs, reward, done, information))
            except Exception as e:
                # 错误处理：将异常信息放入队列
                queue.put((index, None, None, None, {'error': str(e)}))
        
        # 创建并启动所有进程
        for i in range(num_envs):
            p = mp.Process(
                target=worker_process,
                args=(envs[i], action_batch[i], i, result_queue)
            )
            processes.append(p)
            p.start()
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 从队列中收集结果
        results = []
        for _ in range(num_envs):
            try:
                result = result_queue.get(timeout=1)  # 1秒超时
                results.append(result)
            except:
                # 处理超时或队列为空的情况
                results.append((len(results), None, None, None, {'error': 'Timeout or queue empty'}))
        
        # 按原始索引排序结果
        results.sort(key=lambda x: x[0])
        
        # 分离各个批量的结果
        obs_batch = [result[1] for result in results]
        reward_batch = [result[2] for result in results]
        done_batch = [result[3] for result in results]
        information_batch = [result[4] for result in results]
        return obs_batch, reward_batch, done_batch, information_batch
    
    
    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2
        import xenoverse.mazeworld
        from xenoverse.mazeworld import MazeTaskSampler, Resampler, MazeStaticSampler
        from xenoverse.mazeworld.agents import OracleAgent

        max_steps = 11000
        learning_steps = self.learning_steps
        test_steps = self.test_steps
        n_range = (15,16)
        
        
        # self.N_envs = 8 # TODO:
        self.envs = []
        for i in range(self.N_envs):
            # maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
            self.envs.append(gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128)))
            
        print(f"------start with learning steps {learning_steps}------------")
        for batch_id, (batch_data, folder_names) in enumerate(self.dataloader):

            batch_obs = []
            batch_cmd = []
            batch_command2save = []
            
            dones = []
            batch_sum_reward = []
            label_agents = []
            maze_output_folders = []

            for i in range(self.N_envs):
                folder_name = folder_names[i] # batch size is N_env
                new_task_path = batch_data[i]
                new_task = pickle.load(open(new_task_path, 'rb'))
                new_task = Resampler(new_task) 
                self.envs[i].set_task(new_task)

                output_root = self.output_folder_path
                maze_output_folder = os.path.join(output_root, folder_name)
                if not os.path.exists(maze_output_folder):
                    os.makedirs(maze_output_folder)
                maze_output_folders.append(maze_output_folder)
                print(f"{batch_id} : output folder: {maze_output_folder}")
                print("-----------------------------")
                done = False
                sum_reward = 0.0
                dones.append(done)
                batch_sum_reward.append(sum_reward)  
                # label_agent = OracleAgent(maze_env=self.envs[i], render=False)
                obsi, infoi = self.envs[i].reset()
                label_agents.append(OracleAgent(maze_env=self.envs[i], render=False))
                self.envs[i].refresh_command() # To start a new command to record
                batch_obs.append(obsi.copy())            
                
                batch_command2save.append(self.envs[i].maze_core.get_command())
                # command = np.repeat(self.envs[i].maze_core.get_command(), 256, axis=0)   
                batch_cmd.append(np.repeat(self.envs[i].maze_core.get_command(), 256, axis=0))         
            
            batch_obs = np.array(batch_obs)
            batch_cmd = np.array(batch_cmd)
            batch_command2save = np.array(batch_command2save)
            batch_sum_reward = np.array(batch_sum_reward)

            self.model.module.reset()
            self.temp_scheduler = LinearScheduler(self.config.temp_scheduler, 
                                self.config.temp_value)
            
            observation_list = [batch_obs.copy()]
            cmd_list = [batch_command2save.copy()]
            bact_id_list = []
            lact_id_list = []
            bact_val_list = []
            lact_val_list = []
            bact_type_list = []
            reward_list = []
            cache = None
            for step in range(test_steps):
                if done:
                    print(f"done at step {step}")
                    break
                
                # pred_obs, actions, cache = self.model.module.policy(command, np.transpose(observation, (2, 0, 1)), cache=cache, temperature=self.temp_scheduler())
                pred_obs, actions, cache = self.model.module.MutiBatchPolicy(batch_cmd, np.transpose(batch_obs, (0, 3, 1, 2)), cache=cache, temperature=self.temp_scheduler())
                self.temp_scheduler.step()
                new_batch_action = np.array([actions[i, 0] for i in range(self.N_envs)])
                
                
                # can be parallel
                new_batch_oracle_action = []
                new_batch_obs = []
                new_batch_cmd = []
                new_batch_command2save = []
                new_batch_reward = []
                new_batch_done = []
                for i in range(self.N_envs):
                    action = new_batch_action[i]
                    if action == 16:
                        action = 0
                    oracle_action = label_agents[i].step(batch_obs[i], 0)
                    obs_i, reward_i, done_i, information_i = self.envs[i].step(action)
                    # obs_i, reward_i, done_i, information_i = self.envs[i].step(oracle_action)
                    
                    
                    new_batch_oracle_action.append(oracle_action)
                    new_batch_obs.append(obs_i.copy())
                    new_batch_command2save.append(self.envs[i].maze_core.get_command())
                    new_batch_cmd.append(np.repeat(self.envs[i].maze_core.get_command(), 256, axis=0))
                    new_batch_reward.append(reward_i)
                    new_batch_done.append(done_i)
                
                # TODO: new_batch_oracle_action, new_batch_obs, new_batch_command2save, new_batch_cmd, new_batch_reward, new_batch_done = ????
                
                
                new_batch_oracle_action = np.array(new_batch_oracle_action)
                new_batch_obs = np.array(new_batch_obs)
                new_batch_command2save = np.array(new_batch_command2save)
                new_batch_cmd = np.array(new_batch_cmd)
                new_batch_reward = np.array(new_batch_reward)
                new_batch_done = np.array(new_batch_done)
                
                bact_id_list.append(new_batch_action)
                lact_id_list.append(new_batch_oracle_action)
                # bact_val_list.append([self.envs[0].list_actions[a] for a in new_batch_action])
                # lact_val_list.append([self.envs[0].list_actions[oa] for oa in new_batch_oracle_action])
                bact_type_list.append(["Dagger" for i in range(self.N_envs)])
                observation_list.append(new_batch_obs)
                reward_list.append(new_batch_reward)
                cmd_list.append(new_batch_command2save)
                
                batch_obs = np.array(new_batch_obs, dtype=np.uint8)
                batch_cmd = new_batch_cmd
                batch_command2save = new_batch_command2save
                dones = new_batch_done
                batch_sum_reward += new_batch_reward
            
            try:
                observation_list2save = np.transpose(np.array(observation_list), (1, 0, 2, 3, 4)) # from  (N_step, N_env, C, H, W) to (N_env, N_step, C, H, W) 
                cmd_list2save = np.transpose(np.array(cmd_list), (1, 0, 2)) # from  (N_step, N_env, C) to (N_env, N_step, C) 
                bact_id_list2save = np.transpose(np.array(bact_id_list), (1, 0)) # from  (N_step, N_env) to (N_env, N_step) 
                lact_id_list2save = np.transpose(np.array(lact_id_list), (1, 0)) # from  (N_step, N_env) to (N_env, N_step) 
                # bact_val_list2save = np.transpose(np.array(bact_val_list), (1, 0, 2)) # from  (N_step, N_env) to (N_env, N_step) 
                # lact_val_list2save = np.transpose(np.array(lact_val_list), (1, 0, 2)) # from  (N_step, N_env) to (N_env, N_step) 
                bact_type_list2save = np.transpose(np.array(bact_type_list), (1, 0)) # from  (N_step, N_env) to (N_env, N_step) 
                reward_list2save = np.transpose(np.array(reward_list), (1, 0)) # from (N_step, N_env) to (N_env, N_step) 
            except:
                print(f"observation_list: {np.array(observation_list).shape}")
                print(f"cmd_list: {np.array(cmd_list).shape}")
                print(f"bact_id_list: {np.array(bact_id_list).shape}")
                print(f"lact_id_list: {np.array(lact_id_list).shape}")
                # print(f"bact_val_list: {np.array(bact_val_list).shape}")
                # print(f"lact_val_list: {np.array(lact_val_list).shape}")
                print(f"bact_type_list: {np.array(bact_type_list).shape}")
                print(f"reward_list: {np.array(reward_list).shape}")
                raise Exception("Error in transpose")
            
            print(f"Total reward: {batch_sum_reward}")
            # Save the data for every maze_env
            for i in range(self.N_envs):
                maze_output_folder = maze_output_folders[i]
                observation_list = observation_list2save[i]
                cmd_list = cmd_list2save[i]
                bact_id_list = bact_id_list2save[i]
                lact_id_list = lact_id_list2save[i]
                # bact_val_list = bact_val_list2save[i]
                # lact_val_list = lact_val_list2save[i]
                bact_type_list = bact_type_list2save[i]
                reward_list = reward_list2save[i]
                
                # observation_list
                np.save(os.path.join(maze_output_folder, "observations.npy"), np.array(observation_list))
                # cmd_list
                np.save(os.path.join(maze_output_folder, "commands.npy"), np.array(cmd_list))
                # bact_id_list
                np.save(os.path.join(maze_output_folder, "actions_behavior_id.npy"), np.array(bact_id_list))
                # lact_id_list 
                np.save(os.path.join(maze_output_folder, "actions_label_id.npy"), np.array(lact_id_list))
                # # bact_val_list 
                # np.save(os.path.join(maze_output_folder, "actions_behavior_val.npy"), np.array(bact_val_list))
                # # lact_val_list 
                # np.save(os.path.join(maze_output_folder, "actions_label_val.npy"), np.array(lact_val_list))
                # bact_type_list 
                np.save(os.path.join(maze_output_folder, "actions_behavior_prior.npy"), np.array(bact_type_list))
                # reward_list 
                np.save(os.path.join(maze_output_folder, "rewards.npy"), np.array(reward_list))
                
                # Save the task.pkl
                pickle.dump(new_task, open(os.path.join(maze_output_folder, "task.pkl"), "wb"))
                print(f"Saved maze_history to {os.path.join(maze_output_folder, 'maze_history.pkl')}")
                self.envs[i].save_trajectory(os.path.join(maze_output_folder, f"trajectory.png"))
                print(f"Saved trajectory to", os.path.join(maze_output_folder, f"trajectory.png"))



def flatten_memory(caches):
    N_mem_layer = 18
    None_count = 0
    flat_memorys = []
    for cache in caches:
        flat_layers = []
        if cache is None:
            continue
        for n_mem_layer in range(N_mem_layer):
            flat_memory = np.append(cache[n_mem_layer]['recurrent_state'][0].flatten().cpu().numpy().T, cache[n_mem_layer]['recurrent_state'][1].flatten().cpu().numpy().T)
            flat_layers.append(flat_memory)
        flat_layers = np.array(flat_layers)
        flat_memorys.append(flat_layers)
    flat_memorys = np.array(flat_memorys)
    return flat_memorys


def process_into_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, list):
        return [process_into_numpy(d) for d in data]
    elif isinstance(data, dict):
        return {k: process_into_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(process_into_numpy(d) for d in data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        assert False, f"Unsupported data type {type(data)}"
    return data


def plotLongDemo(predictions, reals, save_path):
    import os
    import PIL.Image as Image
    frame_count = 0
    whole_ARimage = None
    # To concatenate the whole images by 2xN 
    
    real = reals
    icsl_predict = predictions
    if len(real.shape) == 5:
        real = real.squeeze(0) # (B, T, C, H, W) to (T, C, H, W)
    if len(icsl_predict.shape) == 5:
        icsl_predict = icsl_predict.squeeze(0)
    N = icsl_predict.shape[0]
    assert real.shape == icsl_predict.shape, f"Shape mismatch: {real.shape} vs {icsl_predict.shape}"
    assert len(real.shape) == 4, f"Invalid shape: {real.shape}, expected (T, C, H, W)"
    for i in range(N):
        real_frames = real[i]
        icsl_frames = icsl_predict[i]
        # rotete the real_frames and icsl_frames by 90 degrees counterclockwise
        real_frames = np.transpose(real_frames, (1, 2, 0))[:, :, ::-1] # (C, H, W) to (H, W, C)
        icsl_frames = np.transpose(icsl_frames, (1, 2, 0))[:, :, ::-1] # (C, H, W) to (H, W, C)
        rotated_real_frames = np.rot90(real_frames, 1, axes=(0, 1)) # (H, W, C) to (W, H, C)
        rotated_icsl_frames = np.rot90(icsl_frames, 1, axes=(0, 1)) # (H, W, C) to (W, H, C)
        # (H, W, C) 
        concatenated_img = np.vstack((rotated_real_frames, rotated_icsl_frames))
        img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
        ARimage = np.clip(img, 0, 255).astype(np.uint8)
        # concatenate the ARimage and ARreal up and down
        if frame_count == 0:
            whole_ARimage = ARimage
        else:
            whole_ARimage = np.hstack((whole_ARimage, ARimage))
        frame_count += 1
    # use Image to save the image to PNG
    img = Image.fromarray(whole_ARimage)
    img.save(os.path.join(save_path))
    print("Save the image to " + save_path)


def video_visualization(pred_obs_list_with_initial, real, output_folder):
    # (B, T, C, H, W)
    import cv2
    video_folder = os.path.join(output_folder, f'video')
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

            
    assert len(pred_obs_list_with_initial) == len(real), f"Length mismatch: {len(pred_obs_list_with_initial)} vs {len(real)}"
    
    pred_obs_list_with_initial = process_into_numpy(pred_obs_list_with_initial)
    real = process_into_numpy(real)
    assert len(pred_obs_list_with_initial.shape) == 6, f"Invalid shape: {pred_obs_list_with_initial.shape}"
    assert len(real.shape) == 6, f"Invalid shape: {real.shape}" # (N, B, T, C, H, W)

    N = pred_obs_list_with_initial.shape[0]
    B = pred_obs_list_with_initial.shape[1]
    T = pred_obs_list_with_initial.shape[2]
    assert real.shape == pred_obs_list_with_initial.shape, f"Shape mismatch: {real.shape} vs {pred_obs_list_with_initial.shape}"
    assert B == 1, f"Batch size should be 1, we don't handle mutiple batch by now for generator, but got {B}"
    video_filename = os.path.join(video_folder, f"pred_obs_video{0}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    frame_height, frame_width = pred_obs_list_with_initial[0].shape[-2:]
    video_writer = cv2.VideoWriter(video_filename, fourcc, 10.0, (frame_width * 2, frame_height))
    frame_count = 0

    for real_frames, pred_frames in zip(real, pred_obs_list_with_initial):
        # (B, T, C, H, W) to (H, W, C) just pick up the first frame of T, and we default B=1

        real_frame = real_frames[0,0].transpose(1, 2, 0)
        pred_frame = pred_frames[0,0].transpose(1, 2, 0)
        rotated_real = cv2.rotate(real_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_pred = cv2.rotate(pred_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        concatenated_img = np.hstack((rotated_real, rotated_pred))
        # save the concatenated image

        img = np.clip(concatenated_img, 0, 255).astype(np.uint8)
        if frame_count % 100 == 0:
            if T > 1:
                ARimageFolder = os.path.join(video_folder, f"ARimage_{frame_count}")
                if not os.path.exists(ARimageFolder):
                    os.makedirs(ARimageFolder)
                whole_ARimage = None
                for i in range(T):
                    ARimage = pred_frames[0,i].transpose(1, 2, 0)
                    rotated_ARimage = cv2.rotate(ARimage, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ARimage = np.clip(rotated_ARimage, 0, 255).astype(np.uint8)
                    ARreal = real_frames[0,i].transpose(1, 2, 0)
                    rotated_ARreal = cv2.rotate(ARreal, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ARimage = np.clip(rotated_ARreal, 0, 255).astype(np.uint8)
                    # concatenate the ARimage and ARreal up and down
                    ARconcatenated_img = np.vstack((rotated_ARreal, rotated_ARimage))
                    if i == 0:
                        whole_ARimage = ARconcatenated_img
                    else:
                        whole_ARimage = np.hstack((whole_ARimage, ARconcatenated_img))
                    ARimage = np.clip(ARconcatenated_img, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(ARimageFolder, f"ARframe_{i}.png"), ARimage)
                cv2.imwrite(os.path.join(ARimageFolder, f"whole_ARimage.png"), whole_ARimage)
                
            cv2.imwrite(os.path.join(video_folder, f"frame_{frame_count}.png"), img)
        frame_count += 1
        video_writer.write(img)

    
    video_writer.release() 

    print(f"Saved video with {len(real)} frames to {video_filename}")


class prediction_coding_generator(GeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        self.K_step = self.config.K_step
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"


    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
    def __call__(self, epoch_id, rank):
        import cv2
        # nohup python -m projects.MazeWorld.generator_test ./generator-configs/blockTest.yaml > static_cache.log 2>&1 &
        batch_size = 1 # TODO
        pred_len = self.pred_len
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape ")
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
            loss_records = []
            pred_records = []
            real_records = []

            for in_context_len in [1, 10, 100, 1000]:
                pred_len = 1
                effect_len = 2
                print(f"pred_len: {pred_len}")
                print(f"in_context_len: {in_context_len}")
                mask_points = range(in_context_len + 1, min(in_context_len + self.end_position, states.shape[1] - 1), 10)
                print(f"record points: {mask_points}")
                # folder_count = 0
                output_folder_pred = os.path.join(output_folder_path, f"context_{in_context_len}")
                if not os.path.exists(output_folder_pred):
                    os.makedirs(output_folder_pred)
                
                map_loss_record = []
                
                for check_point in mask_points: # the check point will be masked by the prediction of check_point - 1
                    history_cache = None # TODO
                    history_before_cache = None
                    last_cache = None
                    start_point = check_point - in_context_len
                    end_point = min(check_point + effect_len, states.shape[1] - 1)
                    loss_record = {}
                    inference_record = {}
                    pred_len = 1
                    print(f"check_point: {check_point}, start_point: {start_point}, end_point: {end_point}")
                    for i in range(start_point, end_point):
                        if i == check_point - 1:
                            pred_len = self.K_step # To change the K when predicting the check point
                        end = min(i, states.shape[1] - 1)
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
                        print(f"check_point {i} with pred_obs_list shape: {pred_obs_list.shape}")
                        print(f"sum of real: {torch.sum(real)}")
                        mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                                loss_type="mse",
                                                gt=real, 
                                                need_cnt=True,
                                                )
                        mse_loss = mse_loss/255/255
                        print(f"check_point {i} with mse_loss: {mse_loss/cnt}, cnt: {cnt}")
                            
                        
                        if i == check_point - 1: # the check point will be masked by the prediction of check_point - 1
                            print("record the history cache")
                            history_before_cache = history_cache.copy()
                            

                        if i >= check_point - 1:
                            real = states[i+1:i+1+pred_len]
                            # mse loss for every state
                            loss_record[i] = mse_loss.detach().cpu().numpy()
                            inference_record[i] = pred_obs_list[:, 0]

                    # the check point will be masked by the prediction of check_point - 1
                    masked_loss_record = {}
                    state_copy = states.clone() 
                    state_copy[:, check_point:check_point+1] = inference_record[check_point - 1]
                    history_cache = history_before_cache
                    effect_loss_sum = 0
                    masked_loss_sum = 0
                    for i in range(check_point, end_point):
                        end = i
                        pred_obs_list, history_cache = self.model.module.generate_states_only(
                                prompts=commands[:, end:end+pred_len],
                                current_observation=state_copy[:, end:end+1], 
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
                        real = state_copy[:, end+1:end+1+pred_len]
                        print(f"sum of real: {torch.sum(real)}")
                        mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                                loss_type="mse",
                                                gt=real, 
                                                need_cnt=True,
                                                )
                        mse_loss = mse_loss/255/255
                        masked_loss_record[i] = mse_loss
                        masked_loss_sum += mse_loss
                        effect_loss_sum += loss_record[i]
                    print(f"masked_loss_sum: {masked_loss_sum}, effect_loss_sum: {effect_loss_sum}")
                    relative_loss_diff = (masked_loss_sum - effect_loss_sum) / effect_loss_sum
                    relative_loss_diff = relative_loss_diff.detach().cpu().numpy()
                    # loss_record_context[check_point] = (masked_loss_sum, effect_loss_sum, relative_loss_diff)
                    data_pair = (loss_record[check_point - 1], relative_loss_diff)
                    np.save(os.path.join(output_folder_pred, f"point_{check_point}.npy"), data_pair)
                    print(f"Saved point to {os.path.join(output_folder_pred, f'point_{check_point}.npy')}")
                    

    def epoch_end(self, epoch_id, batch_id):
        pass
        # stat_res = self.stat()
        # if not hasattr(self, 'logger'):
        #     self.logger = None
        # if(self.logger is not None):
        #     self.logger(stat_res["validate_worldmodel_raw"]["mean"],
        #             epoch=epoch_id)



class fixed_context_generator(GeneratorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        # self.K_step = self.config.K_step
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"


    def preprocess(self):
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main, max_maze = self.max_maze, folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")
    def __call__(self, epoch_id, rank):
        import cv2
        # nohup python -m projects.MazeWorld.generator_test ./generator-configs/blockTest.yaml > static_cache.log 2>&1 &
        batch_size = 1 # TODO
        pred_len = self.pred_len
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[0]
                sub_name = folder_name.split("/")[1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape ")
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
            if not isinstance(self.in_context_len, list):
                self.in_context_len = [self.in_context_len]
            for in_context_len in [5]:
                loss_records = []
                pred_records = []
                real_records = []
                pred_len = 1
                print(f"pred_len: {pred_len}")
                print(f"in_context_len: {in_context_len}")

                output_folder_pred = os.path.join(output_folder_path, f"context_{in_context_len}")
                if not os.path.exists(output_folder_pred):
                    os.makedirs(output_folder_pred)
                
                map_loss_record = []
                
                for check_point in self.record_points: # the check point will be masked by the prediction of check_point - 1
                    history_cache = None # TODO
                    start_point = max(check_point - in_context_len, 0)
                    end_point = min(check_point + 1, states.shape[1] - 1)
                    loss_record = []
                    inference_record = {}
                    pred_len = 1
                    print(f"check_point: {check_point}, start_point: {start_point}, end_point: {end_point}")
                    for i in range(start_point, end_point):
                        # if i == check_point - 1:
                        #     pred_len = self.K_step # To change the K when predicting the check point
                        end = min(i, states.shape[1] - 1)
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
                        print(f"check_point {i} with pred_obs_list shape: {pred_obs_list.shape}")
                        print(f"sum of real: {torch.sum(real)}")
                        mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                                                loss_type="mse",
                                                gt=real, 
                                                need_cnt=True,
                                                )
                        mse_loss = mse_loss/255/255
                        if i == check_point:
                            pred_records.append(pred_obs_list[0].cpu().detach().numpy())
                            real_records.append(real.cpu().detach().numpy())
                            loss_records.append(mse_loss.cpu().detach().numpy())
                            print(f"check_point {check_point} with mse_loss: {loss_records[-1]})")

                    
                    np.save(os.path.join(output_folder_pred, f"loss_{check_point}.npy"), loss_records[-1])
                    print(f"Saved point to {os.path.join(output_folder_pred, f'loss_{check_point}.npy')}")
                    np.save(os.path.join(output_folder_pred, f"pred_{check_point}.npy"), pred_records[-1])
                    print(f"Saved point to {os.path.join(output_folder_pred, f'pred_{check_point}.npy')}")
                    np.save(os.path.join(output_folder_pred, f"real_{check_point}.npy"), real_records[-1])
                    print(f"Saved point to {os.path.join(output_folder_pred, f'real_{check_point}.npy')}")

                    

    def epoch_end(self, epoch_id, batch_id):
        stat_res = self.stat()
        if not hasattr(self, 'logger'):
            self.logger = None
        if(self.logger is not None):
            self.logger(stat_res["validate_worldmodel_raw"]["mean"],
                    epoch=epoch_id)



class general_generator(GeneratorBase): #TODO   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        # self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval

        # self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        # self.start_points = [i for i in range(0, 9000, 2000)]
        # self.record_points = []
        # for s in self.start_points:
        #     for i in range(s, s+1800):
        #         self.record_points.append(i)
        self.record_points = np.array([1, 10, 100, 1000, 8000])
        # np.array([i for i in range(10000)]) # range(1, 9000, 10) # [1, 100, 1000, 5000]
        print(f"record points: {self.record_points}")
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        
        # if self.end_position > self.config.seq_len_causal:
        #     assert False, "end_position should be smaller than seq_len_causal"
        
        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

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
        video_generate = False #  True
        # history_cache = None
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            print(folder_name)
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[-2]
                sub_name = folder_name.split("/")[-1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))
                output_folder_path = os.path.join(self.output_root, parent_folder, sub_name)
            else:
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

            # last_real = states[:, 0:1].clone().cpu().detach().numpy()
            for checkpoint_id in range(0, self.end_position):
                # if checkpoint_id in self.start_points:
                #     print(f"checkpoint_id: {checkpoint_id} start_points")
                #     history_cache = None
                #     self.model.module.reset()
                if checkpoint_id in self.record_points:
                    pred_len = 10
                else:
                    pred_len = 1
                end = min(checkpoint_id, states.shape[1] - 1)
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
                # mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                #                         loss_type="mse",
                #                         gt=real, 
                #                         need_cnt=True,
                #                         )
                # calculate mse loss by T, the pred_obs_list is (B, T, C, H, W) loss is (B, T)
                mse_loss = np.mean(np.square(pred_obs_list.cpu().detach().numpy() - real.cpu().detach().numpy()), axis=(2, 3, 4))
                
                
                mse_loss = mse_loss/255/255
                # print(f"check_point {checkpoint_id} with mse_loss: {mse_loss/cnt}, cnt: {cnt}")
                # loss_records.append(mse_loss.detach().numpy()/cnt)  
                import copy
                if checkpoint_id in self.record_points:
                    if cache_generate == True:
                        np.save(os.path.join(output_folder_path, f"cache_{checkpoint_id}.npy"), history_cache)
                        print(f"Saved cache to {os.path.join(output_folder_path, f'cache_{checkpoint_id}.npy')}")
                    # if o_generate == True:
                    #     o_list = self.model.module.get_o_list()
                    #     o_list = copy.deepcopy(o_list)
                    #     o_list = o_list.cpu().detach().numpy()
                    #     np.save(os.path.join(output_folder_path, f"o_list_{checkpoint_id}.npy"), o_list)
                    #     print(f"Saved o_list to {os.path.join(output_folder_path, f'o_list_{checkpoint_id}.npy')}")
                    pred_records.append(pred_obs_list.cpu().detach().numpy())
                    real = real.clone().cpu().detach().numpy()
                    real_records.append(real)
                    loss_records.append(mse_loss)
                    print(pred_obs_list.cpu().detach().numpy().shape, real.shape, mse_loss)
                    # print(pred_obs_list.cpu().detach().numpy().shape, real.shape)
                    if "procthor" in output_folder_path:
                        plotLongDemo(np.rot90(pred_obs_list.cpu().detach().numpy(), k=2, axes=(2, 3)), 
                            np.rot90(real, k=2, axes=(2, 3)), 
                            os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))
                    else:
                        plotLongDemo(pred_obs_list.cpu().detach().numpy(), 
                            real, 
                            os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))
            loss_records = np.array(loss_records)

            # save the loss record to npy 
            np.save(os.path.join(output_folder_path, f"losses.npy"), loss_records)
            print(f"Saved losses to {os.path.join(output_folder_path, f'losses.npy')}")
            
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)
            # save the pred_records and real_records to npy
            # np.save(os.path.join(output_folder_path, f"pred_records.npy"), pred_records)
            # print(f"Saved pred_records to {os.path.join(output_folder_path, f'pred_records.npy')}")
            # np.save(os.path.join(output_folder_path, f"real_records.npy"), real_records)
            # print(f"Saved real_records to {os.path.join(output_folder_path, f'real_records.npy')}")

            if video_generate == True:
                
                video_visualization(pred_records, real_records, output_folder_path)

        # loss_batch = np.array(loss_batch) # (B, N_record)
        # bsz = loss_batch.shape[0]
        # seg_num = loss_batch.shape[1] // self.downsample_length
        # valid_seq_len = seg_num * self.downsample_length
        # loss_batch = np.mean(loss_batch[:, :valid_seq_len].reshape(bsz, seg_num, -1), axis=-1)
        # self.stat.gather(self.device,
        #         validate_worldmodel_raw=loss_batch[0], 
        #         count=bsz)

    def epoch_end(self, epoch_id, batch_id):
        pass
        # stat_res = self.stat()
        # if not hasattr(self, 'logger'):
        #     self.logger = None
        # if(self.logger is not None):
        #     self.logger(stat_res["validate_worldmodel_raw"]["mean"],
        #             epoch=epoch_id)
        # if(self.extra_info is not None):
        #     if(self.extra_info.lower() == 'validate' and self.main):
        #         if not os.path.exists(self.config.output):
        #             os.makedirs(self.config.output)
        #         for key_name in stat_res:
        #             res_text = string_mean_var(self.downsample_length, stat_res[key_name])
        #             file_path = f'{self.config.output}/result_{key_name}.txt'
        #             if os.path.exists(file_path):
        #                 os.remove(file_path)
        #             with open(file_path, 'w') as f_model:
        #                 f_model.write(res_text)



class context_swapping_generator(GeneratorBase): #TODO   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        # self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval

        self.record_points = np.array([1, 10, 100, 1000, 8000])
        # np.array([i for i in range(10000)]) # range(1, 9000, 10) # [1, 100, 1000, 5000]
        print(f"record points: {self.record_points}")
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"

        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

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
        import random
        batch_size = 1 # TODO
        pred_len = self.pred_len
        loss_batch = []
        cache_generate = False
        o_generate = False
        video_generate = False #  True
        # history_cache = None

        def partial_shuffle_sequence(original, chaos_rate, use_numpy=False):
            original_seq = original #  list(range(0, N))
            N = len(original_seq)
            
            if chaos_rate == 0:
                return original_seq
            elif chaos_rate == 1:
                # 完全打乱
                if use_numpy:
                    return np.random.permutation(original_seq).tolist()
                else:
                    random.shuffle(original_seq)
                    return original_seq
            
            # 计算需要打乱的元素数量
            n_to_shuffle = int(chaos_rate * N)
            # print(n_to_shuffle)
            return _numpy_partial_shuffle(original_seq, n_to_shuffle)


        def _numpy_partial_shuffle(seq, n_to_shuffle):
            n = len(seq)
            if n_to_shuffle == 0 or n_to_shuffle == 1:
                return seq
            
            seq_array = np.array(seq)
            seq_orig = seq_array.copy()
            
            # 随机选择要打乱的位置[1](@ref)
            pos_to_shuffle = np.random.permutation(n)[:n_to_shuffle]
            
            # 打乱这些下标：加一个随机偏移确保彻底打乱[1](@ref)
            meta_index = np.arange(n_to_shuffle)
            rnd_shift = np.random.randint(1, n_to_shuffle)  # 随机偏移
            shuffled_pos = pos_to_shuffle[(meta_index + rnd_shift) % n_to_shuffle]
            
            # 确保所有选中的位置都被打乱[1](@ref)
            assert (pos_to_shuffle != shuffled_pos).all()
            
            # 执行打乱操作
            seq_array[pos_to_shuffle] = seq_orig[shuffled_pos]
            
            return seq_array.tolist()

        def verify_shuffle_rate(original, shuffled):
            """验证实际打乱比例"""
            n = len(original)
            changed_positions = sum(1 for i in range(n) if original[i] != shuffled[i])
            return changed_positions / n


        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            print(folder_name)
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[-2]
                sub_name = folder_name.split("/")[-1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))
                output_folder_path = os.path.join(self.output_root, parent_folder, sub_name)
            else:
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
            
            
            shuffle_rate = 0
            print(f"shuffle_rate: ", shuffle_rate)
            # context-swapping between self.record_points 
            shuffled_index = list(range(0, self.record_points[0]))
            for i in range(len(self.record_points[:]) - 1):
                shuffled_index.extend(list(range(self.record_points[i], self.record_points[i] + 2)))
                to_extend = partial_shuffle_sequence(list(range(self.record_points[i] + 2, self.record_points[i+1])), shuffle_rate, use_numpy=True)
                shuffled_index.extend(to_extend)
            shuffled_index.extend(list(range(self.record_points[-1], states.shape[1])))    
            shuffled_index = np.array(shuffled_index)
            for rp in self.record_points:
                assert shuffled_index[rp] == rp, print(shuffled_index[rp-5:rp+5], rp)
            states = states[:, shuffled_index]
            for checkpoint_id in range(0, self.end_position):
                if checkpoint_id in self.record_points:
                    pred_len = 1
                else:
                    pred_len = 1
                end = min(checkpoint_id, states.shape[1] - 1)
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
                # mse_loss, cnt = weighted_loss(pred_obs_list.cpu(), 
                #                         loss_type="mse",
                #                         gt=real, 
                #                         need_cnt=True,
                #                         )
                # calculate mse loss by T, the pred_obs_list is (B, T, C, H, W) loss is (B, T)
                mse_loss = np.mean(np.square(pred_obs_list.cpu().detach().numpy() - real.cpu().detach().numpy()), axis=(2, 3, 4))
                
                mse_loss = mse_loss/255/255
                import copy
                if checkpoint_id in self.record_points:
                    pred_records.append(pred_obs_list.cpu().detach().numpy())
                    real = real.clone().cpu().detach().numpy()
                    real_records.append(real)
                    loss_records.append(mse_loss)
                    print(pred_obs_list.cpu().detach().numpy().shape, real.shape, mse_loss)
                    if "procthor" in output_folder_path:
                        plotLongDemo(np.rot90(pred_obs_list.cpu().detach().numpy(), k=2, axes=(2, 3)), 
                            np.rot90(real, k=2, axes=(2, 3)), 
                            os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))
                    else:
                        plotLongDemo(pred_obs_list.cpu().detach().numpy(), 
                            real, 
                            os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))
            loss_records = np.array(loss_records)

            # save the loss record to npy 
            np.save(os.path.join(output_folder_path, f"losses.npy"), loss_records)
            print(f"Saved losses to {os.path.join(output_folder_path, f'losses.npy')}")
            
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)
            # save the pred_records and real_records to npy
            # np.save(os.path.join(output_folder_path, f"pred_records.npy"), pred_records)
            # print(f"Saved pred_records to {os.path.join(output_folder_path, f'pred_records.npy')}")
            # np.save(os.path.join(output_folder_path, f"real_records.npy"), real_records)
            # print(f"Saved real_records to {os.path.join(output_folder_path, f'real_records.npy')}")

            if video_generate == True:
                video_visualization(pred_records, real_records, output_folder_path)
                
    def epoch_end(self, epoch_id, batch_id):
        pass




            

class stupid_general_generator(GeneratorBase): #TODO   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        # self.data_root = self.config.data_path
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval

        self.record_points = [i for i in range(10000)] # np.array([1, 10, 100, 1000, 8000])
        print(f"record points: {self.record_points}")
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        
        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

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
        batch_size = 1 # TODO
        pred_len = self.pred_len
        loss_batch = []
        cache_generate = False
        video_generate = False #  True
        # history_cache = None
        for batch_id, (batch_data, folder_name) in enumerate(self.dataloader):
            print(folder_name)
            folder_name = folder_name[0] # batch size is 1
            if len(folder_name.split("/")) > 1:
                parent_folder = folder_name.split("/")[-2]
                sub_name = folder_name.split("/")[-1]
                if not os.path.exists(os.path.join(self.output_root, parent_folder)):
                    os.makedirs(os.path.join(self.output_root, parent_folder))
                output_folder_path = os.path.join(self.output_root, parent_folder, sub_name)
            else:
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
            
            for checkpoint_id in range(0, self.end_position):
                
                if checkpoint_id < 10:
                    fixed_context_window = 0
                elif checkpoint_id < 100:
                    fixed_context_window = 1
                elif checkpoint_id < 400:
                    fixed_context_window = 5
                else:
                    fixed_context_window = None
                
                
                
                if checkpoint_id in self.record_points:
                    pred_len = 1
                else:
                    pred_len = 1
                end = min(checkpoint_id, states.shape[1] - 1)
                
                if fixed_context_window is not None:
                    # start to learn the fixed context window
                    start = max(0, end - fixed_context_window)
                    history_cache = None
                    for id2learn in range(start, end+1):
                        pred_obs_list, history_cache = self.model.module.generate_states_only(
                                prompts=commands[:, id2learn:id2learn+pred_len],
                                current_observation=states[:, id2learn:id2learn+1], 
                                action_trajectory=actions[:, id2learn:id2learn+pred_len],
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
                else:
                    id2learn = end
                    pred_obs_list, history_cache = self.model.module.generate_states_only(
                            prompts=commands[:, id2learn:id2learn+pred_len],
                            current_observation=states[:, id2learn:id2learn+1], 
                            action_trajectory=actions[:, id2learn:id2learn+pred_len],
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
                mse_loss = np.mean(np.square(pred_obs_list.cpu().detach().numpy() - real.cpu().detach().numpy()), axis=(2, 3, 4))
                
                
                mse_loss = mse_loss/255.0/255.0
                import copy
                if checkpoint_id in self.record_points:
                    if cache_generate == True:
                        np.save(os.path.join(output_folder_path, f"cache_{checkpoint_id}.npy"), history_cache)
                        print(f"Saved cache to {os.path.join(output_folder_path, f'cache_{checkpoint_id}.npy')}")
                    pred_records.append(pred_obs_list.cpu().detach().numpy())
                    real = real.clone().cpu().detach().numpy()
                    real_records.append(real)
                    loss_records.append(mse_loss)
                    print(pred_obs_list.cpu().detach().numpy().shape, real.shape, mse_loss)
                    if "procthor" in output_folder_path:
                        plotLongDemo(np.rot90(pred_obs_list.cpu().detach().numpy(), k=2, axes=(2, 3)), 
                            np.rot90(real, k=2, axes=(2, 3)), 
                            os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))
                    else:
                        plotLongDemo(pred_obs_list.cpu().detach().numpy(), 
                            real, 
                            os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))
            loss_records = np.array(loss_records)
            # save the loss record to npy 
            np.save(os.path.join(output_folder_path, f"losses.npy"), loss_records)
            print(f"Saved losses to {os.path.join(output_folder_path, f'losses.npy')}")
            
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)
            if video_generate == True:
                
                video_visualization(pred_records, real_records, output_folder_path)

    def epoch_end(self, epoch_id, batch_id):
        pass
     



class ARgeneral_generator(GeneratorBase): #TODO   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.pred_len = self.config.pred_len
        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        self.record_points = [1, 100, 1000, 9000]
        print(f"record points: {self.record_points}")
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None

        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output folder {self.output_root}")
        else:
            assert False, "output_root is required for general_generator"
        if self.end_position > self.config.seq_len_causal:
            assert False, "end_position should be smaller than seq_len_causal"
        
        self.logger_keys = ["validate_worldmodel_raw"]
        self.stat = DistStatistics(*self.logger_keys)
        if(self.config.has_attr("downsample_length")):
            self.downsample_length = self.config.downsample_length
        else:
            self.downsample_length = 10

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
        video_generate = False
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
            # last_real = states[:, 0:1].clone().cpu().detach().numpy()
            for checkpoint_id in range(0, self.end_position):
                # if checkpoint_id in self.start_points:
                #     print(f"checkpoint_id: {checkpoint_id} start_points")
                #     history_cache = None
                #     self.model.module.reset()
                if checkpoint_id in self.record_points:
                    pred_len = 10
                    pred_obs_list, act_out, _ = self.model.module.generate_states_and_action(
                    prompts=commands[:, end:end+pred_len],
                    current_observation=states[:, end:end+1], 
                    future_steps=pred_len,
                    history_observation=None, #states[start:end],
                    history_action=None, #actions[start:end],
                    history_update_memory=False, 
                    autoregression_update_memory=False, # TOTEST
                    cache=history_cache,
                    single_batch=True,
                    history_single_step=False,
                    # future_single_step=False,
                    raw_images=True,
                    need_numpy=False)
                    real = states[:, end:end+1+pred_len]
                    assert pred_obs_list.shape[1] == pred_len + 1, f"pred_obs_list shape: {pred_obs_list.shape}, pred_len: {pred_len}"
                    assert real.shape[1] == pred_len + 1, f"real shape: {real.shape}, pred_len: {pred_len}"
                    plotLongDemo(pred_obs_list.cpu().detach().numpy(), 
                        real.clone().cpu().detach().numpy(), 
                        os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))

                

                pred_len = 1
                end = min(checkpoint_id, states.shape[1] - 1)
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
                # print(f"check_point {checkpoint_id} with mse_loss: {mse_loss/cnt}, cnt: {cnt}")
                loss_records.append(mse_loss.detach().numpy()/cnt)  
                import copy
                if checkpoint_id in self.record_points:
                    if cache_generate == True:
                        np.save(os.path.join(output_folder_path, f"cache_{checkpoint_id}.npy"), history_cache)
                        print(f"Saved cache to {os.path.join(output_folder_path, f'cache_{checkpoint_id}.npy')}")
                    # if o_generate == True:
                    #     o_list = self.model.module.get_o_list()
                    #     o_list = copy.deepcopy(o_list)
                    #     o_list = o_list.cpu().detach().numpy()
                    #     np.save(os.path.join(output_folder_path, f"o_list_{checkpoint_id}.npy"), o_list)
                    #     print(f"Saved o_list to {os.path.join(output_folder_path, f'o_list_{checkpoint_id}.npy')}")
                    pred_records.append(pred_obs_list.cpu().detach().numpy())
                    real = real.clone().cpu().detach().numpy()
                    real_records.append(real)
                    # print(pred_obs_list.cpu().detach().numpy().shape, real.shape)
                    # plotLongDemo(pred_obs_list.cpu().detach().numpy(), 
                    #     real, 
                    #     os.path.join(output_folder_path, f"demo_{checkpoint_id}.png"))

            loss_records = np.array(loss_records)

            # save the loss record to npy 
            np.save(os.path.join(output_folder_path, f"losses.npy"), loss_records)
            print(f"Saved losses to {os.path.join(output_folder_path, f'losses.npy')}")
            
            loss_batch.append(loss_records)
            real_records = np.array(real_records)
            pred_records = np.array(pred_records)
            # save the pred_records and real_records to npy
            # np.save(os.path.join(output_folder_path, f"pred_records.npy"), pred_records)
            # print(f"Saved pred_records to {os.path.join(output_folder_path, f'pred_records.npy')}")
            # np.save(os.path.join(output_folder_path, f"real_records.npy"), real_records)
            # print(f"Saved real_records to {os.path.join(output_folder_path, f'real_records.npy')}")

            if video_generate == True:
                
                video_visualization(pred_records, real_records, output_folder_path)

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
            