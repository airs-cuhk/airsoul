import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager
from airsoul.utils import noam_scheduler, LinearScheduler
from airsoul.dataloader import MazeDataSet, PrefetchDataLoader

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

    def preprocess(self):
        if(self.is_training):
            self.sigma_scheduler = LinearScheduler(self.config.sigma_scheduler, 
                                                   self.config.sigma_value)
            self.lambda_scheduler = LinearScheduler(self.config.lambda_scheduler, 
                                                    self.config.lambda_value)
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_vae, verbose=self.main),
            batch_size=self.config.batch_size_vae,
            rank=self.rank,
            world_size=self.world_size
            )
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_vae_stop')):
            if(epoch_id >= self.config.epoch_vae_stop):
                return False
        return True

    def compute(self, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, bev_arr,
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
                    count=loss["count"])
            
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["reconstruction_error"]["mean"], 
                        stat_res["kl_divergence"]["mean"], 
                        epoch=epoch_id)

@EpochManager
class MazeEpochCausal:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MazeDataSet
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
            self.logger_keys = ["validate_worldmodel_raw",
                        "validate_worldmodel_latent",
                        "validate_policymodel"]
            self.stat = DistStatistics(*self.logger_keys)
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
            self.reduce_dim = None
            
    def valid_epoch(self, epoch_id): # Add epoch control for VAE training
        if(self.config.has_attr('epoch_causal_start')):
            if(epoch_id < self.config.epoch_causal_start):
                return False
        return True

    def preprocess(self):
        # use customized dataloader
        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, self.config.seq_len_causal, verbose=self.main),
            batch_size=self.config.batch_size_causal,
            rank=self.rank,
            world_size=self.world_size
            )

    def compute(self, obs_arr, behavior_actid_arr, label_actid_arr, 
                behavior_act_arr, label_act_arr, 
                rew_arr, bev_arr,
                epoch_id=-1, 
                batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, seg_obs, seg_behavior_act, seg_label_act, seg_bev in segment_iterator(
                                self.config.seq_len_causal, self.config.seg_len_causal, self.device, 
                                (obs_arr, 1), behavior_actid_arr, label_actid_arr, bev_arr):

            # Permute (B, T, H, W, C) to (B, T, C, H, W)
            seg_obs = seg_obs.permute(0, 1, 4, 2, 3)
            seg_bev = seg_bev.permute(0, 1, 4, 2, 3)

            loss = self.model.module.sequential_loss(
                                    seg_obs, 
                                    seg_behavior_act, 
                                    seg_label_act, 
                                    seg_bev,
                                    state_dropout=0.20,
                                    use_loss_weight=self.is_training,
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
            loss_wm_r = torch.cat([loss["wm-raw"] / loss["count_wm"] for loss in losses], dim=1)
            loss_wm_l = torch.cat([loss["wm-latent"] / loss["count_wm"] for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / loss["count_pm"] for loss in losses], dim=1)
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
        
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_worldmodel_raw"]["mean"], 
                        stat_res["validate_worldmodel_latent"]["mean"], 
                        stat_res["validate_policymodel"]["mean"],
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
