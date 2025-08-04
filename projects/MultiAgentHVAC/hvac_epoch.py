import os
import torch
import numpy

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import DistStatistics, downsample
from airsoul.utils import EpochManager, GeneratorBase, Logger
from airsoul.dataloader import MultiAgentDataSetVetorized

def string_mean_var(downsample_length, res):
    string=""
    if(numpy.size(res["mean"]) > 1):
        for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
            string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    else:
        string =  f'{0}\t{res["mean"]}\t{res["bound"]}\n'
    return string

@EpochManager
class HVACEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=MultiAgentDataSetVetorized
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "loss_worldmodel_state", 
                        "loss_worldmodel_other_agent",
                        "loss_worldmodel_reward", 
                        "loss_policymodel",
                        "entropy"]
            self.stat = DistStatistics()
            self.reduce = 1
        else:
            self.logger_keys = ["validation_state_pred", 
                        "validation_other_agent_pred",
                        "validation_reward_pred", 
                        "validation_policy",
                        "validation_entropy"]
            self.stat = DistStatistics()
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100
        if(self.config.has_attr('state_dropout')):
            self.state_dropout = self.config.state_dropout
        else:
            self.state_dropout = 0.20
        if(self.config.has_attr('reward_dropout')):
            self.reward_dropout = self.config.reward_dropout
        else:
            self.reward_dropout = 0.20

    def compute(self, seq_arr, label_arr,
                        epoch_id=-1, 
                        batch_id=-1):
        """
        Defining the computation function for each batch
        """
        state_dropout = 0.0
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"
            state_dropout = self.state_dropout
        else:
            state_dropout = 0.0

        losses = []
        for sub_idx, seq, label in segment_iterator(self.config.seq_len, self.config.seg_len, self.device, seq_arr, label_arr):
            loss = self.model.module.sequential_loss(
                    seq, 
                    label, 
                    use_loss_weight=self.is_training,
                    update_memory=True,
                    reduce_dim=self.reduce)
            losses.append(loss)
            obs_pre_step = loss["count_s"]/loss["count_p"]
            agent_pre_step = loss["count_a"]/loss["count_p"]
            if(self.is_training):
                syn_loss = (self.config.lossweight_worldmodel_states * loss["wm_obs"] / obs_pre_step
                        + self.config.lossweight_worldmodel_actions * loss["wm_agent"] / agent_pre_step
                        + self.config.lossweight_policymodel * loss["policy"]
                        + self.config.lossweight_worldmodel_rewards * loss["reward"]
                        + self.config.lossweight_entropy * loss["ent"]
                        + self.config.lossweight_l2 * loss["causal-l2"])
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    loss_worldmodel_state = loss["wm_obs"] / loss["count_s"],
                    loss_worldmodel_other_agent = loss["wm_agent"] / loss["count_a"],
                    loss_worldmodel_reward = loss["reward"] / loss["count_p"],
                    loss_policymodel = loss["pm"] / loss["count_p"],
                    entropy = -loss["ent"] / loss["count_p"],
                    count = loss["count_p"])
                
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["loss_worldmodel_state"]["mean"], 
                        stat_res["loss_worldmodel_other_agent"]["mean"],
                        stat_res["loss_worldmodel_reward"]["mean"], 
                        stat_res["loss_policymodel"]["mean"], 
                        stat_res["entropy"]["mean"],
                        epoch=epoch_id,
                        iteration=batch_id)
        else:
            loss_wm_s = torch.cat([loss["wm_obs"] / torch.clamp_min(loss["count_s"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_a = torch.cat([loss["wm_agent"] / torch.clamp_min(loss["count_a"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_wm_r = torch.cat([loss["wm-r"] / torch.clamp_min(loss["count_p"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_pm = torch.cat([loss["pm"] / torch.clamp_min(loss["count_p"], 1.0e-3) 
                    for loss in losses], dim=1)
            loss_ent = torch.cat([-loss["ent"] / torch.clamp_min(loss["count_p"], 1.0e-3) 
                    for loss in losses], dim=1)
            counts = torch.cat([loss["count_p"] for loss in losses], dim=1)

            bsz = loss_wm_s.shape[0]

            loss_wm_s = downsample(loss_wm_s, self.downsample_length * obs_pre_step)
            loss_wm_a = downsample(loss_wm_a, self.downsample_length * obs_pre_step)
            loss_wm_r = downsample(loss_wm_r, self.downsample_length)
            loss_pm = downsample(loss_pm, self.downsample_length)
            loss_ent = downsample(loss_ent, self.downsample_length)
            counts = downsample(counts, self.downsample_length)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validation_state_pred=loss_wm_s[i], 
                        validation_other_agent_pred=loss_wm_a[i],
                        validation_reward_pred=loss_wm_r[i], 
                        validation_policy=loss_pm[i],
                        validation_entropy=loss_ent[i],
                        count=counts[i])
    
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validation_state_pred"]["mean"], 
                        stat_res["validation_other_agent_pred"]["mean"],
                        stat_res["validation_reward_pred"]["mean"], 
                        stat_res["validation_policy"]["mean"],
                        stat_res["validation_entropy"]["mean"],
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