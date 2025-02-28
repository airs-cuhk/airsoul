import os
import torch
import torch.optim as optim

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager
from airsoul.dataloader import LMDataSet

def string_mean_var(downsample_length, res):
    string=""
    for i, (xm,xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

@EpochManager
class LMEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType=LMDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate", 
                        "train_perplexity"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.reduce = 1
        else:
            self.logger_keys = ["validate_perplexity"]
            self.stat = DistStatistics(*self.logger_keys)
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 100

    def compute(self, feas, labs, epoch_id=-1, batch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, fea, lab in segment_iterator(
                    self.config.seq_len, self.config.seg_len, self.device, 
                    feas, labs):
            loss = self.model.module.perplexity(
                    fea, lab,
                    use_loss_weight=self.is_training,
                    reduce_dim=self.reduce) # Do not use loss weight for evaluation
            losses.append(loss)
            if(self.is_training):
                syn_loss = loss["perplexity"]
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    train_perplexity=syn_loss / loss["count"],
                    count = loss["count"])
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["train_perplexity"]["mean"],
                        epoch=epoch_id,
                        iteration=batch_id)
        else:
            perpl = torch.cat([loss["perplexity"] / loss["count"] for loss in losses], dim=1)
            counts = torch.cat([loss["count"] for loss in losses], dim=1)

            bsz = perpl.shape[0]
            seg_num = perpl.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            perpl = torch.mean(perpl[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)

            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_perplexity=perpl[i],
                        count=counts[i])
            
    def epoch_end(self, epoch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_perplexity"]["mean"], 
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
