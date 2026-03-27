import os
import torch
import torch.optim as optim
import numpy as np
from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler, CosineScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager
from airsoul.dataloader import LMDataSet
import pickle
from types import NoneType


def string_mean_var(downsample_length, res):
    string = ""
    for i, (xm, xb) in enumerate(zip(res["mean"], res["bound"])):
        string += f'{downsample_length * i}\t{xm}\t{xb}\n'
    return string

def cache2numpy(cache):
    all_paras = torch.tensor([])
    for layer_id in range(len(cache)):
        for key in cache[layer_id].keys():
            if type(cache[layer_id][key]) is NoneType:
                continue
            if type(cache[layer_id][key]) is tuple:
                for i in range(len(cache[layer_id][key])):
                    c = cache[layer_id][key][i].cpu()
                    all_paras = torch.cat((all_paras, c.flatten()))
            else:
                c = cache[layer_id][key].cpu()
                all_paras = torch.cat((all_paras, c.flatten()))
    all_paras = all_paras.cpu().numpy()
    return all_paras

@EpochManager
class LMEpoch:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.DataType = LMDataSet
        if(self.is_training):
            self.logger_keys = ["learning_rate",
                        "train_cross_entropy",
                        "train_perplexity"]
            self.stat = DistStatistics(*self.logger_keys[1:])
            self.reduce = 1
        else:
            # self.logger_keys = ["validate_cross_entropy", "validate_perplexity"]
            self.logger_keys = ["validate_cross_entropy"]
            
            self.stat = DistStatistics(*self.logger_keys)
            self.reduce = None
            if(self.config.has_attr("downsample_length")):
                self.downsample_length = self.config.downsample_length
            else:
                self.downsample_length = 1
        # if(self.is_training):
        #     self.lr_scheduler = CosineScheduler(self.config.lr_T_max, self.config.lr_warmup_step, self.config.lr_max, self.config.lr_min)

    def compute(self, feas, labs, masks,
                local_batch_id=-1,
                global_batch_id=-1,
                global_epoch_id=-1):
        """
        Defining the computation function for each batch
        """
        if(self.is_training):
            assert self.optimizer is not None, "optimizer is required for training"

        losses = []
        for sub_idx, fea, lab, mask in segment_iterator(
                    self.config.seq_len, self.config.seg_len, self.device,
                    feas, labs, masks):
            loss, _ = self.model.module.perplexity(
                    fea, lab,
                    use_loss_weight=self.is_training,
                    reduce_dim=self.reduce,
                    masks=mask) # Do not use loss weight for evaluation
            losses.append(loss)
            if(self.is_training):
                syn_loss = loss["ce_loss"]
                if(self.scaler is not None):
                    self.scaler.scale(syn_loss).backward()
                else:
                    syn_loss.backward()
                self.stat.gather(self.device,
                    train_cross_entropy=syn_loss / loss["count"],
                    train_perplexity= torch.exp(syn_loss / loss["count"]),
                    count=loss["count"])
        
        # if(self.is_training):
        #     # save cache
        #     cache2save, position = self.model.module.get_mem()
        #     numpy_cache = cache2numpy(cache2save)
        #     assert cache2save is not None
        #     cache_folder = "/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/03-03-memory_cache_statefull_long"
        #     if not os.path.exists(cache_folder):
        #         os.makedirs(cache_folder)
        #     cache_path = os.path.join(cache_folder, f"cache_{global_epoch_id}_{local_batch_id}.pkl")
        #     numpy_cache_path = os.path.join(cache_folder, f"numpy_cache_{global_epoch_id}_{local_batch_id}.npy")
        #     np.save(numpy_cache_path, numpy_cache)
        #     pickle.dump(cache2save, open(cache_path, 'wb'))
        
        if(self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(self.optimizer.param_groups[0]['lr'],
                        stat_res["train_cross_entropy"]["mean"],
                        stat_res["train_perplexity"]["mean"],
                        epoch=global_epoch_id,
                        iteration=local_batch_id)
        else:
            ce_loss = torch.cat([loss["ce_loss"] / loss["count"] for loss in losses], dim=1)
            perpl = torch.cat([torch.exp(loss["ce_loss"] / loss["count"]) for loss in losses], dim=1)
            counts = torch.cat([loss["count"] for loss in losses], dim=1)

            bsz = ce_loss.shape[0]
            seg_num = ce_loss.shape[1] // self.downsample_length
            valid_seq_len = seg_num * self.downsample_length

            ce_loss = torch.mean(ce_loss[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            perpl = torch.mean(perpl[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            counts = torch.mean(counts[:, :valid_seq_len].view(bsz, seg_num, -1), dim=-1)
            
            for i in range(bsz):
                self.stat.gather(self.device,
                        validate_cross_entropy=ce_loss[i],
                        # validate_perplexity=perpl[i],
                        count=counts[i])

    def epoch_end(self, epoch_id, batch_id):
        if(not self.is_training):
            stat_res = self.stat()
            if(self.logger is not None):
                self.logger(stat_res["validate_cross_entropy"]["mean"],
                            # stat_res["validate_perplexity"]["mean"],
                        epoch=epoch_id, iteration=batch_id)
            if(self.extra_info is not None):
                if(self.extra_info.lower() == 'validate' and self.main):
                    if not os.path.exists(os.path.basename(self.config.output)):
                        os.makedirs(os.path.basename(self.config.output))
                    res_output_folder = os.path.join(os.path.basename(self.config.output), f'epoch_{epoch_id}')
                    
                    if not os.path.exists(res_output_folder):
                        os.makedirs(res_output_folder)
                    for key_name in stat_res:
                        res_text = string_mean_var(self.downsample_length, stat_res[key_name])
                        file_path = f'{res_output_folder}/result_{key_name}.txt'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        with open(file_path, 'w') as f_model:
                            f_model.write(res_text)
