import sys
import torch
import torch.distributed as dist
import torch.nn.utils.rnn as rnn_utils
from types import SimpleNamespace
from copy import deepcopy
from dateutil.parser import parse
from collections import defaultdict
from .tools import log_debug, log_warn, log_fatal

class DistStatistics(object):
    """
    Provide distributed statistics over GPUs
    """
    def __init__(self, *args, **kwargs):
        self.reset()

    def reset(self):
        self._sum = dict()
        self._sum2 = dict()
        self._count = dict()

    def gather(self, device, count=None, **kwargs):
        """
        Count being regarded as the number of samples behind each of the value
        if value is an array, then it is regarded as the number of samples behind each of the value
        """
        with torch.no_grad():
            # Reshape the count to 1-dimensional tensor
            if(count is None):
                fcount = torch.Tensor([1]).to(device)
            elif(isinstance(count, list) or isinstance(count, tuple)):
                fcount = torch.Tensor(count).to(device)
            elif(isinstance(count, torch.Tensor)):
                fcount = count.clone().to(device)
            else:
                fcount = torch.Tensor([count]).to(device)
            assert fcount.ndim < 2, f"count must be float/int or a list"
            if(fcount.ndim < 1):
                fcount = fcount.unsqueeze(0)
            
            for key, value in kwargs.items():
                # Reshape value to 1-dimensional tensor
                if isinstance(value, list) or isinstance(value, tuple):
                    fvalue = torch.stack(value, dim=0).to(device)
                elif(isinstance(value, torch.Tensor)):
                    fvalue = value.clone().to(device)
                else:
                    fvalue = torch.Tensor(value).to(device)
                assert fvalue.ndim < 2, f"requires value dimension < 2, get {fvalue.shape}"
                if(fvalue.ndim < 1):
                    fvalue = fvalue.unsqueeze(0)

                # Expand such that the count and the value have the same dimension
                c_dim = fcount.numel()
                v_dim = fvalue.numel()
                if(c_dim == 1 and c_dim < v_dim):
                    fcount_e = fcount.expand(v_dim).clone().to(device)
                elif(c_dim > 1 and c_dim != v_dim):
                    log_fatal(f"dimension mismatch between statistic count {fcount.shape} and value {fvalue.shape}")
                else:
                    fcount_e = fcount.clone()

                illegal = torch.isinf(fvalue) | torch.isnan(fvalue)
                if illegal.any():
                    log_warn(f"'Device:{device}' stating '{key}' has inf/NaN")
                    fvalue = torch.where(illegal, fvalue, torch.zeros_like(fvalue))
                
                if(key not in self._count):
                    self._sum[key] = fvalue * fcount_e
                    self._sum2[key] = fvalue ** 2 * fcount_e
                    self._count[key] = fcount_e
                else:
                    if(self._count[key].shape[0] < v_dim):
                        expand_l = v_dim - self._count[key].shape[0]
                        self._count[key] = torch.cat((self._count[key], torch.zeros((expand_l,), device=device)), dim=0)
                        self._sum[key] = torch.cat((self._sum[key], torch.zeros((expand_l,), device=device)), dim=0)
                        self._sum2[key] = torch.cat((self._sum2[key], torch.zeros((expand_l,), device=device)), dim=0)
                    self._sum[key][:v_dim] += fvalue * fcount_e
                    self._sum2[key][:v_dim] += fvalue ** 2 * fcount_e
                    self._count[key][:v_dim] += fcount_e

    def _stat(self, key):
        # Gather the statistics from different cards
        max_length = torch.tensor([self._count[key].shape[0]], dtype=torch.int64, device=self._count[key].device)
        dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
        max_length = max_length.item()

        # Padding the current statistics to the maximum length if needed
        if(max_length > self._count[key].shape[0]):
            expand_l = max_length - self._count[key].shape[0]
            self._count[key] = torch.cat((self._count[key], torch.zeros((expand_l,), device=self._count[key].device)), dim=0)
            self._sum[key] = torch.cat((self._sum[key], torch.zeros((expand_l,), device=self._sum[key].device)), dim=0)
            self._sum2[key] = torch.cat((self._sum2[key], torch.zeros((expand_l,), device=self._sum2[key].device)), dim=0)
        
        # Gather the statistics from different cards
        sum_cnt = self._count[key].clone().to(self._count[key].device)
        sum_mean = self._sum[key].clone().to(self._count[key].device)
        sum_mean2 = self._sum2[key].clone().to(self._count[key].device)

        dist.all_reduce(sum_cnt, dist.ReduceOp.SUM)
        dist.all_reduce(sum_mean, dist.ReduceOp.SUM)
        dist.all_reduce(sum_mean2, dist.ReduceOp.SUM)

        x_mean = sum_mean / sum_cnt
        x2_mean = sum_mean2 / sum_cnt
        var = torch.sqrt(x2_mean - x_mean ** 2)

        return x_mean, var, sum_cnt

    def __call__(self, reset=True):
        stat_res = dict()
        with torch.no_grad():
            for key in self._count:
                mean,std,cnt = self._stat(key)
                # 95% Confidence Bound For Mean
                bound = 2.0 * std / torch.sqrt(cnt)
                if(mean.numel() < 2):
                    mean = mean.squeeze().item()
                    std = std.squeeze().item()
                    bound = bound.squeeze().item()
                else:
                    mean = mean.squeeze().tolist()
                    std = std.squeeze().tolist()
                    bound = bound.squeeze().tolist()
                stat_res[key] = {"mean":mean,"std":std,'cnt':cnt,
                        'bound':bound}
            if(reset):
                self.reset()
        return stat_res
