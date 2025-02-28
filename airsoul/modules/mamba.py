import torch
from torch import nn
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.utils.generation import InferenceParams
from airsoul.utils import log_warn

class MambaBlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                d_state: int=16,
                d_conv: int=4,
                max_position_encoding: int=60000,
                layer_idx: int=0,
                expand: int=2):
        super().__init__()
        self.hidden_size = io_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.layer_idx = layer_idx
        self.max_position_encoding = max_position_encoding
        self.encoder = Mamba(
                  d_model=io_size,
                  d_state=d_state,
                  d_conv=d_conv,
                  expand=expand,
                  layer_idx=self.layer_idx)
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache):
            if(cache is None):
                cache = InferenceParams(max_seqlen=self.max_position_encoding, max_batch_size=x.shape[0])
            else:
                n_cache = InferenceParams(max_seqlen=self.max_position_encoding, max_batch_size=x.shape[0])
                n_cache.key_value_memory_dict[self.layer_idx] = (cache[0].clone(), cache[1].clone())
                cache = n_cache
                log_warn("Attention!!! Mamba-ssm has bugs for chunk-wise inference")
        else:
            cache = None

        out = self.encoder(x, inference_params=cache)

        if(need_cache):
            return out, cache.key_value_memory_dict[self.layer_idx]
        else:
            return out, None
