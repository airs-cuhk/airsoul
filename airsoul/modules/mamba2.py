import torch
from torch import nn
from fla.models.mamba2.modeling_mamba2 import Mamba2Cache, Mamba2Block
from fla.models.mamba2.configuration_mamba2 import Mamba2Config
from airsoul.utils import memory_cpy

class Mamba2Layer(nn.Module):
    def __init__(self,
                io_size: int=512,
                expand: int=2, # Expanding factor used to determine the intermediate size.
                num_heads: int=4,
                use_segment_input: bool=True,
                num_hidden_layers: int=1,
                layer_idx: int=0
                ):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        self.use_segment_input = use_segment_input
        if use_segment_input:
            num_hidden_layers = 1
            layer_idx = 0

        self.config = Mamba2Config(hidden_size = io_size,
                                   expand = expand,
                                   head_dim = int(expand*io_size // num_heads),
                                   use_segment_input = use_segment_input,
                                   residual_in_fp32 = True,
                                   num_hidden_layers = num_hidden_layers, # Each layer declare its own Mamba2Cache
                                   chunk_size = 96,
                                   )
        self.encoder = Mamba2Block(config=self.config, 
                                          layer_idx=layer_idx) # Manage cache outside the fla lib
        
    def forward(self, x, cache=None, need_cache=False):
        if self.use_segment_input:
            if(need_cache and cache is None):
                cache = Mamba2Cache(self.config, x.size(0), device=x.device, dtype=x.dtype)
            elif(cache is not None):
                # avoid in-place modification of the cache
                cache = memory_cpy(cache)
            # Notice that cache is changed in-place
            out, new_cache = self.encoder(hidden_states=x, cache_params=cache)
            return out, new_cache
        
        out = self.encoder(hidden_states=x)
        return out, None