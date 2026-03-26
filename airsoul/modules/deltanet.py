import torch
from torch import nn
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.utils import Cache
from airsoul.utils import format_cache, memory_cpy, log_warn 

class GatedDeltaNet(nn.Module):
    def __init__(self,
                io_size: int=512,
                intermediate_size: int=1024,
                num_heads: int=4,
                expand_v: int=2,
                layer_idx: int=0,
                is_generate: bool=False):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        if(not is_generate):
            mode = 'chunk'
        else:
            mode = 'fused_recurrent'
        self.config = GatedDeltaNetConfig(attn_mode = mode,
                                          hidden_size = io_size,
                                          intermediate_size = intermediate_size,
                                          num_heads = num_heads,
                                          head_dim = int(0.75*io_size//num_heads),
                                          vocab_size = 32000,
                                          expand_v = expand_v, # default 2
                                          conv_size = 4)
        self.encoder = GatedDeltaNetBlock(config=self.config, 
                                          layer_idx=0) # manage cache outside the fla lib
        
    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = Cache.from_legacy_cache(None)
        elif(cache is not None):
            # avoid in-place modification of the cache
            cache = Cache.from_legacy_cache([memory_cpy(cache)])
    
        use_cache = (cache is not None)

        # Notice that cache is changed in-place
        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        return out, new_cache.states[0]