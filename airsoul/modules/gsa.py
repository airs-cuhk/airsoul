import torch
from torch import nn
from fla.layers.gsa import GatedSlotAttention
from fla.layers.gla import GatedLinearAttention
from fla.models.utils import Cache
from airsoul.utils import format_cache, memory_cpy, log_warn 

class GLABlock(nn.Module):
    def __init__(self,
                io_size: int=512,
                num_heads: int=4,
                layer_idx: int=0,
                is_generate: bool=False):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        if(not is_generate):
            mode = 'chunk'
        else:
            mode = 'fused_recurrent'
        self.encoder = GatedLinearAttention(
                  mode=mode,
                  hidden_size=io_size,
                  num_heads=num_heads,
                  layer_idx=0)
        
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
    
class GSABlock(GLABlock):
    def __init__(self,
                io_size: int=512,
                num_heads: int=4,
                num_slots: int=4,
                gate_bound: float=50,
                layer_idx: int=0,
                is_generate: bool=False):
        super().__init__()
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        if(not is_generate):
            mode = 'chunk'
        else:
            mode = 'fused_recurrent'
        self.encoder = GatedSlotAttention(
                  mode=mode,
                  hidden_size=io_size,
                  num_heads=num_heads,
                  num_slots=num_slots,
                  gate_bound=gate_bound,
                  layer_idx=0)
