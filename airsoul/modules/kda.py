import torch
from torch import nn
# from fla.models.rwkv7.modeling_rwkv7 import RWKV7Block
# from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.kda.modeling_kda import KDABlock
from fla.models.kda.configuration_kda import KDAConfig
from fla.models.utils import Cache
from airsoul.utils import format_cache, memory_cpy, log_warn


class KDALayer(nn.Module):
    def __init__(self,
                io_size: int=512,
                expand_v: float = 1.0,
                num_heads: int = 4,
                layer_idx: int = 0):
        super().__init__()
        head_dim = io_size // num_heads
        self.config = KDAConfig(
                  hidden_size=io_size,
                  expand_v=expand_v,
                  head_dim=head_dim,
                  num_heads=num_heads)
        self.layer_idx = layer_idx
        if layer_idx == 0:
            is_first_layer = True
        else:
            is_first_layer = False
        self.encoder = KDABlock(
                  self.config,
                  layer_idx=0,)
                #   is_first_layer = is_first_layer)

    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = Cache.from_legacy_cache(None)
        elif(cache is not None):
            # avoid in-place modification of the cache
            cache = Cache.from_legacy_cache([memory_cpy(cache)])


        use_cache = (cache is not None)

        out, _, new_cache_ = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        # new_cache = (new_cache_.states[0], v_first)
        # new_cache = (new_cache_.layers[0].state, v_first)
        # new_cache = (new_cache_[0], v_first)
        

        return out, new_cache_[0]
