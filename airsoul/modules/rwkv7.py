import torch
from torch import nn
from fla.models.rwkv7.modeling_rwkv7 import RWKV7Block
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.utils import Cache
from airsoul.utils import format_cache, memory_cpy, log_warn


class RWKV7Layer(nn.Module):
    def __init__(self,
                io_size: int=512,
                intermediate_size: int = 1024,
                num_heads: int = 4,
                layer_idx: int = 0):
        super().__init__()
        self.config = RWKV7Config(
                  hidden_size=io_size,
                  intermediate_size=intermediate_size,
                  num_heads=num_heads)
        self.layer_idx = layer_idx
        if layer_idx == 0:
            is_first_layer = True
        else:
            is_first_layer = False
        self.encoder = RWKV7Block(
                  self.config,
                  layer_idx=0,
                  is_first_layer = is_first_layer)

    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache_ = Cache.from_legacy_cache(None)
        elif(cache is not None):
            # avoid in-place modification of the cache
            cache_copy = memory_cpy(cache)
            cache_ = Cache.from_legacy_cache([cache_copy[0]])

        if cache is None:
            v_first = v_first = torch.zeros_like(x)
        else:
            v_first = cache_copy[1]

        use_cache = (cache_ is not None)

        out, _, new_cache_, v_first = self.encoder(hidden_states=x, past_key_values=cache_, use_cache=use_cache, v_first=v_first)

        new_cache = (new_cache_.states[0], v_first)

        return out, new_cache
