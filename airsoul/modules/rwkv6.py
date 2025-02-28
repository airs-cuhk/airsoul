import torch
from torch import nn
from fla.models.rwkv6.modeling_rwkv6 import RWKV6Block
from fla.models.rwkv6.configuration_rwkv6 import RWKV6Config
from fla.models.utils import Cache
from airsoul.utils import format_cache, memory_cpy, log_warn


class RWKV6Layer(nn.Module):
    def __init__(self,
                io_size: int=512,
                expand_k: float = 0.5,
                expand_v: float = 1,
                hidden_ratio: float = 3.5,
                intermediate_size: int = 1024,
                num_heads: int = 4,
                layer_idx: int = 0,
                gate_bound: float=50.0):
        super().__init__()
        self.config = RWKV6Config(
                  hidden_size=io_size,
                  expand_k=expand_k,
                  expand_v=expand_v,
                  hidden_ratio=hidden_ratio,
                  intermediate_size=intermediate_size,
                  num_heads=num_heads,
                  gate_bound=gate_bound)
        self.layer_idx = layer_idx
        self.encoder = RWKV6Block(
                  self.config,
                  layer_idx=0)

    def forward(self, x, cache=None, need_cache=False):
        if(need_cache and cache is None):
            cache = Cache.from_legacy_cache(None)
        elif(cache is not None):
            # avoid in-place modification of the cache
            cache = Cache.from_legacy_cache([memory_cpy(cache)])

        use_cache = (cache is not None)

        out, _, new_cache = self.encoder(hidden_states=x, past_key_values=cache, use_cache=use_cache)

        return out, new_cache.states[0]
