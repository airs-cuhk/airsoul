import torch
from torch import nn
# from fla.models.rwkv7.modeling_rwkv7 import RWKV7Block
# from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
from fla.models.kda.modeling_kda import KDABlock
from fla.models.kda.configuration_kda import KDAConfig
from fla.models.utils import Cache
from airsoul.utils import format_cache, memory_cpy, log_warn
from .dual_track import DualTrackMixin


class KDALayer(nn.Module):
    def __init__(self,
                io_size: int=512,
                expand_v: float = 1.0,
                num_heads: int = 16,
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

class DualTrackKDA(KDALayer, DualTrackMixin):
    """
    继承 KDALayer 获得完整的 forward 功能
    混入 DualTrackMixin 获得 merge_memory 功能
    """
    def __init__(self,
                 io_size: int = 512,
                 expand_v: float = 1.0,
                 num_heads: int = 4,
                 layer_idx: int = 0,
                 use_memory_merge: bool = False,
                 fusion_gate_init_bias: float = -2.0,
                 use_adaptive_merge: bool = True,
                 short_term_config: dict = None,
                 long_term_config: dict = None):
        
        # 初始化父类 KDALayer
        super().__init__(io_size, expand_v, num_heads, layer_idx)
        
        self.use_memory_merge = use_memory_merge
        
        # 创建配置
        head_dim = io_size // num_heads
        
        short_cfg = short_term_config or {}
        self.short_term_config = KDAConfig(
            hidden_size=short_cfg.get('hidden_size', io_size),
            expand_v=short_cfg.get('expand_v', expand_v),
            head_dim=short_cfg.get('head_dim', head_dim),
            num_heads=short_cfg.get('num_heads', num_heads)
        )
        
        long_cfg = long_term_config or {}
        self.long_term_config = KDAConfig(
            hidden_size=long_cfg.get('hidden_size', io_size),
            expand_v=long_cfg.get('expand_v', expand_v),
            head_dim=long_cfg.get('head_dim', head_dim),
            num_heads=long_cfg.get('num_heads', num_heads)
        )
        
        # 初始化融合层
        if self.use_memory_merge:
            head_v_dim_short = int(self.short_term_config.head_dim * self.short_term_config.expand_v)
            head_v_dim_long = int(self.long_term_config.head_dim * self.long_term_config.expand_v)
            head_k_dim_short = self.short_term_config.head_dim
            head_k_dim_long = self.long_term_config.head_dim
            
            self._init_merge_layers(
                head_v_dim_short, head_v_dim_long,
                head_k_dim_short, head_k_dim_long,
                fusion_gate_init_bias, use_adaptive_merge
            )