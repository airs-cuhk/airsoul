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

class DualTrackGatedDeltaNet(GatedDeltaNet):
    def __init__(self,
                io_size: int=512,
                intermediate_size: int=1024,
                num_heads: int=4,
                expand_v: int=2,
                layer_idx: int=0,
                is_generate: bool=False,
                use_memory_merge: bool=False,
                fusion_gate_init_bias: float = -2.0,
                use_adaptive_merge: bool = True):
        super().__init__(io_size, 
                         intermediate_size, 
                         num_heads, 
                         expand_v, 
                         layer_idx, 
                         is_generate)
        self.use_memory_merge = use_memory_merge

        if use_memory_merge:
            self._init_merge_layers(fusion_gate_init_bias, use_adaptive_merge)
    
    def _init_merge_layers(self, fusion_gate_init_bias: float = -2.0, 
                            use_adaptive_merge: bool = True):
        """
        简化版：直接门控融合，不压缩
        
        Args:
            fusion_gate_init_bias: 门控负偏置，初始时融合强度接近 0
            use_adaptive_merge: 是否使用自适应融合强度（由门控决定），
                            False 则使用固定系数 0.1
        """
        self.use_adaptive_merge = use_adaptive_merge
        
        # 维度信息
        self.head_v_dim_short = int(self.short_term_config.head_dim * self.short_term_config.expand_v)
        self.head_v_dim_long = int(self.long_term_config.head_dim * self.long_term_config.expand_v)
        self.head_k_dim_short = self.short_term_config.head_dim
        self.head_k_dim_long = self.long_term_config.head_dim
        
        # 1. V 维度投影（如果不同）
        if self.head_v_dim_short != self.head_v_dim_long:
            self.v_proj = nn.Linear(self.head_v_dim_short, self.head_v_dim_long, bias=False)
            self.v_proj_reverse = nn.Linear(self.head_v_dim_long, self.head_v_dim_short, bias=False)
        else:
            self.v_proj = nn.Identity()
            self.v_proj_reverse = nn.Identity()
        
        # 2. K 维度投影（如果不同）
        if self.head_k_dim_short != self.head_k_dim_long:
            self.k_proj = nn.Linear(self.head_k_dim_short, self.head_k_dim_long, bias=False)
        else:
            self.k_proj = nn.Identity()
        
        # 3. Gate MLP（输入：ST均值 + ST标准差 + LT均值投影）
        gate_input_dim = 3 * self.head_v_dim_short
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, self.head_v_dim_short),
            nn.Sigmoid()  # 简化为单层
        )
        
        # 4. 负偏置初始化（初始时融合强度接近 0）
        nn.init.constant_(self.gate_mlp[0].bias, fusion_gate_init_bias)

    def merge_memory(self, short_term_cache, long_term_cache, 
                    return_gate_stats=False, detach_fusion=False):
        """
        简化版：直接门控融合
        
        Args:
            short_term_cache: short-term 的 cache（字典）
            long_term_cache: long-term 的 cache（字典）
            return_gate_stats: 是否返回门控统计值（用于调试）
            detach_fusion: 是否冻结融合模块（用于消融实验）
        
        Returns:
            updated_cache: 更新后的 long_term_cache
            gate_stats: (可选) 包含门控统计值的字典
        """
        if not self.use_memory_merge or short_term_cache is None or long_term_cache is None:
            return (long_term_cache, {}) if return_gate_stats else long_term_cache
        
        S_short = short_term_cache.get('recurrent_state')
        S_long = long_term_cache.get('recurrent_state')
        
        if S_short is None or S_long is None:
            return (long_term_cache, {}) if return_gate_stats else long_term_cache
        
        # ============ 1. 计算门控 ============
        # ST 统计信息
        st_mean = S_short.mean(dim=-2)  # [B, H, V_short]
        st_std = S_short.std(dim=-2)    # [B, H, V_short]
        
        # LT 均值投影到 ST 维度
        lt_mean = S_long.mean(dim=-2)   # [B, H, V_long]
        lt_mean_proj = self.v_proj_reverse(lt_mean)  # [B, H, V_short]
        
        # 门控
        gate_input = torch.cat([st_mean, st_std, lt_mean_proj], dim=-1)
        g = self.gate_mlp(gate_input).unsqueeze(-2)  # [B, H, 1, V_short]
        
        # ============ 2. 应用门控 ============
        gated_z = g * S_short  # 直接使用 S_short，不压缩
        
        if detach_fusion:
            gated_z = gated_z.detach()
        
        # ============ 3. 投影到 LT 维度 ============
        # V 维度投影: [B, H, K_short, V_short] -> [B, H, K_short, V_long]
        gated_z = self.v_proj(gated_z)

        # K 维度投影: [B, H, K_short, V_long] -> [B, H, K_long, V_long]
        gated_z = self.k_proj(gated_z.transpose(-1, -2)).transpose(-1, -2)
        
        # ============ 4. 增量更新 ============
        if self.use_adaptive_merge:
            # 自适应融合：由门控决定融合强度
            g_global = g.mean(dim=-1, keepdim=True)  # [B, H, 1, 1]
            S_long_updated = S_long + g_global * gated_z
        else:
            # 固定系数融合：使用 0.1 保护 LT 记忆
            S_long_updated = S_long + 0.1 * gated_z
        
        # ============ 5. 返回 ============
        updated_cache = dict(long_term_cache)
        updated_cache['recurrent_state'] = S_long_updated
        
        if return_gate_stats:
            gate_stats = {
                'g_mean': g.mean().item(),
                'g_std': g.std().item()
            }
            return updated_cache, gate_stats
        
        return updated_cache

class DualTrackGatedDeltaNet1(nn.Module):
    """
    双轨线性注意力模块：包含 short-term 和 long-term 两个并行的 GatedDeltaNet。
    
    Cache 结构说明:
        cache = {
            'recurrent_state': Tensor[B, H, K, V],  # 长程记忆，需要 merge
            'conv_state': Tuple(Tensor, Tensor, Tensor)  # 短期滑动窗口，不处理
        }
    """
    
    def __init__(self,
                 io_size: int = 512,
                 intermediate_size: int = 1024,
                 num_heads: int = 4,
                 expand_v: int = 2,
                 layer_idx: int = 0,
                 is_generate: bool = False,
                 short_term_config: dict = None,
                 long_term_config: dict = None,
                 use_memory_merge: bool = True,
                 fusion_gate_init_bias: float = -2.0,
                 use_adaptive_merge: bool = True,):
        super().__init__()
        
        self.hidden_size = io_size
        self.layer_idx = layer_idx
        self.use_memory_merge = use_memory_merge
        
        mode = 'chunk' if not is_generate else 'fused_recurrent'
        
        # Short-term encoder 配置
        short_cfg = short_term_config or {}
        self.short_term_config = GatedDeltaNetConfig(
            attn_mode=mode,
            hidden_size=short_cfg.get('hidden_size', io_size),
            intermediate_size=short_cfg.get('intermediate_size', intermediate_size),
            num_heads=short_cfg.get('num_heads', num_heads),
            head_dim=short_cfg.get('head_dim', int(0.75 * io_size // num_heads)),
            vocab_size=32000,
            expand_v=short_cfg.get('expand_v', expand_v),
            conv_size=short_cfg.get('conv_size', 4)
        )
        
        # Long-term encoder 配置
        long_cfg = long_term_config or {}
        self.long_term_config = GatedDeltaNetConfig(
            attn_mode=mode,
            hidden_size=long_cfg.get('hidden_size', io_size),
            intermediate_size=long_cfg.get('intermediate_size', intermediate_size),
            num_heads=long_cfg.get('num_heads', num_heads),
            head_dim=long_cfg.get('head_dim', int(0.75 * io_size // num_heads)),
            vocab_size=32000,
            expand_v=long_cfg.get('expand_v', expand_v),
            conv_size=long_cfg.get('conv_size', 4)
        )
        
        # 创建 encoders
        self.short_term_encoder = GatedDeltaNetBlock(
            config=self.short_term_config, layer_idx=layer_idx)
        self.long_term_encoder = GatedDeltaNetBlock(
            config=self.long_term_config, layer_idx=layer_idx)
        
        # Memory Merge 模块
        if use_memory_merge:
            self._init_merge_layers(fusion_gate_init_bias, use_adaptive_merge)
    
    def _init_merge_layers(self, fusion_gate_init_bias: float = -2.0, 
                            use_adaptive_merge: bool = True):
        """
        简化版：直接门控融合，不压缩
        
        Args:
            fusion_gate_init_bias: 门控负偏置，初始时融合强度接近 0
            use_adaptive_merge: 是否使用自适应融合强度（由门控决定），
                            False 则使用固定系数 0.1
        """
        self.use_adaptive_merge = use_adaptive_merge
        
        # 维度信息
        self.head_v_dim_short = int(self.short_term_config.head_dim * self.short_term_config.expand_v)
        self.head_v_dim_long = int(self.long_term_config.head_dim * self.long_term_config.expand_v)
        self.head_k_dim_short = self.short_term_config.head_dim
        self.head_k_dim_long = self.long_term_config.head_dim
        
        # 1. V 维度投影（如果不同）
        if self.head_v_dim_short != self.head_v_dim_long:
            self.v_proj = nn.Linear(self.head_v_dim_short, self.head_v_dim_long, bias=False)
            self.v_proj_reverse = nn.Linear(self.head_v_dim_long, self.head_v_dim_short, bias=False)
        else:
            self.v_proj = nn.Identity()
            self.v_proj_reverse = nn.Identity()
        
        # 2. K 维度投影（如果不同）
        if self.head_k_dim_short != self.head_k_dim_long:
            self.k_proj = nn.Linear(self.head_k_dim_short, self.head_k_dim_long, bias=False)
        else:
            self.k_proj = nn.Identity()
        
        # 3. Gate MLP（输入：ST均值 + ST标准差 + LT均值投影）
        gate_input_dim = 3 * self.head_v_dim_short
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, self.head_v_dim_short),
            nn.Sigmoid()  # 简化为单层
        )
        
        # 4. 负偏置初始化（初始时融合强度接近 0）
        nn.init.constant_(self.gate_mlp[0].bias, fusion_gate_init_bias)

    
    def forward(self, x_short, x_long, cache_short=None, cache_long=None, need_cache=False):
        """双轨并行处理"""
        # Short-term
        if need_cache and cache_short is None:
            cache_short = Cache.from_legacy_cache(None)
        elif cache_short is not None:
            cache_short = Cache.from_legacy_cache([memory_cpy(cache_short)])
        
        short_out, _, new_cache_short = self.short_term_encoder(
            hidden_states=x_short, past_key_values=cache_short,
            use_cache=(cache_short is not None))
        cache_short_out = new_cache_short.states[0] if new_cache_short else None
        
        # Long-term
        if need_cache and cache_long is None:
            cache_long = Cache.from_legacy_cache(None)
        elif cache_long is not None:
            cache_long = Cache.from_legacy_cache([memory_cpy(cache_long)])
        
        long_out, _, new_cache_long = self.long_term_encoder(
            hidden_states=x_long, past_key_values=cache_long,
            use_cache=(cache_long is not None))
        cache_long_out = new_cache_long.states[0] if new_cache_long else None
        
        return short_out, long_out, cache_short_out, cache_long_out
    
    def merge_memory(self, short_term_cache, long_term_cache, 
                    return_gate_stats=False, detach_fusion=False):
        """
        简化版：直接门控融合
        
        Args:
            short_term_cache: short-term 的 cache（字典）
            long_term_cache: long-term 的 cache（字典）
            return_gate_stats: 是否返回门控统计值（用于调试）
            detach_fusion: 是否冻结融合模块（用于消融实验）
        
        Returns:
            updated_cache: 更新后的 long_term_cache
            gate_stats: (可选) 包含门控统计值的字典
        """
        if not self.use_memory_merge or short_term_cache is None or long_term_cache is None:
            return (long_term_cache, {}) if return_gate_stats else long_term_cache
        
        S_short = short_term_cache.get('recurrent_state')
        S_long = long_term_cache.get('recurrent_state')
        
        if S_short is None or S_long is None:
            return (long_term_cache, {}) if return_gate_stats else long_term_cache
        
        # ============ 1. 计算门控 ============
        # ST 统计信息
        st_mean = S_short.mean(dim=-2)  # [B, H, V_short]
        st_std = S_short.std(dim=-2)    # [B, H, V_short]
        
        # LT 均值投影到 ST 维度
        lt_mean = S_long.mean(dim=-2)   # [B, H, V_long]
        lt_mean_proj = self.v_proj_reverse(lt_mean)  # [B, H, V_short]
        
        # 门控
        gate_input = torch.cat([st_mean, st_std, lt_mean_proj], dim=-1)
        g = self.gate_mlp(gate_input).unsqueeze(-2)  # [B, H, 1, V_short]
        
        # ============ 2. 应用门控 ============
        gated_z = g * S_short  # 直接使用 S_short，不压缩
        
        if detach_fusion:
            gated_z = gated_z.detach()
        
        # ============ 3. 投影到 LT 维度 ============
        # V 维度投影: [B, H, K_short, V_short] -> [B, H, K_short, V_long]
        gated_z = self.v_proj(gated_z)

        # K 维度投影: [B, H, K_short, V_long] -> [B, H, K_long, V_long]
        gated_z = self.k_proj(gated_z.transpose(-1, -2)).transpose(-1, -2)
        
        # ============ 4. 增量更新 ============
        if self.use_adaptive_merge:
            # 自适应融合：由门控决定融合强度
            g_global = g.mean(dim=-1, keepdim=True)  # [B, H, 1, 1]
            S_long_updated = S_long + g_global * gated_z
        else:
            # 固定系数融合：使用 0.1 保护 LT 记忆
            S_long_updated = S_long + 0.1 * gated_z
        
        # ============ 5. 返回 ============
        updated_cache = dict(long_term_cache)
        updated_cache['recurrent_state'] = S_long_updated
        
        if return_gate_stats:
            gate_stats = {
                'g_mean': g.mean().item(),
                'g_std': g.std().item()
            }
            return updated_cache, gate_stats
        
        return updated_cache