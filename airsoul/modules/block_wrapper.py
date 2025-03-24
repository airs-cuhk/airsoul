import torch.nn as nn
from torch.nn import functional as F

"""
Wraps Self-Attention, Mamba etc with a Residual Connection and FeedForward to form a Transformer-like structure
"""

class BlockWrapper(nn.Module):
    def __init__(self, temporal_module, 
                        hidden:int=512, 
                        fc_hidden:int=512, 
                        fc_dropout:float=0.10,
                        layer_idx:int=0,
                        **kwargs):
        super(BlockWrapper, self).__init__()
        self.temporal_encoder = temporal_module(**kwargs, layer_idx=layer_idx)

        self.linear1 = nn.Linear(hidden, fc_hidden)
        self.dropout = nn.Dropout(fc_dropout)
        self.linear2 = nn.Linear(fc_hidden, hidden)

        self.norm1 = nn.LayerNorm(hidden, eps=1.0e-5)
        self.norm2 = nn.LayerNorm(hidden, eps=1.0e-5)
        self.layer_idx = layer_idx

        self.activation = nn.GELU()

    def forward(self, src, cache=None, need_cache=False):
        # Residual Connection
        norm_src = self.norm1(src)
        outputs, cache = self.temporal_encoder(norm_src, cache=cache, need_cache=need_cache)

        outputs = outputs + src

        # FeedForward + Residual
        outputs = outputs + self.dropout(
                                self.linear2(
                                    self.dropout(
                                        self.activation(
                                            self.linear1(
                                                self.norm2(outputs)
                                                )
                                            )
                                        )
                                    )
                                )

        return outputs, cache

class MultiBlocks(nn.Module):
    def __init__(self, temporal_module, 
                 num_layers, 
                 need_block_wrapper=True, 
                 **kwargs):
        super(MultiBlocks, self).__init__()
        self.num_layers = num_layers
        self.position = 0 # This is used to keep compatible with BlockRecurison
        if(need_block_wrapper):
            self.layers = nn.ModuleList(
                [BlockWrapper(temporal_module, layer_idx=layer_idx, **kwargs) 
                    for layer_idx in range(self.num_layers)])
        else:
            self.layers = nn.ModuleList(
                [temporal_module(layer_idx=layer_idx, **kwargs) 
                    for layer_idx in range(self.num_layers)])

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        # Residual Connection
        if(need_cache):
            new_cache = []
        else:
            new_cache = None

        output = src

        for i, layer in enumerate(self.layers):
            if(cache is None):
                l_cache = None
            else:
                l_cache = cache[i]
            output, n_cache = layer(output, cache=l_cache, need_cache=True)
            if(need_cache):
                new_cache.append(n_cache)

        return output, new_cache
    

if __name__=='__main__':
    from .recursion import SimpleLSTM
    import torch
    inputs = torch.randn(4, 8, 64)
    model = MultiBlocks(64, 64, 256, SimpleLSTM, 3)
    outputs, mems = model(inputs, need_cache=True)
    print(outputs.shape, mems[0][0].shape, mems[0][1].shape, mems[1][0].shape, mems[1][1].shape)
