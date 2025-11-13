import copy
import torch
from torch import nn
from airsoul.utils import log_warn
from native_sparse_attention_pytorch import SparseAttention
        
class NSATransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(NSATransformerEncoderLayer, self).__init__()
        self.self_attn = SparseAttention(
                dim = d_model,
                dim_head = d_model // nhead,
                heads = nhead,
                sliding_window_size = 2,
                compress_block_size = 4,
                compress_block_sliding_stride = 2,
                selection_block_size = 4,
                causal=True,
                num_selected_blocks = 2
            )

        # Define other layers (e.g., Feedforward, LayerNorm, Dropout) here
        # Norm First = True
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1.0e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1.0e-5)
        self.activation = nn.GELU()

    def forward(self, 
                src : torch.Tensor, 
                cache=None):
        """
        Cache: B, NT, H
        SRC: Other Parts
        """
        # Self Attention

        output = self.norm1(src)        
        output = self.self_attn(output)

        # Residual Connection
        output = src + output

        # FeedForward + Residual
        output = output + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(self.norm2(output))))))

        # Apply other layers and return output
        return output

class NSATransformerEncoder(nn.Module):
    def __init__(self, 
            num_layers : int, 
            d_model : int, 
            nhead : int, 
            dim_feedforward : int=2048, 
            dropout : float=0.10):
        super(NSATransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_head = d_model // nhead
        ar_layer = NSATransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.layers = nn.ModuleList([copy.deepcopy(ar_layer) for i in range(num_layers)])

    def forward(self, src, cache=None, need_cache=False, checkpoints_density=-1):
        # Every checkpoints_density we arrange a checkpoint
        # If checkpoints_density < 1 we do not use checkpoints
        # Calculate Cache Size

        output=src
        for i, layer in enumerate(self.layers):
            output = layer(output)
        return output, None