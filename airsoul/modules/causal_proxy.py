import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from .recursion import PRNN, SimpleLSTM
from .block_wrapper import MultiBlocks
from .transformers import ARTransformerEncoder
from .mamba import MambaBlock
from .blockrec_wrapper import BlockRecurrentWrapper
from .gsa import GLABlock, GSABlock
from .rwkv6 import RWKV6Layer
from .rwkv7 import RWKV7Layer
from .deltanet import GatedDeltaNet
from .mamba2 import Mamba2Layer
from .sparse_attention import NSATransformerEncoder

class CausalBlock(nn.Module):
    """
    Take Observations and actions, output d_models
    """
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type.lower()

        if(config.has_attr("is_generate")):
            is_generate = config.is_generate
        else:
            is_generate = False

        if(self.model_type == "transformer"):
            main_encoder = ARTransformerEncoder(
                config.num_layers, 
                config.hidden_size, 
                config.nhead, 
                config.position_encoding_size, 
                dim_feedforward=config.inner_hidden_size, 
                dropout=config.dropout, 
                context_window=config.context_window
            )
        elif(self.model_type == "nsa"):
            main_encoder = NSATransformerEncoder(
                config.num_layers, 
                config.hidden_size, 
                config.nhead, 
                dim_feedforward=config.inner_hidden_size, 
                dropout=config.dropout, 
            )
        elif(self.model_type == "gsa"):
            main_encoder = MultiBlocks(
                GSABlock,
                config.num_layers,
                hidden=config.hidden_size,
                fc_hidden=config.inner_hidden_size,
                fc_dropout=config.dropout,
                io_size=config.hidden_size,
                num_heads=config.nhead,
                num_slots=config.memory_length,
                is_generate=is_generate
            )
        elif(self.model_type == "gla"):
            main_encoder = MultiBlocks(
                GLABlock,
                config.num_layers,
                hidden=config.hidden_size,
                fc_hidden=config.inner_hidden_size,
                fc_dropout=config.dropout,
                io_size=config.hidden_size,
                num_heads=config.nhead,
                is_generate=is_generate
            )
        elif(self.model_type == "mamba"):
            main_encoder = MultiBlocks(
                # This module uses roughly 3 * expand * d_model^2 parameters
                MambaBlock,
                config.num_layers,
                hidden=config.hidden_size,
                fc_hidden=config.inner_hidden_size,
                fc_dropout=config.dropout,
                io_size=config.hidden_size,
                d_state=config.d_state,
                d_conv=config.d_conv,
                max_position_encoding=config.position_encoding_size,
                expand=config.expand,    # Block expansion factor
            )
        elif(self.model_type == "mamba2"):
            use_segment_input = config.use_segment_input
            if not config.use_blockrecurrence:
                use_segment_input = False
            main_encoder = MultiBlocks(
                Mamba2Layer,
                config.num_layers,
                need_block_wrapper=False,
                io_size=config.hidden_size,
                expand=config.inner_hidden_size/config.hidden_size,
                num_heads=config.nhead,
                use_segment_input=use_segment_input,
                num_hidden_layers=config.num_layers,
            )
        elif(self.model_type == "rwkv6"):
            main_encoder = MultiBlocks(
                RWKV6Layer,
                config.num_layers,
                need_block_wrapper=False,
                io_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                gate_bound=config.gate_bound,
                hidden_ratio=config.hidden_ratio,
                intermediate_size=config.inner_hidden_size,
                num_heads=config.nhead,
            )
        elif(self.model_type == "rwkv7"):
            main_encoder = MultiBlocks(
                RWKV7Layer,
                config.num_layers,
                need_block_wrapper=False,
                io_size=config.hidden_size,
                intermediate_size=config.inner_hidden_size,
                num_heads=config.nhead
            )
        elif(self.model_type == "deltanet"):
            main_encoder = MultiBlocks(
                GatedDeltaNet,
                config.num_layers,
                need_block_wrapper=False,
                io_size=config.hidden_size,
                intermediate_size=config.inner_hidden_size,
                num_heads=config.nhead,
                expand_v = config.expand_v
            )           
        else:
            raise Exception("No such causal model: %s" % config.model_type)
        
        self.need_reset = False
        if(config.use_blockrecurrence):
            main_encoder = BlockRecurrentWrapper(main_encoder, config.memory_length, 
                    memory_type = config.memory_type)
            self.need_reset = True

        if(config.use_layer_norm):
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1.0e-5)
        else:
            self.layer_norm = nn.Identity()

        self.layers = main_encoder
        self.checkpoints_density = config.checkpoints_density

        if(config.has_attr('is_fronzen')):
            if(config.is_frozen):
                for param in self.parameters():
                    param.requires_grad_(False)

    @property
    def position(self):
        if(hasattr(self.layers, 'position')):
            return self.layers.position
        else:
            return 0

    def forward(self, *args, **kwargs):
        kwargs["checkpoints_density"] = self.checkpoints_density
        out, cache = self.layers.forward(*args, **kwargs)
        return self.layer_norm(out), cache
    
    def reset(self):
        if(self.need_reset):
            self.layers.reset()
