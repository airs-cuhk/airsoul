#!/usr/bin/env python
# coding=utf8
# File: models.py
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from modules import ARTransformerStandard
from utils import ce_loss_mask, img_pro, img_post

class LMBase(nn.Module):
    def __init__(self, config):
        super().__init__()

        context_warmup = config.loss_context_warmup
        vocab_size = config.vocabulary_size
        n_trn_block = config.n_transformer_block
        hidden_size = config.transformer_hidden_size
        nhead = config.transformer_nhead
        max_time_step = config.max_time_step
        if(hasattr(config, "transformer_checkpoints_density")):
            checkpoints_density = config.transformer_checkpoints_density
        else:
            checkpoints_density = -1
        self.transformer = ARTransformerStandard(vocab_size, n_trn_block, hidden_size, nhead, max_time_step, checkpoints_density=checkpoints_density)
        loss_mask = torch.cat((
                torch.linspace(0.0, 1.0, context_warmup).unsqueeze(0),
                torch.full((1, max_time_step - context_warmup, ), 1.0)), dim=1)
        self.register_buffer('loss_mask', loss_mask)


    def forward(self, inputs, cache=None, need_cache=True):
        """
        Inputs: [B, NT]
        Outputs: [B, NT, H]
        """
        
        # Temporal Encoders
        output, new_cache = self.transformer(inputs, cache=cache, need_cache=need_cache)

        return output, new_cache



if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = LMBase(config.model_config)
    inputs = torch.randint(32, (4, 128)) 

    losses = model.perplexity(inputs[:-1], inputs[1:])
    outputs = model.inference_seg(inputs, 4)
    print(losses.shape)
    print(outputs.shape)
