#!/usr/bin/env python
# coding=utf8
# File: models.py
import sys
import torch
from torch import nn
from torch.nn import functional as F
from airsoul.utils import weighted_loss
from airsoul.modules import CausalBlock, DualTrackCausalBlock, ResidualMLPDecoder, MLPEncoder
from airsoul.utils import parameters_regularization, count_parameters


class StatefulLM(nn.Module):
    def __init__(self, config, verbose):
        super().__init__()

        self.word_emb = MLPEncoder(config.word_embeddings)
        self.causal_model = CausalBlock(config.causal_block)
        self.output_mapping = ResidualMLPDecoder(config.output_layers)
        print("StatefulLM initialized, total params: {}".format(count_parameters(self)))
 

    def forward(self, inputs, cache=None, need_cache=True, T=1.0, update_memory=True):
        """
        Inputs: [B, NT]
        Outputs: [B, NT, H]
        """ 
        
        outputs = self.word_emb(inputs)
        # print(f"Embedding dtype: {outputs.dtype}")
        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache)
        outputs = self.output_mapping(outputs, T=T)
        return outputs, new_cache
    
    def get_mem(self):
        return self.causal_model.get_mem()
    
    def set_mem(self, mem_dict):
        self.causal_model.set_mem(mem_dict)

    def reset(self):
        self.causal_model.reset()

    def perplexity(self, inputs, outputs, use_loss_weight=True, update_memory=True, reduce_dim=1, masks=None):
        logits, cache = self.forward(inputs, need_cache=False, update_memory=update_memory)

        loss = dict()
        if use_loss_weight:
            loss["ce_loss"], loss["count"] = weighted_loss(logits, gt=outputs, loss_type="ce", gamma=0, 
                             loss_wht=masks, reduce_dim=reduce_dim, need_cnt=True)
        else:
            loss["ce_loss"], loss["count"] = weighted_loss(logits, gt=outputs, loss_type="ce", gamma=0, 
                             reduce_dim=reduce_dim, need_cnt=True)
        # loss["perplexity"] = torch.exp(loss["ce_loss"])
        # # if loss has inf or nan, then print the loss
        # if torch.isinf(loss["perplexity"]).any() or torch.isnan(loss["perplexity"]).any():
        #     print(loss["perplexity"])
        #     print(loss["ce_loss"])
        #     print(loss["count"])
        #     print("----------------------------")
        # assert loss["perplexity"].shape == loss["ce_loss"].shape, print(loss["perplexity"].shape, loss["ce_loss"].shape)
        
        return loss, cache
    
    def inference_seg(self, inputs, L, 
                      temp_default=1.0, 
                      temp_setting=None, 
                      cache=None):
        with torch.no_grad():
            sampled_outputs = inputs
            outputs = inputs
            T = temp_default
            # import pdb;pdb.set_trace()
            for _ in range(L):
                logits, cache = self.forward(sampled_outputs, cache=cache, need_cache=True, T=T)
                logits = logits.view(-1, logits.shape[-1])
                sampled_outputs = torch.multinomial(logits, num_samples=1)
                sampled_outputs = sampled_outputs.view(inputs.shape[0], -1)
                outputs = torch.cat([outputs, sampled_outputs], dim=-1)
                sampled_outputs = sampled_outputs[:,-1:]
                if(temp_setting is not None):
                    assert sampled_outputs.shape[0] == 1, "T_setting is only for batch_size=1"
                    token = sampled_outputs[0][-1].item()
                    if token in temp_setting:
                        T = temp_setting[token]
                    else:
                        T = temp_default
        return outputs

class DualTrackStatefulLM(StatefulLM):
    def __init__(self, config, verbose):
        super().__init__(config, verbose)
        self.causal_model = DualTrackCausalBlock(config.causal_block)
    def merge_memory(self):
        self.causal_model.merge_memory()
    
    def reset(self, stateful_reset: bool):
        self.causal_model.reset(stateful_reset=stateful_reset)

if __name__=="__main__":
    from airsoul.utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = StatefulLM(config.model_config).cuda()
    inputs = torch.randint(32, (4, 256)).cuda()
    losses = model.perplexity(inputs[:, :-1], inputs[:, 1:])
    outputs = model.inference_seg(inputs, 4)
    print(losses)
    print(outputs.shape)
