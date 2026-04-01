#!/usr/bin/env python
# A wrapper that wraps the model with block-recurrence

# coding=utf8
# File: models.py
import sys
import random
import torch
import numpy
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from airsoul.utils import memory_cpy, format_cache, log_warn, log_fatal


class BlockRecurrentWrapper(nn.Module):
    """
    Wrapping a temporal modeler with a memory cache to make it block-recurrent
    """
    def __init__(self, temporal_module, memory_length, memory_type='kv'):
        """
        Memory_Type: "kv", "mem"
        """
        super().__init__()

        self.reset()
        self.temporal_module = temporal_module
        self.mem_len = memory_length
        self.memory_type = memory_type.lower()

    def reset(self):
        # This will clear the memory and the cache
        self.memory = None
        # Position will be synchronized with the memory
        self.position = 0
        
    def merge_memory_in_cache(self, cache):
        if(self.memory_type == "kv"):
            if(cache is not None and self.memory is not None):
                new_cache = []
                for mem, ca in zip(self.memory, cache):
                    new_cache.append(torch.cat((mem, ca), dim=1))
            elif(self.memory is not None):
                new_cache = self.memory
            else:
                new_cache = memory_cpy(cache)
            return new_cache
        elif(self.memory_type == "mem"):
            if(cache is not None):
                new_cache = memory_cpy(cache)
            else:
                new_cache = self.memory
            return new_cache

    def update_memory_cache(self, cache):
        # Updates the Memory and Cache
        # For KV cache, in case the memory + cache > 2 * memory_length, we update the memory
        # Else, we keep the cache and the memory
        # We always keep memory detached and independent from the computation graph
        if(self.memory_type == "kv"):
            if(cache is not None):
                self.memory = [c[:, -self.mem_len:].detach().clone() for c in cache]
            else:
                self.memory = None
        elif(self.memory_type == "mem"):
            # Just update the memory and the cache
            self.memory = memory_cpy(cache)
        else:
            log_fatal(f"No such memory type: {self.memory_type}")
        return None

    def update_cache_only(self, cache):
        if(self.memory_type == 'kv'):
            if(self.memory is None):
                return memory_cpy(cache)
            else:
                new_cache = []
                for m,c in zip(self.memory, cache):
                    m_len = m.shape[1]
                    new_cache.append(c[m_len:].detach().clone())
                return new_cache
        elif(self.memory_type == "mem"):
            return memory_cpy(cache)
        else:
            log_fatal(f"No such memory type: {self.memory_type}")
    def get_o_list(self):
        return self.temporal_module.get_o_list()
    def forward(self, src, cache=None, need_cache=False, verbose=True, checkpoints_density=-1, update_memory=True):
        # when update memory = False, inference won't update the memory, but will update the cache
        # by default the shape of src should be (batch_size, seq_len, dim)

        output, new_cache = self.temporal_module.forward(
                src, 
                cache=self.merge_memory_in_cache(cache), 
                need_cache=True, 
                checkpoints_density=checkpoints_density)
        # print("block-recurrent-wrapper: new_cache", new_cache[0].keys())
        # print(len(new_cache[0]['recurrent_state']))
        # print(new_cache[0]['recurrent_state'][0].shape)

        if(update_memory):
            new_cache = self.update_memory_cache(new_cache)
            # Update the position at the same time
            self.position += src.shape[1]
            if need_cache:
                new_cache = memory_cpy(self.memory)
        elif(need_cache):
            new_cache = self.update_cache_only(new_cache)
        else:
            new_cache = None

        return output, new_cache
    
    def get_mem(self):
        return {
            'memory': memory_cpy(self.memory), 
            'position': self.position
        }

    def set_mem(self, memory_dict):
        self.memory = memory_dict.get('memory')
        self.position = memory_dict.get('position', 0)

    def merge_memory(self):
        pass

class DualTrackBlockRecurrentWrapper(nn.Module):
    """
    DualTrack Block-Recurrent Wrapper: Package the dual-track timing module to manage 
    the memory of short-term and long-term respectively
    """
    
    def __init__(self, temporal_module, memory_length_short, memory_length_long, 
                 memory_type='kv'):
        """
        Args:
            temporal_module: DualTrackGatedDeltaNet 
            memory_length_short: short-term 
            memory_length_long: long-term 
            memory_type: 'kv' or 'mem'
        """
        super().__init__()
        
        self.temporal_module = temporal_module
        
        self.wrapper_short = BlockRecurrentWrapper(
            temporal_module.short_term_encoder, memory_length_short, memory_type)
        self.wrapper_long = BlockRecurrentWrapper(
            temporal_module.long_term_encoder, memory_length_long, memory_type)

    def forward(self, src, cache=None, need_cache=False, 
                checkpoints_density=-1, update_memory=True):
        
        # Short-term
        out_short, new_cache_short = self.wrapper_short.forward(
            src, cache=cache, need_cache=need_cache,
            checkpoints_density=checkpoints_density, update_memory=update_memory)
        
        # Long-term
        out_long, new_cache_long = self.wrapper_long.forward(
            src, cache=cache, need_cache=need_cache,
            checkpoints_density=checkpoints_density, update_memory=update_memory)
        
        return out_short, out_long, new_cache_short, new_cache_long
    
    def get_o_list(self):
        return self.temporal_module.get_o_list()

    def get_mem(self):
        mem_dict_short= self.wrapper_short.get_mem()
        mem_dict_long = self.wrapper_long.get_mem()
        return {
            'mem_dict_short': mem_dict_short, 
            'mem_dict_long': mem_dict_long
        }

    def set_mem(self, mem_dict):
        if mem_dict.get('mem_dict_short') is not None:
            self.wrapper_short.set_mem(mem_dict['mem_dict_short'])
        if mem_dict.get('mem_dict_long') is not None:
            self.wrapper_long.set_mem(mem_dict['mem_dict_long'])
    
    def reset(self, stateful_reset=True):
        self.wrapper_short.reset()
        if not stateful_reset:
            self.wrapper_long.reset()

    def merge_memory(self, return_gate_stats=False, detach_fusion=False):
        """
        融合 short-term memory 到 long-term memory
        遍历每一层，逐层调用 DualTrackGatedDeltaNet.merge_memory
        """
        mem_dict = self.get_mem()
        short_memory_list = mem_dict['mem_dict_short']['memory']  # list of cache dict
        long_memory_list = mem_dict['mem_dict_long']['memory']    # list of cache dict
        
        multi_blocks = self.wrapper_long.temporal_module
        layers = multi_blocks.layers
        num_layers = len(layers)
        merged_long_memory_list = []
        all_gate_stats = []

        num_layers = len(short_memory_list)
        merged_long_memory_list = []
        all_gate_stats = []
        
        for layer_idx in range(num_layers):
            # 每个 layer 是 BlockWrapper，其 temporal_encoder 是 DualTrackGatedDeltaNet
            layer = layers[layer_idx]
            dual_track_attention_block_module = layer.temporal_encoder  # DualTrackGatedDeltaNet
            
            short_cache = short_memory_list[layer_idx]
            long_cache = long_memory_list[layer_idx]
            
            # 调用该层的 DualTrackGatedDeltaNet.merge_memory
            result = dual_track_attention_block_module.merge_memory(
                short_cache, 
                long_cache,
                return_gate_stats=return_gate_stats,
                detach_fusion=detach_fusion
            )
            
            if return_gate_stats:
                merged_cache, gate_stats = result
                all_gate_stats.append(gate_stats)
            else:
                merged_cache = result
            
            merged_long_memory_list.append(merged_cache)
        
        # 更新 long-term memory
        self.wrapper_long.set_mem({
            'memory': merged_long_memory_list,
            'position': mem_dict['mem_dict_long'].get('position', 0)
        })
        
        if return_gate_stats:
            summary_stats = {
                'g_mean': sum(s['g_mean'] for s in all_gate_stats) / len(all_gate_stats),
                'g_std': sum(s['g_std'] for s in all_gate_stats) / len(all_gate_stats)
            }
            return summary_stats
        
        return None