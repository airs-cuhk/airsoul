import os
import re
import sys
import yaml
import numpy
import torch
from torch.nn.utils import clip_grad_norm_
from restools.logging import log_warn, log_debug, log_progress, log_fatal, Logger
from restools.configure import Configure

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def import_with_caution(module_name):
    try:
        module = __import__(module_name)
    except ImportError:
        module = None
        print(f'Warning: module {module_name} does not exist')
    return module

def log_sum_parameters_grad(model, rank=None):
    l1 = 0.0
    l2 = 0.0

    for p in model.parameters():
        if(p.grad is not None):
            l2 += torch.sqrt(torch.sum(p.grad**2)).item()
            l1 += torch.sum(torch.abs(p.grad)).item()

    log_debug(f"parameter grad for {rank}: l1={l1}, l2={l2}")

def safety_check(tensor, replacement=None, msg=None, on=True):
    has_inf = torch.isinf(tensor)
    has_nan = torch.isnan(tensor)
    has_large = tensor ** 2 > 1.0e+6

    risk_level = 0
    if(has_large.any()):
        risk_level=max(risk_level, 1)
        if(replacement is None):
            pass
        else:
            tensor[has_large] = replacement
        if(msg is not None):
            log_warn(f"{msg}\tl2_norm={l2_norm}", on=on)
    if has_inf.any():
        risk_level=max(risk_level, 2)
        if(replacement is None):
            pass
        else:
            tensor[has_inf] = replacement
        if(msg is not None):
            log_warn(f"{msg}\tINF encounted", on=on)
    if has_nan.any():
        risk_level=max(risk_level, 3)
        if(replacement is None):
            pass
        else:
            tensor[has_nan] = replacement
        if(msg is not None):
            log_warn(f"{msg}\tNAN encounted", on=on)

    return tensor, risk_level

def format_cache(cache, prefix=''):
    if(cache is None):
        return prefix + ' None'
    elif(isinstance(cache, numpy.ndarray) or isinstance(cache, torch.Tensor)):
        return prefix + ' ' + str(cache.shape)
    elif(isinstance(cache, list) or isinstance(cache, tuple)):
        ret_str = prefix + f'List of length {len(cache)}:['
        for subc in cache:
            ret_str += format_cache(subc)
            ret_str += ';'
        ret_str += ']'
        return ret_str
    elif(isinstance(cache, dict)):
        ret_str = prefix + 'Dicts{'
        for key in cache:
            ret_str += format_cache(cache[key], prefix + f'{key}:')
            ret_str += ';'
        ret_str += '}'
        return ret_str
    else:
        return prefix + ' ' + str(type(cache))

def memory_cpy(cache):
    if(cache is None):
        return None
    elif(isinstance(cache, torch.Tensor)):
        return cache.detach().clone()
    elif(isinstance(cache, list)):
        return [memory_cpy(c) for c in cache]
    elif(isinstance(cache, dict)):
        return {k:memory_cpy(cache[k]) for k in cache}
    elif(isinstance(cache, tuple)):
        return tuple([memory_cpy(c) for c in cache])
    elif(hasattr(cache, 'clone')):
        return cache.clone()
    else:
        return cache

def model_path(save_model_path, epoch_id, *optimizers):
    directory_path = '%s/ckpt_%02d/' % (save_model_path, epoch_id)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if(len(optimizers) < 1):
        return f'{directory_path}/model.pth'
    else:
        ret_str = [f'{directory_path}/model.pth']
        for opt in optimizers:
            ret_str.append(f'{directory_path}/{opt}.pth')
    return tuple(ret_str)

def reset_optimizer_state(optimizer):
    state = optimizer.state
    for k in list(state.keys()):
        for sk in list(state[k].keys()):
            state[k][sk].zero_()

def apply_gradient_safely(model, optimizer, scaler=None, clip_norm=1.0):
    # Clip graident first
    clip_grad_norm_(model.parameters(), clip_norm)

    overflow=False
    for name, param in model.named_parameters():
        if (param.grad is not None):
            _, risk = safety_check(param.grad, msg=f"gradient_[{name}]")
            if(risk > 1):
                param.grad.zero_()
                overflow=True
    if(overflow):
        reset_optimizer_state(optimizer)
        if(scaler is not None):
            scaler.unscale_(optimizer)

    if(scaler is None):
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()

def print_memory(info="Default"):
    log_debug(info, "Memory allocated:",
            torch.cuda.memory_allocated(), 
            "Memory cached:", 
            torch.cuda.memory_cached())

def custom_load_model(model, 
                      state_dict_path, 
                      black_list=[], 
                      strict_check=False, 
                      verbose=False):  
    """
    Load model from path with safety
    black_list: a list of strings that will be ignored when loading (partial matching)
    strick_check: if true, parameters with NAN/INF and mismatching shape will cause error
        otherwise, they will be replaced by zero and matched with maximum shared parts
    """
    saved_state_dict = torch.load(state_dict_path, weights_only=False)  
    matched_state_dict = {} 

    # Notice: load only trainable parameters
    for param_name, param_tensor in model.named_parameters():
        model_param_shape = param_tensor.shape  
        if param_name in saved_state_dict:
            # check wheter parameters is in black list 
            hits_black = False
            for name in black_list:
                if(param_name.find(name) > -1):
                    log_warn(f"Loading parameter {param_name} hits blacklist name {name}, skip", on=verbose)
                    hits_black = True
            if(hits_black):
                continue

            # check whether there are abnormal parameters
            if(strict_check):
                safe_param, risk = safety_check(saved_state_dict[param_name], 
                                                msg=f"loading parameters: {param_name}",
                                                on=verbose)
                if(risk > 1):
                    log_fatal(e, "Quit Job...")
            else:
                safe_param, risk = safety_check(saved_state_dict[param_name], 
                                                replacement=0, 
                                                msg=f"loading parameters with replacement: {param_name}",
                                                on=verbose)
                if(risk > 1):
                    log_warn(f"keep Loading...", on=verbose)

            # check whehter the shapes match
            # if not, we truncate the tensor to match the model's shape
            if model_param_shape == safe_param.shape:  
                matched_state_dict[param_name] = safe_param
            else:  
                e = f"Loading {param_name} Shape mismatch: requires {model_param_shape}; gets {safe_param.shape}"  
                if(strict_check):
                    log_fatal(e, "Quit Job...")
                else:
                    if(safe_param.ndim != len(model_param_shape)):
                        log_warn(e, "Skipping loading...", on=verbose)
                    else:
                        minimal_shape = []
                        for ns,nt in zip(safe_param.shape, model_param_shape):
                            minimal_shape.append(min(ns,nt))
                        minimal_match = param_tensor.clone()
                        match_inds = tuple(slice(0, n) for n in minimal_shape)
                        minimal_match[match_inds] = safe_param[match_inds]
                        matched_state_dict[param_name] = minimal_match
                        log_warn(e, 
                                f"Apply fractional loading with shape {minimal_shape}...",
                                on=verbose)
        else:  
            e = f"Current parameters {param_name} not found in the checkpoint"
            if(strict_check):
                raise log_fatal(e, "Quit Job...")
            else:
                log_warn(e, "Skipping loading...", on=verbose)
      
    model.load_state_dict(matched_state_dict, strict=False)  
    log_debug("-" * 20, f"Load model success", "-" * 20, on=verbose)
    return model  

def check_model_validity(model, verbose=False, level=1):
    """
    Check the validity of model in RunTime
    level: 0 -> only accept all parameters l2 norm < 1e+6
           1 -> only accept all valid parameters
    """
    param_isnormal = dict()
    max_risk = 0
    for param_name, param_tensor in model.named_parameters():
        if(not param_tensor.requires_grad):
            continue # Neglect non-trainable parameters

        safe_param, risk = safety_check(param_tensor, 
                                        msg=f"checking parameters: {param_name}",
                                        on=verbose)
        max_risk = max(risk, max_risk)

    return (max_risk > level)

