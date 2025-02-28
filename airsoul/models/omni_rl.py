#!/usr/bin/env python
# coding=utf8
# File: models.py
import sys
import random
import torch
import numpy
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint  
from l3c_baselines.utils import weighted_loss, sa_dropout, img_pro, img_post
from l3c_baselines.utils import parameters_regularization, count_parameters
from l3c_baselines.utils import log_debug, log_warn, log_fatal
from l3c_baselines.modules import ImageEncoder, ImageDecoder
from .decision_model import OPTARDecisionModel

class OmniRL(OPTARDecisionModel):
    def __init__(self, config, verbose=False): 
        super().__init__(config)

        # Loss weighting
        loss_weight = torch.cat( (torch.linspace(1.0e-3, 1.0, config.context_warmup),
                                  torch.full((config.max_position_loss_weighting - config.context_warmup,), 1.0)
                                  ), 
                                dim=0)
        loss_weight = loss_weight / torch.sum(loss_weight)

        self.register_buffer('loss_weight', loss_weight)

        self.nactions = config.action_dim
        self.state_dtype = config.state_encode.input_type
        self.reward_dtype = config.reward_encode.input_type
        self.action_dtype = config.action_encode.input_type

        if(config.reward_encode.input_type == "Discrete"):
            self.default_r = torch.full(config.reward_encode.input_size, (1, 1), dtype=torch.int64)
        elif(self.config.reward_encode.input_type == "Continuous"):
            self.default_r = torch.zeros((1, 1, config.reward_encode.input_size))
        else:
            raise ValueError("Invalid reward encoding type", config.reward_encoding)
        
        if(config.action_encode.input_type == "Discrete"):
            self.default_a = torch.full((1, 1), config.action_encode.input_size, dtype=torch.int64)
        elif(config.action_encode.input_type == "Continuous"):
            self.default_a = torch.zeros((1, 1, config.action_encode.input_size))
        else:
            raise ValueError("Invalid reward encoding type", config.action_encoding)

        if(verbose):
            log_debug("RSA Decision Model initialized, total params: {}".format(count_parameters(self)))
            log_debug("Causal Block Parameters： {}".format(count_parameters(self.causal_model)))

    def sequential_loss(self, observations, 
                            prompts,
                            tags,
                            behavior_actions, 
                            rewards, 
                            label_actions, 
                            state_dropout=0.0,
                            update_memory=True,
                            use_loss_weight=True,
                            reduce_dim=1):
        bsz = behavior_actions.shape[0]
        seq_len = behavior_actions.shape[1]
        # Pay attention position must be acquired before calling forward()
        ps = self.causal_model.position // self.rsa_occ
        pe = ps + seq_len
        o_in = sa_dropout(observations[:, :-1].clone())
        # Predict the latent representation of action and next frame (World Model)
        s_pred, a_pred, r_pred, _ = self.forward(
                o_in, prompts, tags, behavior_actions, rewards,
                cache=None, need_cache=False,
                update_memory=update_memory)

        # Calculate the loss information
        loss = dict()
        # Mask out the invalid actions
        loss_weight_s = None
        loss_weight_a = (label_actions.ge(0) * label_actions.lt(self.nactions)).to(
                    self.loss_weight.dtype)
        if(use_loss_weight):
            loss_weight_s = self.loss_weight[ps:pe]
            loss_weight_a = loss_weight_a * self.loss_weight[ps:pe].unsqueeze(0)

        # World Model Loss - States and Rewards
        if self.state_dtype == "Continuous":
            if observations.dim() == 2:
                gt = observations[:, 1:].view(*observations.shape, -1)
            else:
                gt = observations[:, 1:]
            loss["wm-s"], loss["count_s"] = weighted_loss(s_pred, 
                                        gt=gt, 
                                        loss_type="mse",
                                        loss_wht=loss_weight_s, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)
        else:
            loss["wm-s"], loss["count_s"] = weighted_loss(s_pred, 
                                        gt=observations[:, 1:], 
                                        loss_type="ce",
                                        loss_wht=loss_weight_s, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)
        loss["wm-r"] = weighted_loss(r_pred, 
                                     gt=rewards.view(*rewards.shape,1), 
                                     loss_type="mse",
                                     loss_wht=loss_weight_a,
                                     reduce_dim=reduce_dim)

        # Policy Model
        if self.action_dtype == "Continuous":
            if label_actions.dim() == 2:
                gt = label_actions.view(*rewards.shape, 1)
            else:
                gt = label_actions
            loss["pm"], loss["count_a"] = weighted_loss(a_pred, 
                                       gt=gt, 
                                       loss_type="mse",
                                       loss_wht=loss_weight_a, 
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)
        else:
            loss["pm"], loss["count_a"] = weighted_loss(a_pred, 
                                    gt=label_actions, 
                                    loss_type="ce",
                                    loss_wht=loss_weight_a, 
                                    reduce_dim=reduce_dim,
                                    need_cnt=True)
        # Entropy Loss
        loss["ent"] = weighted_loss(a_pred, 
                                    loss_type="ent", 
                                    loss_wht=loss_weight_a,
                                    reduce_dim=reduce_dim)
        loss["causal-l2"] = parameters_regularization(self)
        return loss
    
    def generate(self, observation,
                    prompt,
                    tag,
                    temp,
                    need_numpy=True,
                    single_batch=True,
                    future_prediction=False):
        """
        Generating Step By Step Action and Next Frame Prediction
        Args:
            observation 
            prompts: None if not included
            tags: None if not included
            temp: temperature for sampling
            single_batch: if true, add additional batch to input tensor
        Returns:
            o_pred: predicted states, only valid if future_prediction is True
            a_pred: predicted actions 
            r_pred: predicted rewards, only valid if future_prediction is True
        """
        device = next(self.parameters()).device

        # Prepare the input prompts
        if(not self.p_included):
            pro_in = None
        elif(not isinstance(prompt, torch.Tensor)):
            pro_in = torch.tensor([prompt], dtype=torch.int64).to(device)
        else:
            pro_in = prompt.to(device)
        
        # Prepare the input tags
        if(not self.t_included):
            tag_in = None
        elif(not isinstance(tag, torch.Tensor)):
            tag_in = torch.tensor([tag], dtype=torch.int64).to(device)
        else:
            tag_in = tag.to(device)

        # Prepare the input observations
        if(not isinstance(observation, torch.Tensor)):
            obs_in = torch.tensor([observation], dtype=torch.int64).to(device)
        else:
            obs_in = observation.to(device)

        if(single_batch):
            if(pro_in is not None):
                pro_in = pro_in.unsqueeze(0)
            if(tag_in is not None):
                tag_in = tag_in.unsqueeze(0)
            obs_in = obs_in.unsqueeze(0)

        if(self.r_included):
            default_r = self.default_r.to(device)
        else:
            default_r = None
        default_a = self.default_a.to(device)

        o_pred, a_pred, r_pred, _ = self.forward(
            obs_in,
            pro_in,
            tag_in,
            default_a,
            default_r,
            T=temp,
            update_memory=False,
            need_cache=False)
        
        if(self.a_discrete):
            act_in = a_pred / a_pred.sum(dim=-1, keepdim=True)
            act_in = torch.multinomial(act_in.squeeze(1), num_samples=1)
            act_out = act_in.squeeze()
        else:
            act_in = a_pred
            act_out = act_in.squeeze()

        act_out = act_out.detach().cpu().squeeze()
        if(need_numpy):
            act_out = act_out.numpy()
            if(act_out.size < 2):
                act_out = act_out.item()

        if(future_prediction):
            o_pred, a_pred, r_pred, _ = self.forward(
                obs_in,
                pro_in,
                tag_in,
                act_in,
                default_r,
                T=temp,
                update_memory=False,
                need_cache=False)
            
            state = o_pred.detach().cpu().squeeze()
            reward = r_pred.detach().cpu().squeeze()

            if(need_numpy):
                state = state.numpy()
                if(state.size < 2):
                    state = state.item()
                reward = reward.numpy()
                if(reward.size < 2):
                    reward = reward.item()

        else:
            state = None
            reward = None

        return state, act_out, reward

    def in_context_learn(self, observation,
                    prompts,
                    tags,
                    action,
                    reward,
                    cache=None,
                    need_cache=False,
                    single_batch=True,
                    single_step=True):
        """
        In Context Reinforcement Learning Through an Sequence of Steps
        """
        device = next(self.parameters()).device
        pro_in = None
        obs_in = None

        def proc(x):
            if(x is None):
                return x
            if(single_batch and single_step):
                return x.unsqueeze(0).unsqueeze(0).to(device)
            elif(single_batch):
                return x.unsqueeze(0).to(device)
            elif(single_step):
                return x.unsqueeze(1).to(device)
            return x.to(device)

        obs_in = observation
        pro_in = prompts
        tag_in = tags
        act_in = action
        rew_in = reward

        if(not isinstance(obs_in, torch.Tensor)):
            obs_in = torch.tensor(obs_in)
        obs_in = proc(obs_in)
        if(pro_in is not None and not isinstance(pro_in, torch.Tensor)):
            pro_in = torch.tensor(pro_in)
        pro_in = proc(pro_in)
        if(tag_in is not None and not isinstance(tag_in, torch.Tensor)):
            tag_in = torch.tensor(tag_in)
        tag_in = proc(tag_in)
        if(not isinstance(action, torch.Tensor)):
            act_in = torch.tensor(act_in)
        act_in = proc(act_in)
        if(not isinstance(reward, torch.Tensor) and reward is not None):
            rew_in = torch.tensor(rew_in)
        rew_in = proc(rew_in)

        if self.reward_dtype == "Continuous":
            rew_in = rew_in.to(torch.float32)
        else:
            rew_in = rew_in.to(torch.int32)

        # observation, prompt, tag, action, reward; update memory = true
        _, _, _, new_cache = self.forward(
            obs_in,
            pro_in,
            tag_in,
            act_in,
            rew_in,
            need_cache=need_cache,
            update_memory=True)
        
        return new_cache

if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = AnyMDPRSA(config.model_config)

    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(None, observation, reward, action, action)
    rec_img, img_out, act_out, cache = model.inference_step_by_step(
            observation[:, :5], action[:, :4], 1.0, 0, observation.device)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out[0].shape, act_out.shape)
    print(len(cache))
    print(cache[0].shape)
