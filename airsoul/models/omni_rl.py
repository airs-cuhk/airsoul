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
from airsoul.utils import weighted_loss, sa_dropout, img_pro, img_post
from airsoul.utils import parameters_regularization, count_parameters
from airsoul.utils import log_debug, log_warn, log_fatal
from airsoul.modules import ImageEncoder, ImageDecoder
from .decision_model import POTARDecisionModel

class OmniRL(POTARDecisionModel):
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
            log_debug("Causal Block Parameters: {}".format(count_parameters(self.causal_model)))

    def sequential_loss(self, observations, 
                            prompts,
                            tags,
                            behavior_actions, 
                            rewards, 
                            label_actions, 
                            state_dropout=0.0,
                            update_memory=True,
                            use_loss_weight=True,
                            is_training=True,
                            reduce_dim=1):
        bsz = behavior_actions.shape[0]
        seq_len = behavior_actions.shape[1]
        # Pay attention position must be acquired before calling forward()
        ps = self.causal_model.position // self.rsa_occ
        pe = ps + seq_len
        o_in = sa_dropout(observations[:, :-1].clone())
        # Predict the latent representation of action and next frame (World Model)
        wm_out, pm_out, _ = self.forward(
                o_in, prompts, tags, behavior_actions, rewards,
                cache=None, need_cache=False,
                update_memory=update_memory)
        s_pred, a_pred, r_pred = self.post_decoder(wm_out, pm_out)
        # Calculate the loss information
        loss = dict()
        # Mask out the invalid actions
        if(self.loss_weight.shape[0] < pe):
            log_fatal(f"Loss weight (shape {self.loss_weight.shape[0]}) should be longer" +
                    f" than sequence length {pe}")

        loss_weight_s = None
        loss_weight_a = None
        if(use_loss_weight):
            loss_weight_s = self.loss_weight[ps:pe]
            if self.action_dtype == "Discrete":
                loss_weight_a = (label_actions.ge(0) * label_actions.lt(self.nactions)).to(self.loss_weight.dtype)
                loss_weight_a = loss_weight_a * self.loss_weight[ps:pe].unsqueeze(0)
            elif self.action_dtype == "Continuous":
                loss_weight_a = self.loss_weight[ps:pe]

        # World Model Loss - States and Rewards
        if self.state_dtype == "Discrete":
            loss["wm-s"], loss["count_s"] = weighted_loss(s_pred, 
                                        gt=observations[:, 1:], 
                                        loss_type="ce",
                                        loss_wht=loss_weight_s, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)       
        elif self.state_dtype == "Continuous" and self.config.state_diffusion.enable:
            if is_training: # If training
                if self.config.state_diffusion.prediction_type == "sample":
                    s_latent = self.s_diffusion.loss_DDPM(x0=self.s_encoder(observations[:, 1:]), cond=wm_out)
                    s_pred = self.s_decoder(s_latent)
                    loss["wm-s"], loss["count_s"] = weighted_loss(s_pred, 
                                                gt=observations[:, 1:], 
                                                loss_type="mse",
                                                loss_wht=loss_weight_s, 
                                                reduce_dim=reduce_dim,
                                                need_cnt=True)
                else:   
                    loss["wm-s"], loss["count_s"] = self.s_diffusion.loss_DDPM(x0=self.s_encoder(observations[:, 1:]),
                                                cond=wm_out,
                                                mask=loss_weight_s,
                                                reduce_dim=reduce_dim,
                                                need_cnt=True)
            else: # If testing 
                s_latent = self.s_diffusion.inference(cond=wm_out)[-1]
                s_pred = self.s_decoder(s_latent)
                loss["wm-s"], loss["count_s"] = weighted_loss(s_pred, 
                                        gt=observations[:, 1:], 
                                        loss_type="mse",
                                        loss_wht=loss_weight_s, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)

        elif self.state_dtype == "Continuous" and not self.config.state_diffusion.enable:
            loss["wm-s"], loss["count_s"] = weighted_loss(s_pred,
                loss_type="mse",
                gt=observations[:, 1:],
                loss_wht=loss_weight_s,
                reduce_dim=reduce_dim,
                need_cnt=True)

        # loss["wm-r"] = weighted_loss(r_pred,
        #                                  gt=rewards.view(*rewards.shape,1),
        #                                  loss_type="mse",
        #                                  loss_wht=loss_weight_a,
        #                                  reduce_dim=reduce_dim)

        loss["wm-r"] = 0

        # Policy Model
        if self.action_dtype == "Discrete":
            loss["pm"], loss["count_a"] = weighted_loss(a_pred, 
                                    gt=label_actions, 
                                    loss_type="ce",
                                    loss_wht=loss_weight_a, 
                                    reduce_dim=reduce_dim,
                                    need_cnt=True)
        elif self.action_dtype == "Continuous" and self.config.action_diffusion.enable:
            if is_training: # If training
                if self.config.action_diffusion.prediction_type == "sample":
                    a_latent = self.a_diffusion.loss_DDPM(x0=self.a_encoder(label_actions),cond=pm_out)
                    a_pred = self.a_decoder(a_latent)
                    loss["pm"], loss["count_a"] = weighted_loss(a_pred, 
                                        gt=label_actions, 
                                        loss_type="mse",
                                        loss_wht=loss_weight_a, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)
                else: 
                    loss["pm"], loss["count_a"] = self.a_diffusion.loss_DDPM(x0=self.a_encoder(label_actions),
                                                cond=pm_out,
                                                mask=loss_weight_a,
                                                reduce_dim=reduce_dim,
                                                need_cnt=True)
            else: # If testing
                a_latent = self.a_diffusion.inference(cond=pm_out)[-1]
                a_pred = self.a_decoder(a_latent)
                loss["pm"], loss["count_a"] = weighted_loss(a_pred, 
                                       gt=label_actions, 
                                       loss_type="mse",
                                       loss_wht=loss_weight_a, 
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)

        elif self.action_dtype == "Continuous" and not self.config.action_diffusion.enable:
            loss["pm"], loss["count_a"] = weighted_loss(a_pred,
                loss_type="mse",
                gt=label_actions,
                loss_wht=loss_weight_a,
                reduce_dim=reduce_dim,
                need_cnt=True)
        # Entropy Loss
        if self.action_dtype == "Discrete" :
            loss["ent"] = weighted_loss(a_pred, 
                                        loss_type="ent", 
                                        loss_wht=loss_weight_a,
                                        reduce_dim=reduce_dim)
        else:
            loss["ent"] = 0.0
        
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
        elif not isinstance(prompt, torch.Tensor):
            # Handle non-tensor input (e.g., list or scalar) for batch=1
            pro_in = torch.tensor(prompt, dtype=torch.float32).to(device)
            # Ensure at least 2D input (dim_p,) -> (1, dim_p)
            if pro_in.dim() < 2:
                # (dim_p,) -> (1, dim_p)
                pro_in = pro_in.unsqueeze(0)
            # (1, dim_p) -> (1, 1, dim_p)
            pro_in = pro_in.unsqueeze(1)
        else:
            pro_in = prompt.to(device)
            # Ensure (batch, dim_p) -> (batch, 1, dim_p)
            pro_in = pro_in.unsqueeze(1)
        
        # Prepare the input tags
        if(not self.t_included):
            tag_in = None
        elif(not isinstance(tag, torch.Tensor)):
            # Handle non-tensor input (e.g., list [0, 1, 2, 3, 4, 5, 6, 7])
            tag_in = torch.tensor(tag, dtype=torch.int64).to(device)
            # Ensure (batch,) -> (batch, 1)
            if tag_in.dim() == 1:
                tag_in = tag_in.unsqueeze(-1)
            # (batch, 1) -> (batch, 1, 1)
            tag_in = tag_in.unsqueeze(1)
        else:
            tag_in = tag.to(device)
            # Handle (batch,) or (batch, 1) or (1, batch)
            if tag_in.dim() == 1:
                 # (batch,) -> (batch, 1)
                tag_in = tag_in.unsqueeze(-1)
            elif tag_in.shape[0] == 1 and tag_in.shape[1] > 1:
                 # (1, batch) -> (batch, 1)
                tag_in = tag_in.transpose(0, 1)
            # (batch, 1) -> (batch, 1, 1)
            tag_in = tag_in.unsqueeze(1)

        # Prepare the input observations
        if(not isinstance(observation, torch.Tensor)):
            # Handle non-tensor input (e.g., list) for batch=1
            obs_in = torch.tensor(observation, dtype=torch.float32).to(device)
            # Ensure at least 2D input (dim,) -> (1, dim)
            if obs_in.dim() < 2:
                # (dim,) -> (1, dim)
                obs_in = obs_in.unsqueeze(0)
            # (1, dim) -> (1, 1, dim)
            obs_in = obs_in.unsqueeze(1)
        else:
            obs_in = observation.to(device)
            # Ensure (batch, dim) -> (batch, 1, dim)
            obs_in = obs_in.unsqueeze(1)

        B, T = obs_in.shape[:2]

        # Prepare default rewards
        if(self.r_included):
            if(self.reward_dtype == "Discrete"):
                # Shape (batch, 1)
                default_r = self.default_r.to(device=device).expand(B, T)
            elif(self.reward_dtype == "Continuous"):
                # Shape (batch, 1, R)
                default_r = self.default_r.to(device=device).expand(B, T, -1)
        else:
            default_r = None

        # Prepare default actions
        if(self.action_dtype == "Discrete"):
            # Shape (batch, 1)
            default_a = self.default_a.to(device).expand(B, T)
        elif(self.action_dtype == "Continuous"):
            # Shape (batch, 1, A)
            default_a = self.default_a.to(device).expand(B, T, -1)

        wm_out, pm_out, _ = self.forward(
            obs_in,
            pro_in,
            tag_in,
            default_a,
            default_r,
            T=temp,
            update_memory=False,
            need_cache=False)
        
        o_pred, a_pred, r_pred = self.post_decoder(wm_out, pm_out, T=temp)
        
        if not self.config.action_diffusion.enable:
            if(self.a_discrete):
                act_in = a_pred / a_pred.sum(dim=-1, keepdim=True)
                act_in = torch.multinomial(act_in.squeeze(1), num_samples=1)
                act_out = act_in.squeeze()
            else:
                act_in = a_pred
                act_out = act_in.squeeze()
        else:
            a_latent = self.a_diffusion.inference(cond=pm_out)[-1]
            act_in = self.a_decoder(a_latent)
            act_out = act_in.squeeze()

        act_out = act_out.detach().cpu().squeeze()
        if(need_numpy):
            act_out = act_out.numpy()
            if(act_out.size < 2):
                act_out = act_out.item()

        if(future_prediction):
            wm_out, pm_out, _ = self.forward(
                obs_in,
                pro_in,
                tag_in,
                act_in,
                default_r,
                T=temp,
                update_memory=False,
                need_cache=False)
            
            o_pred, a_pred, r_pred = self.post_decoder(wm_out, pm_out, T=temp)
            
            if not self.config.state_diffusion.enable:
                state = o_pred.detach().cpu().squeeze()
            else:
                o_latent = self.s_diffusion.inference(cond=wm_out)[-1]
                o_pred = self.s_decoder(o_latent)
                state = o_pred.detach().cpu().squeeze()
                
            if r_pred is not None:
                reward = r_pred.detach().cpu().squeeze()
                if(need_numpy):
                    state = state.numpy()
                    if(state.size < 2):
                        state = state.item()
                    reward = reward.numpy()
                    if(reward.size < 2):
                        reward = reward.item()
            else:
                reward = 0.0

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
        _, _, new_cache = self.forward(
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

    model = OmniRL(config.model_config)

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
