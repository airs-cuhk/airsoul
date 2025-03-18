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
            self.default_r = torch.full((1, 1), 0, dtype=torch.int64)  
        elif(self.config.reward_encode.input_type == "Continuous"):
            self.default_r = torch.zeros((1, 1, config.reward_encode.input_size), dtype=torch.float32)
            print(f"debug - default_r type: {self.default_r.dtype if self.default_r is not None else 'None'}")
        else:
            raise ValueError("Invalid reward encoding type", config.reward_encoding)
            
        
        if(config.action_encode.input_type == "Discrete"):
            self.default_a = torch.full((1, 1), config.action_encode.input_size, dtype=torch.int64)
        elif(config.action_encode.input_type == "Continuous"):
            self.default_a = torch.zeros((1, 1, config.action_encode.input_size), dtype=torch.float32)
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
        print(f"sequential_loss o_in shape: {o_in.shape}, dtype: {o_in.dtype}")
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
        loss_weight_a = (label_actions.ge(0) * label_actions.lt(self.nactions)).to(
                    self.loss_weight.dtype)
        if(use_loss_weight):
            loss_weight_s = self.loss_weight[ps:pe]
            if self.action_dtype == "Discrete":
                loss_weight_a = loss_weight_a * self.loss_weight[ps:pe].unsqueeze(0)
            elif self.action_dtype == "Continuous":
                loss_weight_a = loss_weight_a * self.loss_weight[ps:pe].unsqueeze(0).unsqueeze(-1)
                loss_weight_a = torch.mean(loss_weight_a, dim=-1, keepdim=True).squeeze(-1)
        else:
            if self.action_dtype == "Continuous":
                loss_weight_a = torch.sum(loss_weight_a, dim=-1, keepdim=True).squeeze(-1)

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
                
        loss["wm-r"] = weighted_loss(r_pred, 
                                     gt=rewards.view(*rewards.shape,1), 
                                     loss_type="mse",
                                     loss_wht=loss_weight_a,
                                     reduce_dim=reduce_dim)

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
            if not self.config.state_diffusion.enable:
                obs_in = torch.tensor([observation], dtype=torch.int64).to(device)
            else:
                obs_in = torch.tensor([observation], dtype=torch.float32).to(device)
        else:
            obs_in = observation.to(device)

        print(f"first obs_in shape: {obs_in.shape}, dtype: {obs_in.dtype}, value: {obs_in}")

        if(single_batch):
            if(pro_in is not None):
                pro_in = pro_in.unsqueeze(0)
            if(tag_in is not None):
                tag_in = tag_in.unsqueeze(0)
            obs_in = obs_in.unsqueeze(0)
        
        print(f"after unsqueeze obs_in shape: {obs_in.shape}, dtype: {obs_in.dtype}")

        # Prepare reward input
        if(self.r_included):
            if self.reward_dtype == "Continuous":
                default_r = self.default_r.to(dtype=torch.float32, device=device)
                print(f"default_r type: {default_r.dtype if default_r is not None else 'None'}")
            else:
                default_r = self.default_r.to(dtype=torch.int64, device=device)
        else:
            default_r = None
        
        # Prepare action input
        if self.action_dtype == "Continuous":
            default_a = self.default_a.to(dtype=torch.float32, device=device)
        else:
            default_a = self.default_a.to(dtype=torch.int64, device=device)
        
        # Ensure proper shape matching with observation batch
        if single_batch:
            B = obs_in.shape[0]
            NT = obs_in.shape[1]
            # match batch and time dimensions
            if self.action_dtype == "Continuous":
                default_a = default_a.expand(B, NT, -1)
            else: 
                default_a = default_a.expand(B, NT)
        
        print(f"default_a shape: {default_a.shape}, dtype: {default_a.dtype}")

        # First forward pass for action prediction
        wm_out, pm_out, _ = self.forward(
            obs_in,
            pro_in,
            tag_in,
            default_a,  
            default_r,
            T=temp,
            update_memory=False,
            need_cache=False)
        
        # Rest of method continues as before...
        o_pred, a_pred, r_pred = self.post_decoder(wm_out, pm_out, T=temp)
        
        # Generate action based on policy output
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
            act_out = self.a_decoder(a_latent)
            # Make sure act_in has the right dimensions [B, NT, feature_dim]
            act_in = act_out.clone()
            if act_in.dim() == 3:  # If already has right dimensions
                pass
            elif act_in.dim() == 2:  # If missing sequence dimension
                act_in = act_in.unsqueeze(1)
            elif act_in.dim() == 1:  # If missing batch and sequence dimensions
                act_in = act_in.unsqueeze(0).unsqueeze(0)

        # Postprocess action output
        act_out_cpu = act_out.detach().cpu().squeeze()
        if(need_numpy):
            act_out_cpu = act_out_cpu.numpy()
            if(act_out_cpu.size < 2):
                act_out_cpu = act_out_cpu.item()

        if(future_prediction):
            if isinstance(act_in, torch.Tensor):
                if act_in.dim() == 1:
                    act_in = act_in.unsqueeze(0)
                if act_in.dim() == 2 and single_batch:
                    act_in = act_in.unsqueeze(1)
                if act_in.dim() == 4:  
                    act_in = act_in.squeeze(1)
                
                print(f"future_prediction - act_in shape: {act_in.shape}")
                print(f"future_prediction - obs_in shape: {obs_in.shape}")
                
                # Second forward pass to predict next state and reward
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
        
                if self.state_dtype == "Discrete":
                    # For discrete states, return probability distribution
                    state_pred = o_pred
                else:
                    # For continuous state spaces
                    if not self.config.state_diffusion.enable:
                        state_pred = o_pred.detach().cpu().squeeze()
                    else:
                        o_latent = self.s_diffusion.inference(cond=wm_out)[-1]
                        o_pred = self.s_decoder(o_latent)
                        state_pred = o_pred.detach().cpu().squeeze()
                
                reward = r_pred.detach().cpu().squeeze()
                
                if need_numpy:
                    if self.state_dtype == "Continuous":
                        state_pred = state_pred.numpy()
                    reward = reward.numpy()
                    if reward.size < 2:
                        reward = reward.item()
            else:
                state_pred = None
                reward = None
            
            return state_pred, act_out_cpu, reward

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

        def proc(x, is_action=False):
            if x is None:
                return x
                
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            
            if is_action:
                if self.action_dtype == "Continuous":
                    x = x.to(torch.float32)
                    # If action is a scalar or 1D tensor without action dimension
                    if x.dim() == 0 or (x.dim() == 1 and x.size(0) != self.nactions):
                        if self.nactions > 1:
                            if x.dim() == 0:
                                x = x.view(1).expand(self.nactions).contiguous()
                            else:
                                x = x.view(-1, 1).expand(-1, self.nactions).contiguous()
                else:
                    # For discrete actions
                    x = x.to(torch.int64)
            elif isinstance(x, torch.Tensor) and x.dtype == torch.float64:
                if x.dtype != torch.int64:  
                    x = x.to(torch.float32)
                
            if single_batch and single_step:
                return x.unsqueeze(0).unsqueeze(0).to(device)
            elif single_batch:
                return x.unsqueeze(0).to(device)
            elif single_step:
                return x.unsqueeze(1).to(device)
            return x.to(device)

        # Process inputs
        if not isinstance(observation, torch.Tensor):
            obs_in = torch.tensor(observation)
        else:
            obs_in = observation
            
        print(f"in_context_learn before proc - obs_in shape: {obs_in.shape if hasattr(obs_in, 'shape') else 'scalar'}, dtype: {obs_in.dtype}, value: {obs_in}")
        
        if obs_in.dtype == torch.float64:
            obs_in = obs_in.to(torch.float32)
            
        obs_in = proc(obs_in)
        
        print(f"in_context_learn after proc - obs_in shape: {obs_in.shape}, dtype: {obs_in.dtype}")

        if prompts is not None and not isinstance(prompts, torch.Tensor):
            pro_in = torch.tensor(prompts)
        else:
            pro_in = prompts
        pro_in = proc(pro_in)
        
        if tags is not None and not isinstance(tags, torch.Tensor):
            tag_in = torch.tensor(tags)
        else:
            tag_in = tags
        tag_in = proc(tag_in)
        
        if not isinstance(action, torch.Tensor):
            act_in = torch.tensor(action)
        else:
            act_in = action
        act_in = proc(act_in, is_action=True)  
        
        if reward is not None and not isinstance(reward, torch.Tensor):
            rew_in = torch.tensor(reward)
        else:
            rew_in = reward
        
        if rew_in is not None:
            if self.reward_dtype == "Continuous":
                rew_in = rew_in.to(torch.float32)
            else:
                rew_in = rew_in.to(torch.int64)
        rew_in = proc(rew_in)

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