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
from airsoul.utils import weighted_loss, img_pro, img_post
from airsoul.utils import parameters_regularization, count_parameters
from airsoul.modules import ImageEncoder, ImageDecoder, VAE
from .decision_model import SADecisionModel, POTARDecisionModel

class E2EObjNavSA(nn.Module):
    def __init__(self, config, verbose=False): 
        super().__init__()

        self.config = config
        
        self.img_encoder = ImageEncoder(config.image_encoder_block)

        self.img_decoder = ImageDecoder(config.image_decoder_block)

        self.decision_model = POTARDecisionModel(config.decision_block)

        self.vae = VAE(config.vae_latent_size, self.img_encoder, self.img_decoder) 

        loss_weight = torch.cat((
                    torch.linspace(0.0, 1.0, config.context_warmup),
                    torch.full((config.max_position_loss_weighting - config.context_warmup,), 1.0)), dim=0)
        loss_weight = loss_weight / torch.sum(loss_weight)

        self.register_buffer('loss_weight', loss_weight)

        self.nactions = config.action_dim
        self.state_dtype = config.decision_block.state_encode.input_type
        self.action_dtype = config.decision_block.action_encode.input_type

        self.policy_loss = config.policy_loss_type.lower()

        if(verbose):
            print("E2EObjNavSA initialized, total params: {}".format(count_parameters(self)))
            print("image_encoder params: {}; image_decoder_params: {}; decision_model_params: {}".format(
                count_parameters(self.img_encoder), 
                count_parameters(self.img_decoder), 
                count_parameters(self.decision_model)))
        
    def forward(self, observations, 
                    prompts,
                    tags,
                    actions,
                    rewards,
                    cache=None, need_cache=True, state_dropout=0.0,update_memory=True):
        """
        Input Size:
            observations:[B, NT, C, W, H]
            actions:[B, NT / (NT - 1)] 
            cache: [B, NC, H]
        """
        
        # Encode with VAE
        B = actions.shape[0]
        NT = actions.shape[1]
        with torch.no_grad():
            z_rec, _ = self.vae(observations)
        wm_out, pm_out, new_cache = self.decision_model(
                z_rec, prompts, tags, actions, rewards,
                cache=cache, need_cache=need_cache, state_dropout=state_dropout, 
                update_memory=update_memory)

        return z_rec, wm_out, pm_out, new_cache

    def vae_loss(self, observations, _sigma=1.0, seq_len=None):
        self.vae.requires_grad_(True)
        self.img_encoder.requires_grad_(True)
        self.img_decoder.requires_grad_(True)
        return self.vae.loss(img_pro(observations), _sigma=_sigma, seq_len=seq_len)

    def reset(self):
        self.decision_model.reset()

    def sequential_loss(self, prompts, 
                        observations,
                        tags,
                        behavior_actions, 
                        rewards, 
                        label_actions, 
                        additional_info=None, # Kept for passing additional information
                        state_dropout=0.0, 
                        update_memory=True,
                        use_loss_weight=True,
                        is_training=True,
                        reduce_dim=1):
                        
        # print("label_actions  ",label_actions.size())
        self.img_encoder.requires_grad_(False)
        self.img_decoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.decision_model.requires_grad_(True)

        inputs = img_pro(observations)

        bsz = behavior_actions.shape[0]
        seq_len = behavior_actions.shape[1]

        # Pay attention the position must be acquired before calling forward()
        ps = self.decision_model.causal_model.position
        pe = ps + seq_len

        # Predict the latent representation of action and next frame (World Model)
        z_rec, wm_out, pm_out, cache = self.forward(
                inputs[:, :-1], 
                prompts,
                tags,
                behavior_actions, 
                rewards,
                cache=None, 
                need_cache=False, 
                state_dropout=state_dropout,
                update_memory=update_memory)
        
        z_pred, a_pred, r_pred = self.decision_model.post_decoder(wm_out, pm_out)
        
        # Encode the last frame to latent space
        with torch.no_grad():
            z_rec_l, _ = self.vae(inputs[:, -1:])
            z_rec_l = torch.cat((z_rec, z_rec_l), dim=1)

        # Calculate the loss information
        loss = dict()

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

        if not self.config.decision_block.state_diffusion.enable:
            # World Model Loss - Latent Space
            loss["wm-latent"], loss["count_wm"] = weighted_loss(z_pred, 
                                            loss_type="mse",
                                            gt=z_rec_l[:, 1:], 
                                            loss_wht=loss_weight_s, 
                                            reduce_dim=reduce_dim,
                                            need_cnt=True)

            # World Model Loss - Raw Image
            obs_pred = self.vae.decoding(z_pred)
            loss["wm-raw"] = weighted_loss(obs_pred, 
                                        loss_type="mse",
                                        gt=inputs[:, 1:], 
                                        loss_wht=loss_weight_s, 
                                        reduce_dim=reduce_dim)
        else:
            if is_training:
                if self.config.decision_block.state_diffusion.prediction_type == "sample":
                    z_pred = self.decision_model.s_diffusion.loss_DDPM(x0=z_rec_l[:, 1:],
                                                        cond=wm_out,
                                                        mask=loss_weight_s,
                                                        reduce_dim=reduce_dim,
                                                        need_cnt=True)
                    # World Model Loss - Latent Space
                    loss["wm-latent"], loss["count_wm"] = weighted_loss(z_pred, 
                                                    loss_type="mse",
                                                    gt=z_rec_l[:, 1:], 
                                                    loss_wht=loss_weight_s, 
                                                    reduce_dim=reduce_dim,
                                                    need_cnt=True)

                    # World Model Loss - Raw Image
                    obs_pred = self.vae.decoding(z_pred)
                    loss["wm-raw"] = weighted_loss(obs_pred, 
                                                loss_type="mse",
                                                gt=inputs[:, 1:], 
                                                loss_wht=loss_weight_s.unsqueeze(0), 
                                                reduce_dim=reduce_dim)
                else:
                    loss["wm-latent"], loss["count_wm"] = self.decision_model.s_diffusion.loss_DDPM(x0=z_rec_l[:, 1:],
                                                    cond=wm_out,
                                                    mask=loss_weight_s,
                                                    reduce_dim=reduce_dim,
                                                    need_cnt=True)
                    loss["wm-raw"] = 0.0
            else:
                z_pred = self.decision_model.s_diffusion.inference(cond=wm_out)[-1]
                loss["wm-latent"], loss["count_wm"] = weighted_loss(z_pred, 
                                                loss_type="mse",
                                                gt=z_rec_l[:, 1:], 
                                                loss_wht=loss_weight_s, 
                                                reduce_dim=reduce_dim,
                                                need_cnt=True)
                obs_pred = self.vae.decoding(z_pred)
                loss["wm-raw"] = weighted_loss(obs_pred, 
                                            loss_type="mse",
                                            gt=inputs[:, 1:], 
                                            loss_wht=loss_weight_s, 
                                            reduce_dim=reduce_dim)

        # Decision Model Loss
        if self.action_dtype == "Discrete":
            if(self.policy_loss == 'crossentropy'):
                assert label_actions.dtype in [torch.int64, torch.int32, torch.uint8]
                truncated_actions = torch.clip(label_actions, 0, self.nactions - 1)
                loss["pm"], loss["count_pm"] = weighted_loss(a_pred,
                                        loss_type="ce",
                                        gt=truncated_actions, 
                                        loss_wht=loss_weight_a, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)
            elif(self.policy_loss == 'mse'):
                if(use_loss_weight):
                    loss_weight_a = self.loss_weight[ps:pe]
                else:
                    loss_weight_a = None
                loss["pm"], loss["count_pm"] = weighted_loss(a_pred,
                                        loss_type="mse", 
                                        gt=label_actions, 
                                        loss_wht=loss_weight_a, 
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)
            else:
                log_fatal(f"no such policy loss type: {self.policy_loss}")
        elif self.config.decision_block.action_diffusion.enable and self.config.decision_block.action_encode.input_type == "Continuous":
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
            else:
                a_latent = self.a_diffusion.inference(cond=pm_out)[-1]
                a_pred = self.a_decoder(a_latent)
                loss["pm"], loss["count_a"] = weighted_loss(a_pred, 
                                       gt=label_actions, 
                                       loss_type="mse",
                                       loss_wht=loss_weight_a, 
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)
            
        loss["causal-l2"] = parameters_regularization(self.decision_model)

        return loss
    
    def preprocess_others(self, 
                   vals, 
                   single_batch=True, 
                   single_step=True,
                   default_dim=None):
        if(vals is None):
            return None

        if(isinstance(vals, numpy.ndarray)):
            vals = torch.tensor(vals, device=next(self.parameters()).device)
        elif(isinstance(vals, torch.Tensor)):
            vals = vals.to(next(self.parameters()).device)
        else:
            raise TypeError(f"Unsupported type of values: {type(vals)}")
        
        if(single_batch):
            vals = vals.unsqueeze(0)
        if(single_step):
            vals = vals.unsqueeze(1)
        
        if(default_dim is not None):
            assert vals.dim == default_dim + 2, f"Input dimension of actions must be {default_dim + 2}, acquire {vals.dim}"

        return vals

    def preprocess_observation(self, 
                   observations, 
                   single_batch=True, 
                   single_step=True, 
                   raw_images=True):
        if(observations is None):
            return None

        if(isinstance(observations, numpy.ndarray)):
            obs = torch.tensor(observations, device=next(self.parameters()).device)
        elif(isinstance(observations, torch.Tensor)):
            obs = observations.to(next(self.parameters()).device)
        else:
            raise TypeError(f"Unsupported type of observations: {type(observations)}")
        
        if(single_batch):
            obs = obs.unsqueeze(0)
        if(single_step):
            obs = obs.unsqueeze(1)

        if(raw_images):
            assert obs.dim == 5, f"Input dimension of observations of raw images must be 5, acquire {obs.dim}"
            with torch.no_grad():
                z_rec, _ = self.vae(obs)
        else:
            z_rec = obs

        return z_rec
    
    def preprocess(self, 
                observations, 
                prompts,
                tags,
                actions,
                rewards, 
                single_batch=True, 
                single_step=True, 
                raw_images=True):
        z_rec = self.preprocess_observation(observations, single_batch=single_batch, single_step=single_step, raw_images=raw_images)
        act = self.preprocess_others(actions, single_batch=single_batch, single_step=single_step, default_dim=0)
        prompts = self.preprocess_others(prompts, single_batch=single_batch, single_step=single_step, default_dim=1)
        tags = self.preprocess_others(tags, single_batch=single_batch, single_step=single_step, default_dim=0)
        rewards = self.preprocess_others(rewards, single_batch=single_batch, single_step=single_step, default_dim=0)

        return z_rec, prompts, tags, actions, rewards

    def in_context_learn(self, 
                         observations,
                         prompts,
                         tags, 
                         actions,
                         rewards,
                         cache=None,
                         need_cache=False,
                         single_batch=True,
                         single_step=False,
                         raw_images=True):
        # Inputs:
        #   observations: [B, NT, C, W, H]
        #   actions: [B, NT]
        # Outputs:
        #   new_cache: [B, NC, H] if need_cache is True

        o,p,t,a,r = self.preprocess(
                        observations,
                        prompts,
                        tags, 
                        actions,
                        rewards,
                        single_batch=single_batch,
                        single_step=single_step, 
                        raw_images=raw_images)

        _, _, new_cache = self.decision_model(
                o, p, t, a, r, 
                cache=cache, need_cache=need_cache, 
                update_memory=True)

        return new_cache
    
    def sample_action_discrete(self, logits, temperature = 1.0):
        # Inputs:
        #   logits: [B, D]
        # Outputs:
        #   action: [B]
        return torch.multinomial(logits / temperature, num_samples=1)
    
    def generate_states_only(self, 
                            current_observation, 
                            action_trajectory,
                            history_observation=None,
                            history_action=None,
                            history_update_memory=True, 
                            autoregression_update_memory=False,
                            cache=None,
                            single_batch=True,
                            history_single_step=False,
                            future_single_step=False,
                            raw_images=True,
                            need_numpy=True):
        # Generate state autoregressively in latent space
        # Inputs:
        #   history_observations: o_1, o_2, ..., o_t
        #   history_actions: a_1, a_2, ..., a_t
        #   current_observation: o_{t+1}
        #   action_trajectory: a_{t+1}, a_{t+2}, ..., a_{t+n}
        # Outputs:
        #   predict_observations: o_{t+1}, ..., o_{t+n}

        his_obs, his_act = self.preprocess(
                    history_observation, 
                    history_action, 
                    single_batch=single_batch, 
                    single_step=history_single_step, 
                    raw_images=raw_images)

        obs = self.preprocess_observation(
            observation, 
            single_batch=single_batch, 
            single_step=True, 
            raw_images=raw_images)
        
        act = self.preprocess_others(action_trajectory, 
                                     single_batch=single_batch,
                                     single_step=future_single_step)

        if(autoregression_update_memory and not history_update_memory):
            raise ValueError("Autoregression update memory cannot be True when history update memory is False")

        # If do not update memory, then we need to cache it to keep consistency
        if(not history_update_memory):
            history_need_cache = True
        if(not autoregression_update_memory):
            autoregression_need_cache = True


        if(his_obs is not None):
            with torch.no_grad():
                _, _, cache = self.decision_model(his_obs,    
                                                    his_act, 
                                                    cache=cache,need_cache=history_need_cache, 
                                                    update_memory=history_update_memory)
        
        obs_out = [obs]
        for i in range(act.shape[1]):
            with torch.no_grad():
                wm_out, pm_out, cache = self.decision_model(obs_out[-1], 
                                                act[:, i:i+1], 
                                                cache=cache, need_cache=autoregression_need_cache, 
                                                update_memory=autoregression_update_memory)
                
                obs_n, _, _ = self.decision_model.post_decoder(wm_out, pm_out)
                if self.config.decision_block.state_diffusion.enable:
                    obs_n = self.decision_model.s_diffusion.inference(cond=wm_out)[-1]
                obs_out.append(obs_n)

        # Post Processing
        obs_out = torch.cat(obs_out, dim=1)
        if(raw_images):
            obs_out = img_post(self.vae.decoding(obs_out))
        if(need_numpy):
            obs_out = obs_out.cpu().detach().numpy()

        return obs_out, cache
    
    def generate_states_and_action(self, 
                            current_observation, 
                            future_steps=1,
                            history_observation=None,
                            history_action=None,
                            history_update_memory=True, 
                            autoregression_update_memory=False,
                            cache=None,
                            single_batch=True,
                            history_single_step=False,
                            raw_images=True,
                            need_predict_states=True,
                            need_numpy=True):
        # Generate state autoregressively in latent space
        # Inputs:
        #   history_observations: o_1, o_2, ..., o_t
        #   history_actions: a_1, a_2, ..., a_t
        #   current_observation: o_{t+1}
        #   future_steps: n
        # Outputs:
        #   predict_observations: o_{t+1}, ..., o_{t+n}
        #   predict_actions: a_{t}, ..., a_{t+n-1}
        #   if(n = 1) and need_predict_states is False:
        #       return predict_actions a_{t} only

        his_obs, his_act = self.preprocess(
                    history_observation, 
                    history_action, 
                    single_batch=single_batch, 
                    single_step=history_single_step, 
                    raw_images=raw_images)

        obs = self.preprocess_observation(
            observation, 
            single_batch=single_batch, 
            single_step=True, 
            raw_images=raw_images)

        ext_act = torch.zeros((1, 1), dtype=torch.int64).to(next(self.parameters()).device)

        if(autoregression_update_memory and not history_update_memory):
            raise ValueError("Autoregression update memory cannot be True when history update memory is False")

        # If do not update memory, then we need to cache it to keep consistency
        if(not history_update_memory):
            history_need_cache = True
        if(not autoregression_update_memory):
            autoregression_need_cache = True

        if(his_obs is not None):
            with torch.no_grad():
                _, _, cache = self.decision_model(his_obs,    
                                                    his_act, 
                                                    cache=cache,need_cache=history_need_cache, 
                                                    update_memory=history_update_memory)
        
        obs_out = [obs]
        act_out = []
        for i in range(future_steps):
            with torch.no_grad():
                # Step 1: Predict the next_action, do not update memory and cache
                wm_out, pm_out, _ = self.decision_model(obs_out[-1], 
                                                ext_act, 
                                                cache=cache, need_cache=False, 
                                                update_memory=False)
                _, act_pred, _ = self.decision_model.post_decoder(wm_out, pm_out)
                if self.config.decision_block.action_diffusion.enable:
                    act_pred = self.decision_model.s_diffusion.inference(cond=pm_out)[-1]
                act_out.append(self.sample_action_discrete(act_pred))

                # Step 2: Predict the next_observation
                if(future_steps > 1 or need_predict_states):
                    wm_out, pm_out, cache = self.decision_model(obs_out[-1], 
                                                    act_out[-1], 
                                                    cache=cache, need_cache=autoregression_need_cache, 
                                                    update_memory=autoregression_update_memory)
                    obs_n, _, _ = self.decision_model.post_decoder(wm_out, pm_out)
                    if self.config.decision_block.state_diffusion.enable:
                        obs_n = self.decision_model.s_diffusion.inference(cond=wm_out)[-1]
                    obs_out.append(obs_n)

        # Post Processing
        if(len(obs_out) > 1):
            obs_out = torch.cat(obs_out[1:], dim=1)
            if(raw_images):
                obs_out = img_post(self.vae.decoding(obs_out))
        else:
            obs_out = None
        act_out = torch.cat(act_out, dim=1)
        if(need_numpy):
            obs_out = obs_out.cpu().detach().numpy()
            act_out = act_out.cpu().detach().numpy()

        return obs_out, act_out, cache

if __name__=="__main__":
    from utils import Configure
    config=Configure()
    config.from_yaml(sys.argv[1])

    model = E2EObjNavSA(config.model_config)

    observation = torch.randn(8, 33, 3, 128, 128)
    action = torch.randint(4, (8, 32)) 
    reward = torch.randn(8, 32)
    local_map = torch.randn(8, 32, 3, 7, 7)

    vae_loss = model.vae_loss(observation)
    losses = model.sequential_loss(observation, action, action)
    rec_img, img_out, act_out, cache = model.inference_step_by_step(
            observation[:, :5], action[:, :4], 1.0, 0, observation.device)
    print("vae:", vae_loss, "sequential:", losses)
    print(img_out[0].shape, act_out.shape)
    print(len(cache))
    print(cache[0].shape)
