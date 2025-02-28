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
from l3c_baselines.utils import weighted_loss, img_pro, img_post
from l3c_baselines.utils import parameters_regularization, count_parameters
from l3c_baselines.modules import ImageEncoder, ImageDecoder, VAE
from .decision_model import SADecisionModel

class E2EObjNavSA(nn.Module):
    def __init__(self, config, verbose=False): 
        super().__init__()

        self.config = config
        
        self.img_encoder = ImageEncoder(config.image_encoder_block)

        self.img_decoder = ImageDecoder(config.image_decoder_block)

        self.decision_model = SADecisionModel(config.decision_block)

        self.vae = VAE(config.vae_latent_size, self.img_encoder, self.img_decoder) 

        loss_weight = torch.cat((
                    torch.linspace(0.0, 1.0, config.context_warmup),
                    torch.full((config.max_position_loss_weighting - config.context_warmup,), 1.0)), dim=0)
        loss_weight = loss_weight / torch.sum(loss_weight)

        self.register_buffer('loss_weight', loss_weight)

        self.nactions = config.action_dim

        self.policy_loss = config.policy_loss_type.lower()

        if(verbose):
            print("E2EObjNavSA initialized, total params: {}".format(count_parameters(self)))
            print("image_encoder params: {}; image_decoder_params: {}; decision_model_params: {}".format(
                count_parameters(self.img_encoder), 
                count_parameters(self.img_decoder), 
                count_parameters(self.decision_model)))
        
    def forward(self, observations, actions, cache=None, need_cache=True, state_dropout=0.0, update_memory=True):
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

        z_pred, a_pred, new_cache = self.decision_model(z_rec, actions, 
                cache=cache, need_cache=need_cache, state_dropout=state_dropout, 
                update_memory=update_memory)

        return z_rec, z_pred, a_pred, new_cache

    def vae_loss(self, observations, _sigma=1.0, seq_len=None):
        self.vae.requires_grad_(True)
        self.img_encoder.requires_grad_(True)
        self.img_decoder.requires_grad_(True)
        return self.vae.loss(img_pro(observations), _sigma=_sigma, seq_len=seq_len)

    def reset(self):
        self.decision_model.reset()

    def sequential_loss(self, observations, behavior_actions, label_actions, 
                        additional_info=None, # Kept for passing additional information
                        state_dropout=0.0, 
                        update_memory=True,
                        use_loss_weight=True,
                        reduce_dim=1):
        
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
        z_rec, z_pred, a_pred, cache = self.forward(inputs[:, :-1], behavior_actions, 
                cache=None, need_cache=False, state_dropout=state_dropout,
                update_memory=update_memory)
        
        # Encode the last frame to latent space
        with torch.no_grad():
            z_rec_l, _ = self.vae(inputs[:, -1:])
            z_rec_l = torch.cat((z_rec, z_rec_l), dim=1)

        # Calculate the loss information
        loss = dict()

        if(use_loss_weight):
            loss_weight = self.loss_weight[ps:pe]
        else:
            loss_weight = None

        # World Model Loss - Latent Space
        loss["wm-latent"], loss["count_wm"] = weighted_loss(z_pred, 
                                          loss_type="mse",
                                          gt=z_rec_l[:, 1:], 
                                          loss_wht=loss_weight, 
                                          reduce_dim=reduce_dim,
                                          need_cnt=True)

        # World Model Loss - Raw Image
        obs_pred = self.vae.decoding(z_pred)
        loss["wm-raw"] = weighted_loss(obs_pred, 
                                       loss_type="mse",
                                       gt=inputs[:, 1:], 
                                       loss_wht=loss_weight, 
                                       reduce_dim=reduce_dim)

        # Decision Model Loss
        if(self.policy_loss == 'crossentropy'):
            assert label_actions.dtype in [torch.int64, torch.int32, torch.uint8]
            loss_weight = (label_actions.ge(0) * label_actions.lt(self.nactions)).to(self.loss_weight.dtype)
            if(use_loss_weight):
                loss_weight = loss_weight * self.loss_weight[ps:pe]
            truncated_actions = torch.clip(label_actions, 0, self.nactions - 1)
            loss["pm"], loss["count_pm"] = weighted_loss(a_pred,
                                       loss_type="ce",
                                       gt=truncated_actions, 
                                       loss_wht=loss_weight, 
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)
        elif(self.policy_loss == 'mse'):
            if(use_loss_weight):
                loss_weight = self.loss_weight[ps:pe]
            else:
                loss_weight = None
            loss["pm"], loss["count_pm"] = weighted_loss(a_pred,
                                       loss_type="mse", 
                                       gt=label_actions, 
                                       loss_wht=loss_weight, 
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)
        else:
            log_fatal(f"no such policy loss type: {self.policy_loss}")
        loss["causal-l2"] = parameters_regularization(self.decision_model)

        return loss

    def inference_step_by_step(self, observations, actions, 
                               temp, start_position, device, 
                               n_step=1, cache=None, verbose=True):
        """
        Given: cache - from s_0, a_0, ..., s_{tc}, a_{tc}
               observations: s_{tc}, ... s_{t}
               actions: a_{tc}, ..., a_{t-1}
               temp: temperature
        Returns:
            obs_pred: numpy.array [n, W, H, C], s_{t+1}, ..., s_{t+n}
            act_pred: numpy.array [n], a_{t}, ..., a_{t+n-1}
            new_cache: torch.array caches up to s_0, a_0, ..., s_{t}, a_{t} (Notice not to t+n, as t+1 to t+n are imagined)
        """
        obss = numpy.array(observations, dtype=numpy.float32)
        acts = numpy.array(actions, dtype=numpy.int64)
        Nobs, W, H, C = obss.shape
        (No,) = acts.shape

        assert Nobs == No + 1

        valid_obs = torch.from_numpy(img_pro(obss)).float().to(device)
        valid_obs = valid_obs.permute(0, 3, 1, 2).unsqueeze(0)

        if(No < 1):
            valid_act = torch.zeros((1, 1), dtype=torch.int64).to(device)
        else:
            valid_act = torch.from_numpy(acts).int()
            valid_act = torch.cat((valid_act, torch.zeros((1,), dtype=torch.int64)), dim=0).unsqueeze(0).to(device)

        # Update the cache first
        # Only use ground truth
        if(Nobs > 1):
            with torch.no_grad():
                z_rec, z_pred, a_pred, valid_cache  = self.forward(
                        valid_obs[:, :-1], valid_act[:, :-1], 
                        cache=cache, need_cache=True,
                        update_memory=True)
        else:
            valid_cache = cache

        # Inference Action First
        pred_obs_list = []
        pred_act_list = []
        updated_cache = valid_cache
        n_act = valid_act[:, -1:]
        z_rec, _ = self.vae(valid_obs[:, -1:])

        for step in range(n_step):
            with torch.no_grad():
                # Temporal Encoders
                z_pred, a_pred, _ = self.decision_model(z_rec, n_act, cache=updated_cache, need_cache=True)

                action = torch.multinomial(a_pred[:, 0], num_samples=1).squeeze(1)

                n_act[:, 0] = action
                pred_act_list.append(action.squeeze(0).cpu().numpy())
                if(verbose):
                    print(f"Action: {valid_act[:, -1]} Raw Output: {a_pred[:, -1]}")

                # Inference Next Observation based on Sampled Action
                # Do not update the memory based on current imagination
                z_pred, a_pred, updated_cache = self.decision_model(z_rec, n_act, 
                        cache=updated_cache, need_cache=True, 
                        update_memory=(step==0))

                # Only the first step uses the ground truth
                if(step == 0):
                    valid_cache = updated_cache

                # Decode the prediction
                pred_obs = self.vae.decoding(z_pred)
                pred_obs = img_post(pred_obs)

                pred_obs_list.append(pred_obs.squeeze(1).squeeze(0).permute(1, 2, 0).cpu().numpy())

                # Do auto-regression for n_step
                z_rec = z_pred

        return pred_obs_list, pred_act_list, valid_cache
        

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
