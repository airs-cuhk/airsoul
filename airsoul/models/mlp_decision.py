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

class MLPDecision(nn.Module):
    def __init__(self, config, verbose=False): 
        super().__init__()

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

        self.hidden_size = config.causal_block.hidden_size
        self.causal_model = ResidualMLPDecoder(config.causal_block)

        self.rsa_choice =  ["poar", "oar", "oa", "poa"]
        if(config.rsa_type.lower() not in self.rsa_choice):
            log_fatal(f"rsa_type must be one of the following: {self.rsa_choice}, get {self.rsa_type}")

        if(self.rsa_type.find('r') > -1):
            self.r_decoder = ResidualMLPDecoder(config.reward_decode)
        else:
            self.r_decoder = None

        if(config.action_diffusion.enable):
            self.a_diffusion = DiffusionLayers(config.action_diffusion)
            self.a_mapping = FixedEncoderDecoder(low_dim=config.action_encode.input_size,
                                                 high_dim=config.action_encode.hidden_size)
            self.a_encoder = self.a_mapping.encoder
            self.a_decoder = self.a_mapping.decoder
        else:
            self.a_encoder = MLPEncoder(config.action_encode, reserved_ID=True)
            self.a_decoder = ResidualMLPDecoder(config.action_decode)
            
        if(config.state_diffusion.enable):
            self.s_diffusion = DiffusionLayers(config.state_diffusion)
            self.s_mapping = FixedEncoderDecoder(low_dim=config.state_encode.input_size,
                                                 high_dim=config.state_encode.hidden_size)
            self.s_encoder = self.s_mapping.encoder
            self.s_decoder = self.s_mapping.decoder
        else:
            self.s_encoder = MLPEncoder(config.state_encode, reserved_ID=True)
            self.s_decoder = ResidualMLPDecoder(config.state_decode)

        if(self.config.state_encode.input_type == "Discrete"):
            self.s_discrete = True
            self.s_dim = self.config.state_encode.input_size
        else:
            self.s_discrete = False

        if(self.config.action_encode.input_type == "Discrete"):
            self.a_discrete = True
            self.a_dim = self.config.action_encode.input_size
        else:
            self.a_discrete = False

        if("p" in self.rsa_type):
            self.p_encoder = MLPEncoder(config.prompt_encode)
            self.p_included = True
            if(self.config.prompt_encode.input_type == "Discrete"):
                self.p_discrete = True
            else:
                self.p_discrete = False
        else:
            self.p_included = False

        if("r" in self.rsa_type):
            self.r_encoder = MLPEncoder(config.reward_encode, reserved_ID=True)
            self.r_included = True
            if(self.config.reward_encode.input_type == "Discrete"):
                self.r_discrete = True
                self.r_dim = self.config.reward_encode.input_size
            else:
                self.r_discrete = False
        else:
            self.r_included = False

        if(verbose):
            log_debug("RSA Decision Model initialized, total params: {}".format(count_parameters(self)))
            log_debug("Causal Block Parameters: {}".format(count_parameters(self.causal_model)))

    def forward(self, o_arr, p_arr, t_arr, a_arr, r_arr, 
                cache=None, need_cache=True, state_dropout=0.0, T=1.0, update_memory=True):
        """
        Input Size:
            observations:[B, NT, H], float
            actions:[B, NT, H], float
            prompts: [B, NT, H], float or None
            rewards:[B, NT, X], float or None
            cache: [B, NC, H]
        """
        B = o_arr.shape[0]
        NT = o_arr.shape[1]

        assert a_arr.shape[:2] == o_arr.shape[:2]

        if(self.p_included):
            assert p_arr is not None
            assert p_arr.shape[:2] == o_arr.shape[:2]
        if(self.r_included):
            assert r_arr is not None
            assert r_arr.shape[:2] == o_arr.shape[:2]
            if(self.r_discrete):
                r_arr = torch.where(r_arr<0, torch.full_like(r_arr, self.r_dim), r_arr)
            else:
                if(r_arr.dim() < 3):
                    r_arr = r_arr.view(B, NT, 1)

        if(self.s_discrete):
            observation_in = torch.where(observation_in<0, torch.full_like(observation_in, self.s_dim), observation_in)
        if(self.a_discrete):
            a_arr = torch.where(a_arr<0, torch.full_like(a_arr, self.a_dim), a_arr)

        var_dict = {}
        var_dict["o_in"] = self.s_encoder(observation_in).view(B, NT, -1)
        var_dict["a_in"] = self.a_encoder(a_arr).view(B, NT, -1)

        if(self.p_included):
            p_in = self.p_encoder(p_arr)
            var_dict["p_in"] = p_in.view(B, NT, -1)
        if(self.r_included):
            r_in = self.r_encoder(r_arr)
            var_dict["r_in"] = r_in.view(B, NT, -1)

        inputs = []
        for char in self.rsa_type:
            var_name = f"{char}_in"
            inputs.append(var_dict[var_name])

        # [B, NT, 2-5, H]
        outputs = torch.cat(inputs, dim=2)

        # Temporal Encoders
        outputs = self.causal_model(outputs)

        # Extract world models outputs
        s_out = self.s_decoder(outputs)
        a_out = self.a_decoder(outputs)
        if(self.r_decoder is not None):
            r_out = self.r_decoder(outputs)
        else:
            r_out = None

        return s_out, a_out, r_out

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
        s_pred, a_pred, r_pred = self.forward(
                o_in, prompts, tags, behavior_actions, rewards,
                cache=None, need_cache=False,
                update_memory=update_memory)
                
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

        if(r_pred is not None):
            loss["wm-r"] = weighted_loss(r_pred, 
                                         gt=rewards.view(*rewards.shape,1), 
                                         loss_type="mse",
                                         loss_wht=loss_weight_a,
                                         reduce_dim=reduce_dim)
        else:
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