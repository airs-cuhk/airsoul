import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from airsoul.modules import MLPEncoder, ResidualMLPDecoder, CausalBlock, DiffusionLayers, FixedEncoderDecoder
from airsoul.utils import format_cache, log_fatal

class SADecisionModel(nn.Module):
    """
    Take Observations and actions; output next state and action.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.causal_model = CausalBlock(config.causal_block)

        # 创建Type向量[1, 1, NP, C]
        type_embeddings = torch.randn(1, 1, 2, config.causal_block.hidden_size)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)
        mask_embeddings = torch.randn(1, 1, config.state_encode.input_size)
        self.mask_query = nn.Parameter(mask_embeddings, requires_grad=True)

        self.s_encoder = MLPEncoder(config.state_encode)
        self.a_encoder = MLPEncoder(config.action_encode)
        
        if(config.action_diffusion.enable):
            self.a_diffusion = DiffusionLayers(config.action_diffusion)
        else:
            self.a_decoder = ResidualMLPDecoder(config.action_decode)

        if(config.state_diffusion.enable):
            self.s_diffusion = DiffusionLayers(config.state_diffusion)
        else:
            self.s_decoder = ResidualMLPDecoder(config.state_decode)

    def forward(self, s_arr, a_arr, cache=None, need_cache=True, state_dropout=0.0, T=1.0, update_memory=True):
        """
        Input Size:
            observations:[B, NT, H], float
            actions:[B, NT, H], float
            cache: [B, NC, H]
        """
        B = s_arr.shape[0]
        NT = s_arr.shape[1]
        H = s_arr.shape[2]

        assert s_arr.shape[:2] == a_arr.shape[:2]

        # Add state dropouts
        device = s_arr.device
        p_noise = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        p_mask = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
        eps = torch.randn((B, NT, H)).to(device)
        dp_eps = torch.bernoulli(p_noise)
        dp_mask = torch.bernoulli(p_mask)

        # Calculate dropout for mazes: 50% * state_dropout add noise, 50% * state_dropout are directly masked
        observation_in = s_arr + eps * dp_eps
        observation_in = observation_in * (1 - dp_mask) + self.mask_query * dp_mask
        observation_in = self.s_encoder(observation_in).view(B, NT, 1, -1)

        # Input actions: [B, NT, 1, H]
        action_in = self.a_encoder(a_arr).view(B, NT, 1, -1)

        # [B, NT, 2, H]
        outputs = torch.cat([observation_in, action_in], dim=2)

        # Add Type Embedding
        outputs = outputs + self.type_query

        # Concatenate [s_0, a_0, s_1, a_1, s_2, ...] to acquire the size of [B, NT * 2, H]
        outputs = outputs.view(B, NT * 2, -1)

        # Temporal Encoders
        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        # Acqure Outputs: [a_0, s_1, a_1, ...]
        outputs = outputs.reshape(B, NT, 2, -1)

        wm_out = outputs[:, :, 1]
        pm_out = outputs[:, :, 0]

        return wm_out, pm_out, new_cache

    def post_decoder(self, wm_out, pm_out, T=1.0):
        obs_output, act_output = None, None
        if not self.config.state_diffusion.enable:
            obs_output = self.s_decoder(wm_out)
        if not self. config.action_diffusion.enable:
            act_output = self.a_decoder(pm_out, T=T)
        return obs_output, act_output

    def reset(self):
        self.causal_model.reset()


class POTARDecisionModel(nn.Module):
    """
    Take Observations, actions and reward; output next state, action and reward.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.causal_model = CausalBlock(config.causal_block)
        self.hidden_size = config.causal_block.hidden_size

        self.rsa_choice =  ["podtawr", "podtaw", "podaw", "odawr", "odtawr", "odtaw", "odaw",
                            "opdtawr"]
        if(config.rsa_type.lower() not in self.rsa_choice):
            log_fatal(f"rsa_type must be one of the following: {self.rsa_choice}, get {self.rsa_type}")
        positon_map, self.rsa_type = self.process_string(config.rsa_type)
        self.rsa_occ = len(self.rsa_type)
        self.pm_pos = self.rsa_type.find(positon_map['d'])
        self.wm_pos = self.rsa_type.find(positon_map['w'])      

        if(self.rsa_type.find('r') > -1):
            self.r_decoder = ResidualMLPDecoder(config.reward_decode)

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

        type_embeddings = torch.randn(1, 1, self.rsa_occ, self.hidden_size)
        self.type_query = nn.Parameter(type_embeddings, requires_grad=True)
        mask_embeddings = torch.randn(1, 1, config.state_encode.input_size)
        self.mask_query = nn.Parameter(mask_embeddings, requires_grad=True)

        if("p" in self.rsa_type):
            self.p_encoder = MLPEncoder(config.prompt_encode)
            self.p_included = True
            if(self.config.prompt_encode.input_type == "Discrete"):
                self.p_discrete = True
            else:
                self.p_discrete = False
        else:
            self.p_included = False

        if("t" in self.rsa_type):
            self.t_encoder = MLPEncoder(config.tag_encode)
            self.t_included = True
            if(self.config.tag_encode.input_type == "Discrete"):
                self.t_discrete = True
                self.t_dim = self.config.tag_encode.input_size
            else:
                self.t_discrete = False
        else:
            self.t_included = False

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

        # print("s_discrete", self.s_discrete)
        # print("state_dropout: ", state_dropout)
        # print("obs : ", o_arr)
        # print("prompt : ", p_arr)
        # print("action : ", a_arr)
        # print("action : ", a_arr.dtype)


        assert a_arr.shape[:2] == o_arr.shape[:2], (a_arr.shape, o_arr.shape)

        if(self.p_included):
            assert p_arr is not None
            assert p_arr.shape[:2] == o_arr.shape[:2], (p_arr.shape, o_arr.shape)
        if(self.r_included):
            assert r_arr is not None
            assert r_arr.shape[:2] == o_arr.shape[:2], (r_arr.shape, o_arr.shape)
            if(self.r_discrete):
                r_arr = torch.where(r_arr<0, torch.full_like(r_arr, self.r_dim), r_arr)
            else:
                if(r_arr.dim() < 3):
                    r_arr = r_arr.view(B, NT, 1)
        if(self.t_included):
            assert t_arr is not None
            assert t_arr.shape[:2] == o_arr.shape[:2]            

        # Add state dropouts
        if not self.s_discrete and state_dropout > 0.0:
            
            H = o_arr.shape[2]
            device = o_arr.device
            p_noise = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
            p_noise = ( state_dropout ) * torch.ones(B, NT, 1).to(device)
            
            p_mask = (0.5 * state_dropout * torch.rand((B, 1, 1)) * torch.ones(B, NT, 1)).to(device)
            sigma = torch.abs(torch.randn(B, NT, H)).to(device)
            eps = torch.normal(0.0, sigma).to(device) # torch.randn((B, NT, H)).to(device)
            dp_eps = torch.bernoulli(p_noise)
            dp_mask = torch.bernoulli(p_mask)
            # print("random state")
            # print("p_noise: ", p_noise)
            # print("dp_eps: ", dp_eps)
            # print("dp_mask: ", dp_mask)
            # print("eps * dp_eps: ", eps * dp_eps)
            
            # Calculate dropout for mazes: 50% * state_dropout add noise, 50% * state_dropout are directly masked
            observation_in = o_arr + eps * dp_eps
            # observation_in = observation_in * (1 - dp_mask) + self.mask_query * dp_mask
            
            # print("observation diff: ", (observation_in - o_arr==0).all())
            # print("-----------------------------------")

        else:
            observation_in = o_arr

        if(self.s_discrete):
            observation_in = torch.where(observation_in<0, torch.full_like(observation_in, self.s_dim), observation_in)
        if(self.a_discrete):
            a_arr = torch.where(a_arr<0, torch.full_like(a_arr, self.a_dim), a_arr)

        var_dict = {}
        var_dict["o_in"] = self.s_encoder(observation_in).view(B, NT, 1, -1)
        var_dict["a_in"] = self.a_encoder(a_arr).view(B, NT, 1, -1)
        
        

        # Insert Prompts, tags and rewards if necessary
        if(self.t_included):
            if self.t_discrete:
                t_arr = torch.where(t_arr<0, torch.full_like(t_arr, self.t_dim - 1), t_arr)
            t_in = self.t_encoder(t_arr)
            var_dict["t_in"] = t_in.view(B, NT, 1, -1)
        if(self.p_included):
            p_in = self.p_encoder(p_arr)
            var_dict["p_in"] = p_in.view(B, NT, 1, -1)
        if(self.r_included):
            r_in = self.r_encoder(r_arr)
            var_dict["r_in"] = r_in.view(B, NT, 1, -1)

        inputs = []
        for char in self.rsa_type:
            var_name = f"{char}_in"
            inputs.append(var_dict[var_name])

        # print("var_dict: ", var_dict)
        # [B, NT, 2-5, H]
        outputs = torch.cat(inputs, dim=2)

        # Add Type Embedding
        outputs = outputs + self.type_query
        

        # Concatenate [r_0, s_0, a_0, r_1, s_1, a_1, r_2, ...] to acquire the size of [B, NT * 3, H]
        outputs = outputs.view(B, NT * self.rsa_occ, -1)

        # print("after output: ", outputs)
        # print("cache: ", cache)
        # Temporal Encoders
        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        # Acqure Outputs: [r0_, a_0, s_1, r_1, ...]
        outputs = outputs.reshape(B, NT, self.rsa_occ, -1)

        # Extract world models outputs
        wm_out = outputs[:, :, self.wm_pos]
        pm_out = outputs[:, :, self.pm_pos]
        # print("pm_out: ",pm_out)
        return wm_out, pm_out, new_cache
    def get_o_list(self):
        return self.causal_model.get_o_list()
    def post_decoder(self, wm_out, pm_out, T=1.0):
        obs_output, act_output, rew_output = None, None, None
        if(not self.config.state_diffusion.enable):
            # Predict s_1, s_2, ..., s_{t+1}
            obs_output = self.s_decoder(wm_out)

        if(not self.config.action_diffusion.enable):
            # Predict a_0, a_1, ..., a_t
            act_output = self.a_decoder(pm_out, T=T)
        
        if(self.rsa_type.find('r') > -1):
            rew_output = self.r_decoder(wm_out)
        
        return obs_output, act_output, rew_output

    def reset(self):
        self.causal_model.reset()
    
    def process_string(self, s):
        char_map = {}
        filtered_str = []
        s = s.lower()
        
        for idx, char in enumerate(s):
            if char in {'d', 'w'}:
                if idx > 0 and char not in char_map:
                    char_map[char] = s[idx-1]
            else:
                filtered_str.append(char)
        
        return char_map, ''.join(filtered_str)

if __name__=='__main__':
    import sys
    from airsoul.utils import Configure
    config = Configure()
    config.from_yaml(sys.argv[1])
    RSADM = SADecisionModel(config.model_config.decision_block)
    cache = None
    for seg in range(10):
        i_s = torch.rand((4, 64, 128))
        i_a = torch.randint(0, 4, (4, 64))
        i_r = torch.rand((4, 64, 1))
        o_r, o_s, o_a, cache = RSADM.forward(i_r, i_s, i_a, cache=cache, update_memory=False)
        print(seg, o_s.shape, o_a.shape, o_r.shape)
        print(format_cache(cache, "Cache"))
        print(format_cache(RSADM.causal_model.layers.memory, "Memory"))
