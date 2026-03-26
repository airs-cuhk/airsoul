import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from airsoul.modules import MLPEncoder, ResidualMLPDecoder, CausalBlock
from airsoul.utils import format_cache
from airsoul.utils import weighted_loss
from airsoul.utils import count_parameters
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import parameters_regularization, count_parameters

class MultiAgentModel(nn.Module):
    '''
    Autoregressive Larguage_model
    '''
    def __init__(self, config, verbose=False):
        super().__init__()

        self.word_emb = MLPEncoder(config.word_embeddings)

        # TODO, in config, input size of word_emb = vocab_size + 1. Last one is padding value.

        self.nvocab = config.vocab_size
        
        self.causal_model = CausalBlock(config.causal_block)

        self.output_mapping = ResidualMLPDecoder(config.output_layers)

        if(verbose):
            print("Language Model initialized, total params: {}".format(count_parameters(self)))

    def forward(self, inputs, cache=None, need_cache=True, T=1.0, update_memory=True):
        """
        Input Size:
            inputs:[B, NT], int
        """

        outputs = self.word_emb(inputs)

        outputs, new_cache = self.causal_model(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        outputs = self.output_mapping(outputs, T=T)

        return outputs, new_cache
    
    def reset(self):
        self.causal_model.reset()

class OmniRL_MultiAgent(MultiAgentModel):
    """
    Input format: List[idx]
    All of the id and value are transform into discrete integer values, forming the vocabular.
    - i.e., we have m obs_id, n action_id -> m + n words
        idx_policy, idx_tag, idx_a_self, idx_reward, idx_end_timestep, idx_reset_env -> 6 words
        t tag_value -> t words 
        value of obs, action, and reward ~ [-15, 15], resolution 0.1 -> 300 words + 2 upper and lower words = 302 words
        off_action_id -> 1 word
        idx_padding -> 1 word
        Then the total vocabular size = m + n + t + 310
        

    - For one timestep the sequence is arranged as: 
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep ]

        World Model (obs) position: idx_o1, idx_o3, idx_o4, ...
        World Model (action of other agent): idx_a1, idx_a2, idx_a4, ...
        Policy Model: idx_policy
        World Model (reward): idx_reward

    """
    def __init__(self, config, verbose=False): 
        super().__init__(config)

        loss_weight = torch.cat((
                torch.linspace(0.0, 1.0, config.context_warmup),
                torch.full((config.max_position_loss_weighting - config.context_warmup,), 1.0)), dim=0)
        loss_weight = loss_weight / torch.sum(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

        self.nobs = config.nobs
        self.nagent = config.nagent
        self.ntag = config.ntag
        self.value_num = config.value_num
        self.default_tag = int(config.default_tag)

        # 6=idx_policy, idx_tag, idx_a_self, idx_reward, idx_end_timestep, idx_reset_env; 
        # 2=value blow botom bound and value above upper bound; 2=off_action_id, idx_padding
        vocab_size = self.nobs + self.nagent + 6 + self.ntag + self.value_num + 2 + 2
        if not (config.word_embeddings.input_size == config.vocab_size == vocab_size):
            log_fatal(f"Word embeddings input size {config.word_embeddings.input_size} should be equal to vocab size {config.vocab_size} and {vocab_size}")

        self._init_vocab_offsets()

        if(verbose):
            log_debug("RSA Decision Model initialized, total params: {}".format(count_parameters(self)))
            log_debug("Causal Block Parameters: {}".format(count_parameters(self.causal_model)))

    def _init_vocab_offsets(self):
        self.OBS_IDX_OFFSET = 0
        self.AGENT_IDX_OFFSET = self.nobs
        self.SPECIAL_TOKENS_OFFSET = self.AGENT_IDX_OFFSET + self.nagent
        self.SPECIAL_TOKENS = {
            'idx_policy': 0,
            'idx_tag': 1,
            'idx_a_self': 2,
            'idx_reward': 3,
            'idx_end_timestep': 4,
            'idx_reset_env': 5
        }
        self.TAG_BASE = self.SPECIAL_TOKENS_OFFSET + len(self.SPECIAL_TOKENS)
        self.VALUE_BASE = self.TAG_BASE + self.ntag
        self.ACTION_OFF_BASE = self.VALUE_BASE + self.value_num + 2

    def find_position(self, inputs):
        """
        inputs: [batch_size, seq_len]
        World Model (obs) position: value in [0, nobs)
        World Model (action of other agent): value in [nobs, nagent)
        Policy Model: value == nobs + nagent
        World Model (reward): value == nobs + nagent + 3; (idx_policy, idx_tag, idx_a_self, idx_reward)
        return world_model_obs_out, world_model_action_out, policy_out, reward_out
        """
        world_model_obs_mask = (inputs < self.AGENT_IDX_OFFSET)
        world_model_action_mask = (inputs >= self.AGENT_IDX_OFFSET) & (inputs < self.SPECIAL_TOKENS_OFFSET)
        policy_mask = (inputs == self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_policy'])
        reward_mask = (inputs == (self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_reward']))

        return world_model_obs_mask, world_model_action_mask, policy_mask, reward_mask

    def sequential_loss(self, inputs, label_actions, use_loss_weight=True, update_memory=True, reduce_dim=1):
        """
        label_actions should have same shape as inputs, and replace the idx_policy with label action.
        """
        seq_len = inputs.shape[1]
        ps = self.causal_model.position
        pe = ps + seq_len - 1
        if(self.loss_weight.shape[0] < pe):
            log_fatal(f"Loss weight (shape {self.loss_weight.shape[0]}) should be longer" +
                    f" than sequence length {pe}")

        outputs, _ = self.forward(inputs, need_cache=False, update_memory=update_memory) #outputs: [batch_size, seq_len, vocab_size]
        outputs = outputs[:, :-1, :]
        world_model_obs_mask, world_model_action_mask, policy_mask, reward_mask = self.find_position(inputs[:,:-1])
    
        
        loss_weight_wm_obs = world_model_obs_mask.float()
        loss_weight_wm_action = world_model_action_mask.float()
        loss_weight_policy = policy_mask.float()
        loss_weight_reward = reward_mask.float()
        if use_loss_weight:
            loss_weight_wm_obs *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_wm_action *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_policy *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_reward *= self.loss_weight[ps:pe].unsqueeze(0)
        
        loss = dict()
        loss["wm_obs"], loss["count_s"] = weighted_loss(outputs, 
                                                        gt=inputs[:, 1:], 
                                                        loss_type="ce",
                                                        loss_wht=loss_weight_wm_obs, 
                                                        reduce_dim=reduce_dim,
                                                        need_cnt=True)
        loss["wm_agent"], loss["count_a"] = weighted_loss(outputs,
                                          gt=inputs[:, 1:],
                                          loss_type="ce",
                                          loss_wht=loss_weight_wm_action,
                                          reduce_dim=reduce_dim,
                                          need_cnt=True)
        loss["policy"], loss["count_p"] = weighted_loss(outputs,
                                       gt=label_actions[:,:-1],
                                       loss_type="ce",
                                       loss_wht=loss_weight_policy,
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)
        loss["reward"] = weighted_loss(outputs,
                                       gt=inputs[:, 1:],
                                       loss_type="ce",
                                       loss_wht=loss_weight_reward,
                                       reduce_dim=reduce_dim,
                                       need_cnt=False)
        loss["ent"] = weighted_loss(outputs, 
                                        loss_type="ent", 
                                        loss_wht=loss_weight_policy,
                                        reduce_dim=reduce_dim)
        loss["causal-l2"] = parameters_regularization(self)
        return loss
        
    def generate(self, inputs, need_numpy=True, single_batch=True, reward_prediction=False):
        """
        0, inputs : tensor with shape [BT, NT], 
            if agents have different seq lenth, padding with value self.nvocab: 
            [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
            idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
            idx_policy, idx_padding ...]
        1, Forward with inputs, update memory = False. Get wd_obs, wd_action, and action.
        2, if reward_prediction:
                Form the sequences with action:
                    [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
                    idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
                    idx_policy, idx_tag, tag, idx_a_self, action, idx_reward]
                Forward again and get wd_reward.
                return wd_obs, wd_action, action, wd_reward.
            else:
                reutrn wd_obs, wd_action, action
        """
        # inputs: [BT, NT]
        BT = inputs.size(0)
        outputs, _ = self.forward(inputs, need_cache=False, update_memory=False)
        
        def get_value(output): 
            output = F.softmax(output, dim=-1)  # [B, NT, D]
            B, NT, D = output.shape
            output = output.view(-1, D)  # [B*NT, D]

            mask = torch.zeros_like(output, dtype=torch.bool)
            start_idx = self.VALUE_BASE
            end_idx = self.VALUE_BASE + self.value_num + 2 # upper and lower value take two tokens
            mask[:, start_idx:end_idx] = True
            masked_output = output * mask

            row_sums = masked_output.sum(dim=1, keepdim=True)
            zero_mask = (row_sums == 0)
            if zero_mask.any():
                uniform_value = 1.0 / D
                zero_indices = zero_mask.squeeze().nonzero(as_tuple=True)[0]
                masked_output[zero_indices] = uniform_value
                masked_output[zero_indices] *= mask
                row_sums = masked_output.sum(dim=1, keepdim=True)
                

            normalized_output = masked_output / row_sums
            samples = torch.multinomial(normalized_output, num_samples=1)  # [B*NT, 1]
            output = samples.view(B, NT)  # [B, NT]

            return output
        
        outputs = get_value(outputs)
        world_model_obs_mask, world_model_action_mask, policy_mask, _ = self.find_position(inputs)
        world_model_obs = []
        world_model_action = []
        action = []
        for i in range(BT):
            world_model_obs.append(outputs[i][world_model_obs_mask[i]].detach().cpu())
            world_model_action.append(outputs[i][world_model_action_mask[i]].detach().cpu())
            action.append(outputs[i][policy_mask[i]].detach().cpu())

        if need_numpy:
            world_model_obs = [obs.numpy() for obs in world_model_obs]
            world_model_action = [act.numpy() for act in world_model_action]
            action = [a.numpy() for a in action]

        if not reward_prediction:
            return world_model_obs, world_model_action, action
        else:
            new_value = torch.tensor([
                [self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_tag'], self.default_tag, # idx_tag, tag
                self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_a_self'], int(action[i].item()), # idx_a_self, a_self
                self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_reward']] # idx_reward
                for i in range(BT)], dtype=torch.int64, device=inputs.device)
            policy_idx = self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_policy']
            new_nt = inputs.size(1) + 5
            new_inputs = torch.zeros((BT, new_nt), dtype=torch.int64, device=inputs.device)
            for i in range(BT):
                pos = torch.where(inputs[i] == policy_idx)[0]
                assert pos >= 0 and pos < inputs.size(1), f"idle position for policy_idx: {pos}"
                new_inputs[i, :pos+1] = inputs[i, :pos+1]
                new_inputs[i, pos+1:pos+6] = new_value[i]
                if pos < inputs.size(1) -1:
                    new_inputs[i, pos+6:] = inputs[i, pos+1:]
            outputs, _ = self.forward(new_inputs, need_cache=False, update_memory=False)
            outputs = get_value(outputs)
            _, _, _, reward_mask = self.find_position(new_inputs)
            world_model_reward = []
            for i in range(BT):
                world_model_reward.append(outputs[reward_mask[i]].detach().cpu())
            if need_numpy:
                world_model_reward = [reward.numpy() for reward in world_model_reward]
            return world_model_obs, world_model_action, action, world_model_reward


    def incontext_learn(self, inputs, need_cache = False):
        """
        inputs : tensor with shape [1, NT], only support 1 batch for now: 
            [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
            idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
            idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward]
        """
        _, cache = self.forward(inputs, need_cache=need_cache, update_memory=True)
        if need_cache:
            return cache                       

if __name__=='__main__':
    pass
