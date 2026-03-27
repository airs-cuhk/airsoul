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
        idx_prompt, idx_a_self, idx_end_timestep, idx_reset_env -> 4 words
        (If reward is included, idx_prompt, idx_a_self, idx_end_timestep, idx_reset_env, idx_reward -> 5 words)
        p prompt_value -> p words
        value of obs, action, and reward ~ [-16, 16], temperature_resolution 0.1 -> 320 words + 2 upper and lower words = 322 words
        off_action_id -> 1 word
        idx_padding -> 1 word
        Then the total vocabular size = m + n + p + 333
        (If reward is included, Then the total vocabular size = m + n + p + 334)


    - For one timestep the sequence is arranged as: 
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_prompt, prompt, idx_a_self, a_self, idx_end_timestep]
        (If reward is included:
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_prompt, prompt, idx_a_self, a_self, idx_reward, reward_value, idx_end_timestep])

        World Model (obs) position: idx_o1, idx_o3, idx_o4, ...
        World Model (action of other agent): idx_a1, idx_a2, idx_a4, ...
        Policy Model: idx_a_self
        (If reward is included, World Model (reward) position: idx_reward)

    """
    def __init__(self, config, verbose=False): 
        super().__init__(config)
        self.config = config
        if config.use_context_warmup:
            loss_weight = torch.cat((
                    torch.linspace(0.0, 1.0, config.context_warmup),
                    torch.full((config.max_position_loss_weighting - config.context_warmup,), 1.0)), dim=0)
            loss_weight = loss_weight / torch.sum(loss_weight)
            self.register_buffer('loss_weight', loss_weight)

        self.nobs = config.nobs
        self.nagent = config.nagent
        self.nprompt = config.nprompt
        self.temperature_value_num = config.temperature_value_num
        self.temperature_resolution = config.temperature_resolution
        self.policy_value_num = config.policy_value_num
        self.policy_resolution = config.policy_resolution
        self.include_reward = config.include_reward

        # 4=idx_prompt, idx_a_self, idx_end_timestep, idx_reset_env; include reward -> 4 + 1 = 5
        # 2=value blow botom bound and value above upper bound; 2=off_action_id, idx_padding
        if self.include_reward:
            vocab_size = self.nobs + self.nagent + 5 + self.nprompt + (self.temperature_value_num + 3) + (self.policy_value_num + 1) + 2
        else:
            vocab_size = self.nobs + self.nagent + 4 + self.nprompt + (self.temperature_value_num + 3) + (self.policy_value_num + 1) + 2
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
        if not self.include_reward:
            self.SPECIAL_TOKENS = {
                'idx_prompt': 0,
                'idx_a_self': 1,
                'idx_end_timestep': 2,
                'idx_reset_env': 3
            }
        else:
            self.SPECIAL_TOKENS = {
                'idx_prompt': 0,
                'idx_a_self': 1,
                'idx_end_timestep': 2,
                'idx_reset_env': 3,
                'idx_reward': 4
            }
        self.PROMPT_BASE = self.SPECIAL_TOKENS_OFFSET + len(self.SPECIAL_TOKENS)
        self.TEMPERATURE_VALUE_BASE = self.PROMPT_BASE + self.nprompt
        self.POLICY_VALUE_BASE = self.TEMPERATURE_VALUE_BASE + self.temperature_value_num + 3 # Include lower and upper value
        self.ACTION_OFF_BASE = self.POLICY_VALUE_BASE + self.policy_value_num + 1

    def find_position(self, inputs):
        """
        inputs: [batch_size, seq_len]
        World Model (obs) position: value in [0, nobs)
        World Model (action of other agent): value in [nobs, nagent)
        Policy Model: value == self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_a_self']
        (If reward is inclued, World Model (reward): value == self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_reward'])
        return world_model_obs_out, world_model_action_out, policy_out, reward_out
        """
        world_model_obs_mask = (inputs < self.AGENT_IDX_OFFSET)
        world_model_action_mask = (inputs >= self.AGENT_IDX_OFFSET) & (inputs < self.SPECIAL_TOKENS_OFFSET)
        policy_mask = (inputs == self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_a_self'])
        if self.include_reward:
            reward_mask = (inputs == self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_reward'])
            return world_model_obs_mask, world_model_action_mask, policy_mask, reward_mask

        return world_model_obs_mask, world_model_action_mask, policy_mask

    def check_kl_loss_alignment(self, outputs, label_actions, loss_weight_policy, debug=False):
        """
        检查 KL loss 计算时，loss_wht 和 label_actions 的位置是否对齐
        
        参数:
            outputs: 模型输出 [batch_size, seq_len, vocab_size]
            label_actions: label action 分布 [batch_size, seq_len, action_dim]
            loss_weight_policy: policy 位置的权重掩码 [batch_size, seq_len]
            debug: 是否打印详细调试信息
        """
        if not debug:
            return
        
        print("\n" + "="*80)
        print("检查 KL Loss 位置对齐")
        print("="*80)
        
        # 1. 打印 shapes
        print(f"outputs shape: {outputs.shape}")
        print(f"label_actions shape: {label_actions.shape}")
        print(f"loss_weight_policy shape: {loss_weight_policy.shape}")
        
        batch_size, seq_len = loss_weight_policy.shape
        
        # 2. 获取 loss_weight_policy 不为 0 的位置
        policy_positions = (loss_weight_policy != 0)
        policy_indices = torch.nonzero(policy_positions, as_tuple=False)
        
        print(f"\nloss_weight_policy 不为 0 的位置:")
        print(f"  - 总共有 {policy_indices.shape[0]} 个位置")
        if policy_indices.shape[0] > 0:
            print(f"  - 前 5 个位置: {policy_indices[:5].tolist()}")
        
        # 3. 获取 label_actions prob 和不为零的位置
        # label_actions 是概率分布，shape: [batch_size, seq_len, action_dim]
        # 检查哪些位置的概率和 > 0（说明有有效的分布）
        prob_sums = label_actions.sum(dim=-1)  # [batch_size, seq_len]
        valid_prob_positions = (prob_sums > 0)
        valid_prob_indices = torch.nonzero(valid_prob_positions, as_tuple=False)
        
        print(f"\nlabel_actions 概率和不为零的位置:")
        print(f"  - 总共有 {valid_prob_indices.shape[0]} 个位置")
        if valid_prob_indices.shape[0] > 0:
            print(f"  - 前 5 个位置: {valid_prob_indices[:5].tolist()}")
        
        # 4. 检查对齐情况
        # 找出在 policy_positions 中但不在 valid_prob_positions 中的位置
        mismatch_positions = policy_positions & ~valid_prob_positions
        mismatch_indices = torch.nonzero(mismatch_positions, as_tuple=False)
        
        print(f"\n对齐检查结果:")
        print(f"  - policy_positions 中的位置数: {policy_indices.shape[0]}")
        print(f"  - valid_prob_positions 中的位置数: {valid_prob_indices.shape[0]}")
        print(f"  - 不匹配的位置数: {mismatch_indices.shape[0]}")
        
        if mismatch_indices.shape[0] > 0:
            print(f"  ⚠️  警告: 发现 {mismatch_indices.shape[0]} 个不匹配的位置！")
            print(f"  这些位置在 loss_weight_policy 中不为 0，但 label_actions 的概率和为 0:")
            print(f"  前 10 个不匹配位置: {mismatch_indices[:10].tolist()}")
            
            # 打印一些不匹配的详细信息
            for i in range(min(3, mismatch_indices.shape[0])):
                b, s = mismatch_indices[i].tolist()
                print(f"    位置 (batch={b}, seq={s}):")
                print(f"      loss_weight_policy = {loss_weight_policy[b, s].item()}")
                print(f"      label_actions prob sum = {prob_sums[b, s].item()}")
                print(f"      label_actions values = {label_actions[b, s, :5].tolist()}...")  # 只显示前5个值
        else:
            print(f"  ✅ 所有位置对齐正确！")
        
        # 5. 详细分析：检查 batch 维度上的分布
        print(f"\nBatch 维度分析:")
        for b in range(min(3, batch_size)):
            policy_cnt = policy_positions[b].sum().item()
            valid_cnt = valid_prob_positions[b].sum().item()
            mismatch_cnt = mismatch_positions[b].sum().item()
            print(f"  Batch {b}: policy_positions={policy_cnt}, valid_prob_positions={valid_cnt}, mismatch={mismatch_cnt}")
        
        print("="*80 + "\n")
        
        return mismatch_indices.shape[0] == 0  # 返回是否对齐


    def sequential_loss(self, inputs, label_actions, use_loss_weight=True, update_memory=True, use_kl=False, reduce_dim=1):
        """
        label_actions should have same shape as inputs, and replace the idx_a_self with label action.
        """
        seq_len = inputs.shape[1]
        ps = self.causal_model.position
        pe = ps + seq_len - 1
        
        if not self.config.use_context_warmup:
            use_loss_weight = False

        if(use_loss_weight and (self.loss_weight.shape[0] < pe)):
            log_fatal(f"Loss weight (shape {self.loss_weight.shape[0]}) should be longer" +
                    f" than sequence length {pe}")

        outputs, _ = self.forward(inputs, need_cache=False, update_memory=update_memory) #outputs: [batch_size, seq_len, vocab_size]
        outputs = outputs[:, :-1, :]
        if self.include_reward:
            world_model_obs_mask, world_model_action_mask, policy_mask, reward_mask = self.find_position(inputs[:,:-1])
        else:
            world_model_obs_mask, world_model_action_mask, policy_mask= self.find_position(inputs[:,:-1])
    
        
        loss_weight_wm_obs = world_model_obs_mask.float()
        loss_weight_wm_action = world_model_action_mask.float()
        loss_weight_policy = policy_mask.float()
        if self.include_reward:
            loss_weight_reward = reward_mask.float()
        if use_loss_weight:
            loss_weight_wm_obs *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_wm_action *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_policy *= self.loss_weight[ps:pe].unsqueeze(0)
            if self.include_reward:
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
        if use_kl:
            self.check_kl_loss_alignment(outputs, label_actions[:,:-1], loss_weight_policy, debug=True)
            loss["policy"], loss["count_p"] = weighted_loss(outputs,
                                       gt=label_actions[:,:-1],
                                       loss_type="kl",
                                       loss_wht=loss_weight_policy,
                                       reduce_dim=reduce_dim,
                                       need_cnt=True)
        else:
            loss["policy"], loss["count_p"] = weighted_loss(outputs,
                                        gt=label_actions[:,:-1],
                                        loss_type="ce",
                                        loss_wht=loss_weight_policy,
                                        reduce_dim=reduce_dim,
                                        need_cnt=True)
        if self.include_reward:
            loss["reward"] = weighted_loss(outputs,
                                        gt=inputs[:, 1:],
                                        loss_type="ce",
                                        loss_wht=loss_weight_reward,
                                        reduce_dim=reduce_dim,
                                        need_cnt=False)
        else:
            loss["reward"] = 0.0
        loss["ent"] = weighted_loss(outputs, 
                                        loss_type="ent", 
                                        loss_wht=loss_weight_policy,
                                        reduce_dim=reduce_dim)
        loss["causal-l2"] = parameters_regularization(self)
        return loss
        
    def generate(self, inputs, need_numpy=True, single_batch=True, reward_prediction=False, Temp=1.0):
        """
        0, inputs : tensor with shape [BT, NT], 
            if agents have different seq lenth, padding with value self.nvocab: 
            [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
            idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
            idx_padding ...]
        1, Forward with inputs, update memory = False. Get wd_obs, wd_action, and action.
        2, if reward_prediction:
                Form the sequences with action:
                    [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
                    idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
                    idx_prompt, prompt, idx_a_self, action]
                return wd_obs, wd_action, action.
            else:
                reutrn wd_obs, wd_action, action
        """
        # inputs: [BT, NT]
        device = next(self.parameters()).device
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs).to(device)
        else:
            inputs = inputs.to(device)
        BT = inputs.size(0)
        outputs, _ = self.forward(inputs, need_cache=False, update_memory=False, T=Temp)
        
        def sample_value(mask, output_in, deterministic, B, NT, D):
            output = output_in.clone()
            masked_output = output * mask
            row_sums = masked_output.sum(dim=1, keepdim=True)
            zero_rows = (row_sums.squeeze(1) == 0)
            if zero_rows.any():
                uniform_value = 1.0 / mask.sum(dim=1, keepdim=True)
                valid_mask = mask & (uniform_value > 0)
                masked_output[zero_rows] = uniform_value[zero_rows] * valid_mask[zero_rows].float()
                row_sums = masked_output.sum(dim=1, keepdim=True)
                
            normalized_output = masked_output / row_sums
            if deterministic:
                samples = torch.argmax(normalized_output, dim=1, keepdim=True)
            else:
                samples = torch.multinomial(normalized_output, num_samples=1)  # [B*NT, 1]
            output = samples.view(B, NT)  # [B, NT]
            return output
        
        def get_value(output, deterministic=False, 
                      policy_mask=None,
                      wm_obs_mask=None,
                      wm_action_mask=None): 
            origin_output = output.clone()
            B, NT, D = output.shape
            output = output.view(-1, D)  # [B*NT, D]

            mask_policy = torch.zeros_like(output, dtype=torch.bool)
            mask_wm_obs = torch.zeros_like(output, dtype=torch.bool)
            mask_wm_action = torch.zeros_like(output, dtype=torch.bool)
            start_idx_policy = self.POLICY_VALUE_BASE
            end_idx_policy = self.ACTION_OFF_BASE 
            start_idx_temp = self.TEMPERATURE_VALUE_BASE
            end_idx_temp = self.POLICY_VALUE_BASE

            mask_policy[:, start_idx_policy:end_idx_policy] = True
            mask_wm_obs[:, start_idx_temp:end_idx_temp] = True
            mask_wm_action[:, start_idx_temp:end_idx_temp] = True
            mask_wm_action[:, self.ACTION_OFF_BASE] = True
            
            # [B, NT]
            output_policy = sample_value(mask_policy, output, deterministic, B, NT, D)
            output_wm_obs = sample_value(mask_wm_obs, output, deterministic, B, NT, D)
            output_wm_action = sample_value(mask_wm_action, output, deterministic, B, NT, D)

            output_result = torch.zeros_like(output_policy)
            output_result[policy_mask] = output_policy[policy_mask]
            output_result[wm_obs_mask] = output_wm_obs[wm_obs_mask]
            output_result[wm_action_mask] = output_wm_action[wm_action_mask]
            

            if policy_mask is not None:
                if policy_mask.shape != (B, NT):
                    raise ValueError(f"policy_mask shape {policy_mask.shape} does not match output shape {B, NT}")
                value_range_probs = (origin_output * mask_policy.view(B, NT, D)).sum(dim=-1)  # [B, NT]
                policy_probs = value_range_probs[policy_mask] # [num_policy_positions]
                policy_value_prob_avg = policy_probs.mean().item()
                print(f"policy_value_prob_avg: {policy_value_prob_avg}")
                
            return output_result
        
        
        if self.include_reward:
            world_model_obs_mask, world_model_action_mask, policy_mask, reward_mask = self.find_position(inputs)
        else:
            world_model_obs_mask, world_model_action_mask, policy_mask= self.find_position(inputs)
        
        outputs = get_value(outputs, deterministic=False, 
                            policy_mask=policy_mask,
                            wm_obs_mask=world_model_obs_mask,
                            wm_action_mask=world_model_action_mask)
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

        if not reward_prediction or not self.include_reward:
            return world_model_obs, world_model_action, action
        else:
            new_value = torch.tensor([
                [int(action[i].item()), #a_self
                self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_reward']] # idx_reward
                for i in range(BT)], dtype=torch.int64, device=inputs.device)
            policy_idx = self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS['idx_a_self']
            new_nt = inputs.size(1) + 2
            new_inputs = torch.zeros((BT, new_nt), dtype=torch.int64, device=inputs.device)
            for i in range(BT):
                pos = torch.where(inputs[i] == policy_idx)[0]
                assert pos >= 0 and pos < inputs.size(1), f"idle position for policy_idx: {pos}"
                new_inputs[i, :pos+1] = inputs[i, :pos+1]
                new_inputs[i, pos+1:pos+3] = new_value[i]
                if pos < inputs.size(1) -1:
                    new_inputs[i, pos+3:] = inputs[i, pos+1:]
            outputs, _ = self.forward(new_inputs, need_cache=False, update_memory=False, T=Temp)
            outputs = get_value(outputs)
            _, _, _, reward_mask = self.find_position(new_inputs)
            world_model_reward = []
            for i in range(BT):
                world_model_reward.append(outputs[reward_mask[i]].detach().cpu())
            if need_numpy:
                world_model_reward = [reward.numpy() for reward in world_model_reward]
            return world_model_obs, world_model_action, action, world_model_reward


    def incontext_learn(self, inputs, cache=None, need_cache = False):
        """
        inputs : tensor with shape [1, NT], only support 1 batch for now: 
            [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
            idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
            idx_prompt, prompt, idx_a_self, a_self, (idx_reward, reward,) end_timestep]
        """
        device = next(self.parameters()).device
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs).to(device)
        else:
            inputs = inputs.to(device)

        _, cache = self.forward(inputs, cache=cache, need_cache=need_cache, update_memory=True)
        if need_cache:
            return cache                       

if __name__=='__main__':
    pass
