import os
import glob
import sys
import random
import torch
from tqdm import tqdm
import shutil
import multiprocessing
from multiprocessing import Pool, Manager
import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MultiAgentLoadDateSet(Dataset):
    def __init__(self, directory, time_step, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def _load_and_process_data(self, path):
        try:
            sequence = np.load(path + '/sequence.npy')
            label_action = np.load(path + '/label_action.npy')

            max_t = min(sequence.shape[0], 
                        label_action.shape[0])

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            return (sequence[n_b:n_e], 
                    label_action[n_b:n_e])
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 2
        
    def __getitem__(self, index):
        path = self.file_list[index]

        data = self._load_and_process_data(path)
        
        if any(arr is None for arr in data):
            return None

        seq_arr = torch.from_numpy(data[0].astype("int32")).long() 
        lact_arr = torch.from_numpy(data[1].astype("int32")).long() 

        # Orders: Seqnence and Action Label
        return seq_arr, lact_arr
    
class MultiAgentDistributionLoadDateSet(Dataset):
    def __init__(self, directory, time_step, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def _load_and_process_data(self, path):
        try:
            sequence = np.load(path + '/sequence.npy')
            label_action_distribution = np.load(path + '/label_action_distribution.npy')

            max_t = min(sequence.shape[0], 
                        label_action_distribution.shape[0])

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            return (sequence[n_b:n_e], 
                    label_action_distribution[n_b:n_e])
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 2
        
    def __getitem__(self, index):
        path = self.file_list[index]

        data = self._load_and_process_data(path)
        
        if any(arr is None for arr in data):
            return None

        seq_arr = torch.from_numpy(data[0].astype("int32")).long() 
        lact_arr = torch.from_numpy(data[1].astype("float32")).float()

        # Orders: Seqnence and Label Action Distribution
        return seq_arr, lact_arr


class MultiAgentDataSet(Dataset):
    """
    All of the id and value are transform into discrete integer values, forming the vocabular.
    - i.e., we have m obs_id, n agent_id -> m + n words
        idx_prompt, idx_a_self, idx_end_timestep, idx_reset_env -> 4 words
        (If reward is included, idx_prompt, idx_a_self, idx_end_timestep, idx_reset_env, idx_reward -> 5 words)
        p prompt_value -> p words (p=4) [0,1,2,unknown]
        value of temperature for obs sensor and agent action ~ [-16, 16], temperature_resolution 0.1, upper and lower -> 323 words
        value of policy action ~ [-3, 3], resolution 0.5 -> 13 words
        (value of reward ... )
        off_action_id -> 1 word
        idx_padding -> 1 word
        Then the total vocabular size = m + n + p + 4 + 323 + 13 + 1 + 1
        
    - For one timestep the sequence is arranged as: 
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_prompt, prompt, idx_a_self, a_self, idx_end_timestep ]
        (If reward is included:
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_prompt, prompt, idx_a_self, a_self, idx_reward, reward_value, idx_end_timestep])

    - For obs connection 2D matrix, the line is obs_id, the column is agent_idx,
        value is 1 if the obs is useful for agent.

    - For agent connection 2D matrix, the line is agent_idx, the column is agent_idx,
        value is 1 if the two agent have connection.    

    - The data is save as observations_*.npy, actions_behavior_*.npy, actions_behavior_*.npy, prompts_*.npy, rewards.npy
        The sequential data is constructed base on two connection matrix above as each time steps, and each agent can form a sequence.
    """
    def __init__(self, 
                 directory, 
                 time_step, 
                 max_obs_num, 
                 max_agent_num, 
                 prompt_num, 
                 temperature_value_num, 
                 temperature_resolution,
                 policy_value_num,
                 policy_resolution,
                 vocab_size, 
                 verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step
        self.max_obs_num = max_obs_num
        self.max_agent_num = max_agent_num
        self.prompt_num = prompt_num
        self.temperature_value_num = temperature_value_num
        self.temperature_resolution = temperature_resolution
        self.policy_value_num = policy_value_num
        self.policy_resolution = policy_resolution
        self.vocab_size = vocab_size
        self.temperature_min_value = - temperature_resolution * temperature_value_num / 2
        self.temperature_max_value = - self.temperature_min_value
        self.policy_min_value = - policy_resolution * policy_value_num / 2
        self.policy_max_value = - self.policy_min_value
        

        self.include_reward = False

        self._init_vocab_offsets()

        if(verbose):
            print(self.file_list)
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        # TODO multiple sequence in one file
        return len(self.file_list)

    def _init_vocab_offsets(self):
        self.OBS_IDX_OFFSET = 0
        self.AGENT_IDX_OFFSET = self.max_obs_num
        self.SPECIAL_TOKENS_OFFSET = self.AGENT_IDX_OFFSET + self.max_agent_num
        if self.include_reward:
            self.SPECIAL_TOKENS = {
                'idx_prompt': 0,
                'idx_a_self': 1,
                'idx_end_timestep': 2,
                'idx_reset_env': 3,
                'idx_reward': 4
            }
        else:
           self.SPECIAL_TOKENS = {
                'idx_prompt': 0,
                'idx_a_self': 1,
                'idx_end_timestep': 2,
                'idx_reset_env': 3
            }
        self.PROMPT_BASE = self.SPECIAL_TOKENS_OFFSET + len(self.SPECIAL_TOKENS)
        self.TEMPERATURE_VALUE_BASE = self.PROMPT_BASE + self.prompt_num
        self.POLICY_VALUE_BASE = self.TEMPERATURE_VALUE_BASE + self.temperature_value_num + 3 # Include lower and upper value
        self.ACTION_OFF_BASE = self.POLICY_VALUE_BASE + self.policy_value_num + 1 # Include 0, total discrete action num is policy_value_num + 1
        self.PADDING_IDX = self.vocab_size - 1

    def _validate_policy_value(self, value):
        discrete_values = np.arange(
            self.policy_min_value,
            self.policy_max_value + self.policy_resolution/2,
            self.policy_resolution
        )
        valid_mask = np.isclose(
            value[:, None], 
            discrete_values,
            atol=1e-5,
            rtol=1e-5
        ).any(axis=1)
        invalid_indices = np.where(~valid_mask)[0]
        if len(invalid_indices) > 0:
            error_info = []
            for idx in invalid_indices[:5]:
                actual = value[idx]
                closest_idx = np.abs(discrete_values - actual).argmin()
                closest_val = discrete_values[closest_idx]
                error_info.append(
                    f"idx {idx}: value {actual:.4f} invalid, "
                    f"Closest valid value: {closest_val:.4f}"
                )
            error_msg = (
                f"Fine {len(invalid_indices)} invalid values\n"
                f"Valid range:: [{self.policy_min_value:.4f}, {self.policy_max_value:.4f}]\n"
                f"Resolution: {self.policy_resolution:.4f}\n"
                f"Error msg:\n" + "\n".join(error_info)
            )
            raise ValueError(error_msg)
        return True
    
    def vocabularize(self, type, value, behavior_value=None,value_previous=None,use_diff_action=False):
        handler = getattr(self, f'_handle_{type}', None)
        if handler:
            if behavior_value is not None:
                return handler(value,behavior_value, use_diff_action=use_diff_action)
            if value_previous is not None:
                return handler(value,value_previous, use_diff_action=use_diff_action)
            return handler(value, use_diff_action=use_diff_action)
        raise ValueError(f"Invalid type: {type}")

    def _handle_obs_id(self, value, use_diff_action=False):
        return value

    def _handle_agent_id(self, value, use_diff_action=False):
        return self.AGENT_IDX_OFFSET + value

    def _handle_special_token(self, token_type, use_diff_action=False):
        return self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS[token_type]

    def _handle_prompt_value(self, value, use_diff_action=False):
        return self.PROMPT_BASE + value

    def _handle_value(self, value, behavior_value = None, use_diff_action=False):
        # value in (batch, timestep, value) shape
        scalar_input = False
        if not isinstance(value, np.ndarray):
            value = np.array(value)
            scalar_input = True

        if value.shape[2] == 2:
            raw_values = value[:, :, 0:1] 
            flags = value[:, :, 1:2] 

            result = np.zeros_like(raw_values, dtype=np.int64)
            on_mask = flags > 0.5
            padding_mask = flags < -0.5
            off_mask = ~on_mask & ~padding_mask
            if np.any(on_mask):
                on_values = raw_values[on_mask]
                if use_diff_action:
                    if behavior_value is None:
                        flags_previous_off = np.zeros_like(on_mask, dtype=bool)
                        flags_previous_off[:,0] = True
                        flags_previous_off[:,1:] = ~on_mask[:,:-1]
                        on_values_previous_off = flags_previous_off & on_mask
                        diff_values = np.zeros_like(raw_values)
                        diff_values[:,1:] = raw_values[:,1:] - raw_values[:,:-1]
                        diff_values[on_values_previous_off] = raw_values[on_values_previous_off]
                        on_values = diff_values[on_mask]
                    else:
                        raw_values_behavior = behavior_value[:, :, 0:1]
                        flags_behavior = behavior_value[:, :, 1:2]
                        behavior_on_mask = flags_behavior > 0.5
                        behavior_previous_off = np.zeros_like(behavior_on_mask, dtype=bool)
                        behavior_previous_off[:,0] = True
                        behavior_previous_off[:,1:] = ~behavior_on_mask[:,:-1]
                        behavior_previous_off = behavior_previous_off & on_mask
                        diff_values = np.zeros_like(raw_values)
                        diff_values[:,1:] = raw_values[:, 1:] - raw_values_behavior[:, :-1]
                        diff_values[behavior_previous_off] = raw_values[behavior_previous_off]
                        on_values = diff_values[on_mask]
                    
                    temp_result = np.zeros_like(on_values, dtype=np.int64)
                    self._validate_policy_value(on_values)
                    quantized = np.round((on_values - self.policy_min_value) / self.policy_resolution).astype(np.int64)
                    temp_result = self.POLICY_VALUE_BASE + quantized
                    result[on_mask] = temp_result
                else:
                    below_min = on_values < self.temperature_min_value
                    above_max = on_values > self.temperature_max_value
                    in_range = ~(below_min | above_max)
                    temp_result = np.zeros_like(on_values, dtype=np.int64)
                    if np.any(below_min):
                        temp_result[below_min] = self.TEMPERATURE_VALUE_BASE
                    if np.any(above_max):
                        temp_result[above_max] = self.TEMPERATURE_VALUE_BASE + self.temperature_value_num + 2
                    if np.any(in_range):
                        clipped = np.clip(on_values[in_range], self.temperature_min_value, self.temperature_max_value)
                        quantized = np.round((clipped - self.temperature_min_value) / self.temperature_resolution).astype(np.int64)
                        temp_result[in_range] = self.TEMPERATURE_VALUE_BASE + quantized + 1
                    result[on_mask] = temp_result

            if np.any(off_mask):
                result[off_mask] = self.ACTION_OFF_BASE

            if np.any(padding_mask):
                result[padding_mask] = self.PADDING_IDX
            
            return result.item() if scalar_input else result
        elif value.shape[2] == 1:
            raw_values = value
            result = np.zeros_like(raw_values, dtype=np.int64)
            below_min = raw_values < self.temperature_min_value
            above_max = raw_values > self.temperature_max_value
            in_range = ~(below_min | above_max)
            if np.any(below_min):
                result[below_min] = self.TEMPERATURE_VALUE_BASE
            if np.any(above_max):
                result[above_max] = self.TEMPERATURE_VALUE_BASE + self.temperature_value_num + 2
            if np.any(in_range):
                clipped = np.clip(raw_values[in_range], self.temperature_min_value, self.temperature_max_value)
                quantized = np.round((clipped - self.temperature_min_value) / self.temperature_resolution).astype(np.int64)
                result[in_range] = self.TEMPERATURE_VALUE_BASE + quantized + 1
            return result.item() if scalar_input else result

    def _handle_policy_action(self, value, use_diff_action=False):
        if value.shape[2] == 2:
            raw_values = value[:, :, 0:1] 
            flags = value[:, :, 1:2] 

            result = np.zeros_like(raw_values, dtype=np.int64)
            on_mask = flags > 0.5
            padding_mask = flags < -0.5
            off_mask = ~on_mask & ~padding_mask
            if np.any(on_mask):
                on_values = raw_values[on_mask]
                temp_result = np.zeros_like(on_values, dtype=np.int64)
                self._validate_policy_value(on_values)
                quantized = np.round((on_values - self.policy_min_value) / self.policy_resolution).astype(np.int64)
                temp_result = self.POLICY_VALUE_BASE + quantized
                result[on_mask] = temp_result
            if np.any(off_mask):
                result[off_mask] = self.ACTION_OFF_BASE
            if np.any(padding_mask):
                result[padding_mask] = self.PADDING_IDX
            return result
        else:
            raise ValueError("Invalid shape: {}".format(value.shape))
            
    
    def _handle_action_vocab(self, vocab, use_diff_action=False):
        # vocab: [num_of_agent, 1, 1]
        # result: [num_of_agent, 1, 2]
        if not hasattr(vocab, 'shape'):
            vocab = np.array(vocab)
        num_agents, timesteps = vocab.shape[:2]
        result = np.ones((num_agents, timesteps, 2))
        off_mask = (vocab == self.ACTION_OFF_BASE)
        on_mask = ~ off_mask
        result[off_mask] = [0.0,0.0]

        quantized_index = vocab[on_mask] - self.POLICY_VALUE_BASE
        restored_values = self.policy_min_value + quantized_index * self.policy_resolution
        result[on_mask] = np.column_stack([np.ones_like(restored_values), restored_values])

        return result

    def _load_and_process_data(self, path):
        try:
            observations = np.load(path + '/diff_observations.npy')
            actions_behavior = np.load(path + '/diff_actions_behavior.npy')
            actions_label = np.load(path + '/diff_actions_label.npy')
            prompts = np.load(path + '/prompts.npy')
            rewards = np.load(path + '/rewards.npy')
            resets = np.load(path + '/resets.npy')

            obs_matrix = np.load(path + '/obs_graph.npy')
            agent_matrix = np.load(path + '/agent_graph.npy')
            num_obs = observations.shape[0]
            num_agent = actions_behavior.shape[0]

            max_t = min(actions_label[0].shape[0], 
                        rewards.shape[0], 
                        actions_behavior[0].shape[0],
                        observations[0].shape[0],
                        prompts[0].shape[0])

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            sequence_list = []
            for agent_id in range(num_agent):
                relevant_obs = np.where(obs_matrix[agent_id] == 1)[0]
                connected_agents = np.where(agent_matrix[agent_id] == 1)[0]
                agent_sequence = []
                for t in range(n_b, n_e):
                    time_step_seq = []
                    # idx_o1, o1, idx_o3, o3, idx_o4, o4, ...,
                    for obs_id in relevant_obs:
                        obs_value = observations[obs_id][t]
                        time_step_seq.append(self.vocabularize('obs_id', obs_id))
                        time_step_seq.append(self.vocabularize('value', obs_value))
                    # idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
                    for other_agent_id in connected_agents:
                        other_agent_value = actions_behavior[other_agent_id][t]
                        time_step_seq.append(self.vocabularize('agent_id', other_agent_id))
                        time_step_seq.append(self.vocabularize('value', other_agent_value))
                    # idx_prompt, prompt, idx_a_self, a_self, (idx_reward, reward,) idx_end_timestep
                    time_step_seq.append(self.vocabularize('special_token', 'idx_prompt'))
                    time_step_seq.append(self.vocabularize('prompt_value', prompts[agent_id][t]))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                    time_step_seq.append(self.vocabularize('value', actions_behavior[agent_id][t]))
                    if self.include_reward:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_reward'))
                        time_step_seq.append(self.vocabularize('value', rewards[t]))
                    if resets[t]:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_reset_env'))
                    else:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_end_timestep'))
                    agent_sequence.append(time_step_seq)

                agent_sequence = np.concatenate(agent_sequence)

                policy_position_mask = (agent_sequence == self.vocabularize('special_token', 'idx_a_self'))
                label_vocabularize = self.vocabularize('value', actions_label[agent_id][n_b:n_e], )
                if np.sum(policy_position_mask) != len(label_vocabularize):
                    raise ValueError(
                        f"Agent {agent_id} poilicy position count ({np.sum(policy_position_mask)}) "
                        f"not equal to label length ({len(label_vocabularize)})"
                    )
                label_action_array = np.full(agent_sequence.shape, self.PADDING_IDX, dtype=np.int64)
                label_action_array[policy_position_mask] = label_vocabularize  

                sequence_list.append((agent_sequence, label_action_array))      

            return sequence_list
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 6
        
class MultiAgentDataSetVetorized(MultiAgentDataSet):
    def __init__(self, 
                 directory, 
                 time_step, 
                 max_obs_num, 
                 max_agent_num, 
                 prompt_num, 
                 temperature_value_num, 
                 temperature_resolution,
                 policy_value_num,
                 policy_resolution, 
                 vocab_size, 
                 use_policy_action=True,
                 use_action_mask=True,
                 verbose=False):
        super().__init__(directory, 
                         time_step, 
                         max_obs_num, 
                         max_agent_num, 
                         prompt_num, 
                         temperature_value_num, 
                         temperature_resolution,
                         policy_value_num,
                         policy_resolution,  
                         vocab_size, 
                         verbose)
        self.use_policy_action = use_policy_action
        self.use_action_mask = use_action_mask

    def _handle_obs_id(self, value, use_diff_action=False):
        return value.astype(np.int64) if isinstance(value, np.ndarray) else int(value)

    def _handle_agent_id(self, value, use_diff_action=False):
        return (self.AGENT_IDX_OFFSET + value).astype(np.int64) if isinstance(value, np.ndarray) else self.AGENT_IDX_OFFSET + value

    def _handle_prompt_value(self, value, use_diff_action=False):
        return (self.PROMPT_BASE + value).astype(np.int64) if isinstance(value, np.ndarray) else self.PROMPT_BASE + value

    def _interleave_columns(self, arrays):
        
        assert len(arrays) > 0, "At least one array needs to be input"
        shapes = {arr.shape for arr in arrays}
        assert len(shapes) == 1, "All arrays must have the same shape"
        n_rows, n_cols = arrays[0].shape
        n_arrays = len(arrays)
        
        merged = np.empty((n_rows, n_cols * n_arrays), dtype=arrays[0].dtype)
        
        for col_idx in range(n_cols):
            target_col = col_idx * n_arrays
            for arr_idx, arr in enumerate(arrays):
                merged[:, target_col + arr_idx] = arr[:, col_idx]
        
        return merged    

    def _generate_action_mask(self, self_action):
        # self_action: (1, timesteps, 2)
        # action_mask: (1, timesteps)
        timesteps = self_action.shape[1]
        action_mask = np.ones((1, timesteps), dtype=np.float32)
        sample_mask_ratio = np.random.uniform(low=0.0, high=1.0)
        if sample_mask_ratio > 0.5:
            partially_mask_ratio =  np.random.uniform(low=0.0, high=0.5)
            num_masked_steps = int(partially_mask_ratio * timesteps)
            if num_masked_steps > 0:
                masked_indices = np.random.choice(
                    timesteps, 
                    size=num_masked_steps, 
                    replace=False
                )
                action_mask[0, masked_indices] = 0
        elif sample_mask_ratio > 0.3:
            action_mask[0, :] = 0
        else:
            pass # no mask
        return action_mask
    
    def _mask_action_data(self, action_data, action_mask):
        # action_data: (1, timesteps, 2)
        # action_mask: (1, timesteps)
        # masked_action_data: (1, timesteps, 2)
        masked_action_data = action_data.copy()
        expanded_mask = action_mask[..., np.newaxis]
        replacement = np.stack([
            action_data[..., 0],
            np.where(action_mask == 0, -1, action_data[..., 1]) 
        ], axis=-1) 
        masked_action_data = np.where(expanded_mask == 1, action_data, replacement)
        return masked_action_data

    def _load_and_process_data(self, path, use_relative_idx=True):
        try:
            observations = np.load(path + '/diff_observations.npy')
            actions_behavior = np.load(path + '/diff_actions_behavior.npy')
            actions_label = np.load(path + '/diff_actions_label.npy')
            actions_behavior_policy = np.load(path + '/actions_behavior_policy.npy')
            actions_label_policy = np.load(path + '/actions_label_policy.npy')
            label_action_distribution = np.load(path + '/label_action_distribution.npy')
            
            prompts = np.load(path + '/prompts.npy')
            rewards = np.load(path + '/rewards.npy')
            resets = np.load(path + '/resets.npy')
            obs_matrix = np.load(path + '/obs_graph.npy')
            agent_matrix = np.load(path + '/agent_graph.npy')

            max_t = min(actions_label[0].shape[0], 
                        rewards.shape[0], 
                        actions_behavior[0].shape[0],
                        observations[0].shape[0],
                        prompts.shape[0])
            num_agents = actions_behavior.shape[0]

            # Convert to a numpy array and unify the dimensions
            observations = np.stack(observations, axis=0)       # (num_obs, total_timesteps, 1)
            actions_behavior = np.stack(actions_behavior, axis=0) # (num_agents, total_timesteps, 2)
            actions_behavior_policy = np.stack(actions_behavior_policy, axis=0) # (num_agents, total_timesteps, 2)
            

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            time_slice = slice(n_b, n_e)
            time_slice_last_action = slice(n_b, n_e)
            time_slice_current_action = slice(n_b + 1, n_e + 1)
            num_timesteps = n_e - n_b
            # Get the associated and sorted indexes of all agents
            obs_conn_mask = []
            agent_conn_mask = []
            for agent_id in range(num_agents):
                sensor_indices = np.where(obs_matrix[agent_id] > 0)[0]
                sensor_weights = obs_matrix[agent_id][sensor_indices]
                sensor_sorted_indices = np.argsort(sensor_weights)
                sensor_indices = sensor_indices[sensor_sorted_indices]
                agent_indices = np.where(agent_matrix[agent_id] > 0)[0]
                agent_weights = agent_matrix[agent_id][agent_indices]
                agent_sorted_indices = np.argsort(agent_weights)
                agent_indices = agent_indices[agent_sorted_indices]
                obs_conn_mask.append(sensor_indices)
                agent_conn_mask.append(agent_indices)

            sequence_list = []
            for agent_id in range(num_agents):
                # Obtain the association relationship of the current agent
                obs_mask = obs_conn_mask[agent_id]
                agent_mask = agent_conn_mask[agent_id]

                # Obs -> [obs_id, value] × timesteps
                obs_data = observations[obs_mask][:, time_slice] # (num_relative_obs, timesteps)
                relevant_obs = obs_mask # (num_relative_obs)
                obs_idx_vocabularize = self.vocabularize('obs_id', relevant_obs)
                obs_value_vocabularize = self.vocabularize('value', obs_data).squeeze()
                if use_relative_idx:
                    obs_idx_vocabularize = np.arange(len(obs_idx_vocabularize))
                obs_idx_vocabularize = np.broadcast_to(obs_idx_vocabularize[:, np.newaxis],
                                                   shape=obs_value_vocabularize.shape).astype(np.int64)  # (num_relative_obs, timesteps)
                obs_value_vocabularize = obs_value_vocabularize.T  # (timesteps, num_relative_obs)
                obs_idx_vocabularize = obs_idx_vocabularize.T  # (timesteps, num_relative_obs)
                # Merge into (timesteps, num_relative_obs * 2)
                obs_pairs = self._interleave_columns([obs_idx_vocabularize, obs_value_vocabularize])

                # Other agent -> [agent_id, value] × timesteps
                agent_data = actions_behavior[agent_mask][:, time_slice_last_action] # (num_relative_agents, timesteps)
                relevant_agents = agent_mask # (num_relative_agents)
                # Self last action
                self_last_action = np.expand_dims(actions_behavior[agent_id][time_slice_last_action], axis=0) # (1, timesteps)
                # create self action mask, mask out self action in self_last_action and agent_data_vocabularize
                if self.use_action_mask:
                    action_mask = self._generate_action_mask(self_last_action)
                    self_last_action = self._mask_action_data(self_last_action, action_mask)
                agent_data = np.concatenate([self_last_action, agent_data], axis=0) # (1 + num_relative_agents, timesteps)
                relevant_agents = np.insert(relevant_agents, 0, agent_id) # (1 + num_relative_agents)

                agent_idx_vocabularize = self.vocabularize('agent_id', relevant_agents)
                agent_value_vocabularize = self.vocabularize('value', agent_data, use_diff_action=False).squeeze()
                if use_relative_idx:
                    agent_idx_vocabularize = np.arange(len(agent_idx_vocabularize)) + self.AGENT_IDX_OFFSET
                agent_idx_vocabularize = np.broadcast_to(agent_idx_vocabularize[:, np.newaxis],
                                                         shape=agent_value_vocabularize.shape).astype(np.int64)  # (num_relative_agents, timesteps)
                agent_value_vocabularize = agent_value_vocabularize.T  # (timesteps, num_relative_agents)
                agent_idx_vocabularize = agent_idx_vocabularize.T  # (timesteps, num_relative_agents)
                # Merge into (timesteps, num_relative_agents * 2)
                agent_pairs = self._interleave_columns([agent_idx_vocabularize, agent_value_vocabularize])
                
                # meta_data: idx_prompt, prompt, idx_a_self, a_self, (idx_reward, reward,) idx_end_timestep/idx_reset(if reset)
                # meta_pairs in (num_timesteps)
                idx_prompt_vocabularize = self.vocabularize('special_token', 'idx_prompt')
                idx_prompt_vocabularize = np.full((num_timesteps, 1), idx_prompt_vocabularize, dtype=np.int64)
                idx_a_self_vocabularize = self.vocabularize('special_token', 'idx_a_self')
                idx_a_self_vocabularize = np.full((num_timesteps, 1), idx_a_self_vocabularize, dtype=np.int64)
                idx_end_timestep_vocabularize = self.vocabularize('special_token', 'idx_end_timestep')
                idx_reset_vocabularize = self.vocabularize('special_token', 'idx_reset_env')
                idx_end_timestep_vocabularize = np.where(resets.reshape(-1, 1), idx_reset_vocabularize, idx_end_timestep_vocabularize)
                prompts_vocabularize = self.vocabularize('prompt_value', prompts[time_slice]).squeeze()
                prompts_vocabularize = prompts_vocabularize[:, np.newaxis]
                if self.use_policy_action:
                    policy_action = actions_behavior_policy[agent_id][time_slice_current_action,:][np.newaxis, :]
                    if self.use_action_mask:
                        policy_action = self._mask_action_data(policy_action, action_mask)
                    agent_data_vocabularize = self.vocabularize('policy_action', policy_action).squeeze()
                else:
                    policy_action = actions_behavior[agent_id][time_slice_current_action,:][np.newaxis, :]
                    if self.use_action_mask:
                        policy_action = self._mask_action_data(policy_action, action_mask)
                    agent_data_vocabularize = self.vocabularize('value', policy_action, use_diff_action=True).squeeze()
                agent_data_vocabularize = agent_data_vocabularize[:, np.newaxis]
                if self.include_reward:
                    idx_reward_vocabularize = self.vocabularize('special_token', 'idx_reward')
                    idx_reward_vocabularize = np.full((num_timesteps, 1), idx_reward_vocabularize, dtype=np.int64)
                    rewards_vocabularize = self.vocabularize('value', rewards[time_slice][np.newaxis, :]).squeeze()
                    rewards_vocabularize = rewards_vocabularize[:, np.newaxis]
                    meta_pairs = np.concatenate([idx_prompt_vocabularize, prompts_vocabularize,
                                                idx_a_self_vocabularize, agent_data_vocabularize,
                                                idx_reward_vocabularize, rewards_vocabularize,
                                                idx_end_timestep_vocabularize], axis= 1) # (num_timesteps, 7) 
                else:
                    meta_pairs = np.concatenate([idx_prompt_vocabularize, prompts_vocabularize,
                                                idx_a_self_vocabularize, agent_data_vocabularize,
                                                idx_end_timestep_vocabularize], axis= 1) # (num_timesteps, 5)               

                # Merge all data
                agent_seq = np.concatenate([agent_pairs, obs_pairs, meta_pairs], axis=1) # (timesteps, num_relative_obs * 2 + num_relative_agents * 2 + 5or7)
                agent_seq = agent_seq.reshape(-1) # (timesteps * (num_relative_obs * 2 + num_relative_agents * 2 + 5or7))
                
                policy_position_mask = (agent_seq == self.vocabularize('special_token', 'idx_a_self'))
                if self.use_policy_action:
                    label_vocabularize = self.vocabularize('policy_action', actions_label_policy[agent_id][time_slice_current_action][np.newaxis, :]).squeeze()
                else:
                    label_vocabularize = self.vocabularize('value', actions_label[agent_id][time_slice_current_action][np.newaxis, :],
                                                        behavior_value = actions_behavior[agent_id][time_slice_current_action,:][np.newaxis, :],
                                                        use_diff_action=True).squeeze()
                if np.sum(policy_position_mask) != len(label_vocabularize):
                    raise ValueError(
                        f"Agent {agent_id} poilicy position count ({np.sum(policy_position_mask)}) "
                        f"not equal to label length ({len(label_vocabularize)})"
                    )
                label_action_array = np.full(agent_seq.shape, self.PADDING_IDX, dtype=np.int64)
                label_action_array[policy_position_mask] = label_vocabularize

                discrete_action_dim = label_action_distribution.shape[2]
                label_action_distribution_array = np.full(
                    (agent_seq.shape[0], discrete_action_dim), 
                    0.0, 
                    dtype=np.float32
                )
                label_action_distribution_array[policy_position_mask] = label_action_distribution[agent_id][time_slice_last_action]
                
                sequence_list.append((agent_seq, label_action_array, label_action_distribution_array))

            return sequence_list
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 3

def process_subdir(args):
    """
    A parallel task function for handling a single subfolder
    """
    sub_dir, load_dir, save_dir, counter, lock, params = args
    sub_path = os.path.join(load_dir, sub_dir)
    
    time_step = params['time_step']
    max_obs_num = params['max_obs_num']
    max_agent_num = params['max_agent_num']
    prompt_num = params['prompt_num']
    temperature_value_num = params['temperature_value_num']
    temperature_resolution = params['temperature_resolution']
    policy_value_num = params['policy_value_num']
    policy_resolution = params['policy_resolution']
    vocab_size = params['vocab_size']
    
    try:
        # Each process creates its own instance of the dataset
        dataset = MultiAgentDataSetVetorized(
            directory=load_dir,
            time_step=time_step,
            max_obs_num=max_obs_num,
            max_agent_num=max_agent_num,
            prompt_num=prompt_num,
            temperature_value_num=temperature_value_num,
            temperature_resolution=temperature_resolution,
            policy_value_num=policy_value_num,
            policy_resolution=policy_resolution,
            vocab_size=vocab_size,
            verbose=False
        )
        
        results = dataset._load_and_process_data(sub_path)
        
        # Use locks to ensure the security of the counter
        with lock:
            current_counter = counter.value
            # Pre-assign all the record numbers required for this task
            record_ids = list(range(current_counter, current_counter + len(results)))
            counter.value += len(results)
        
        for i, (sequence, label_action, label_action_distribution) in enumerate(results):
            record_name = f"record_{record_ids[i]:06d}"
            record_path = os.path.join(save_dir, record_name)
            os.makedirs(record_path, exist_ok=True)
            
            np.save(os.path.join(record_path, "sequence.npy"), sequence)
            np.save(os.path.join(record_path, "label_action.npy"), label_action)
            np.save(os.path.join(record_path, "label_action_distribution.npy"), label_action_distribution)
            
            # Save origin data (optional)
            # src_files = glob.glob(os.path.join(sub_path, "*"))
            # for src_file in src_files:
            #     if os.path.isfile(src_file):
            #         shutil.copy2(src_file, record_path)
            
            print(f"Process {os.getpid()}: Saved record {record_name} from {sub_dir} agent {i}")
        
        return len(results)
    
    except Exception as e:
        print(f"Error processing {sub_dir}: {str(e)}")
        return 0

def main(load_dir, 
         save_dir, 
         time_step, 
         max_obs_num, 
         max_agent_num, 
         prompt_num, 
         temperature_value_num, 
         temperature_resolution, 
         policy_value_num, 
         policy_resolution,
         vocab_size, 
         num_workers=None):
    os.makedirs(save_dir, exist_ok=True)
    
    sub_dirs = [d for d in os.listdir(load_dir) 
                if os.path.isdir(os.path.join(load_dir, d))]
    
    print(f"Found {len(sub_dirs)} subdirectories to process")
    
    params = {
        'time_step': time_step,
        'max_obs_num': max_obs_num,
        'max_agent_num': max_agent_num,
        'prompt_num': prompt_num,
        'temperature_value_num': temperature_value_num,
        'temperature_resolution': temperature_resolution,
        'policy_value_num': policy_value_num,
        'policy_resolution': policy_resolution,
        'vocab_size': vocab_size
    }
    
    tasks = [(sub_dir, load_dir, save_dir, None, None, params) 
             for sub_dir in sub_dirs]
    
    total_records = 0
    
    if num_workers == 1:
        print("Processing in main thread (num_workers=1)...")
        results = []
        for task in tqdm(tasks, desc="Processing subdirectories"):
            counter = type('Counter', (), {'value': 0})
            lock = type('Lock', (), {'acquire': lambda: None, 'release': lambda: None})
            task = (task[0], task[1], task[2], counter, lock, task[5])
            
            result = process_subdir(task)
            results.append(result)
        
        total_records = sum(results)
    else:
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = min(num_workers, multiprocessing.cpu_count())
        
        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        
        tasks = [(sub_dir, load_dir, save_dir, counter, lock, params) 
                 for sub_dir in sub_dirs]
        
        print(f"Using {num_workers} worker processes...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_subdir, tasks), 
                               total=len(tasks),
                               desc="Processing subdirectories"))
            
            total_records = sum(results)
    
    print(f"\nProcessing completed! Total records saved: {total_records}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process multi-agent dataset in parallel")
    parser.add_argument("--load_dir", type=str, required=True, 
                        help="Directory containing subdirectories with data")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save processed records")
    parser.add_argument("--time_step", type=int, required=True,
                        help="Time step parameter")
    parser.add_argument("--max_obs_num", type=int, required=True,
                        help="Maximum observations parameter")
    parser.add_argument("--max_agent_num", type=int, required=True,
                        help="Maximum agents parameter")
    parser.add_argument("--prompt_num", type=int, required=True,
                        help="Maximum prompts parameter")
    parser.add_argument("--temperature_value_num", type=int, required=True,
                        help="Value number parameter")
    parser.add_argument("--temperature_resolution", type=float, required=True,
                        help="Resolution parameter")
    parser.add_argument("--policy_value_num", type=int, required=True,
                        help="Value number parameter")
    parser.add_argument("--policy_resolution", type=float, required=True,
                        help="Resolution parameter")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Vocab size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes to use (default: all cores)")
    args = parser.parse_args()
    
    main(
        args.load_dir,
        args.save_dir,
        args.time_step,
        args.max_obs_num,
        args.max_agent_num,
        args.prompt_num,
        args.temperature_value_num,
        args.temperature_resolution,
        args.policy_value_num,
        args.policy_resolution,
        args.vocab_size,
        num_workers=args.num_workers
    )

    # Develop
    # os.makedirs(args.save_dir, exist_ok=True)
    
    # sub_dirs = [d for d in os.listdir(args.load_dir) 
    #             if os.path.isdir(os.path.join(args.load_dir, d))]
    
    # print(f"Found {len(sub_dirs)} subdirectories to process")
    # dataset = MultiAgentDataSetVetorized(
    #     directory=args.load_dir,
    #     time_step=args.time_step,
    #     max_obs_num=args.max_obs_num,
    #     max_agent_num=args.max_agent_num,
    #     prompt_num=args.prompt_num,
    #     temperature_value_num=args.temperature_value_num,
    #     temperature_resolution=args.temperature_resolution,
    #     policy_value_num=args.policy_value_num,
    #     policy_resolution=args.policy_resolution,
    #     vocab_size=args.vocab_size,
    #     verbose=True
    # )
    # sub_path = os.path.join(args.load_dir, sub_dirs[0])
    # results = dataset._load_and_process_data(sub_path)
    # for i, (sequence, label_action, label_action_distribution) in enumerate(results):
    #         record_name = f"record_000000"
    #         record_path = os.path.join(args.save_dir, record_name)
    #         os.makedirs(record_path, exist_ok=True)
            
    #         np.save(os.path.join(record_path, "sequence.npy"), sequence)
    #         np.save(os.path.join(record_path, "label_action.npy"), label_action)
    #         np.save(os.path.join(record_path, "label_action_distribution.npy"), label_action_distribution)

