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

class MultiAgentDataSet(Dataset):
    """
    All of the id and value are transform into discrete integer values, forming the vocabular.
    - i.e., we have m obs_id, n agent_id -> m + n words
        idx_policy, idx_tag, idx_a_self, idx_reward, idx_end_timestep, idx_reset_env -> 6 words
        t tag_value -> t words 
        value of obs, action, and reward ~ [-10, 10], resolution 0.1 -> 200 words
        off_action_id -> 1 word
        idx_padding -> 1 word
        Then the total vocabular size = m + n + t + 208
        
    - For one timestep the sequence is arranged as: 
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep ]

    - For obs connection 2D matrix, the line is obs_id, the column is agent_idx,
        value is 1 if the obs is useful for agent.

    - For agent connection 2D matrix, the line is agent_idx, the column is agent_idx,
        value is 1 if the two agent have connection.    

    - The data is save as observations_*.npy, actions_behavior_*.npy, actions_behavior_*.npy, tags_*.npy, rewards.npy
        The sequential data is constructed base on two connection matrix above as each time steps, and each agent can form a sequence.
    """
    def __init__(self, directory, time_step, max_obs_num, max_agent_num, tag_num, value_num, resolution, vocab_size, verbose=False):
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
        self.tag_num = tag_num
        self.resolution = resolution
        self.vocab_size = vocab_size
        self.min_value = - resolution * value_num / 2
        self.max_value = - self.min_value
        self.value_num = value_num

        self._init_vocab_offsets()

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        # TODO multiple sequence in one file
        return len(self.file_list)

    def _init_vocab_offsets(self):
        self.OBS_IDX_OFFSET = 0
        self.AGENT_IDX_OFFSET = self.max_obs_num
        self.SPECIAL_TOKENS_OFFSET = self.AGENT_IDX_OFFSET + self.max_agent_num
        self.SPECIAL_TOKENS = {
            'idx_policy': 0,
            'idx_tag': 1,
            'idx_a_self': 2,
            'idx_reward': 3,
            'idx_end_timestep': 4,
            'idx_reset_env': 5
        }
        self.TAG_BASE = self.SPECIAL_TOKENS_OFFSET + len(self.SPECIAL_TOKENS)
        self.VALUE_BASE = self.TAG_BASE + self.tag_num
        self.ACTION_OFF_BASE = self.VALUE_BASE + self.value_num + 2 # 2: lower and upper value

    def vocabularize(self, type, value, behavior_value=None):
        handler = getattr(self, f'_handle_{type}', None)
        if handler:
            if behavior_value is not None:
                return handler(value,behavior_value)
            else:
                return handler(value)
        raise ValueError(f"Invalid type: {type}")

    def _handle_obs_id(self, value):
        return value

    def _handle_agent_id(self, value):
        return self.AGENT_IDX_OFFSET + value

    def _handle_special_token(self, token_type):
        return self.SPECIAL_TOKENS_OFFSET + self.SPECIAL_TOKENS[token_type]

    def _handle_tag_value(self, value):
        return self.TAG_BASE + value

    def _handle_value(self, value, behavior_value = None, use_diff_action = True):
        scalar_input = False
        if not isinstance(value, np.ndarray):
            value = np.array(value)
            scalar_input = True

        if value.shape[2] == 2:
            raw_values = value[:, :, 0:1] 
            flags = value[:, :, 1:2] 

            result = np.zeros_like(raw_values, dtype=np.int64)
            on_mask = flags > 0.5
            off_mask = ~on_mask
            if np.any(on_mask):
                on_values = raw_values[on_mask]
                if use_diff_action:
                    if behavior_value is None:
                        flags_previous_off = np.zeros_like(on_mask, dtype=bool)
                        flags_previous_off[0] = True
                        flags_previous_off[1:] = ~on_mask[:-1]
                        on_values_previous_off = flags_previous_off[on_mask]
                        diff_values = np.zeros_like(on_values)
                        diff_values[1:] = on_values[1:] - on_values[:-1]
                        diff_values[on_values_previous_off] = on_values[on_values_previous_off]
                        on_values = diff_values
                    else:
                        raw_values_behavior = behavior_value[:, :, 0:1][on_mask]
                        flags_behavior = behavior_value[:, :, 1:2]
                        behavior_on_mask = flags_behavior > 0.5
                        behavior_previous_off = np.zeros_like(behavior_on_mask, dtype=bool)
                        behavior_previous_off[0] = True
                        behavior_previous_off[1:] = ~behavior_on_mask[:1]
                        behavior_previous_off = behavior_previous_off[on_mask]
                        diff_values = np.zeros_like(on_values)
                        diff_values[1:] = on_values[1:] - raw_values_behavior[:-1]
                        diff_values[behavior_previous_off] = on_values[behavior_previous_off]
                        on_values = diff_values

                below_min = on_values < self.min_value
                above_max = on_values > self.max_value
                in_range = ~(below_min | above_max)
                temp_result = np.zeros_like(on_values, dtype=np.int64)
                if np.any(below_min):
                    temp_result[below_min] = self.VALUE_BASE
                if np.any(above_max):
                    temp_result[above_max] = self.VALUE_BASE + self.value_num + 1
                if np.any(in_range):
                    clipped = np.clip(on_values[in_range], self.min_value, self.max_value)
                    quantized = np.round((clipped - self.min_value) / self.resolution).astype(np.int64)
                    temp_result[in_range] = self.VALUE_BASE + quantized + 1
                result[on_mask] = temp_result

            if np.any(off_mask):
                result[off_mask] = self.ACTION_OFF_BASE
            
            return result.item() if scalar_input else result
        elif value.shape[2] == 1:
            raw_values = value
            result = np.zeros_like(raw_values, dtype=np.int64)
            below_min = raw_values < self.min_value
            above_max = raw_values > self.max_value
            in_range = ~(below_min | above_max)
            if np.any(below_min):
                result[below_min] = self.VALUE_BASE
            if np.any(above_max):
                result[above_max] = self.VALUE_BASE + self.value_num + 1
            if np.any(in_range):
                clipped = np.clip(raw_values[in_range], self.min_value, self.max_value)
                quantized = np.round((clipped - self.min_value) / self.resolution).astype(np.int64)
                result[in_range] = self.VALUE_BASE + quantized + 1
            return result.item() if scalar_input else result

    def _handle_action_vocab(self, vocab, value_previous=None, use_diff_action=True):
        # vocab: [num_of_agent, T, value]
        num_agents, timesteps = vocab.shape[:2]
        result = np.zeros((num_agents, timesteps, 2))
        off_mask = (vocab == self.ACTION_OFF_BASE)
        on_mask = ~ off_mask
        result[off_mask] = [0.0,0.0]
        
        vocab_on = vocab[on_mask]
        below_min_mask = (vocab_on == self.VALUE_BASE)
        above_max_mask = (vocab_on == self.VALUE_BASE + self.value_num + 1)
        in_range_mask = ~(below_min_mask | above_max_mask)
        result[on_mask][below_min_mask] = [self.min_value, 1.0]
        result[on_mask][above_max_mask] = [self.max_value, 1.0]
        if np.any(in_range_mask):
            quantized_index = vocab_on[in_range_mask] - self.VALUE_BASE - 1
            restored_values = self.min_value + quantized_index * self.resolution
            result[on_mask][in_range_mask] = np.column_stack([restored_values, np.ones_like(restored_values)])
        
        if use_diff_action and value_previous is not None:
            off_mask_previous = (value_previous[:, :, 1] < 0.5)
            on_mask_previous = ~ off_mask_previous
            both_on_mask = on_mask_previous & on_mask
            diff_value = result[:, :, 0][both_on_mask] - value_previous[:, :, 0][both_on_mask]
            result[:, :, 0][both_on_mask]= diff_value

        return result

    def _load_and_process_data(self, path):
        try:
            observations = np.load(path + '/diff_observations.npy')
            actions_behavior = np.load(path + '/diff_actions_behavior.npy')
            actions_label = np.load(path + '/diff_actions_label.npy')
            tags = np.load(path + '/tags.npy')
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
                        tags[0].shape[0])

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
                    # idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep
                    time_step_seq.append(self.vocabularize('special_token', 'idx_policy'))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_tag'))
                    time_step_seq.append(self.vocabularize('tag_value', tags[agent_id][t]))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                    time_step_seq.append(self.vocabularize('value', actions_behavior[agent_id][t]))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_reward'))
                    time_step_seq.append(self.vocabularize('value', rewards[t]))
                    if resets[t]:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_reset_env'))
                    else:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_end_timestep'))
                    agent_sequence.append(time_step_seq)

                agent_sequence = np.concatenate(agent_sequence)

                policy_position_mask = (agent_sequence == self.vocabularize('special_token', 'idx_policy'))
                label_vocabularize = self.vocabularize('value', actions_label[agent_id][n_b:n_e], )
                if np.sum(policy_position_mask) != len(label_vocabularize):
                    raise ValueError(
                        f"Agent {agent_id} poilicy position count ({np.sum(policy_position_mask)}) "
                        f"not equal to label length ({len(label_vocabularize)})"
                    )
                pad_token = self.vocab_size - 1 
                label_action_array = np.full(agent_sequence.shape, pad_token, dtype=np.int64)
                label_action_array[policy_position_mask] = label_vocabularize  

                sequence_list.append((agent_sequence, label_action_array))      

            return sequence_list
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 6
        
class MultiAgentDataSetVetorized(MultiAgentDataSet):
    def __init__(self, directory, time_step, max_obs_num, max_agent_num, tag_num, value_num, resolution, vocab_size, verbose=False):
        super().__init__(directory, time_step, max_obs_num, max_agent_num, tag_num, value_num, resolution, vocab_size, verbose)

    def _handle_obs_id(self, value):
        return value.astype(np.int64) if isinstance(value, np.ndarray) else int(value)

    def _handle_agent_id(self, value):
        return (self.AGENT_IDX_OFFSET + value).astype(np.int64) if isinstance(value, np.ndarray) else self.AGENT_IDX_OFFSET + value

    def _handle_tag_value(self, value):
        return (self.TAG_BASE + value).astype(np.int64) if isinstance(value, np.ndarray) else self.TAG_BASE + value

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

    def _load_and_process_data(self, path, use_relative_idx=False):
        try:
            observations = np.load(path + '/diff_observations.npy')
            actions_behavior = np.load(path + '/diff_actions_behavior.npy')
            actions_label = np.load(path + '/diff_actions_label.npy')
            tags = np.load(path + '/tags.npy')
            rewards = np.load(path + '/rewards.npy')
            resets = np.load(path + '/resets.npy')
            obs_matrix = np.load(path + '/obs_graph.npy')
            agent_matrix = np.load(path + '/agent_graph.npy')

            max_t = min(actions_label[0].shape[0], 
                        rewards.shape[0], 
                        actions_behavior[0].shape[0],
                        observations[0].shape[0],
                        tags[0].shape[0])
            num_agents = actions_behavior.shape[0]

            # Convert to a numpy array and unify the dimensions
            observations = np.stack(observations, axis=0)       # (num_obs, total_timesteps)
            actions_behavior = np.stack(actions_behavior, axis=0) # (num_agents, total_timesteps)
            tags = np.stack(tags, axis=0)                      # (num_agents, total_timesteps)
            

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            time_slice = slice(n_b, n_e)
            num_timesteps = n_e - n_b
            # Get the associated indexes of all agents
            obs_conn_mask = [obs_matrix[i].astype(bool) for i in range(num_agents)]
            agent_conn_mask = [agent_matrix[i].astype(bool) for i in range(num_agents)]

            sequence_list = []
            for agent_id in range(num_agents):
                # Obtain the association relationship of the current agent
                obs_mask = obs_conn_mask[agent_id]
                agent_mask = agent_conn_mask[agent_id]

                # Obs -> [obs_id, value] × timesteps
                obs_data = observations[obs_mask][:, time_slice] # (num_relative_obs, timesteps)
                relevant_obs = np.where(obs_mask)[0] # (num_relative_obs)
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
                agent_data = actions_behavior[agent_mask][:, time_slice] # (num_relative_agents, timesteps)
                relevant_agents = np.where(agent_mask)[0] # (num_relative_agents)
                
                agent_idx_vocabularize = self.vocabularize('agent_id', relevant_agents)
                agent_value_vocabularize = self.vocabularize('value', agent_data).squeeze()
                if use_relative_idx:
                    agent_idx_vocabularize = np.arange(len(agent_idx_vocabularize)) + self.AGENT_IDX_OFFSET
                agent_idx_vocabularize = np.broadcast_to(agent_idx_vocabularize[:, np.newaxis],
                                                         shape=agent_value_vocabularize.shape).astype(np.int64)  # (num_relative_agents, timesteps)
                agent_value_vocabularize = agent_value_vocabularize.T  # (timesteps, num_relative_agents)
                agent_idx_vocabularize = agent_idx_vocabularize.T  # (timesteps, num_relative_agents)
                # Merge into (timesteps, num_relative_agents * 2)
                agent_pairs = self._interleave_columns([agent_idx_vocabularize, agent_value_vocabularize])
                
                # meta_data: idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep/idx_reset(if reset)
                # meta_pairs in (num_timesteps)
                idx_policy_vocabularize = self.vocabularize('special_token', 'idx_policy')
                idx_policy_vocabularize = np.full((num_timesteps, 1), idx_policy_vocabularize, dtype=np.int64)
                idx_tag_vocabularize = self.vocabularize('special_token', 'idx_tag')
                idx_tag_vocabularize = np.full((num_timesteps, 1), idx_tag_vocabularize, dtype=np.int64)
                idx_a_self_vocabularize = self.vocabularize('special_token', 'idx_a_self')
                idx_a_self_vocabularize = np.full((num_timesteps, 1), idx_a_self_vocabularize, dtype=np.int64)
                idx_reward_vocabularize = self.vocabularize('special_token', 'idx_reward')
                idx_reward_vocabularize = np.full((num_timesteps, 1), idx_reward_vocabularize, dtype=np.int64)
                idx_end_timestep_vocabularize = self.vocabularize('special_token', 'idx_end_timestep')
                idx_reset_vocabularize = self.vocabularize('special_token', 'idx_reset_env')
                idx_end_timestep_vocabularize = np.where(resets.reshape(-1, 1), idx_reset_vocabularize, idx_end_timestep_vocabularize)
                tags_vocabularize = self.vocabularize('tag_value', tags[agent_id][time_slice])
                tags_vocabularize = tags_vocabularize[:, np.newaxis]
                agent_data_vocabularize = self.vocabularize('value', actions_behavior[agent_id][time_slice,:][np.newaxis, :]).squeeze()
                agent_data_vocabularize = agent_data_vocabularize[:, np.newaxis]
                rewards_vocabularize = self.vocabularize('value', rewards[time_slice][np.newaxis, :]).squeeze()
                rewards_vocabularize = rewards_vocabularize[:, np.newaxis]
                meta_pairs = np.concatenate([idx_policy_vocabularize, idx_tag_vocabularize, tags_vocabularize,
                                             idx_a_self_vocabularize, agent_data_vocabularize,
                                             idx_reward_vocabularize, rewards_vocabularize,
                                             idx_end_timestep_vocabularize], axis= 1) # (num_timesteps, 8)               

                # Merge all data
                agent_seq = np.concatenate([obs_pairs, agent_pairs, meta_pairs], axis=1) # (timesteps, num_relative_obs * 2 + num_relative_agents * 2 + 9)
                agent_seq = agent_seq.reshape(-1) # (timesteps * (num_relative_obs * 2 + num_relative_agents * 2 + 9))
                
                policy_position_mask = (agent_seq == self.vocabularize('special_token', 'idx_policy'))
                label_vocabularize = self.vocabularize('value', actions_label[agent_id][time_slice][np.newaxis, :],
                                                       behavior_value = actions_behavior[agent_id][time_slice,:][np.newaxis, :]).squeeze()
                if np.sum(policy_position_mask) != len(label_vocabularize):
                    raise ValueError(
                        f"Agent {agent_id} poilicy position count ({np.sum(policy_position_mask)}) "
                        f"not equal to label length ({len(label_vocabularize)})"
                    )
                pad_token = self.vocab_size - 1
                label_action_array = np.full(agent_seq.shape, pad_token, dtype=np.int64)
                label_action_array[policy_position_mask] = label_vocabularize
                
                sequence_list.append((agent_seq, label_action_array))

            return sequence_list
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 6

def process_subdir(args):
    """
    A parallel task function for handling a single subfolder
    """
    sub_dir, load_dir, save_dir, counter, lock, params = args
    sub_path = os.path.join(load_dir, sub_dir)
    
    time_step = params['time_step']
    max_obs_num = params['max_obs_num']
    max_agent_num = params['max_agent_num']
    tag_num = params['tag_num']
    value_num = params['value_num']
    resolution = params['resolution']
    vocab_size = params['vocab_size']
    
    try:
        # Each process creates its own instance of the dataset
        dataset = MultiAgentDataSetVetorized(
            directory=load_dir,
            time_step=time_step,
            max_obs_num=max_obs_num,
            max_agent_num=max_agent_num,
            tag_num=tag_num,
            value_num=value_num,
            resolution=resolution,
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
        
        for i, (sequence, label_action) in enumerate(results):
            record_name = f"record_{record_ids[i]:06d}"
            record_path = os.path.join(save_dir, record_name)
            os.makedirs(record_path, exist_ok=True)
            
            np.save(os.path.join(record_path, "sequence.npy"), sequence)
            np.save(os.path.join(record_path, "label_action.npy"), label_action)
            
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

def main(load_dir, save_dir, time_step, max_obs_num, max_agent_num, tag_num, value_num, resolution, vocab_size, num_workers=None):
    os.makedirs(save_dir, exist_ok=True)
    
    sub_dirs = [d for d in os.listdir(load_dir) 
                if os.path.isdir(os.path.join(load_dir, d))]
    
    print(f"Found {len(sub_dirs)} subdirectories to process")
    
    params = {
        'time_step': time_step,
        'max_obs_num': max_obs_num,
        'max_agent_num': max_agent_num,
        'tag_num': tag_num,
        'value_num': value_num,
        'resolution': resolution,
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
    parser.add_argument("--tag_num", type=int, required=True,
                        help="Maximum tags parameter")
    parser.add_argument("--value_num", type=int, required=True,
                        help="Value number parameter")
    parser.add_argument("--resolution", type=float, required=True,
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
        args.tag_num,
        args.value_num,
        args.resolution,
        args.vocab_size,
        num_workers=args.num_workers
    )

