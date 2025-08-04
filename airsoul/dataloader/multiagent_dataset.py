import os
import glob
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MultiAgentDataSet(Dataset):
    """
    All of the id and value are transform into discrete integer values, forming the vocabular.
    - i.e., we have m obs_id, n agent_id -> m + n words
        idx_policy, idx_tag, idx_a_self, idx_reward, idx_end_timestep, idx_reset_env -> 6 words
        t tag_value -> t words 
        value of obs, action, and reward ~ [-10, 10], resolution 0.1 -> 200 words
        Then the total vocabular size = m + n + t + 206
        
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
    def __init__(self, directory, time_step, max_obs, max_agent, max_tag, value_num, resolution, verbose=False):
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
        self.max_obs = max_obs
        self.max_agent = max_agent
        self.max_tag = max_tag
        self.resolution = resolution
        self.min_value = - resolution * value_num / 2
        self.max_value = - self.min_value

        self._init_vocab_offsets()

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        # TODO multiple sequence in one file
        return len(self.file_list)

    def _init_vocab_offsets(self):
        self.BASE_OFFSET = self.max_obs + self.max_agent
        self.SPECIAL_TOKENS = {
            'idx_policy': 0,
            'idx_tag': 1,
            'idx_a_self': 2,
            'idx_reward': 3,
            'idx_end_timestep': 4,
            'idx_reset_env': 5
        }
        self.TAG_BASE = self.BASE_OFFSET + 6
        self.VALUE_BASE = self.TAG_BASE + self.max_tag

    def vocabularize(self, type, value):
        handler = getattr(self, f'_handle_{type}', None)
        if handler:
            return handler(value)
        raise ValueError(f"Invalid type: {type}")

    def _handle_obs_id(self, value):
        return value

    def _handle_agent_id(self, value):
        return self.max_obs + value

    def _handle_special_token(self, token_type):
        return self.BASE_OFFSET + self.SPECIAL_TOKENS[token_type]

    def _handle_tag_value(self, value):
        return self.TAG_BASE + value

    def _handle_value(self, value):
        scalar_input = False
        if not isinstance(value, np.ndarray):
            value = np.array(value)
            scalar_input = True
            
        clipped = np.clip(value, self.min_value, self.max_value)
        quantized = np.round((clipped - self.min_value) / self.resolution).astype(np.int64)
        result = self.VALUE_BASE + quantized
        
        return result.item() if scalar_input else result

    def _load_and_process_data(self, path):
        try:
            observation_files = sorted(
                glob.glob(os.path.join(path, 'observations_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            actions_behavior_files = sorted(
                glob.glob(os.path.join(path, 'actions_behavior_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            actions_label_files = sorted(
                glob.glob(os.path.join(path, 'actions_label_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            tags_files = sorted(
                glob.glob(os.path.join(path, 'tags_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            observations = [np.load(f) for f in observation_files]
            actions_behavior = [np.load(f) for f in actions_behavior_files]
            actions_label = [np.load(f) for f in actions_label_files]
            tags = [np.load(f) for f in tags_files]
            rewards = np.load(path + '/rewards.npy')
            resets = np.load(path + '/resets.npy')

            obs_matrix = np.load(path + '/obs_graph.npy')
            agent_matrix = np.load(path + '/agent_graph.npy')
            num_obs = len(observation_files) 
            num_agent = len(actions_behavior_files)

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

            sequences = []
            label_action_seqs = []
            for agent_id in range(num_agent):
                relevant_obs = np.where(obs_matrix[:, agent_id] == 1)[0]
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
                    time_step_seq.append(self.vocabularize('special_token', 'idx_end_timestep'))
                    if resets[t]:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_reset_env'))
                    agent_sequence.append(time_step_seq)

                agent_sequence = np.concatenate(agent_sequence)
                sequences.append(agent_sequence)

                policy_position_mask = (agent_sequence == self.vocabularize('special_token', 'idx_policy'))
                label_vocabularize = self.vocabularize('value', actions_label[agent_id][n_b:n_e])
                if np.sum(policy_position_mask) != len(label_vocabularize):
                    raise ValueError(
                        f"Agent {agent_id} poilicy position count ({np.sum(policy_position_mask)}) "
                        f"not equal to label length ({len(label_vocabularize)})"
                    )
                pad_token = -1
                label_action_array = np.full(agent_sequence.shape, pad_token, dtype=np.int64)
                label_action_array[policy_position_mask] = label_vocabularize
                label_action_seqs.append(label_action_array)         

            return sequences, label_action_seqs
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 6
        
class MultiAgentDataSetVetorized(MultiAgentDataSet):
    def __init__(self, directory, time_step, max_obs, max_agent, max_tag, value_num, resolution, verbose=False):
        super().__init__(directory, time_step, max_obs, max_agent, max_tag, value_num, resolution, verbose)

    def _handle_obs_id(self, value):
        return value.astype(np.int64) if isinstance(value, np.ndarray) else int(value)

    def _handle_agent_id(self, value):
        return (self.max_obs + value).astype(np.int64) if isinstance(value, np.ndarray) else self.max_obs + value

    def _handle_tag_value(self, value):
        return (self.TAG_BASE + value).astype(np.int64) if isinstance(value, np.ndarray) else self.TAG_BASE + value

    def _handle_value(self, value):
        scalar_input = False
        if not isinstance(value, np.ndarray):
            value = np.array(value)
            scalar_input = True
            
        clipped = np.clip(value, self.min_value, self.max_value)
        quantized = np.round((clipped - self.min_value) / self.resolution).astype(np.int64)
        result = self.VALUE_BASE + quantized
        
        return result.item() if scalar_input else result

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

    def _load_and_process_data(self, path):
        try:
            observation_files = sorted(
                glob.glob(os.path.join(path, 'observations_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            actions_behavior_files = sorted(
                glob.glob(os.path.join(path, 'actions_behavior_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            actions_label_files = sorted(
                glob.glob(os.path.join(path, 'actions_label_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            tags_files = sorted(
                glob.glob(os.path.join(path, 'tags_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            observations = [np.load(f) for f in observation_files]
            actions_behavior = [np.load(f) for f in actions_behavior_files]
            actions_label = [np.load(f) for f in actions_label_files]
            tags = [np.load(f) for f in tags_files]
            rewards = np.load(path + '/rewards.npy')
            resets = np.load(path + '/resets.npy')
            obs_matrix = np.load(path + '/obs_graph.npy')
            agent_matrix = np.load(path + '/agent_graph.npy')

            max_t = min(actions_label[0].shape[0], 
                        rewards.shape[0], 
                        actions_behavior[0].shape[0],
                        observations[0].shape[0],
                        tags[0].shape[0])
            num_agents = len(actions_behavior_files)

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
            obs_conn_mask = [obs_matrix[:,i].astype(bool) for i in range(num_agents)]
            agent_conn_mask = [agent_matrix[i].astype(bool) for i in range(num_agents)]

            sequences = []
            label_action_seqs = []
            for agent_id in range(num_agents):
                # Obtain the association relationship of the current agent
                obs_mask = obs_conn_mask[agent_id]
                agent_mask = agent_conn_mask[agent_id]

                # Obs -> [obs_id, value] × timesteps
                obs_data = observations[obs_mask][:, time_slice] # (num_relative_obs, timesteps)
                relevant_obs = np.where(obs_mask)[0] # (num_relative_obs)
                obs_idx_vocabularize = self.vocabularize('obs_id', relevant_obs)
                obs_value_vocabularize = self.vocabularize('value', obs_data)
                obs_idx_vocabularize = np.broadcast_to(obs_idx_vocabularize[:, np.newaxis],
                                                   shape=obs_data.shape).astype(np.int64)  # (num_relative_obs, timesteps)
                obs_value_vocabularize = obs_value_vocabularize.T  # (timesteps, num_relative_obs)
                obs_idx_vocabularize = obs_idx_vocabularize.T  # (timesteps, num_relative_obs)
                # Merge into (timesteps, num_relative_obs * 2)
                obs_pairs = self._interleave_columns([obs_idx_vocabularize, obs_value_vocabularize])

                # Other agent -> [agent_id, value] × timesteps
                agent_data = actions_behavior[agent_mask][:, time_slice] # (num_relative_agents, timesteps)
                relevant_agents = np.where(agent_mask)[0] # (num_relative_agents)
                
                agent_idx_vocabularize = self.vocabularize('agent_id', relevant_agents)
                agent_value_vocabularize = self.vocabularize('value', agent_data)
                agent_idx_vocabularize = np.broadcast_to(agent_idx_vocabularize[:, np.newaxis],
                                                         shape=agent_data.shape).astype(np.int64)  # (num_relative_agents, timesteps)
                agent_value_vocabularize = agent_value_vocabularize.T  # (timesteps, num_relative_agents)
                agent_idx_vocabularize = agent_idx_vocabularize.T  # (timesteps, num_relative_agents)
                # Merge into (timesteps, num_relative_agents * 2)
                agent_pairs = self._interleave_columns([agent_idx_vocabularize, agent_value_vocabularize])
                
                # meta_data: idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep, idx_reset(if reset)
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
                idx_end_timestep_vocabularize = np.full((num_timesteps, 1), idx_end_timestep_vocabularize, dtype=np.int64)
                tags_vocabularize = self.vocabularize('tag_value', tags[agent_id][time_slice])
                tags_vocabularize = tags_vocabularize[:, np.newaxis]
                agent_data_vocabularize = self.vocabularize('value', actions_behavior[agent_id][time_slice])
                agent_data_vocabularize = agent_data_vocabularize[:, np.newaxis]
                rewards_vocabularize = self.vocabularize('value', rewards[time_slice])
                rewards_vocabularize = rewards_vocabularize[:, np.newaxis]
                meta_pairs = np.concatenate([idx_policy_vocabularize, idx_tag_vocabularize, tags_vocabularize,
                                             idx_a_self_vocabularize, agent_data_vocabularize,
                                             idx_reward_vocabularize, rewards_vocabularize,
                                             idx_end_timestep_vocabularize], axis= 1) # (num_timesteps, 8)
                idx_reset_vocabularize = self.vocabularize('special_token', 'idx_reset_env')
                pad_token = -1
                reset_col = np.where(resets.reshape(-1, 1), idx_reset_vocabularize, pad_token)
                meta_pairs = np.concatenate([meta_pairs, reset_col], axis=1) # (num_timesteps, 9)

                # Merge all data
                agent_seq = np.concatenate([obs_pairs, agent_pairs, meta_pairs], axis=1) # (timesteps, num_relative_obs * 2 + num_relative_agents * 2 + 9)
                agent_seq = agent_seq.reshape(-1) # (timesteps * (num_relative_obs * 2 + num_relative_agents * 2 + 9))
                filter_mask = agent_seq != pad_token
                agent_seq = agent_seq[filter_mask]
                sequences.append(agent_seq)
                
                policy_position_mask = (agent_seq == self.vocabularize('special_token', 'idx_policy'))
                label_vocabularize = self.vocabularize('value', actions_label[agent_id][time_slice])
                if np.sum(policy_position_mask) != len(label_vocabularize):
                    raise ValueError(
                        f"Agent {agent_id} poilicy position count ({np.sum(policy_position_mask)}) "
                        f"not equal to label length ({len(label_vocabularize)})"
                    )
                label_action_array = np.full(agent_seq.shape, pad_token, dtype=np.int64)
                label_action_array[policy_position_mask] = label_vocabularize
                
                label_action_seqs.append(label_action_array)

            return sequences, label_action_seqs
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 6
        
############ test ##############

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

class TestMultiAgentDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建临时测试目录
        cls.test_dir = Path(tempfile.mkdtemp())
        cls._generate_test_data(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    @classmethod
    def _generate_test_data(cls, base_dir):
        """生成测试用的虚拟数据"""
        time_steps = 10000
        
        # 观测数据 (3个观测源，5个时间步)
        obs_data = np.array([
            np.random.uniform(-10, 10, size=time_steps),  # 观测0
            np.random.uniform(-10, 10, size=time_steps),  # 观测1
            np.random.uniform(-10, 10, size=time_steps)   # 观测2
        ])
        for i in range(3):
            np.save(cls.test_dir/f'observations_{i}.npy', obs_data[i])

        # 行为动作 (2个agent，5个时间步)
        behavior_data = np.array([
            np.random.uniform(-10, 10, size=time_steps),  # agent0
            np.random.uniform(-10, 10, size=time_steps)   # agent1
        ])
        for i in range(2):
            np.save(cls.test_dir/f'actions_behavior_{i}.npy', behavior_data[i])

        # 标签动作 (同行为动作结构)
        label_data = behavior_data.copy()
        for i in range(2):
            np.save(cls.test_dir/f'actions_label_{i}.npy', label_data[i])

        # 标签数据 (2个agent，time_steps个时间步)
        tag_data = np.random.randint(0, 2, size=(2, time_steps))
        for i in range(2):
            np.save(cls.test_dir/f'tags_{i}.npy', tag_data[i])

        # 奖励和重置信号
        np.save(cls.test_dir/'rewards.npy',  np.random.uniform(-10, 10, size=time_steps))
        resets = np.zeros(time_steps, dtype=bool)
        # 随机设置重置点（约0.5%的概率）
        reset_indices = np.random.choice(
            time_steps, 
            size=int(time_steps*0.005),  # 50个重置点
            replace=False
        )
        resets[reset_indices] = 1
        np.save(cls.test_dir/'resets.npy', np.array(resets, dtype=bool))

        # 连接矩阵
        np.save(cls.test_dir/'obs_graph.npy', np.array([
            [1, 0],   # 观测0的连接情况
            [1, 1],   # 观测1
            [0, 1]    # 观测2
        ]))
        np.save(cls.test_dir/'agent_graph.npy', np.array([
            [0, 1],   # agent0的连接
            [1, 0]    # agent1
        ]))

    def _validate_equivalence(self, base_seq, vec_seq):
        """验证两个序列的等效性（允许1e-5的浮点误差）"""
        self.assertEqual(len(base_seq), len(vec_seq))

        for b_agent, v_agent in zip(base_seq, vec_seq):
            # 转换为基础类的数据类型
            b_clean = b_agent.astype(np.int64)
            v_agent = v_agent.astype(np.int64)

            diff_mask = b_clean != v_agent
            if np.any(diff_mask):
                print(f"\nAgent 序列差异：")
                print("索引 | 基准值 | 测试值")
                for idx in np.where(diff_mask)[0]:
                    print(f"{idx:4d} | {b_clean[idx]:6d} | {v_agent[idx]:6d}")
            
            np.testing.assert_allclose(b_clean, v_agent, atol=1e-5, 
                                    err_msg="序列数据不匹配")

    def test_basic_equivalence(self):
        """基本等效性测试"""
        # 初始化数据集
        base_ds = MultiAgentDataSet(
            directory=str(self.test_dir),
            time_step=10000,
            max_obs=3,
            max_agent=2,
            max_tag=2,
            value_num=200,
            resolution=0.1
        )

        vec_ds = MultiAgentDataSetVetorized(
            directory=str(self.test_dir),
            time_step=10000,
            max_obs=3,
            max_agent=2,
            max_tag=2,
            value_num=200,
            resolution=0.1
        )

        # 加载数据
        base_seq, base_labels = base_ds._load_and_process_data(str(self.test_dir))
        vec_seq, vec_labels = vec_ds._load_and_process_data(str(self.test_dir))

        # 验证序列
        self._validate_equivalence(base_seq, vec_seq)

        # 验证标签
        for b_label, v_label in zip(base_labels, vec_labels):
            np.testing.assert_array_equal(b_label, v_label)


    def test_performance_comparison(self):
        """性能对比测试"""
        import timeit

        base_time = timeit.timeit(
            lambda: MultiAgentDataSet(
                directory=str(self.test_dir),
                time_step=10000,
                max_obs=3,
                max_agent=2,
                max_tag=2,
                value_num=200,
                resolution=0.1
            )._load_and_process_data(str(self.test_dir)),
            number=10
        )

        vec_time = timeit.timeit(
            lambda: MultiAgentDataSetVetorized(
                directory=str(self.test_dir),
                time_step=10000,
                max_obs=3,
                max_agent=2,
                max_tag=2,
                value_num=200,
                resolution=0.1
            )._load_and_process_data(str(self.test_dir)),
            number=10
        )

        print(f"\n性能对比：\n- 原始版本：{base_time:.2f}s\n- 向量化版本：{vec_time:.2f}s")
        self.assertLess(vec_time, base_time, "向量化版本应更快")


if __name__ == '__main__':
    unittest.main(verbosity=2)
