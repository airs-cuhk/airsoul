import os
import torch
import numpy
import pickle
from copy import deepcopy

from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import DistStatistics, downsample
from airsoul.utils import EpochManager, GeneratorBase, Logger
from airsoul.dataloader import MultiAgentLoadDateSet, MultiAgentDataSetVetorized
from data.anyhvac import create_cooler_cooler_graph, create_cooler_sensor_graph
from env_wapper import HVACEnvWrapper, plot_cooler_values


class HVACAPI(GeneratorBase):
    def preprocess(self):

        logger_keys = ["step", "reward", "state_prediction", "action_prediction", "reward_prediction"]
        self.stat = DistStatistics(*logger_keys)
        self.logger = Logger("trail_idx",
                            "total_steps",
                            *logger_keys, 
                            on=self.main, 
                            use_tensorboard=False)
        
        self.dataset = MultiAgentDataSetVetorized(
            directory="./",
            time_step=5000,
            max_obs_num=self.config.vocab.max_obs_num,
            max_agent_num=self.config.vocab.max_agent_num,
            prompt_num=self.config.vocab.prompt_num,
            temperature_value_num=self.config.vocab.temperature_value_num,
            temperature_resolution=self.config.vocab.temperature_resolution,
            policy_value_num=self.config.vocab.policy_value_num,
            policy_resolution=self.config.vocab.policy_resolution,
            vocab_size=self.config.vocab.vocab_size,
            verbose=False
        )
        self.vocabularize = self.dataset.vocabularize
        self.interactive_prompt = self.config.default_prompt
    
    def epoch_end(self, epoch_id):
        pass

    def init(self, target_temperature, related_sensor, related_agent):
        """
        初始化函数
        
        Args:
            target_temperature (float): 环境目标温度值
            related_sensor (numpy array): 每个 agent 的关联传感器，形状为(num_agent, knn)
                - 每行表示一个 agent 关联的传感器索引，并按照关联程度进行排序（关联性大的在前）
                - 索引顺序必须和下方get_action函数的 obs 和 agent_action 顺序对应
                - knn 通常为 3～5
            related_agent (numpy array): 每个 agent 的关联空调器，形状为(num_agent, knn)
                - 每行表示一个 agent 关联的其他 agent 的索引，并按照关联程度进行排序（关联性大的在前）
                - 索引顺序必须和下方get_action函数的 agent_action 顺序对应
                - knn 通常为 3～5
        """

        self.total_step = 0
        self.target_temperature = target_temperature
        self.related_sensor = related_sensor
        self.related_agent = related_agent
        self.agent_num = related_agent.shape[0]
    
    def in_context_learn_from_teacher(self, epoch_id):
        pass # TODO

    def _build_up_vocab_seq_in_batch(self, obs_sensor, obs_agent, current_batch_seq=None, 
                                    action=None, reward=None, reset=False, use_relative_idx=True):
        # [num_agent, time, value]
        # obs_agent and action should contain [:,t-1:t,:] two timestep if t>0.
        if current_batch_seq is None:
            current_batch_seq = []
            # [num, value] -> [num, 1, value]
            obs_sensor = obs_sensor[:,numpy.newaxis,numpy.newaxis]
            obs_sensor_vocabularize = self.vocabularize('value', obs_sensor).squeeze()
            obs_agent_vocabularize = self.vocabularize('value', obs_agent, use_diff_action=False)[:,-1:,:].squeeze()
            for agent_id in range(self.agent_num):
                current_agent_seq = []
                # 1, Related agent idx and value
                if use_relative_idx:
                    current_agent_seq.append(self.vocabularize('agent_id', 0))
                else:
                    current_agent_seq.append(self.vocabularize('agent_id', agent_id))

                current_agent_seq.append(obs_agent_vocabularize[agent_id])

                for i, related_agent in enumerate(self.related_agent[agent_id]):
                    if use_relative_idx:
                        current_agent_seq.append(self.vocabularize('agent_id', i+1))
                    else:
                        current_agent_seq.append(self.vocabularize('agent_id', related_agent))
                    current_agent_seq.append(obs_agent_vocabularize[related_agent])
                    # current_agent_seq.append(self.vocabularize('value', obs_agent[related_agent]))
                # 2, Related sensor idx and value
                for i, related_obs in enumerate(self.related_sensor[agent_id]):
                    if use_relative_idx:
                        current_agent_seq.append(self.vocabularize('obs_id', i))
                    else:
                        current_agent_seq.append(self.vocabularize('obs_id', related_obs))
                    current_agent_seq.append(obs_sensor_vocabularize[related_obs])
                    # current_agent_seq.append(self.vocabularize('value', obs_sensor[related_obs]))
                # 3, Prompt
                current_agent_seq.append(self.vocabularize('special_token', 'idx_prompt'))
                current_agent_seq.append(self.vocabularize('prompt_value', self.interactive_prompt))
                # 4, Self action flag
                current_agent_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                current_batch_seq.append(current_agent_seq)
            current_batch_seq = [[int(x) for x in lst] for lst in current_batch_seq]
            return current_batch_seq
        else:
            agent_action_vocabularize = self.vocabularize('policy_action', action).squeeze()
                
            if self.model.include_reward:
                reward_idx_vocabularize = self.vocabularize('special_token', 'idx_reward')
                reward_vocabularize = self.vocabularize('value', reward)
            timestep_end_vocabularize = self.vocabularize('special_token', 
                                             'idx_reset_env' if reset else 'idx_end_timestep')
            if self.model.include_reward:
                to_add = numpy.array([
                    agent_action_vocabularize,                              # 5, Self action value
                    numpy.full(self.agent_num, reward_idx_vocabularize),   # 6, Reward idx and value
                    numpy.full(self.agent_num, reward_vocabularize),  
                    numpy.full(self.agent_num, timestep_end_vocabularize)  # 7, End
                ], dtype=object).T  # (num_agents, 4)
            else:
                to_add = numpy.array([
                    agent_action_vocabularize,                             # 5, Self action value
                    numpy.full(self.agent_num, timestep_end_vocabularize)  # 6, End
                ], dtype=object).T  # (num_agents, 2)
            for i, seq in enumerate(current_batch_seq):
                seq.extend(to_add[i])
            current_batch_seq = [[int(x) for x in lst] for lst in current_batch_seq]
            return current_batch_seq
    
    def _build_up_env_action(self, action_in_vocab, action_temp_previous):
        # Shape of action_in_vocab: [num, 1, 1]
        # Shape of action_temp_previous: [num, ], action: (temp)

        action_in_value = self.vocabularize('action_vocab', 
                                            action_in_vocab) # [num, 2]
        # Convert actual temperature settings to normalized values 
        action_temp_diff = action_in_value.squeeze()[:,1]
        action_temp_diff = numpy.clip(action_temp_diff, -3.0, 3.0)
        current_action_temp = action_temp_previous + action_temp_diff
        
        return current_action_temp

    def _compute_temperature_deviation(self, obs):
        sensor_readings =  obs["sensor_readings"]
        return sensor_readings - self.target_temperature

    def _convert_env_action_deviation(self, env_action):
        temp_value = env_action["value"]
        diff_temp_value = temp_value - self.target_temperature
        switch = env_action["switch"]
        data = numpy.column_stack((diff_temp_value, switch))
        result = data.reshape((len(diff_temp_value), 1, 2))
        return result
    
    def _convert_diff_action(self, previous_action, current_action):
        prev_temp = previous_action[:, 0, 0]  # shape: (n_agents,)
        prev_switch = previous_action[:, 0, 1]  # shape: (n_agents,)
        curr_temp = current_action[:, 0, 0]  # shape: (n_agents,)
        curr_switch = current_action[:, 0, 1]  # shape: (n_agents,)
        diff_temp = curr_temp - prev_temp
        diff_switch = curr_switch
        data = numpy.column_stack((diff_temp, diff_switch))
        result = data.reshape((len(diff_temp), 1, 2))
        return result
    
    def get_action(self, obs, previous_action):
        """
        根据当前观测和上一步动作，获取当前步骤的控制动作。
        
        Args:
            obs (numpy.ndarray): 传感器观测值数组，形状为 (num_sensor,)
                num_sensor 为传感器数量，每个元素表示对应传感器的温度读数
            previous_action (dict): 上一步的动作，包含以下字段:
                - "value" (numpy.ndarray): 温度值数组，形状为 (num_agent,)
                每个元素表示对应空调器上一步设定的温度值
                - "switch" (numpy.ndarray): 开关状态数组，形状为 (num_agent,)
                每个元素为布尔值, 表示对应空调器是否开启 (True: 开启, False: 关闭）
        
        Returns:
            env_action (numpy.ndarray): 当前步骤的控制温度值数组，形状为 (num_agent,)
                num_agent 为空调器数量，每个元素表示对应空调器当前步骤应设定的温度值, 可忽略关机的 agent 动作
            vocab_seq_batch: 包含当前步骤的控制温度值数组和上一步动作的词表序列
        
        Note:
            OmniRL 模型不决定空调的开关状态，开关需要外部指定。
            OmniRL 每一步的动作范围是 [-3, 3] 度，设定的最低温度为目标温度 - 3 度，最高温度为 32°C
            (请确保温度开高不额外耗能， 无制热功能)
        """

        previous_state_diff = self._compute_temperature_deviation(obs)
        previous_action_diff = self._convert_env_action_deviation(previous_action)
        temp = self._scheduler(self.total_step)
        vocab_seq_batch = self._build_up_vocab_seq_in_batch(previous_state_diff,
                                                            previous_action_diff)

        world_model_obs, world_model_action, action = self.model.generate(
            vocab_seq_batch,
            single_batch=False,
            reward_prediction=False,
            Temp=temp)
        
        env_action = self._build_up_env_action(action, previous_action["value"])   
        return env_action, vocab_seq_batch

    def icl(self, vocab_seq_batch, current_action, previous_action, done, reward=None):

        current_action_diff = self._convert_env_action_deviation(current_action)
        previous_action_diff = self._convert_env_action_deviation(previous_action)
        diff_action = self._convert_diff_action(previous_action_diff, current_action_diff)
        
        # icl call
        vocab_seq_batch = self._build_up_vocab_seq_in_batch(None,
                                                            None,
                                                            current_batch_seq=vocab_seq_batch,
                                                            action=diff_action,
                                                            reward=None,
                                                            reset=done
                                                            )
        cache = self.model.incontext_learn(vocab_seq_batch, need_cache=False)
        self.total_step += 1

        # develop
        # print("diff_action: ", diff_action)
        print("reward: ", reward)
        print("done: ", done)