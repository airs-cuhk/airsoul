import os
import argparse
import numpy as np
import pickle
import random
import multiprocessing
import time
from copy import deepcopy
from pathlib import Path
import shutil
import torch
import re
from typing import List, Tuple
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiffAction
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
from xenoverse.utils import RandomFourier
from rl_trainer_hvac import HVACRLTester


def create_cooler_sensor_graph(env, num_closest_sensors: int = 3):
    """
    Create sensor-cooler relationship graph.

    Args:
        num_closest_sensors (int): The number of closest sensors to consider for each cooler.
                                    Defaults to 3.

    Returns:
        np.ndarray: The sensor-cooler relationship graph.
    """
    # Create a new matrix with the same shape as the original (cooler, sensor) matrix
    obs_graph = np.zeros_like(env.cooler_sensor_topology)
    n_coolers, n_sensors = env.cooler_sensor_topology.shape
    
    for cooler_id in range(n_coolers):
        sensor_weights = env.cooler_sensor_topology[cooler_id, :]
        if n_sensors >= num_closest_sensors:
            # Get the indices of the 'num_closest_sensors' smallest values (closest sensors)
            closest_sensor_idx = np.argpartition(sensor_weights, num_closest_sensors - 1)[:num_closest_sensors]
            sorted_indices = closest_sensor_idx[np.argsort(sensor_weights[closest_sensor_idx])]
        else:
            # If there are fewer than 'num_closest_sensors' sensors, use all sensors
            sorted_indices = np.argsort(sensor_weights)[:n_sensors]
        
        # Set the positions corresponding to the closest sensors to 1
        for rank, sensor_idx in enumerate(sorted_indices, start=1):
            obs_graph[cooler_id, sensor_idx] = rank
    
    return obs_graph

def create_cooler_cooler_graph(env, num_closest_coolers: int = 3):
    """
    Create cooler-cooler relationship graph using KNN.

    Args:
        k_nearest_coolers (int): The number of nearest coolers to consider for each cooler.
                                    Defaults to 3.

    Returns:
        np.ndarray: The cooler-cooler relationship graph.
    """
    agent_graph = np.zeros_like(env.cooler_topology)
    n_coolers, n_coolers = env.cooler_topology.shape
    
    for cooler_id in range(n_coolers):
        cooler_weights = env.cooler_topology[cooler_id, :]
        cooler_weights[cooler_id] = np.inf
        available_coolers = n_coolers - 1
        actual_num_closest = min(num_closest_coolers, available_coolers)
        if actual_num_closest > 0:
            closest_cooler_idx = np.argpartition(cooler_weights, actual_num_closest)[:actual_num_closest]
            sorted_indices = closest_cooler_idx[np.argsort(cooler_weights[closest_cooler_idx])]
            for rank, cooler_idx in enumerate(sorted_indices, start=1):
                agent_graph[cooler_id, cooler_idx] = rank
    
    return agent_graph

class FourierNoiseGenerator:
    """
    Random Fourier noise generator

    Function
    1. Generate random Fourier noise
    2. Automatically calculate the noise range
    3. Provide normalized noise values

    Parameter
    ndim: Output dimension
    max_order: Control the period range
    max_item: Controls the number of superimposed items
    max_steps: Time series length
    box_size: Controls the amplitude
    """
    
    def __init__(self, ndim=1, max_order=32, max_item=3, max_steps=250, box_size=0.15):
        """
        Initialize the noise generator
        """
        self.ndim = ndim
        self.max_order = max_order
        self.max_item = max_item
        self.max_steps = max_steps
        self.box_size = box_size
        
        # Create a random Fourier function
        self.rf = RandomFourier(
            ndim=ndim,
            max_order=max_order,
            max_item=max_item,
            max_steps=max_steps,
            box_size=box_size
        )
        
        # Calculate the noise range
        self.y_min, self.y_max = self._calculate_range()
    
    def _calculate_range(self, max_attempts=10):
        """
        Calculate the noise range and ensure it is valid
        """
        for attempt in range(max_attempts):
            t = np.arange(0, self.max_steps)
            y = self.rf(t)
            y_min, y_max = y.min(), y.max()
            
            if y_min != y_max:
                return y_min, y_max
            
            self.rf = RandomFourier(
                ndim=self.ndim,
                max_order=self.max_order,
                max_item=self.max_item,
                max_steps=self.max_steps,
                box_size=self.box_size
            )
        
        raise ValueError(f"The effective noise range cannot be generated. Number of attempts: {max_attempts}")
    
    def get_noise(self, t):
        """
        Obtain the normalized noise value

        Parameter
        t: Time point or time series

        Return
        Normalized noise value (0-1 range)
        """
        raw_noise = self.rf(t)
        normalized = (raw_noise - self.y_min) / (self.y_max - self.y_min)
        return normalized

    def generate_sequence(self):
        """
        Generate the noise values of the complete time series

        Return
        A normalized noise sequence of length max_steps
        """
        t = np.arange(0, self.max_steps)
        return self.get_noise(t)

class HVACActionNoiseFourier(object):
    def __init__(self, agent_num):

        self.agent_num = agent_num

        self.no_noise = np.random.uniform(low=0.0, high=1.0) > 0.7

        fourier_noise_num_factor = np.random.uniform(low=0.4, high=0.6)
        inverse_noise_num_factor = np.random.uniform(low=0.0, high=0.2)
        self.add_fourier_noise_agent_num = int(round(fourier_noise_num_factor * agent_num))
        add_inverse_noise_num = int(max(1, round(inverse_noise_num_factor * self.add_fourier_noise_agent_num)))
        self.noise_value_factor = np.random.uniform(low=0.25, high=0.75, size=self.add_fourier_noise_agent_num)

        self.mask = np.zeros(agent_num, dtype=bool)
        self.inverse_mask = np.zeros(agent_num, dtype=bool)
        indices = np.random.choice(agent_num, self.add_fourier_noise_agent_num, replace=False)
        inverse_indices = np.random.choice(agent_num, add_inverse_noise_num, replace=False)
        self.mask[indices] = True
        self.inverse_mask[inverse_indices] = True

        ndim = 1
        max_order = 32
        max_item = 3
        max_steps = 4000
        box_size = 0.5
        
        self.generators = [
            FourierNoiseGenerator(
                ndim=ndim,
                max_order=max_order,
                max_item=max_item,
                max_steps=max_steps,
                box_size=box_size
            ) for _ in range(self.add_fourier_noise_agent_num)
        ]
    def add_noise(self, current_step, action):
        noisy_action = action.copy()

        if self.no_noise:
            return noisy_action
        
        noise = []
        for i in range(self.add_fourier_noise_agent_num):
            noise.append(self.generators[i].get_noise(current_step))
        noise = np.array(noise).squeeze()
        noisy_action[self.mask] = self.noise_value_factor * noise + (1 - self.noise_value_factor) * action[self.mask]
        noisy_action[self.inverse_mask] =  1 - noisy_action[self.inverse_mask]
        return noisy_action

class HVACActionNoiseDecay(object):
    def __init__(self, T_ini, T_fin, T_decay_type, T_total_step, mask_change_step=100):
        
        self.T_ini = min(1, T_ini)
        self.T_fin = max(0, T_fin)
        self.T_decay_type = T_decay_type
        self.T_total_step = T_total_step
        self.mask_change_step = mask_change_step
        self.mask = []

        self.dT_linear = (self.T_fin - self.T_ini) / self.T_total_step
        self.dT_exp = np.exp((np.log(max(0.0001, self.T_fin)) - np.log(self.T_ini)) / self.T_total_step)

    
    def add_noise(self, current_step, action):
        if self.T_decay_type == "linear":
            Temp = self.T_ini + min(current_step, self.T_total_step) * self.dT_linear
        elif self.T_decay_type == "exponential":
            Temp = self.T_ini * (self.dT_exp ** min(current_step, self.T_total_step))
        n = len(action)
        k = int(round(Temp * n))
        if k == 0:
            return action
        if len(self.mask) == 0 or current_step % self.mask_change_step == 0:
            self.mask = np.zeros(n, dtype=bool)
            indices = np.random.choice(n, k, replace=False)
            self.mask[indices] = True
        noise = np.random.random(action.shape)
        noisy_action = action.copy()
        noisy_action[self.mask] = Temp * noise[self.mask] + (1 - Temp) * action[self.mask]
        return noisy_action

class HVACDataGenerator:
    def __init__(self,
                 epoch_id=0,
                 task_config_path=None,
                 rl_mode_type='sac',
                 rl_model_path=None,
                 output_path=None,
                 reward_mode=0,
                 max_steps=20000,
                 behavior_change_step=100,
                 T_ini=0.5,
                 T_fin=0.0,
                 T_total_step=10000,
                 mask_change_step=100,
                 random_start=True,
                 use_fourier_noise=True,
                 use_rrd=False,
                 reference_action_deterministic=True,
                 verbose=False
                 ):
        torch.set_num_threads(1)
        pid = os.getpid()
        seed = int(time.time() * 1000) % (2**32 - 1) + pid
        random.seed(seed)
        np.random.seed(seed)

        try:
            with open(task_config_path, "rb") as f:
                task = pickle.load(f)
                self.env = HVACEnvDiffAction(reward_mode=reward_mode)
                self.env.set_task(task,discretize_rl_action_space=True,add_action_cost=False,too_cold_limit=False)
                self.n_sensors = len(self.env.sensors)
                self.n_coolers = len(self.env.coolers)
                self.n_heaters = len(self.env.equipments)
                if random_start:
                    self.env.set_random_start_t(True)
                    self.env.set_generate_record(True)
                self.task_file_path = task_config_path
        except:
            print(f"Load task config from {task_config_path} failed!")

        self.rl_models = []
        print("rl_model_path: ", rl_model_path)
        for i, path in enumerate(rl_model_path):
            try:
                self.rl_models.append(HVACRLTester(path, rl_mode_type, "cpu"))
                print(f"Load rl model from {path} success!")
            except:
                print(f"Load rl model from {path} failed!")
        if use_rrd:
            self.rl_model_oracle_deterministic = HVACRLTester(rl_model_path[0], rl_mode_type, "cpu")
        else:
            self.rl_model_oracle_deterministic = HVACRLTester(rl_model_path[reward_mode], rl_mode_type, "cpu")
        self.reference_action_deterministic = reference_action_deterministic
        
        self.epoch_id = epoch_id
        self.output_path = output_path
        self.reward_mode = reward_mode
        self.other_reward_mode = list(filter(lambda mode: mode != self.reward_mode, [0, 1, 2]))
        self.max_steps = max_steps
        self.use_rrd = use_rrd
        self.verbose = verbose
        
        if use_fourier_noise:
            self.add_noise = HVACActionNoiseFourier(self.n_coolers)
        else:
            T_total_step = int(min(T_total_step, max_steps/2) * random.uniform(0.5, 2))
            T_decay_type = random.choice(["linear", "exponential"])
            T_ini = min(random.uniform(0.5, 1.5) * T_ini, 1.0)
            mask_change_step = int(random.uniform(0.75, 1.25) * mask_change_step)
            self.add_noise = HVACActionNoiseDecay(T_ini, T_fin, T_decay_type, T_total_step, mask_change_step)

        if self.use_rrd:
            self.behavior_dict = [("opt", 0.20), 
                        ("pid", 0.20),  
                        ("rrd",   0.60)]
        else:    
            self.behavior_dict = [("opt", 0.20), 
                        ("opt_w_noise", 0.30),  
                        ("pid_w_noise", 0.20),
                        ("rl1_w_noise", 0.15),
                        ("rl2_w_noise", 0.15)]
        if self.use_rrd:
            self.current_behavior_name, self.rrd_idx = self.sample_behavior_rrd()
        else:
            self.current_behavior_name = self.sample_behavior()
        behavior_change_step = int(behavior_change_step * random.uniform(0.75, 1.25))
        self.behavior_change_point_list = self.generate_change_points(max_steps, behavior_change_step)
        self.overheat_no_reset = random.uniform(0.0, 1.0) > 0.5

    def _reset(self):
        obs, info = self.env.reset()
        self.pid_solver = HVACSolverGTPID(self.env)
        self.pid_solver.acc_diff = np.zeros_like(self.pid_solver.acc_diff)
        for i, rl_model in enumerate(self.rl_models):
            rl_model.reset()
        self.rl_model_oracle_deterministic.reset()
        return obs, info
    
    def _compute_temperature_deviation(self, sensor_readings):
        target_temp = self.env.target_temperature
        if isinstance(target_temp, (list, np.ndarray)):
            temperature_deviation = sensor_readings - np.array(target_temp)
        else:
            temperature_deviation = sensor_readings - target_temp
        
        return temperature_deviation

    def _compute_action_temperature_differences_with_graph(self, behavior_temp_settings, reference_temp_settings, 
                                                          cooler_sensor_graph):
        """
        Use obs_graph to calculate the difference between the temperature setting value of each cooler and the target temperatures of its nearest k sensors
        
        Args:
            behavior_temp_settings: The temperature setting value of the behavior policy (n_coolers,)
            reference_temp_settings: The temperature setting value of the reference policy (n_coolers,)
            obs_graph: sensor-cooler graph (n_sensors, n_coolers), transport to (n_coolers, n_sensors)
        
        Returns:
            diff_behavior: (n_coolers,)
            diff_reference: (n_coolers,)
        """
        n_coolers = self.n_coolers
        diff_behavior = np.zeros(n_coolers, dtype=np.float32)
        diff_reference = np.zeros(n_coolers, dtype=np.float32)
        
        target_temp = self.env.target_temperature
        if isinstance(target_temp, (list, np.ndarray)):
            target_temps = np.array(target_temp)
            for cooler_idx in range(n_coolers):
                connected_sensors = np.where(cooler_sensor_graph[cooler_idx] > 0)[0]
                
                if len(connected_sensors) > 0:
                    avg_target_temp = np.mean(target_temps[connected_sensors])
                else:
                    avg_target_temp = np.mean(target_temps)
                
                diff_behavior[cooler_idx] = behavior_temp_settings[cooler_idx] - avg_target_temp
                diff_reference[cooler_idx] = reference_temp_settings[cooler_idx] - avg_target_temp
        else:
            diff_behavior = behavior_temp_settings - target_temp
            diff_reference = reference_temp_settings - target_temp
      
        return diff_behavior, diff_reference

    def _get_action_list(self, obs):
        action_list = []
        for i, rl_model in enumerate(self.rl_models):
            action = rl_model.predict(obs, deterministic=False)
            action_list.append(action)
        action_list.append(1 - self.pid_solver.policy(obs["sensor_readings"])[self.n_coolers:])

        # Get Reference action and distribution
        action_reference, dist_info, _ = self.rl_model_oracle_deterministic.predict_with_distribution(obs, deterministic=self.reference_action_deterministic)
        action_reference_distribution = np.squeeze(dist_info['probs'])
        action_list.append(action_reference)
        return action_list, action_reference_distribution

    def sample_behavior(self):
        behaviors, probabilities = zip(*self.behavior_dict)
        prob_array = np.array(probabilities)
        normalized_probs = prob_array / prob_array.sum()
        sampled_index = np.random.choice(len(behaviors), p=normalized_probs)
        return behaviors[sampled_index]
    
    def sample_behavior_rrd(self):
        behaviors, probabilities = zip(*self.behavior_dict)
        prob_array = np.array(probabilities)
        normalized_probs = prob_array / prob_array.sum()
        sampled_index = np.random.choice(len(behaviors), p=normalized_probs)

        n_model = len(self.rl_models)
        rrd_idx = np.random.randint(1,n_model)

        return behaviors[sampled_index], rrd_idx
    
    def generate_change_points(self, max_steps, behavior_change_step):
        n_changes = (max_steps // behavior_change_step) + 1
        change_points = []
        current = 0

        for _ in range(n_changes):
            interval = random.randint(
                int(behavior_change_step * 0.8),
                int(behavior_change_step * 1.2)
            )
            current += interval
            if current > self.max_steps:
                change_points.append(current)
                break
            change_points.append(current)

        return change_points
    
    def _get_action(self, action_list, behavior_name, step):
        if behavior_name == "reference":
            return action_list[-1]
        elif behavior_name == "opt":
            return action_list[self.reward_mode]
        elif behavior_name == "opt_w_noise":
            return self.add_noise.add_noise(step, action_list[self.reward_mode])
        elif behavior_name == "pid_w_noise":
            return self.add_noise.add_noise(step, action_list[-2])
        elif behavior_name == "rl1_w_noise":
            return self.add_noise.add_noise(step, action_list[self.other_reward_mode[0]])
        elif behavior_name == "rl2_w_noise":
            return self.add_noise.add_noise(step, action_list[self.other_reward_mode[1]])
        else:
            raise ValueError(f"Invalid behavior name: {behavior_name}")
        
    def _get_action_rrd(self, action_list, behavior_name, rrd_idx):
        if behavior_name == "reference":
            return action_list[-1]
        elif behavior_name == "pid":
            return action_list[-2]
        elif behavior_name == "opt":
            return action_list[0]
        elif behavior_name == "rrd":
            return action_list[rrd_idx]
        else:
            raise ValueError(f"Invalid behavior name: {behavior_name}")

    def generate_data(self):
        torch.set_num_threads(1)
        observations = []
        diff_observations = [] 
        actions_behavior = []
        actions_label = []
        diff_actions_behavior = []
        diff_actions_label = []
        actions_behavior_policy = []
        actions_label_policy = []
        label_action_distribution = []
        prompts = []
        rewards = []
        resets = []

        # Initialize environment
        obs, info = self._reset()
        

        # Statistics
        steps = 0
        num_closest_sensors = random.randint(3,5)
        num_closest_coolers = random.randint(3,5)

        self.agent_sensor_graph = create_cooler_sensor_graph(self.env, num_closest_sensors)
        self.agent_agent_graph = create_cooler_cooler_graph(self.env, num_closest_sensors)

        default_action = self.env.last_action
        default_action_switch = default_action["switch"] 
        default_action_temp = self.env._action_value_to_temp(default_action["value"])
        default_action_array = np.array([default_action_temp, default_action_switch], dtype=np.float32)
        actions_behavior.append(default_action_array)
        actions_label.append(default_action_array)
        
        default_action_diff_behavior, default_action_diff_reference = self._compute_action_temperature_differences_with_graph(
                default_action_temp, 
                default_action_temp,
                self.agent_sensor_graph
            )
        default_action_diff = np.array([default_action_diff_behavior, default_action_switch], dtype=np.float32)
        diff_actions_behavior.append(default_action_diff)
        diff_actions_label.append(default_action_diff)

        default_policy_action_value = np.zeros_like(default_action_temp)
        default_policy_action = np.array([default_policy_action_value, default_action_switch], dtype=np.float32)
        actions_behavior_policy.append(default_policy_action)
        actions_label_policy.append(default_policy_action)
        
        while steps < self.max_steps:
            sensor_readings = obs["sensor_readings"]
            temperature_deviations = self._compute_temperature_deviation(sensor_readings)
            temperature_ori = sensor_readings

            action_list, action_reference_distribution = self._get_action_list(obs)

            if len(self.behavior_change_point_list) > 0 and steps > self.behavior_change_point_list[0]:
                self.behavior_change_point_list.pop(0)
                if self.use_rrd:
                    self.current_behavior_name, self.rrd_idx = self.sample_behavior_rrd()
                else:
                    self.current_behavior_name = self.sample_behavior()

            if self.use_rrd:
                behavior_action = self._get_action_rrd(action_list, self.current_behavior_name, self.rrd_idx)
                reference_action = self._get_action_rrd(action_list, "reference", self.rrd_idx)
            else:
                behavior_action = self._get_action(action_list, self.current_behavior_name, steps)
                reference_action = self._get_action(action_list, "reference", steps)
            
            behavior_action, behavior_policy_action = self.env._diff_action(np.array(behavior_action).flatten())
            reference_action, reference_policy_action = self.env._diff_action(np.array(reference_action).flatten())

            obs, reward_behavior_action, terminated, truncated, info = self.env.step(behavior_action, call_diff_action=False)
            steps += 1

            env_action = deepcopy(self.env.last_action)
            behavior_switch = env_action["switch"]
            behavior_values = env_action["value"]
            behavior_temp_settings = self.env._action_value_to_temp(behavior_values)
            reference_switch = env_action["switch"]
            reference_values = reference_action
            reference_temp_settings = self.env._action_value_to_temp(reference_values)
            diff_temp_behavior, diff_temp_reference = self._compute_action_temperature_differences_with_graph(
                behavior_temp_settings, 
                reference_temp_settings,
                self.agent_sensor_graph
            )

            if self.verbose:
                print(
                    f"current_behavior_name: {self.current_behavior_name}\n"
                    f"temperature_ori: {temperature_ori}\n"
                    f"temperature_deviations: {temperature_deviations}\n"
                    f"prompt: {self.reward_mode}\n"
                    f"behavior_temp_settings: {behavior_temp_settings}\n"
                    f"diff_temp_behavior: {diff_temp_behavior}\n"
                    f"reference_temp_settings: {reference_temp_settings}\n"
                    f"diff_temp_reference: {diff_temp_reference}\n"
                    f"reward: {reward_behavior_action}\n"
                    f"reset: {1 if truncated or terminated else 0}"
                )
            
            behavior_action_array = np.array([behavior_temp_settings, behavior_switch], dtype=np.float32)
            reference_action_array = np.array([reference_temp_settings, reference_switch], dtype=np.float32)
            diff_behavior_action_array = np.array([diff_temp_behavior, behavior_switch], dtype=np.float32)
            diff_reference_action_array = np.array([diff_temp_reference, reference_switch], dtype=np.float32)
            behavior_policy_action_array = np.array([behavior_policy_action, behavior_switch], dtype=np.float32)
            reference_policy_action_array = np.array([reference_policy_action, reference_switch], dtype=np.float32)
            observations.append(temperature_ori)
            diff_observations.append(temperature_deviations)
            actions_behavior.append(behavior_action_array)
            actions_label.append(reference_action_array)
            diff_actions_behavior.append(diff_behavior_action_array)
            diff_actions_label.append(diff_reference_action_array)
            actions_behavior_policy.append(behavior_policy_action_array)
            actions_label_policy.append(reference_policy_action_array)
            label_action_distribution.append(action_reference_distribution)
            rewards.append(reward_behavior_action)
            resets.append(1 if truncated or terminated else 0)

            if terminated or truncated:
                if self.overheat_no_reset and terminated:
                    pass
                obs, info = self._reset()

        observations = np.array(observations, dtype=np.float32)  # Shape: (max_steps, sensor)
        diff_observations = np.array(diff_observations, dtype=np.float32)  # Shape: (max_steps, sensor)
        actions_behavior = np.array(actions_behavior, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        actions_label = np.array(actions_label, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        diff_actions_behavior = np.array(diff_actions_behavior, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        diff_actions_label = np.array(diff_actions_label, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        actions_behavior_policy = np.array(actions_behavior_policy, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        actions_label_policy = np.array(actions_label_policy, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        label_action_distribution = np.array(label_action_distribution, dtype=np.float32)  # Shape: (max_steps, cooler, discrete_action_dim)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)  # Shape: (max_steps, 1)
        prompts = np.full(rewards.shape, self.reward_mode, dtype=rewards.dtype)
        resets = np.array(resets, dtype=np.int32).reshape(-1, 1)   # Shape: (max_steps, 1)

        # Print data shapes for verification
        if self.verbose:
            print(f"\nData shapes:")
            print(f"  observations: {observations.shape}")
            print(f"  diff_observations: {diff_observations.shape}")
            print(f"  actions_behavior: {actions_behavior.shape}")
            print(f"  actions_label: {actions_label.shape}")
            print(f"  diff_actions_behavior: {diff_actions_behavior.shape}")
            print(f"  diff_actions_label: {diff_actions_label.shape}")
            print(f"  actions_behavior_policy: {actions_behavior_policy.shape}")
            print(f"  actions_label_policy: {actions_label_policy.shape}")
            print(f"  rewards: {rewards.shape}")
            print(f"  prompts: {prompts.shape}")
            print(f"  resets: {resets.shape}")
            print(f"  obs_graph: {self.agent_sensor_graph.shape}")
            print(f"  agent_graph: {self.agent_agent_graph.shape}")

        processed_data = {
            "observations": observations,
            "diff_observations": diff_observations,
            "actions_behavior": actions_behavior,
            "actions_label": actions_label,
            "diff_actions_behavior": diff_actions_behavior,
            "diff_actions_label": diff_actions_label,
            "actions_behavior_policy": actions_behavior_policy,
            "actions_label_policy": actions_label_policy,
            "label_action_distribution": label_action_distribution,
            "prompts": prompts,
            "rewards": rewards,
            "resets": resets,
            "obs_graph": self.agent_sensor_graph,
            "agent_graph": self.agent_agent_graph,
        }

        reorganized_data = self.reorganize_data_by_agent(processed_data)
        self.save_multiagent_data(reorganized_data)
        return

    def reorganize_data_by_agent(self, data, max_steps=5040):
        """
        Reorganize the data and take the agent as the first dimension
        
        Target format:
        - observations: (n_agents, timesteps, features_per_agent)
        - actions_behavior/label: (n_agents, timesteps, 2)
        - diff_actions_behavior/label: (n_agents, timesteps, 2)
        - actions_behavior_policy/label_policy: (n_agents, timesteps, 2)
        - prompts: (n_agents, timesteps, 1) -> policy_prompt for each agent
        - rewards: (timesteps, 1)
        - resets: (timesteps, 1)
        """
        n_coolers = self.n_coolers
        n_sensors = self.n_sensors
        timesteps = len(data["observations"])
        
        if self.verbose:
            print(f"\nReorganizing data by agent...")
            print(f"  Number of agents (coolers): {n_coolers}")
            print(f"  Number of sensors: {n_sensors}")
            print(f"  Number of timesteps: {timesteps}")
        
        # 1. observations, diff_observations
        # origin: (timesteps, n_sensors)
        # target: (n_sensors, timesteps, 1)
        observations_by_agent = data["observations"].T[:, :, np.newaxis].astype(np.float32)
        diff_observations_by_agent = data["diff_observations"].T[:, :, np.newaxis].astype(np.float32)

        # 2. actions
        # origin: (timesteps, 2, n_coolers)
        # target: (n_coolers, timesteps, 2)
        actions_behavior_by_agent = np.transpose(data["actions_behavior"], (2, 0, 1)).astype(np.float32)
        actions_label_by_agent = np.transpose(data["actions_label"], (2, 0, 1)).astype(np.float32)
        diff_actions_behavior_by_agent = np.transpose(data["diff_actions_behavior"], (2, 0, 1)).astype(np.float32)
        diff_actions_label_by_agent = np.transpose(data["diff_actions_label"], (2, 0, 1)).astype(np.float32)
        actions_behavior_policy_by_agent = np.transpose(data["actions_behavior_policy"], (2, 0, 1)).astype(np.float32)
        actions_label_policy_by_agent = np.transpose(data["actions_label_policy"], (2, 0, 1)).astype(np.float32)
        
        # 3. label_action_distribution
        # origin: (timesteps, n_coolers, discrete_action_dim)
        # target: (n_coolers, timesteps, discrete_action_dim)
        label_action_distribution_by_agent = np.transpose(data["label_action_distribution"], (1, 0, 2)).astype(np.float32)
        
        prompts = data["prompts"]
        obs_graph = data["obs_graph"]    # (n_coolers，n_sensors)  
        agent_graph = data["agent_graph"]  # (n_coolers, n_coolers)
        rewards = data["rewards"]  # (timesteps, 1)
        resets = data["resets"]  # (timesteps, 1)
        
        reorganized_data = {
            "observations": observations_by_agent,  # (n_sensors, timesteps, 1)
            "diff_observations": diff_observations_by_agent,  # (n_sensors, timesteps, 1)
            "actions_behavior": actions_behavior_by_agent,  # (n_coolers, timesteps, 2)
            "actions_label": actions_label_by_agent,  # (n_coolers, timesteps, 2)
            "diff_actions_behavior": diff_actions_behavior_by_agent,  # (n_coolers, timesteps, 2)
            "diff_actions_label": diff_actions_label_by_agent,  # (n_coolers, timesteps, 2)
            "actions_behavior_policy": actions_behavior_policy_by_agent,  # (n_coolers, timesteps, 2)
            "actions_label_policy": actions_label_policy_by_agent,  # (n_coolers, timesteps, 2)
            "label_action_distribution": label_action_distribution_by_agent, # (n_coolers, timesteps, discrete_action_dim)
            "prompts": prompts,  # (n_coolers, timesteps)
            "rewards": rewards,  # (timesteps, 1)
            "resets": resets,  # (timesteps, 1)
            "obs_graph": obs_graph,  # (n_coolers, n_sensors)
            "agent_graph": agent_graph,  # (n_coolers, n_coolers)
            "metadata": {
                "n_agents": n_coolers,
                "n_sensors": n_sensors,
                "n_timesteps": timesteps,
            }
        }
        
        if self.verbose:
            print(f"\nReorganized data shapes:")
            print(f"  observations: {reorganized_data['observations'].shape}")
            print(f"  diff_observations: {reorganized_data['diff_observations'].shape}")
            print(f"  actions_behavior: {reorganized_data['actions_behavior'].shape}")
            print(f"  actions_label: {reorganized_data['actions_label'].shape}")
            print(f"  diff_actions_behavior: {reorganized_data['diff_actions_behavior'].shape}")
            print(f"  diff_actions_label: {reorganized_data['diff_actions_label'].shape}")
            print(f"  actions_behavior_policy: {reorganized_data['actions_behavior_policy'].shape}")
            print(f"  actions_label_policy: {reorganized_data['actions_label_policy'].shape}")
            print(f"  label_action_distribution: {reorganized_data['label_action_distribution'].shape}")
            print(f"  prompts: {reorganized_data['prompts'].shape}")
            print(f"  rewards: {reorganized_data['rewards'].shape}")
            print(f"  resets: {reorganized_data['resets'].shape}")
            print(f"  obs_graph: {reorganized_data['obs_graph'].shape}")
            print(f"  agent_graph: {reorganized_data['agent_graph'].shape}")
        
        return reorganized_data
    
    def save_multiagent_data(self, data):
        """Save multi-agent data with correct format"""
        epoch_path = Path(self.output_path) / f'record-{self.epoch_id:06d}'
        os.makedirs(epoch_path, exist_ok=True)
        
        np.save(epoch_path / 'observations.npy', data["observations"])
        np.save(epoch_path / 'diff_observations.npy', data["diff_observations"])
        np.save(epoch_path / 'actions_behavior.npy', data["actions_behavior"])
        np.save(epoch_path / 'actions_label.npy', data["actions_label"])
        np.save(epoch_path / 'diff_actions_behavior.npy', data["diff_actions_behavior"])
        np.save(epoch_path / 'diff_actions_label.npy', data["diff_actions_label"])
        np.save(epoch_path / 'actions_behavior_policy.npy', data["actions_behavior_policy"])
        np.save(epoch_path / 'actions_label_policy.npy', data["actions_label_policy"])
        np.save(epoch_path / 'label_action_distribution.npy', data["label_action_distribution"])
        np.save(epoch_path / 'prompts.npy', data["prompts"])
        np.save(epoch_path / 'rewards.npy', data["rewards"])
        np.save(epoch_path / 'resets.npy', data["resets"])
        np.save(epoch_path / 'obs_graph.npy', data["obs_graph"])
        np.save(epoch_path / 'agent_graph.npy', data["agent_graph"])
        task_file_source = Path(self.task_file_path)
        target = epoch_path / task_file_source.name
        shutil.copy(str(task_file_source), str(target))
        print(f"Saved data to {epoch_path}")

def compare_first_last_rewards(log_path: str) -> Tuple[bool, float, float]:
    """
    Read the log file and compare the average of the first three rewards with the average of the last three rewards.
    
    Parameters:
        log_path: Path to the log file
    
    Returns:
        Tuple[bool, float, float]:
            - bool: Returns True if the average of the last three rewards is greater than the average of the first three rewards, otherwise False
            - float: Average of the first three rewards
            - float: Average of the last three rewards
    """

    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    reward_pattern = r'Reward:\s*([-\d.]+)'
    rewards = re.findall(reward_pattern, log_content)
    
    rewards = [float(r) for r in rewards]

    first_three = rewards[:10]
    first_avg = sum(first_three) / 10
    
    last_three = rewards[-10:]
    last_avg = sum(last_three) / 10
    
    is_improved = last_avg > first_avg
    
    return is_improved, first_avg, last_avg

def single_process_test():
    task_folder = "./rl_models/hvac_task_43/"
    output_path = "./rl_models/hvac_task_43/data/"
    rl_mode_type = "rppo"
    use_rrd = False
    reward_mode = 0

    task_folder = Path(task_folder)
    
    task_config_path = None
    for file in task_folder.iterdir():
        if file.suffix == '.pkl':
            task_config_path = str(file)
            break
    
    if not task_config_path:
        print(f"Warning: No .pkl file found in {task_folder}")
        return
    
    rl_model_path = [] 
    if use_rrd:
        rl_model_path.append(task_folder / f"{rl_mode_type}_reward_mode_{reward_mode}.zip")
        rrd_path = task_folder / f"{rl_mode_type}_reward_mode_{reward_mode}_ckpts"
        if rrd_path.exists() and rrd_path.is_dir():
            zip_files = list(rrd_path.glob("*.zip"))
            rl_model_path.extend(zip_files)
            print(f"Add {len(zip_files)} rrd models")
        else:
            print(f"Warning: RRD model path not found: {rrd_path}")
        print(f"Total {len(rl_model_path)} models")
    else:    
        for i in range(3):
            path = task_folder / f"{rl_mode_type}_reward_mode_{i}.zip"
            if not path.exists():
                print(f"Warning: Model file not found: {path}")
                return
            rl_model_path.append(path)
    
    generator = HVACDataGenerator(
        epoch_id=0,
        task_config_path=task_config_path,
        rl_model_path=rl_model_path,
        output_path=output_path,
        rl_mode_type='rppo',
        reward_mode=reward_mode,
        max_steps=200,
        use_rrd=use_rrd,   
        verbose=True
    )
    
    print("Generating data...")
    generator.generate_data()

def process_task(epoch_id, task_folder, rl_mode_type, reward_mode, output_path, max_steps, use_rrd, reference_action_deterministic):
    """
    处理单个任务，包含完整的错误处理
    
    参数:
        epoch_id: 任务ID
        task_folder: 任务文件夹路径
        rl_mode_type: RL模式类型
        reward_mode: 奖励模式
        output_path: 输出路径
        max_steps: 最大步数
        use_rrd: 是否使用RRD
        reference_action_deterministic: 参考动作确定性
    """
    task_folder = Path(task_folder)
    
    try:
        # 查找 pkl 文件
        task_config_path = None
        for file in task_folder.iterdir():
            if file.suffix == '.pkl':
                task_config_path = str(file)
                break
        
        if not task_config_path:
            print(f"⚠️ [Task {epoch_id}] Warning: No .pkl file found in {task_folder}")
            return False
        
        # 查找 RL 模型路径
        rl_model_path = []
        if use_rrd:
            rl_model_path.append(task_folder / f"{rl_mode_type}_reward_mode_{reward_mode}.zip")
            rrd_path = task_folder / f"{rl_mode_type}_reward_mode_{reward_mode}_ckpts"
            if rrd_path.exists() and rrd_path.is_dir():
                zip_files = list(rrd_path.glob("*.zip"))
                rl_model_path.extend(zip_files)
            else:
                print(f"⚠️ [Task {epoch_id}] Warning: RRD model path not found: {rrd_path}")
                return False
        else:
            for i in range(3):
                path = task_folder / f"{rl_mode_type}_reward_mode_{i}.zip"
                if not path.exists():
                    print(f"⚠️ [Task {epoch_id}] Warning: Model file not found: {path}")
                    return False
                rl_model_path.append(path)
        
        print(f"📋 [Task {epoch_id}] Processing task {task_folder.name}, reward_mode={reward_mode}")
        
        # 检查 reward 改善情况
        coach_log_path = task_folder / f"{rl_mode_type}_reward_mode_{reward_mode}.log"
        try:
            is_improved, first_avg, last_avg = compare_first_last_rewards(coach_log_path)
            if not is_improved:
                print(f"⚠️ [Task {epoch_id}] {coach_log_path} is not improved, first_avg={first_avg:.4f}, last_avg={last_avg:.4f}")
                return False
        except Exception as e:
            print(f"⚠️ [Task {epoch_id}] Error checking reward improvement: {str(e)}")
            return False
        
        # 创建生成器
        generator = HVACDataGenerator(
            epoch_id=epoch_id,
            task_config_path=task_config_path,
            rl_model_path=rl_model_path,
            output_path=output_path,
            rl_mode_type=rl_mode_type,
            reward_mode=reward_mode,
            max_steps=max_steps,
            use_rrd=use_rrd,
            reference_action_deterministic=reference_action_deterministic,
            verbose=False
        )
        
        # 生成数据（关键部分：添加 try-catch）
        try:
            generator.generate_data()
            print(f"✅ [Task {epoch_id}] Successfully completed task {task_folder.name}, reward_mode={reward_mode}")
            return True
        except Exception as e:
            print(f"❌ [Task {epoch_id}] Error in generate_data() for task {task_folder.name}, reward_mode={reward_mode}")
            print(f"   Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"❌ [Task {epoch_id}] Unexpected error processing task {task_folder.name}, reward_mode={reward_mode}")
        print(f"   Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def parallel_generate_data(input_folders, output_path, rl_mode_type, max_steps, use_rrd, reference_action_deterministic, 
                           num_records=None, num_processes=None):
    """
    并行生成数据，带有任务失败统计
    
    参数:
        input_folders: 输入文件夹列表
        output_path: 输出路径
        rl_mode_type: RL模式类型
        max_steps: 最大步数
        use_rrd: 是否使用RRD
        reference_action_deterministic: 参考动作确定性
        num_records: 记录数量
        num_processes: 进程数
    """
    if not isinstance(input_folders, list):
        input_folders = [input_folders]
    
    tasks = []
    epoch_id = 0

    for input_folder in input_folders:
        input_folder = Path(input_folder)
        if not input_folder.is_dir():
            print(f"❌ Error: {input_folder} is not a directory")
            continue
        for task_folder in input_folder.iterdir():
            if task_folder.is_dir():
                mode_num = 3  # 0: reward mode 0 + rl_0; 1: reward mode 1 + rl_1; 2: reward mode 2 + rl_2;
                for reward_mode in range(mode_num):
                    tasks.append({
                        "epoch_id": epoch_id,
                        "task_folder": str(task_folder),
                        "rl_mode_type": rl_mode_type,
                        "reward_mode": reward_mode,
                        "output_path": output_path,
                        "max_steps": max_steps,
                        "use_rrd": use_rrd,
                        "reference_action_deterministic": reference_action_deterministic
                    })
                    epoch_id += 1
    
    print(f"📋 Found {len(tasks)} tasks to process")

    if num_records is None or num_records < epoch_id:
        num_records = epoch_id
    else:
        original_task_count = len(tasks)
        full_cycles = num_records // original_task_count
        remainder = num_records % original_task_count
        
        expended_tasks = []
        for cycle in range(full_cycles):
            for task in tasks:
                new_task = task.copy()
                new_task["epoch_id"] = cycle * original_task_count + task["epoch_id"]
                expended_tasks.append(new_task)
        for i in range(remainder):
            task = tasks[i].copy()
            task["epoch_id"] = full_cycles * original_task_count + i
            expended_tasks.append(task)
        tasks = expended_tasks
    
    print(f"📋 Total tasks after expansion: {len(tasks)}")

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = min(num_processes, multiprocessing.cpu_count())
    
    print(f"🔧 Using {num_processes} CPU cores")

    # 使用进程池处理任务
    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(t["epoch_id"], t["task_folder"], t["rl_mode_type"], t["reward_mode"], 
                 t["output_path"], t["max_steps"], t["use_rrd"], t["reference_action_deterministic"]) for t in tasks]
        results = pool.starmap(process_task, args)
    
    # 统计成功/失败数量
    success_count = sum(1 for r in results if r)
    failed_count = len(results) - success_count
    
    print("\n" + "=" * 60)
    print(f"📊 Task Summary:")
    print(f"   Total tasks: {len(results)}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {failed_count}")
    print("=" * 60)
    
    return success_count, failed_count


if __name__ == "__main__":
    # single_process_test()

    parser = argparse.ArgumentParser(description='Generate HVAC data in parallel')
    
    parser.add_argument('--input_folders', nargs='+', type=str, default=None,
                    help='Input folders containing task subdirectories (multiple can be specified)')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder for generated data')
    parser.add_argument('--rl_mode_type', type=str, default='sac',
                        choices=['sac', 'ppo', 'rppo'],
                        help='Type of RL model to use')
    parser.add_argument('--use_rrd', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), 
                        default=False, help='Use RRD to generate behavior sequence')
    parser.add_argument('--reference_action_deterministic', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), 
                        default=False, help='Use deterministic reference action')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--num_records', type=int, default=None, 
                    help='Total number of records to generate. If not set, defaults to task_num * 3')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='Number of parallel processes to use')
    
    args = parser.parse_args()
    
    print("Starting data generation with parameters:")
    print(f"  Input folder: {args.input_folders}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  RL model type: {args.rl_mode_type}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Use RRD: {args.use_rrd}")
    print(f"  Reference action deterministic: {args.reference_action_deterministic}")
    print(f"  Number of records: {args.num_records}")
    print(f"  Number of processes: {args.num_processes}")
    
    parallel_generate_data(
        input_folders=args.input_folders,
        output_path=args.output_folder,
        rl_mode_type=args.rl_mode_type,
        max_steps=args.max_steps,
        use_rrd=args.use_rrd,
        reference_action_deterministic=args.reference_action_deterministic,
        num_records=args.num_records,
        num_processes=args.num_processes
    )


