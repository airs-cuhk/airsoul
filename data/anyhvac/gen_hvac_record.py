import os
import argparse
import numpy as np
import pickle
import random
import multiprocessing
import time
from copy import deepcopy
from pathlib import Path
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiffAction
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
from rl_trainer_hvac import HVACRLTester

class HVACDataGenerator:
    def __init__(self,
                 epoch_id=0,
                 task_config_path=None,
                 rl_mode_type='sac',
                 rl_model_path=None,
                 output_path=None,
                 reward_mode=0,
                 max_steps=20000,
                 self_regression=True,
                 verbose=False
                 ):
        self.self_regression = self_regression
        try:
            with open(task_config_path, "rb") as f:
                task = pickle.load(f)
                if self_regression and reward_mode ==3:
                    env_reward_mode = 1
                    self.pid_self_regression = True
                else:
                    env_reward_mode = reward_mode
                    self.pid_self_regression = False
                self.env = HVACEnvDiffAction(reward_mode=env_reward_mode)
                self.env.set_task(task)
                self.cooler_sensor_topology = self.env.cooler_sensor_topology
                self.n_sensors = len(self.env.sensors)
                self.n_coolers = len(self.env.coolers)
                self.n_heaters = len(self.env.equipments)
                # self.env.set_random_start_t(True)
        except:
            print(f"Load task config from {task_config_path} failed!")

        if not self.pid_self_regression:
            try:
                self.rl_model = HVACRLTester(rl_model_path, rl_mode_type, "cpu")
            except:
                print(f"Load rl model from {rl_model_path} failed!")
        
        self.epoch_id = epoch_id
        self.output_path = output_path
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.verbose = verbose

        pid = os.getpid()
        seed = int(time.time() * 1000) % (2**32 - 1) + pid
        random.seed(seed)
        np.random.seed(seed)

    def _reset(self):
        obs, info = self.env.reset()
        self.pid_solver = HVACSolverGTPID(self.env)
        self.pid_solver.acc_diff = np.zeros_like(self.pid_solver.acc_diff)
        if not self.pid_self_regression:
            self.rl_model.reset()
        return obs, info

    def _create_sensor_cooler_graph(self, num_closest_sensors: int = 3):
        """
        Create sensor-cooler relationship graph.

        Args:
            num_closest_sensors (int): The number of closest sensors to consider for each cooler.
                                       Defaults to 3.

        Returns:
            np.ndarray: The sensor-cooler relationship graph.
        """
        # Create a new matrix with the same shape as the original (cooler, sensor) matrix
        obs_graph_orig = np.zeros_like(self.cooler_sensor_topology)
        n_coolers, n_sensors = self.cooler_sensor_topology.shape
        
        for cooler_id in range(n_coolers):
            sensor_weights = self.cooler_sensor_topology[cooler_id, :]
            if n_sensors >= num_closest_sensors:
                # Get the indices of the 'num_closest_sensors' smallest values (closest sensors)
                closest_sensor_idx = np.argpartition(sensor_weights, num_closest_sensors)[:num_closest_sensors]
            else:
                # If there are fewer than 'num_closest_sensors' sensors, use all sensors
                closest_sensor_idx = np.arange(len(sensor_weights))
            
            # Set the positions corresponding to the closest sensors to 1
            obs_graph_orig[cooler_id, closest_sensor_idx] = 1.0
        
        # Transpose the matrix to match the required shape (sensor, cooler)
        obs_graph = obs_graph_orig.T.astype(np.float32)
        return obs_graph

    def _create_cooler_cooler_graph(self, k_nearest_coolers: int = 3):
        """
        Create cooler-cooler relationship graph using KNN.

        Args:
            k_nearest_coolers (int): The number of nearest coolers to consider for each cooler.
                                     Defaults to 3.

        Returns:
            np.ndarray: The cooler-cooler relationship graph.
        """
        n_coolers = self.n_coolers
        agent_graph = np.zeros((n_coolers, n_coolers), dtype=np.float32)
        
        # Get cooler positions
        # Assuming self.env.coolers is an iterable of objects with a 'loc' attribute
        cooler_positions = np.array([cooler.loc for cooler in self.env.coolers])
        
        # Compute pairwise distances
        for i in range(n_coolers):
            for j in range(n_coolers):
                if i != j:
                    dist = np.linalg.norm(cooler_positions[i] - cooler_positions[j])
                    agent_graph[i, j] = dist
        
        # Convert to KNN graph (k='k_nearest_coolers' nearest neighbors)
        k = min(k_nearest_coolers, n_coolers - 1)
        for i in range(n_coolers):
            # Get k nearest neighbors
            distances = agent_graph[i, :]
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self
            
            # Set connections
            agent_graph[i, :] = 0
            agent_graph[i, nearest_indices] = 1
        
        # Make symmetric and add self-connections (if desired, currently commented out)
        # agent_graph = np.maximum(agent_graph, agent_graph.T)
        # np.fill_diagonal(agent_graph, 1.0) # Uncomment if self-connections are needed
        
        return agent_graph
    
    def _compute_temperature_deviation(self, sensor_readings):
        target_temp = self.env.target_temperature
        if isinstance(target_temp, (list, np.ndarray)):
            temperature_deviation = sensor_readings - np.array(target_temp)
        else:
            temperature_deviation = sensor_readings - target_temp
        
        return temperature_deviation

    def _compute_action_temperature_differences_with_graph(self, behavior_temp_settings, reference_temp_settings, 
                                                          obs_graph):
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
            cooler_sensor_graph = obs_graph.T
            for cooler_idx in range(n_coolers):
                connected_sensors = np.where(cooler_sensor_graph[cooler_idx] == 1.0)[0]
                
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

    def generate_data(self):
        observations = []
        diff_observations = [] 
        actions_behavior = []
        actions_label = []
        diff_actions_behavior = []
        diff_actions_label = []
        prompts = []
        rewards = []
        resets = []

        # Initialize environment
        obs, info = self._reset()
        

        # Statistics
        steps = 0
        num_closest_sensors = random.randint(3,5)
        num_closest_coolers = random.randint(3,5)

        self.obs_graph = self._create_sensor_cooler_graph(num_closest_sensors)
        self.agent_graph = self._create_cooler_cooler_graph(num_closest_sensors)

        default_action = self.env.last_action
        default_action_switch = default_action["switch"] 
        default_action_temp = self.env._action_value_to_temp(default_action["value"])
        default_action_array = np.array([default_action_temp, default_action_switch], dtype=np.float32)
        actions_behavior.append(default_action_array)
        actions_label.append(default_action_array)
        
        default_action_diff_behavior, default_action_diff_reference = self._compute_action_temperature_differences_with_graph(
                default_action_temp, 
                default_action_temp,
                self.obs_graph
            )
        default_action_diff = np.array([default_action_diff_behavior, default_action_switch], dtype=np.float32)
        diff_actions_behavior.append(default_action_diff)
        diff_actions_label.append(default_action_diff)
        

        while steps < self.max_steps:
            sensor_readings = obs["sensor_readings"]
            temperature_deviations = self._compute_temperature_deviation(sensor_readings)
            temperature_ori = sensor_readings

            if not self.self_regression: 
                behavior_action = 1 - self.pid_solver.policy(obs["sensor_readings"])[self.n_coolers:]
                reference_action = self.rl_model.predict(obs)
            else: 
                if self.pid_self_regression:
                    behavior_action = 1 - self.pid_solver.policy(obs["sensor_readings"])[self.n_coolers:]
                else:
                    behavior_action = self.rl_model.predict(obs)
                reference_action = behavior_action
            
            behavior_action = np.array(behavior_action).flatten()
            reference_action = self.env._diff_action(np.array(reference_action).flatten())

            obs, reward_behavior_action, terminated, truncated, info = self.env.step(behavior_action)
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
                self.obs_graph
            )

            if self.verbose:
                print(
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
            observations.append(temperature_ori)
            diff_observations.append(temperature_deviations)
            actions_behavior.append(behavior_action_array)
            actions_label.append(reference_action_array)
            rewards.append(reward_behavior_action)
            diff_actions_behavior.append(diff_behavior_action_array)
            diff_actions_label.append(diff_reference_action_array)
            resets.append(1 if truncated or terminated else 0)

            if truncated or truncated:
                obs, info = self._reset()

        observations = np.array(observations, dtype=np.float32)  # Shape: (max_steps, sensor)
        diff_observations = np.array(diff_observations, dtype=np.float32)  # Shape: (max_steps, sensor)
        actions_behavior = np.array(actions_behavior, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        actions_label = np.array(actions_label, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        diff_actions_behavior = np.array(diff_actions_behavior, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
        diff_actions_label = np.array(diff_actions_label, dtype=np.float32)  # Shape: (max_steps, 2, cooler)
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
            print(f"  rewards: {rewards.shape}")
            print(f"  prompts: {prompts.shape}")
            print(f"  resets: {resets.shape}")
            print(f"  obs_graph: {self.obs_graph.shape}")
            print(f"  agent_graph: {self.agent_graph.shape}")

        processed_data = {
            "observations": observations,
            "diff_observations": diff_observations,
            "actions_behavior": actions_behavior,
            "actions_label": actions_label,
            "diff_actions_behavior": diff_actions_behavior,
            "diff_actions_label": diff_actions_label,
            "prompts": prompts,
            "rewards": rewards,
            "resets": resets,
            "obs_graph": self.obs_graph,
            "agent_graph": self.agent_graph,
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
        
        prompts = data["prompts"]
        obs_graph = data["obs_graph"].T    # (n_coolers，n_sensors)  
        agent_graph = data["agent_graph"]  # (n_coolers, n_coolers)
        rewards = data["rewards"]  # (timesteps, 1)
        resets = data["resets"]  # (timesteps, 1)
        
        reorganized_data = {
            "observations": observations_by_agent,  # (n_sensors, timesteps, 1)
            "diff_observations": diff_observations_by_agent,  # (n_sensors, timesteps, 1)
            "actions_behavior": actions_behavior_by_agent,  # (n_coolers, timesteps, 2)
            "actions_label": actions_label_by_agent,  # (n_coolers, timesteps, 2)
            "diff_actions_behavior": diff_actions_behavior_by_agent,  # (n_coolers, timesteps, 2) - 修改
            "diff_actions_label": diff_actions_label_by_agent,  # (n_coolers, timesteps, 2) - 修改
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
        np.save(epoch_path / 'prompts.npy', data["prompts"])
        np.save(epoch_path / 'rewards.npy', data["rewards"])
        np.save(epoch_path / 'resets.npy', data["resets"])
        np.save(epoch_path / 'obs_graph.npy', data["obs_graph"])
        np.save(epoch_path / 'agent_graph.npy', data["agent_graph"])
        print(f"Saved data to {epoch_path}")



def single_process_test():
    task_config_path = "./rl_models/hvac_task_1185/hvac_task_1185.pkl"
    rl_model_path = "./rl_models/hvac_task_1185/rppo_reward_mode_1.zip"
    output_path = "./rl_models/hvac_task_1185/data/"
    
    generator = HVACDataGenerator(
        epoch_id=0,
        task_config_path=task_config_path,
        rl_model_path=rl_model_path,
        output_path=output_path,
        rl_mode_type='rppo',
        reward_mode=0,
        max_steps=1000,
        self_regression=True,      
        verbose=True
    )
    
    print("Generating data...")
    generator.generate_data()

def process_task(epoch_id, task_folder, rl_mode_type, reward_mode, output_path, max_steps, self_regression):

    task_folder = Path(task_folder)
    
    task_config_path = None
    for file in task_folder.iterdir():
        if file.suffix == '.pkl':
            task_config_path = str(file)
            break
    
    if not task_config_path:
        print(f"Warning: No .pkl file found in {task_folder}")
        return
    
    if reward_mode < 3 or not self_regression:
        rl_model_path = task_folder / f"{rl_mode_type}_reward_mode_{reward_mode}.zip"
        
        if not rl_model_path.exists():
            print(f"Warning: Model file not found: {rl_model_path}")
            return
        
        print(f"Processing task {epoch_id}: {task_folder.name}, reward_mode={reward_mode}")
    else:
        rl_model_path = None
    
    generator = HVACDataGenerator(
        epoch_id=epoch_id,
        task_config_path=task_config_path,
        rl_model_path=str(rl_model_path),
        output_path=output_path,
        rl_mode_type=rl_mode_type,
        reward_mode=reward_mode,
        max_steps=max_steps,
        self_regression=self_regression,
        verbose=False
    )
    
    generator.generate_data()

def parallel_generate_data(input_folder, output_path, rl_mode_type, max_steps, num_processes=None, self_regression=True):
    input_folder = Path(input_folder)
    if not input_folder.is_dir():
        print(f"Error: {input_folder} is not a directory")
        return
    
    tasks = []
    epoch_id = 0
    
    for task_folder in input_folder.iterdir():
        if task_folder.is_dir():
            if self_regression:
                mode_num = 4 # 0: reward mode 0 + rl_0; 1: reward mode 1 + rl_1; 2: reward mode 2 + rl_2; 3: reward mode 1 + pid
            else:
                mode_num = 3 # 0: reward mode 0 + rl_0; 1: reward mode 1 + rl_1; 2: reward mode 2 + rl_2;
            for reward_mode in range(mode_num):
                tasks.append({
                    "epoch_id": epoch_id,
                    "task_folder": str(task_folder),
                    "rl_mode_type": rl_mode_type,
                    "reward_mode": reward_mode,
                    "output_path": output_path,
                    "max_steps": max_steps,
                    "self_regression": self_regression
                })
                epoch_id += 1
    
    print(f"Found {len(tasks)} tasks to process")
    
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = min(num_processes, multiprocessing.cpu_count())
    
    print(f"Using {num_processes} CPU cores")

    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(t["epoch_id"], t["task_folder"], t["rl_mode_type"], t["reward_mode"], t["output_path"], t["max_steps"], t['self_regression']) for t in tasks]
        pool.starmap(process_task, args)

if __name__ == "__main__":
    # single_process_test()

    parser = argparse.ArgumentParser(description='Generate HVAC data in parallel')
    
    parser.add_argument('--input_folder', type=str, default=None,
                        help='Input folder containing task subdirectories')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder for generated data')
    parser.add_argument('--rl_mode_type', type=str, default='sac',
                        choices=['sac', 'ppo', 'rppo'],
                        help='Type of RL model to use')
    parser.add_argument('--max_steps', type=int, default=50,
                        help='Maximum steps per episode')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='Number of parallel processes to use')
    parser.add_argument('--self_regression', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), 
                        default=True, help='Use differential action environment (true/false)')
    
    args = parser.parse_args()
    
    print("Starting data generation with parameters:")
    print(f"  Input folder: {args.input_folder}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  RL model type: {args.rl_mode_type}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Number of processes: {args.num_processes}")
    
    parallel_generate_data(
        input_folder=args.input_folder,
        output_path=args.output_folder,
        rl_mode_type=args.rl_mode_type,
        max_steps=args.max_steps,
        num_processes=args.num_processes,
        self_regression=args.self_regression
    )


