import numpy
import gymnasium
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, PPO, DQN, TD3
from ma_gym.envs.switch import Switch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import imageio

class MapStateToDiscrete:
    def __init__(self, env_name, state_space_dim1, state_space_dim2):
        self.env_name = env_name.lower()
        self.state_space_dim1 = state_space_dim1
        self.state_space_dim2 = state_space_dim2
        
        if self.env_name.find("pendulum") >= 0:
            self.map_state_to_discrete_func = self._map_state_to_discrete_pendulum
        elif self.env_name.find("mountaincar") >= 0:
            self.map_state_to_discrete_func = self._map_state_to_discrete_mountaincar
        else:
            self.map_state_to_discrete_func = self._map_state_to_discrete_default # return origin state
    
    def map_to_discrete(self, value, min_val, max_val, n_interval):
        """
        Maps a continuous value to a discrete integer.

        Parameters:
        value (float): The continuous value to be mapped.
        min_val (float): The minimum value of the continuous range.
        max_val (float): The maximum value of the continuous range.
        n_interval (int): The number of intervals.

        Returns:
        int: The mapped discrete integer [0, n_interval - 1].
        """
        # Create bin edges
        bins = numpy.linspace(min_val, max_val, n_interval + 1)
        
        # Clip the value within the range [min_val, max_val]
        clipped_value = numpy.clip(value, min_val, max_val)
        
        # Digitize the clipped value to get the discrete integer
        discrete_value = numpy.digitize(clipped_value, bins) - 1
        
        # Ensure the discrete value is within the range [0, num_bins-1]
        return numpy.clip(discrete_value, 0, n_interval - 1)
    
    def _map_state_to_discrete_pendulum(self, state):
        """
        Maps a state array to a discrete integer for Pendulum-v1.

        Parameters:
        state (numpy.ndarray): An array containing cos(theta), sin(theta), and speed.

        Returns:
        int: The discretized state integer.
        """
        # Extract cos_theta and sin_theta
        cos_theta = state[0]
        sin_theta = state[1]
        
        # Calculate theta using atan2 to get the correct quadrant
        theta = numpy.arctan2(sin_theta, cos_theta)
        
        # Map theta from [-pi, pi] to [0, 2*pi]
        if theta < 0:
            theta += 2 * numpy.pi
        
        # Define the range and number of intervals for theta
        theta_min, theta_max = 0, 2 * numpy.pi
        n_interval_theta = self.state_space_dim1
        
        # Use the helper function to map theta
        theta_discrete = self.map_to_discrete(theta, theta_min, theta_max, n_interval_theta)
        
        # Define the range and number of intervals for speed
        speed_min, speed_max = -8.0, 8.0
        n_interval_speed = self.state_space_dim2
        
        # Use the helper function to map speed
        speed_discrete = self.map_to_discrete(state[2], speed_min, speed_max, n_interval_speed)
        
        # Calculate the discretized state
        state_discrete = n_interval_speed * theta_discrete + speed_discrete
        
        return state_discrete
    
    def _map_state_to_discrete_mountaincar(self, state):
        """
        Maps a state array to a discrete integer for MountainCar-v0.

        Parameters:
        state (numpy.ndarray): An array containing position and velocity.

        Returns:
        int: The discretized state integer.
        """
        # Define the ranges and number of intervals for position and velocity
        position_min, position_max = -1.2, 0.6
        n_interval_position = self.state_space_dim1
        
        velocity_min, velocity_max = -0.07, 0.07
        n_interval_velocity = self.state_space_dim2
        
        # Use the helper function to map position and velocity
        position_discrete = self.map_to_discrete(state[0], position_min, position_max, n_interval_position)
        velocity_discrete = self.map_to_discrete(state[1], velocity_min, velocity_max, n_interval_velocity)
        
        # Calculate the discretized state
        state_discrete = n_interval_velocity * position_discrete + velocity_discrete
        
        return state_discrete
    
    def _map_state_to_discrete_default(self, state):
        return state
    
    def map_state_to_discrete(self, state):
        """
        Maps a state array to a discrete integer based on the environment.

        Parameters:
        state (numpy.ndarray): An array containing the state variables of the environment.

        Returns:
        int: The discretized state integer.
        """
        return self.map_state_to_discrete_func(state)
    
class MapActionToContinuous:
    def __init__(self, env_name, distribution_type='linear'):
        self.env_name = env_name.lower()
        self.distribution_type = distribution_type
        
        if self.env_name.find("pendulum") >= 0:
            self.map_action_to_continuous_func = self._map_action_to_continous_pendulum
        else:
            self.map_action_to_continuous_func = self._map_action_to_continous_default # return origin action
    
    def map_to_continuous(self, value, min_val, max_val, n_action):
        """
        Maps a discrete integer to a continuous value.

        Parameters:
        value (int): The discrete integer to be mapped.
        min_val (float): The minimum value of the continuous range.
        max_val (float): The maximum value of the continuous range.
        n_interval (int): The number of intervals.

        Returns:
        float: The mapped continuous value within the range [min_val, max_val].
        """
        # Calculate the step size for each interval
        if n_action < 2:
            raise ValueError(f"Invalid number of actions: {n_action}")
        
        if self.distribution_type == 'linear':
            step_size = (max_val - min_val) / (n_action - 1)
            continuous_value = min_val + value * step_size
        elif self.distribution_type == 'sin':
            # Map the discrete value to a normalized range [0, pi]
            normalized_value = (value / (n_action - 1)) * numpy.pi
            # Apply sine function and scale it to the desired range
            continuous_value = min_val + ((numpy.sin(normalized_value) + 1) / 2) * (max_val - min_val)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        return continuous_value
    
    def _map_action_to_continous_pendulum(self, action):
        """
        Maps a discrete action integer to a continuous action for Pendulum-v1.

        Parameters:
        action (int): A discrete action integer from 0 to n_action-1.

        Returns:
        float: The mapped continuous action value.
        """
        min_val, max_val = -2.0, 2.0
        n_action = 5
        
        # Use the helper function to map action
        continuous_action = self.map_to_continuous(action, min_val, max_val, n_action)
        
        return numpy.array([continuous_action])  
    
    def _map_action_to_continous_default(self, action):
        return action
    
    def map_action_to_continuous(self, action):
        """
        Maps a discrete action integer to a continuous action based on the environment.

        Parameters:
        action (int): A discrete action integer from 0 to n_action-1.

        Returns:
        float: The mapped continuous action value.
        """
        return self.map_action_to_continuous_func(action)
    
class DiscreteEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, env_name, action_space=5, state_space_dim1=8, state_space_dim2=8, reward_shaping = False, skip_frame=0):
        super(DiscreteEnvWrapper, self).__init__(env)
        self.env_name = env_name.lower()
        self.action_space = gymnasium.spaces.Discrete(action_space)
        self.observation_space = gymnasium.spaces.Discrete(state_space_dim1 * state_space_dim2)
        self.reward_shaping = reward_shaping
        self.skip_frame = skip_frame
        self.map_state_to_discrete = MapStateToDiscrete(self.env_name, state_space_dim1, state_space_dim2).map_state_to_discrete
        self.map_action_to_continuous = MapActionToContinuous(self.env_name).map_action_to_continuous

    def reset(self, **kwargs):
        continuous_state, info = self.env.reset(**kwargs)
        discrete_state = self.map_state_to_discrete(continuous_state)
        if self.env_name.lower().find("mountaincar") >= 0:
            self.last_energy = 0.5*continuous_state[1]*continuous_state[1] + 0.0025*(numpy.sin(3*continuous_state[0])*0.45+0.55)
            self.last_gamma_vel = 0.0
        return discrete_state, info
        
    def step(self, discrete_action):
        total_reward = 0.0
        continuous_action = self.map_action_to_continuous(discrete_action)
        for _ in range(self.skip_frame + 1):
            continuous_state, reward, terminated, truncated, info = self.env.step(continuous_action)
            if self.reward_shaping:
                if self.env_name.lower().find("mountaincar") >= 0:
                    energy = 0.5*continuous_state[1]*continuous_state[1] + 0.0025*(numpy.sin(3*continuous_state[0])*0.45+0.55)
                    if energy > self.last_energy:
                        reward = 0.01
                    else:
                        reward = -0.01
                    gamma = 0.66
                    reward = -0.01 + 10*(continuous_state[1]*continuous_state[1] + gamma * self.last_gamma_vel)
                    self.last_gamma_vel = continuous_state[1]*continuous_state[1] + gamma * self.last_gamma_vel
                    self.last_energy = energy
            
            if self.env_name.lower().find("cliff") >= 0:
                if reward < -50:
                    truncated = True

            total_reward += reward
            if terminated or truncated:
                break
        discrete_state = self.map_state_to_discrete(continuous_state)
        return discrete_state, total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()

class RolloutLogger(BaseCallback):
    """
    A custom callback for logging the total reward and episode length of each rollout.
    
    :param env_name: Name of the environment.
    :param max_rollout: Maximum number of rollouts to perform.
    :param max_step: Maximum steps per episode.
    :param downsample_trail: Downsample trail parameter.
    :param verbose: Verbosity level: 0 = no output, 1 = info, 2 = debug
    """
    def __init__(self, env_name, max_rollout, max_step, downsample_trail, verbose=0):
        super(RolloutLogger, self).__init__(verbose)
        self.env_name = env_name.lower()
        self.max_rollout = max_rollout
        self.max_steps = max_step
        self.current_rollout = 0
        self.reward_sums = []
        self.step_counts = []
        self.success_rate = []
        self.success_rate_f = 0.0
        self.downsample_trail = downsample_trail
        self.episode_reward = 0
        self.episode_length = 0

    def is_success_fail(self, reward, total_reward, terminated):
        if "lake" in self.env_name:
            return int(reward > 1.0e-3)
        elif "lander" in self.env_name:
            return int(total_reward >= 200)
        elif "mountaincar" in self.env_name:
            return terminated
        elif "cliff" in self.env_name:
            return terminated
        else:
            return 0

    def _on_step(self) -> bool:
        """
        This method is called after every step in the environment.
        Here we update the current episode's reward and length.
        """
        # Accumulate the episode reward
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1
        
        if 'terminated' in self.locals:
            terminated = self.locals['terminated'][0]
        elif 'dones' in self.locals:  # Fallback to 'done' flag
            done = self.locals['dones'][0]
            terminated = done  # Assuming 'done' means the episode has ended, either successfully or due to failure

        if 'truncated' in self.locals:
            truncated = self.locals['truncated'][0]
        elif 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            truncated = info.get('TimeLimit.truncated', False)

        if terminated or truncated:
            # Episode is done, record the episode information
            succ_fail = self.is_success_fail(self.locals['rewards'][0], self.episode_reward, terminated)
            
            if self.current_rollout < self.downsample_trail:
                self.success_rate_f = (1 - 1 / (self.current_rollout + 1)) * self.success_rate_f + succ_fail / (self.current_rollout + 1)
            else:
                self.success_rate_f = (1 - 1 / self.downsample_trail) * self.success_rate_f + succ_fail / self.downsample_trail

            self.reward_sums.append(self.episode_reward)
            self.step_counts.append(self.episode_length)
            self.success_rate.append(self.success_rate_f)

            # Reset episode counters
            self.episode_reward = 0
            self.episode_length = 0

            # Check if we have reached the maximum number of rollouts
            self.current_rollout += 1
            if self.current_rollout >= self.max_rollout:
                if self.verbose >= 1:
                    print(f"Reached maximum rollouts ({self.max_rollout}). Stopping training.")
                self.model.stop_training = True
                return False

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        For algorithms that do not use rollout_buffer, this method can be left empty.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered at the end of training.
        We can perform any final logging here if needed.
        """
        pass


class OnlineRL:
    def __init__(self, env, env_name, model_name, max_trails, max_steps, downsample_trail):
        self.env = env
        self.model_name = model_name
        self.log_callback = RolloutLogger(env_name, max_trails, max_steps, downsample_trail, verbose=1)
        
    def create_model(self):
        model_classes = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN, 'td3': TD3}
        if self.model_name not in model_classes:
            raise ValueError("Unknown policy type: {}".format(self.model_name))
        
        # Create the model with appropriate parameters
        if self.model_name.lower() in ['a2c', 'ppo']:
            self.model = model_classes[self.model_name.lower()](
                policy='MlpPolicy', env=self.env, verbose=1)
        elif self.model_name.lower() == 'dqn':
            self.model = DQN(
                policy='MlpPolicy', env=self.env,
                learning_rate=0.00025, buffer_size=100_000, exploration_fraction=0.1,
                exploration_final_eps=0.01, batch_size=32, tau=0.005,
                train_freq=(4, 'step'), gradient_steps=1, seed=None, optimize_memory_usage=False,
                verbose=1)
        elif self.model_name.lower() == 'td3':
            self.model = TD3(
                policy='MlpPolicy', env=self.env,
                learning_rate=0.0025, buffer_size=1_000_000, train_freq=(1, 'episode'),
                gradient_steps=1, action_noise=None, optimize_memory_usage=False,
                replay_buffer_class=None, replay_buffer_kwargs=None, verbose=1)

    def __call__(self):
        self.create_model()
        self.model.learn(total_timesteps=int(1e6), callback=self.log_callback)
        return (self.log_callback.reward_sums, 
                self.log_callback.step_counts, 
                self.log_callback.success_rate)
    
class AgentVisualizer:
    def __init__(self, save_path, visualize_online=False, skip_episode=0, fig=None, ax=None):
        self.save_path = save_path
        self.visualize_online = visualize_online
        self.skip_episode = skip_episode

        self.G = nx.DiGraph()
        self.fixed_positions = {}
        self.total_reward = 0
        
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots(figsize=(15, 7))
        else:
            self.fig = fig
            self.ax = ax

        self.episode = 0
        self.current_state = None 

    def step(self, last_state, action, reward, next_state, done):
        if not self.G.has_node(last_state):
            self.G.add_node(last_state)
        if not self.G.has_node(next_state):
            self.G.add_node(next_state)
        if not self.G.has_edge(last_state, next_state):
            self.G.add_edge(last_state, next_state, weight=reward)

        # Initialize positions for new nodes
        if last_state not in self.fixed_positions:
            pos = self._compute_new_position()
            self.fixed_positions[last_state] = pos

        if next_state not in self.fixed_positions:
            pos = self._compute_new_position()
            self.fixed_positions[next_state] = pos

        self.total_reward += reward
        
        if done:
            self.episode += (1 + self.skip_episode)
            self.total_reward = 0  

    def _compute_new_position(self):
        # Generate a new position and ensure it's at least 1.0 distance from all existing positions
        while True:
            pos = np.random.rand(2) * 20
            if all(np.linalg.norm(pos - p) >= 0.8 for p in self.fixed_positions.values()):
                return pos

    def update(self, frame):
        last_state, action, reward, next_state, done = frame
        self.step(last_state, action, reward, next_state, done)
        
        self.current_state = next_state

        pos = self.fixed_positions

        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.G.edges(data=True)}

        self.ax.clear()

        nx.draw_networkx_nodes(self.G, pos, node_size=700)
        nx.draw_networkx_edges(self.G, pos, arrowstyle='-')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        nx.draw_networkx_labels(self.G, pos, font_size=16, font_family="sans-serif")

        node_colors = ['orange' if n == self.current_state else 'skyblue' for n in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=700)

        plt.title(f"Anymdp (Episode {self.episode})")
        
        plt.text(0.02, 0.98, f'Episode Reward: {self.total_reward:.2f}', transform=self.ax.transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        
        plt.text(0.98, 0.98, f'Action: {action}', transform=self.ax.transAxes, fontsize=14,
                 verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
        
    def init(self):
        self.ax.clear()
        return []

    def draw(self, frames):
        if self.visualize_online:
            ani = FuncAnimation(self.fig, self.update, frames=frames, init_func=self.init, interval=100, blit=False, repeat=False)
            plt.show()  # Show the animation if visualizing
        else:
            imageio_writer = imageio.get_writer(f'{self.save_path}/anymdp.gif', mode='I', fps=10)
            for frame in frames:
                self.update(frame)  # Update the plot
                self.fig.canvas.draw_idle()  # Draw the canvas and flush it
                frame_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                imageio_writer.append_data(frame_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)))
            imageio_writer.close()
        return

if __name__ == "__main__":
    model_name = "dqn"
    env_name = "lake"
    max_trails = 50
    max_steps = 200
    downsample_trail = 10

    if env_name == "lake":
        env = gymnasium.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    online_rl = OnlineRL(env, env_name, model_name, max_trails, max_steps, downsample_trail)
    reward_sums, step_counts, success_rate = online_rl()
    
    print("Reward Sums:", reward_sums)
    print("Step Counts:", step_counts)
    print("Success Rate:", success_rate)

class Switch2(Switch):

    def __init__(self, full_observable: bool = False, step_cost: float = 0, n_agents: int = 4, max_steps: int = 50,
                 clock: bool = True):
        super().__init__(full_observable, step_cost, n_agents, max_steps, clock)
        self.init_mapping()

    def init_mapping(self):
        position_to_state = {}
        state_counter = 0
        
        for i in range(self._full_obs.shape[0]):
            for j in range(self._full_obs.shape[1]):
                if self._full_obs[i, j] != -1:
                    position_to_state[(i, j)] = state_counter
                    state_counter += 1  
        self.position_to_state = position_to_state
        for position, state in position_to_state.items():
            print(f"Position {position} -> State {state}")

    def get_agent_obs(self):
        _obs = []
        _obs_1dim = []
        for agent_i in range(0, self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = pos
            _obs.append(_agent_i_obs)

        agent1_state = self.position_to_state[tuple(self.agent_pos[0])]
        agent2_state = self.position_to_state[tuple(self.agent_pos[1])]
        agent1_x = self.agent_pos[0][1]
        agent1_y = self.agent_pos[0][0]
        agent2_x = self.agent_pos[1][1]
        agent2_y = self.agent_pos[1][0]
        if self.full_observable:
            # method 1: another agent's x pos (0~6)
            # method 2: relative x position when y1 = y2 & abs(x1-x2)<=2 (0~4)
            # method 3: another agent's area, left \ bridge \ right  (0~2)
            # method 4: another agent on bridge & (x > x_another -> 1 or x < x_another -> 2), else 0
            method = 1 
            if method == 1:
                _obs_1dim.append(agent2_x * 15 + agent1_state)
                _obs_1dim.append(agent1_x * 15 + agent2_state)
            elif method == 2:
                def get_idx(agent_x, another_agent_x):
                    x_diff = agent_x - another_agent_x
                    mapping = {2: 1, 1: 2, -1: 3, -2: 4}
                    return mapping.get(x_diff, 0) 

                if agent1_y != agent2_y:
                    _obs_1dim.append(agent1_state)
                    _obs_1dim.append(agent2_state)
                else:
                    _obs_1dim.append(get_idx(agent1_x, agent2_x) * 15 + agent1_state)
                    _obs_1dim.append(get_idx(agent2_x, agent1_x) * 15 + agent2_state)
            elif method == 3:
                def get_area(another_agent_x):
                    return 0 if another_agent_x < 2 else (1 if another_agent_x < 5 else 2)
                _obs_1dim.append(get_area(agent2_x) * 15 + agent1_state)
                _obs_1dim.append(get_area(agent1_x) * 15 + agent2_state)
            elif method == 4:
                def get_bridge_relative(agent_x, another_agent_x):
                    if another_agent_x in range(2, 5):
                        return 1 if agent_x > another_agent_x else 2
                    return 0
                _obs_1dim.append(get_bridge_relative(agent1_x, agent2_x) * 15 + agent1_state)
                _obs_1dim.append(get_bridge_relative(agent2_x, agent1_x) * 15 + agent2_state)

        else:
            _obs_1dim.append(agent1_state)
            _obs_1dim.append(agent2_state)

        # append original observation
        _obs_1dim.append(self.agent_pos[0])
        _obs_1dim.append(self.agent_pos[1])
        
        return _obs_1dim
    
    def render(self, mode='rgb_array'):
        if mode == 'human':
            super().render(mode=mode)
        elif mode == 'rgb_array':
            return super().render(mode=mode)
        else:
            raise ValueError(f"Unsupported mode: {mode}")