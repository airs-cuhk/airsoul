import numpy
import gymnasium

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

class Switch2:

    def __init__(self, full_observable: bool = False, step_cost: float = 0, n_agents: int = 4, max_steps: int = 50,
                 clock: bool = True):

        try:
            from ma_gym.envs.switch import Switch
        except ImportError as e:
            raise RuntimeError("To use Switch2 class, please install ma-gym: pip install ma-gym") from e

        self.__class__ = type("Switch2", (Switch,), {})
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
           
class BaseEnv(gymnasium.Env):
    def reset(self, seed=None, options=None):
        raise NotImplementedError

    def transit(self, state, action):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        pass

    def deploy_eval(self, ctrl):
        return self.deploy(ctrl)

    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)

            rews.append(rew)
            next_obs.append(ob)

        obs = numpy.array(obs)
        acts = numpy.array(acts)
        next_obs = numpy.array(next_obs)
        rews = numpy.array(rews)

        return obs, acts, next_obs, rews
    
class DarkroomEnv(BaseEnv):
    def __init__(self, dim, horizon, goal=None):
        self.dim = dim
        if goal is None:
            self.goal = self.sample_state()
        else:
            self.check_goal(dim, goal)
        self.horizon = horizon
        self.state_dim = dim * dim
        self.action_dim = 4
        self.observation_space = gymnasium.spaces.Discrete(self.state_dim)
        self.action_space = gymnasium.spaces.Discrete(self.action_dim)

    def check_goal(self, dim, goal):
        if not isinstance(goal, numpy.ndarray):
            goal = numpy.array(goal)
        if goal.ndim != 1 or goal.shape[0] != 2:
            raise ValueError(f"goal must be 2-element 1D array, got shape {goal.shape}")
        if not numpy.issubdtype(goal.dtype, numpy.integer):
            raise TypeError(f"goal must be integer type, got dtype {goal.dtype}")
        if (goal < 0).any() or (goal >= dim).any():
            raise ValueError(f"All elements in goal must be in [0, {dim-1}], got {goal}")
        self.goal = goal
    
    def sample_state(self):
        state_2d = numpy.random.randint(0, self.dim, 2)
        return state_2d

    def sample_action(self):
        a = numpy.random.randint(0, 4)
        return a

    def map_state_to_1D(self, state):
        return state[0] * self.dim + state[1]

    def reset(self, seed=None, options=None):
        if seed is not None:
            numpy.random.seed(seed)

        self.current_step = 0
        self.state_2d = self.sample_state()
        while numpy.array_equal(self.state_2d, self.goal):
            self.state_2d = self.sample_state()
            
        self.state = self.map_state_to_1D(self.state_2d)
        return self.state, {}

    def transit(self, state, action):
        # action = numpy.argmax(action)
        assert action in numpy.arange(self.action_space.n)
        state = numpy.array(state)
        if action == 0:
            state[0] += 1
        elif action == 1:
            state[0] -= 1
        elif action == 2:
            state[1] += 1
        elif action == 3:
            state[1] -= 1
        state = numpy.clip(state, 0, self.dim - 1)

        if numpy.all(state == self.goal):
            reward = 1
        else:
            reward = - 0.01
        return state, reward

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state_2d, r = self.transit(self.state_2d, action)
        self.state = self.map_state_to_1D(self.state_2d)
        self.current_step += 1
        terminated = r == 1
        truncated = self.current_step >= self.horizon
        return self.state.copy(), r, terminated, truncated, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        if state[0] < self.goal[0]:
            action = 0
        elif state[0] > self.goal[0]:
            action = 1
        elif state[1] < self.goal[1]:
            action = 2
        elif state[1] > self.goal[1]:
            action = 3

        zeros = numpy.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros
    