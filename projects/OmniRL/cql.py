import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Tuple, Union
from gymnasium import spaces

class QNetwork(nn.Module):
    def __init__(self, state_space: spaces.Space, action_dim: int, hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        self.state_embed = None
        
        if isinstance(state_space, spaces.Discrete):
            self.state_embed = nn.Embedding(state_space.n, hidden_dim)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(state_space.shape[0], hidden_dim)
            
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        if self.state_embed is not None:
            if state.dim() == 2 and state.size(1) == 1:
                state = state.squeeze(1)  
            x = self.state_embed(state.long())
        else:
            x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, capacity: int, state_space: spaces.Space):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.state_space = state_space

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def add_batch(self, states, actions, rewards, next_states, dones):
        """ Batch add transitions to buffer """
        assert len(states) == len(actions) == len(rewards) == len(next_states) == len(dones), \
            "All input arrays must have same length"
        
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.push(s, a, r, ns, d)

    def sample(self, batch_size: int):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

class StateProcessor:
    def __init__(self, state_space: spaces.Space):
        self.state_space = state_space
        self.normalizer = None
        
        if isinstance(state_space, spaces.Box):
            self.normalizer = Normalizer(state_space.shape[0])
        elif isinstance(state_space, spaces.Discrete):
            self.embed_dim = 128  

    def update(self, state):
        if isinstance(self.state_space, spaces.Box) and self.normalizer:
            self.normalizer.update(state)

    def process(self, state):
        if isinstance(self.state_space, spaces.Discrete):
            return state  
        elif self.normalizer:
            return self.normalizer.normalize(state)
        return state

    @property
    def state_dim(self):
        if isinstance(self.state_space, spaces.Discrete):
            return self.state_space.n
        return self.state_space.shape[0]

class Normalizer:
    def __init__(self, state_dim: int):
        self.sum = np.zeros(state_dim)
        self.sum_sq = np.zeros(state_dim)
        self.count = 1e-4

    def update(self, state):
        self.sum += state
        self.sum_sq += state ** 2
        self.count += 1

    def batch_update(self, states: np.ndarray):
        self.sum += states.sum(axis=0)
        self.sum_sq += (states ** 2).sum(axis=0)
        self.count += states.shape[0]

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sum_sq / self.count - (self.mean)**2, 1e-8))

    def normalize(self, state):
        return (state - self.mean) / (self.std + 1e-8)

class CQL(BaseAlgorithm):
    def __init__(
        self,
        env: GymEnv,
        gamma: float = 0.99,
        alpha: float = 1.0,
        lr: float = 3e-4,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
    ):

        assert isinstance(env.observation_space, (spaces.Box, spaces.Discrete)), \
            "Only Box and Discrete state Spaces are supported"
        assert isinstance(env.action_space, spaces.Discrete), "Requires a discrete action space."

        super().__init__(
            policy=None,
            env=env,
            learning_rate=lr,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=False,
        )
        self.env = env        
        self.state_processor = StateProcessor(env.observation_space)

        self.q_network = QNetwork(
            state_space=env.observation_space,
            action_dim=env.action_space.n,
            hidden_dim=256
        )
        self.target_q_network = QNetwork(
            state_space=env.observation_space,
            action_dim=env.action_space.n,
            hidden_dim=256
        )
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, env.observation_space)
        
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self.policy = self

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        processed = self.state_processor.process(state)
        if isinstance(self.env.observation_space, spaces.Discrete):
            return torch.as_tensor(processed, dtype=torch.long) 
        return torch.as_tensor(processed, dtype=torch.float32)  

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        self.state_processor.update(observation)
        state_tensor = self._preprocess_state(observation).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        action = q_values.argmax().item()
        return np.array([action], dtype=np.int64), state
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
        tb_log_name: str = "CQL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "CQL":
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        state = self.env.reset()[0]
        episode_reward = 0.0

        while self.num_timesteps < total_timesteps:
            action, _ = self.predict(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action[0])
            done = terminated or truncated
            
            self.state_processor.update(next_state)
            self.replay_buffer.push(state, action[0], reward, next_state, done)
            
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                self._update_network(batch)
                self._soft_update_target()
            
            episode_reward += reward
            self.num_timesteps += 1
            
            if done:
                self.logger.record("train/episode_reward", episode_reward)
                state = self.env.reset()[0]
                episode_reward = 0.0
            else:
                state = next_state

            callback.locals.update({
                'rewards': [reward],
                'dones': done,
                'terminated': [terminated],
                'truncated': [truncated]
            })
            if not callback.on_step():
                break

        callback.on_training_end()
        return self

    def add_demo_data(self, states, actions, rewards, next_states, dones):
        if isinstance(self.state_processor.state_space, spaces.Box):
            self.state_processor.normalizer.batch_update(states) 
        
        self.replay_buffer.add_batch(states, actions, rewards, next_states, dones)

    def offline_learn(self, total_steps: int):
        update_freq = 100
        step_counter = 0
        for step in range(1, total_steps+1):
            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                loss = self._update_network(batch)
                step_counter += 1
                if step_counter % update_freq == 0:
                    self._soft_update_target()


    def _update_network(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.as_tensor(
            [self.state_processor.process(s) for s in states],
            dtype=torch.long if isinstance(self.env.observation_space, spaces.Discrete) else torch.float32
        )
        
        next_states = torch.as_tensor(
            [self.state_processor.process(s) for s in next_states],
            dtype=torch.long if isinstance(self.env.observation_space, spaces.Discrete) else torch.float32
        )
        
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).view(-1)
        dones = torch.FloatTensor(dones).view(-1)

        with torch.no_grad():
            next_q = self.target_q_network(next_states)
            max_next_q = torch.max(next_q, dim=1)[0].view(-1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # CQL regular terms
        q_values = self.q_network(states)
        logsumexp = torch.logsumexp(q_values / self.alpha, dim=1)
        policy_probs = F.softmax(q_values.detach(), dim=1)
        policy_q = (q_values * policy_probs).sum(dim=1)
        conservative_loss = (logsumexp - policy_q).mean()

        mse_loss = F.mse_loss(current_q, target_q.detach())
        total_loss = mse_loss + self.alpha * conservative_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def _soft_update_target(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str) -> None:
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_space': self.env.observation_space,
            'processor': self.state_processor.__dict__,
        }, path)

    @classmethod
    def load(cls, path: str, env: GymEnv, **kwargs) -> "CQL":
        checkpoint = torch.load(path)
        model = cls(env=env, **kwargs)
        model.q_network.load_state_dict(checkpoint['q_network'])
        model.target_q_network.load_state_dict(checkpoint['target_q_network'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        model.state_processor.__dict__.update(checkpoint['processor'])
        return model
