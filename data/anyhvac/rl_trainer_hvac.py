# rl_trainer.py
import os
import torch
import numpy as np
import gymnasium as gym
import numbers
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Union, Type

class HVACRLTrainer:
    def __init__(
        self,
        env_maker,  
        algorithm: str = "ppo", 
        stage_steps: int = 10000,
        n_envs: int = 4,
        vec_env_type: str = "dummy",
        vec_env_args: dict = None,
        verbose: int = 1,
        device: str = "auto",
        log_path: str = None
    ):
        # env wrapper
        self.n_envs = n_envs
        self.env = self._make_vec_env(
            env_maker=env_maker,  
            vec_type=vec_env_type.lower(),
            vec_args=vec_env_args or {}
        )
        self.algorithm = algorithm.lower()
        self.stage_steps = stage_steps
        self.stats = {"stage_rewards": []}
        
        # init model
        policy_map = {
            "ppo": PPO,
            "rppo": RecurrentPPO,
            "sac": SAC
        }
        policy_type_map = {
            "ppo": "MlpPolicy",
            "rppo": "MlpLstmPolicy",
            "sac": "MlpPolicy"
        }
        
        if self.algorithm not in policy_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.model_class = policy_map[self.algorithm]
        self.model = self._init_model(policy_type_map[self.algorithm], verbose, device)
        
        # training callback
        self.logger_callback = TrainingLoggerCallback(
            check_freq=stage_steps,
            verbose=verbose,
            log_path=log_path
        )

    def _make_vec_env(self, env_maker, vec_type: str, vec_args: dict):

        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
        
        def _make_env(rank):
            def _init():
                env = env_maker()  
                if hasattr(env, "seed"):
                    env.seed(42 + rank)
                return Monitor(env)
            return _init

        vec_env_classes = {
            "dummy": DummyVecEnv,
            "subproc": SubprocVecEnv
        }
        if vec_type not in vec_env_classes:
            raise ValueError(f"Unsupported vec env type: {vec_type}")
        

        return vec_env_classes[vec_type](
            [ _make_env(i) for i in range(self.n_envs) ],
            **vec_args
        )
    
    def _init_model(self, policy_type: str, verbose: int, device: str):
        """ init RL model """
        common_params = {
            "policy": policy_type,
            "env": self.env,
            "verbose": verbose,
            "device": device
        }
        
        if self.algorithm == "rppo":
            print("Use RecurrentPPO!")
            return RecurrentPPO(
                batch_size= int(32 * self.n_envs / 4),
                n_steps= 32,
                **common_params
            )
        elif self.algorithm == "ppo":
            print("Use PPO!")
            return PPO(
                batch_size= int(32 * self.n_envs / 4),
                n_steps= 32,
                **common_params
            )
        elif self.algorithm == "sac":
            return SAC(
                policy_kwargs={
                    "net_arch": {
                        "pi": [512, 256, 128],  # Actor（策略网络）的隐藏层结构
                        "qf": [512, 512, 256]   # Critic（价值网络）的隐藏层结构
                    },
                    "activation_fn": torch.nn.ReLU  # 高维数据常用ReLU增强非线性表达
                },
                buffer_size = 10000000, # 1e7
                batch_size = 1024,  # 增大批量（利用多核CPU并行计算，同时稳定梯度）
                learning_starts = 10000, # 先采集1万步初始经验（高维环境需要更多初始探索）
                train_freq=(1, "step"),  # 每步都训练（长周期环境需及时更新策略）
                gradient_steps=4,    # 每次训练更新4步（充分利用采样的batch数据）
                gamma=0.995, # 稍大于默认的0.99（长周期环境更重视远期奖励）
                use_sde=True,    # 开启SDE（高维动作空间需要更强的随机探索）
                sde_sample_freq=16,  # 每16步重新采样噪声（平衡探索随机性和稳定性）
                ent_coef="auto",
                **common_params
            )

    def train(self, total_steps: int = 100000):
        self.model.learn(
            total_timesteps=total_steps,
            callback=self.logger_callback,
            reset_num_timesteps=False
        )
        self._update_stats()

    def evaluate(self, n_episodes: int = 10):
        total_rewards = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model = self.model_class.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def _update_stats(self):
        if self.logger_callback.stage_rewards:
            self.stats["stage_rewards"].extend(
                self.logger_callback.stage_rewards
            )

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1, log_path=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.stage_rewards = []
        self.current_stage = []
        self.current_info = []
        self.current_overheat = []
        self.current_over_tolerace = []
        self.log_path = log_path

    def _on_step(self) -> bool:
        current_obs = self.model.env.get_attr("current_obs")[0]
        current_action = self.model.env.get_attr("current_action")[0]
        self.current_stage.append(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        self.current_info.append(info)
        self.current_overheat.append(info["over_heat"])
        self.current_over_tolerace.append(info["over_tolerace"])

        if not hasattr(self, 'info_sums'):
            self.info_sums = {key: 0.0 for key in info.keys()}
            self.info_counts = {key: 0 for key in info.keys()}

        for key, value in info.items():
            if isinstance(value, numbers.Real):
                self.info_sums[key] += value
                self.info_counts[key] += 1
        
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.current_stage)
            cool_power = round(np.sum(info.get("cool_power", 0)),4)
            heat_power = round(np.sum(info.get("heat_power", 0)),4)
            over_heat = np.mean(self.current_overheat)
            over_tolerace = np.mean(self.current_over_tolerace)
            self.current_overheat.clear()
            self.current_over_tolerace.clear()

            info_total = f"energy_cost: {round(info.get('energy_cost', 0), 4)}, " \
                         f"target_cost: {round(info.get('target_cost', 0), 4)}, " \
                         f"switch_cost: {round(info.get('switch_cost', 0), 4)}, " \
                         f"action_cost: {round(info.get('action_cost', 0), 4)}, " \
                         f"cool_power: {cool_power}, heat_power: {heat_power}"

            if self.log_path is None:
                print(f"Step {self.model.num_timesteps} | over_heat:{over_heat} | over_tolerace:{over_tolerace} | Reward: {mean_reward:.2f} | {info_total}", flush=True)
                print("current_obs: ", current_obs)
            else:
                log_line1 = f"Step {self.model.num_timesteps} | over_heat:{over_heat} | over_tolerace:{over_tolerace} | Reward: {mean_reward:.2f} | {info_total}"
                log_line2 = f"current_obs: {current_obs}"
                with open(self.log_path, 'a') as f:
                    f.write(log_line1 + '\n')
                    f.write(log_line2 + '\n')
                    f.flush()


            self.current_stage = []
            self.info_sums = {k:0.0 for k in self.info_sums}
            self.info_counts = {k:0 for k in self.info_counts}
            
        return True

class HVACRLTester:
    def __init__(
        self,
        model_path: str,
        algorithm: str = "ppo",
        device: str = "auto"
    ):
        self.algorithm = algorithm.lower()
        self.model_path = model_path
        self.device = device
        
        self.model = self._load_model()
    
    def _load_model(self):

        model_classes = {
            "ppo": PPO,
            "rppo": RecurrentPPO,
            "sac": SAC
        }
        
        if self.algorithm not in model_classes:
            raise ValueError(f"Unsupported model: {self.algorithm}")
        
        model_class = model_classes[self.algorithm]
        
        model = model_class.load(
            self.model_path,
            device=self.device
        )
        
        print(f"Load {self.algorithm.upper()} model: {self.model_path}")
        return model
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict actions based on observed values
        
        Param:
            obs: Observation value array (one-dimensional)
            deterministic: Whether to use a deterministic strategy
        
        Return:
            Action array (one-dimensional)
        """
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        obs = obs[np.newaxis, :]
        
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        return action[0]
