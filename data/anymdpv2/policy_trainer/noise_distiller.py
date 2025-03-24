import random

class NoiseDistillerWrapper:

    def __init__(self, env, base_policy_data, max_steps=4000):
        self.env = env
        
        # 支持传入策略数据对象或直接状态字典
        if isinstance(base_policy_data, dict) and "state_dict" in base_policy_data:
            self.base_policy_data = base_policy_data
            self.base_policy_state_dict = base_policy_data["state_dict"]
        else:
            # 向后兼容 - 如果直接传入状态字典
            self.base_policy_data = {"state_dict": base_policy_data}
            self.base_policy_state_dict = base_policy_data
        
        # 保存重要的策略元数据
        self.policy_name = self.base_policy_data.get("policy_name", "unknown")
        self.policy_kwargs = self.base_policy_data.get("policy_kwargs", {})
        
        # 如果是LSTM策略，确保保存特定配置
        if "ppo_lstm" in self.policy_name:
            self.lstm_hidden_size = self.base_policy_data.get("lstm_hidden_size", 32)
            self.n_lstm_layers = self.base_policy_data.get("n_lstm_layers", 2)
            self.enable_critic_lstm = self.base_policy_data.get("enable_critic_lstm", True)
        
        # 噪声参数
        self.noise_upper = random.uniform(0.5, 0.9)
        self.noise_lower = random.uniform(0.1, 0.4)
        if self.noise_lower > self.noise_upper:
            self.noise_lower, self.noise_upper = self.noise_upper, self.noise_lower
            
        self.noise_decay_steps = random.randint(max_steps // 4, max_steps)
        self.current_steps = 0


class NoiseDistillerPolicy:
    def __init__(self, base_policy, env, noise_params):
        self.base_policy = base_policy
        self.env = env
        self.noise_upper = noise_params["upper"]
        self.noise_lower = noise_params["lower"]
        self.noise_decay_steps = noise_params["decay_steps"]
        self.noise_decay_rate = (self.noise_upper - self.noise_lower) / self.noise_decay_steps if self.noise_decay_steps > 0 else 0
        self.current_noise = self.noise_upper
        self.steps = 0
        
    def predict(self, observation, state=None, deterministic=True):

        if self.steps < self.noise_decay_steps:
            self.current_noise -= self.noise_decay_rate
            self.current_noise = max(self.noise_lower, self.current_noise)
        self.steps += 1

        if random.random() < self.current_noise:
            return self.env.action_space.sample(), None
        else:
            if state is not None:
                return self.base_policy.predict(observation, state=state, deterministic=deterministic)
            return self.base_policy.predict(observation, deterministic=deterministic)