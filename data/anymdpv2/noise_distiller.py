import random

class NoiseDistillerWrapper:

    def __init__(self, env, base_policy_state_dict, max_steps=4000):
        self.env = env
        self.base_policy_state_dict = base_policy_state_dict
        
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
        self.noise_decay_rate = (self.noise_upper - self.noise_lower) / self.noise_decay_steps
        self.current_noise = self.noise_upper
        self.steps = 0
        
    def predict(self, state, deterministic=True):
        if self.steps < self.noise_decay_steps:
            self.current_noise -= self.noise_decay_rate
            self.current_noise = max(self.noise_lower, self.current_noise)
        self.steps += 1

        if random.random() < self.current_noise:
            return self.env.action_space.sample(), None
        else:
            return self.base_policy.predict(state, deterministic)

