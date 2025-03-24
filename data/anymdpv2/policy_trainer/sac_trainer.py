import numpy as np
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
import os

class SACTrainer:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        self.model = SAC(  
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1000000,
            learning_starts=100,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    qf=[256, 256]
                ),
                activation_fn=nn.ReLU
            ),
        )
        
    def train(self, episodes=10, max_steps=1000):
        if not hasattr(self.model, '_logger'):
            tmp_path = os.path.join(os.getcwd(), "tmp")
            os.makedirs(tmp_path, exist_ok=True)
            self.model._logger = configure(
                os.path.join(tmp_path, "sac"),
                ["stdout", "csv"]
            )
        episode_count = 0
        total_steps = 0
        total_success = 0
        episode_returns = []
        
        while episode_count < episodes and total_steps < max_steps:
            state = self.env.reset()
            if isinstance(state, tuple):  
                state = state[0]
            done = False
            episode_reward = 0
            terminated = False
            
            while not done:
                try:
                    action, _ = self.model.predict(state, deterministic=False)
                    
                    step_result = self.env.step(action)
                    if len(step_result) == 5:  
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:  
                        next_state, reward, done, info = step_result
                        terminated = done
                    
                    self.model.replay_buffer.add(
                        obs=state,
                        action=action,
                        reward=reward,
                        next_obs=next_state,
                        done=done,
                        infos=[info]
                    )
                    
                    episode_reward += reward
                    total_steps += 1
                    state = next_state
                    
                except Exception as e:
                    print(f"Error during step execution for SAC: {e}")
                    done = True
                    break
            
            success = terminated and reward > 0
            total_success += int(success)
            episode_returns.append(episode_reward)
            episode_count += 1
        
        try:
            if self.model.replay_buffer.size() > self.model.batch_size:
                self.model.train(gradient_steps=1)
        except Exception as e:
            print(f"Error during training for SAC: {e}")
            import traceback
            traceback.print_exc()
        
        avg_return = np.mean(episode_returns) if episode_returns else 0
        success_rate = total_success / episode_count if episode_count > 0 else 0
        
        print(f"SAC - episodes={episode_count}, steps={total_steps}, "
              f"success_rate={success_rate:.2%}, avg_return={avg_return:.2f}")
        
        return {
            "avg_return": avg_return,
            "episode_count": episode_count,
            "total_steps": total_steps,
            "total_success": total_success,
            "episode_returns": episode_returns
        }
        
    def get_state_dict(self):
        return self.model.policy.state_dict().copy()