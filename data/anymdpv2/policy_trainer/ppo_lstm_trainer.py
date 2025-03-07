import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

class PPO_LSTM_Trainer:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        self.model = RecurrentPPO(
            "MlpLstmPolicy",      
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs={
                "lstm_hidden_size": 32,
                "n_lstm_layers": 2,
                "enable_critic_lstm": True
            },
            clip_range=0.2,
            seed=seed,
        )
        
    def preprocess_state(self, state):
        if isinstance(state, np.ndarray):
            return state.astype(np.float32)
        elif isinstance(state, list):
            return np.array(state, dtype=np.float32)
        return state
        
    def train(self, episodes=10, max_steps=1000):
        episode_count = 0
        total_steps = 0
        total_success = 0
        episode_returns = []
        
        while episode_count < min(3, episodes) and total_steps < min(max_steps // 3, 1000):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]    
            state = self.preprocess_state(state)
            done = False
            episode_reward = 0
            lstm_states = None
            
            while not done:
                try:
                    action, lstm_states = self.model.predict(
                        state,
                        state=lstm_states,
                        deterministic=False
                    )
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, info = step_result
                    next_state = self.preprocess_state(next_state)
                    episode_reward += reward
                    total_steps += 1
                    state = next_state
                except Exception as e:
                    print(f"Error during evaluation step for PPO_LSTM: {e}")
                    done = True
                    break
            
            success = done and reward > 0
            total_success += int(success)
            episode_returns.append(episode_reward)
            episode_count += 1
        
        try:
            total_timesteps = min(max_steps, 40000)  
            
            callback = CustomCallback()
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=False
            )
            
            print(f"PPO_LSTM training completed with {total_timesteps} timesteps")
            
        except Exception as e:
            print(f"Error during PPO_LSTM training: {e}")
            import traceback
            traceback.print_exc()
        
        post_train_episode_count = 0
        post_train_total_steps = 0
        post_train_total_success = 0
        post_train_episode_returns = []
        
        while post_train_episode_count < min(episodes - episode_count, 5) and post_train_total_steps < min(max_steps - total_steps, 1000):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = self.preprocess_state(state)
            done = False
            episode_reward = 0
            lstm_states = None
            
            while not done:
                try:
                    action, lstm_states = self.model.predict(
                        state,
                        state=lstm_states,
                        deterministic=False
                    )
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, info = step_result
                    next_state = self.preprocess_state(next_state)
                    episode_reward += reward
                    post_train_total_steps += 1
                    state = next_state
                except Exception as e:
                    print(f"Error during post-training evaluation for PPO_LSTM: {e}")
                    done = True
                    break
            
            success = done and reward > 0
            post_train_total_success += int(success)
            post_train_episode_returns.append(episode_reward)
            post_train_episode_count += 1
        
        total_episode_count = episode_count + post_train_episode_count
        all_episode_returns = episode_returns + post_train_episode_returns
        total_success_count = total_success + post_train_total_success
        total_step_count = total_steps + post_train_total_steps + total_timesteps
        
        avg_return = np.mean(all_episode_returns) if all_episode_returns else 0
        success_rate = total_success_count / total_episode_count if total_episode_count > 0 else 0
        
        print(f"PPO_LSTM - episodes={total_episode_count}, steps={total_step_count}, "
              f"success_rate={success_rate:.2%}, avg_return={avg_return:.2f}")
        
        return {
            "avg_return": avg_return,
            "episode_count": total_episode_count,
            "total_steps": total_step_count,
            "total_success": total_success_count,
            "episode_returns": all_episode_returns
        }
        
    def get_state_dict(self):
        return self.model.policy.state_dict().copy()