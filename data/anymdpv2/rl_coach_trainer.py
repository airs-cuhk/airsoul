import argparse
import pickle
import numpy as np
import random
import os
from stable_baselines3 import SAC, PPO
from sb3_contrib import RecurrentPPO
import torch.nn as nn
from xenoverse.anymdpv2 import AnyMDPv2TaskSampler
from xenoverse.anymdpv2 import AnyMDPEnv
from policy_trainer.noise_distiller import NoiseDistillerWrapper, NoiseDistillerPolicy
from stable_baselines3.common.callbacks import BaseCallback
import gym
from policy_trainer.sac_trainer import SACTrainer
from policy_trainer.ppo_mlp_trainer import PPO_MLP_Trainer
from policy_trainer.ppo_lstm_trainer import PPO_LSTM_Trainer

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

class RLCoach:
    def __init__(self, env, n_epochs, mode, task, seed=None, policies_to_use=None):
        self.env = env
        self.seed = seed
        self.n_epochs = n_epochs
        self.mode = mode
        self.task = task

        if policies_to_use is None:
            policies_to_use = ["sac", "ppo_mlp", "ppo_lstm"]
        self.policies_to_use = policies_to_use

        self.policies = {
            "random": lambda x: env.action_space.sample()
        }

        if "sac" in policies_to_use:
            self.policies["sac"] = SACTrainer(env, seed).model
            
        if "ppo_mlp" in policies_to_use:
            self.policies["ppo_mlp"] = PPO_MLP_Trainer(env, seed).model
            
        if "ppo_lstm" in policies_to_use:
            self.policies["ppo_lstm"] = PPO_LSTM_Trainer(env, seed).model
        
        self.trainers = {}
        if "sac" in policies_to_use:
            self.trainers["sac"] = SACTrainer(env, seed)
            
        if "ppo_mlp" in policies_to_use:
            self.trainers["ppo_mlp"] = PPO_MLP_Trainer(env, seed)
            
        if "ppo_lstm" in policies_to_use:
            self.trainers["ppo_lstm"] = PPO_LSTM_Trainer(env, seed)
        
        self.stages = {
            "random": {"epochs": 2, "policy": "random"},
            "early": {"epochs": 3, "policy": "best_of_all"},   
            "middle": {"epochs": 3, "policy": "best_of_all"},  
            "final": {"epochs": 3, "policy": "best_of_all"},
            "finalnoisedistiller": {"epochs": 3, "policy": "noise_distilled"}
        }
        
        self.policy_snapshots = {stage: [] for stage in self.stages.keys()}

    def preprocess_state(self, state):

        if isinstance(state, np.ndarray):
            return state.astype(np.float32)
        elif isinstance(state, list):
            return np.array(state, dtype=np.float32)
        return state

    def check_env_validity(self, num_steps=10):
        """
        Check if the environment is valid by running 10 steps of RANDOM and policy policies
        and comparing their total rewards.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        print("Checking environment validity...")
        
        # Run random policy for num_steps
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        random_rewards = []
        for _ in range(num_steps):
            action = self.policies["random"](state)
            step_result = self.env.step(action)
            if len(step_result) == 5: 
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  
                next_state, reward, done, info = step_result
            random_rewards.append(reward)
            if done:
                break
            state = next_state

        compare_policy = None
        if "ppo_lstm" in self.policies_to_use:
            compare_policy = "ppo_lstm"
        elif "ppo_mlp" in self.policies_to_use:
            compare_policy = "ppo_mlp"
        elif "sac" in self.policies_to_use:
            compare_policy = "sac"
        else:
            print("No RL policies available for validation. Considering environment as valid.")
            return True
            
        print(f"Using {compare_policy.upper()} for environment validation")
        
        # Reset and run the chosen policy for num_steps
        state = self.env.reset()
        if isinstance(state, tuple):  
            state = state[0]
        policy_rewards = []
        lstm_states = None
        
        for _ in range(num_steps):
            if compare_policy == "ppo_lstm":
                action, lstm_states = self.policies[compare_policy].predict(
                    state, 
                    state=lstm_states,  
                    deterministic=False
                )
            else:
                action, _ = self.policies[compare_policy].predict(state, deterministic=False)
                
            step_result = self.env.step(action)
            if len(step_result) == 5: 
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  
                next_state, reward, done, info = step_result
            policy_rewards.append(reward)
            if done:
                break
            state = next_state
        
        # Compare total rewards
        random_total = sum(random_rewards)
        policy_total = sum(policy_rewards)
        
        if policy_total - random_total <= max(3.0 * np.std(random_rewards), 1e-3):
            print(f"Environment invalid: no significant improvements for RL")
            print(f"Random reward: {random_total}, {compare_policy.upper()} reward: {policy_total}")
            return False
        
        print(f"Environment valid - Random={random_total}, {compare_policy.upper()}={policy_total}")
        return True
                
    def train_policies(self, max_steps_per_epoch, max_episodes_per_epoch):
        """
        训练各阶段的策略并保存快照
        
        Args:
            max_steps_per_epoch: 每个epoch的最大步数
            max_episodes_per_epoch: 每个epoch的最大episode数
        
        Returns:
            bool: 训练是否成功
        """
        print("Training policies by stages...")

        from stable_baselines3.common.logger import configure
        tmp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        os.makedirs(tmp_path, exist_ok=True)

        self.env_info = {
            "observation_space": str(self.env.observation_space),
            "action_space": str(self.env.action_space),
            "state_dim": getattr(self.task, 'state_dim', None),
            "action_dim": getattr(self.task, 'action_dim', None),
            "ndim": getattr(self.task, 'ndim', None),
        }
        
        print(f"Environment configuration:")
        print(f"  Observation space: {self.env.observation_space}")
        print(f"  Action space: {self.env.action_space}")
        print(f"  State dim: {getattr(self.task, 'state_dim', None)}")
        print(f"  Action dim: {getattr(self.task, 'action_dim', None)}")
        print(f"  Ndim: {getattr(self.task, 'ndim', None)}")

        self.trainer_configs = {}
        for policy_name, trainer in self.trainers.items():
            if policy_name == "ppo_lstm":
                model = trainer.model
                policy_kwargs = getattr(model, 'policy_kwargs', {})
                self.trainer_configs[policy_name] = {
                    "policy_kwargs": policy_kwargs,
                    "lstm_hidden_size": policy_kwargs.get("lstm_hidden_size", 32),
                    "n_lstm_layers": policy_kwargs.get("n_lstm_layers", 2),
                    "enable_critic_lstm": policy_kwargs.get("enable_critic_lstm", True)
                }
            elif policy_name == "ppo_mlp":
                model = trainer.model
                self.trainer_configs[policy_name] = {
                    "policy_kwargs": getattr(model, 'policy_kwargs', {})
                }
            elif policy_name == "sac":
                model = trainer.model
                self.trainer_configs[policy_name] = {
                    "policy_kwargs": getattr(model, 'policy_kwargs', {})
                }
        
        print(f"Trainer configurations:")
        for name, config in self.trainer_configs.items():
            print(f"  {name}: {config}")

        for stage, config in self.stages.items():
            print(f"\n{'='*50}")
            print(f"Training stage: {stage}")
            print(f"{'='*50}")
            policy_name = config["policy"]

            if policy_name == "random":
                print(f"Random policy stage - generating {config['epochs']} snapshots")
                for epoch in range(config["epochs"]):
                    state = self.env.reset()
                    if isinstance(state, tuple):
                        state = state[0]
                    rewards = []
                    done = False
                    steps = 0
                    max_steps = min(max_steps_per_epoch, 1000)  

                    while not done and steps < max_steps:
                        action = self.policies["random"](state)
                        step_result = self.env.step(action)
                        if len(step_result) == 5:
                            next_state, reward, terminated, truncated, info = step_result
                            done = terminated or truncated
                        else:
                            next_state, reward, done, info = step_result
                        rewards.append(reward)
                        state = next_state
                        steps += 1

                    policy_data = {
                        "stage": stage,
                        "epoch": epoch,
                        "policy_name": "random",
                        "state_dict": None,  
                        "avg_reward": sum(rewards) if rewards else 0,
                        "steps": steps
                    }
                    self.policy_snapshots[stage].append(policy_data)
                    print(f"Added random policy snapshot {epoch} - avg_reward: {policy_data['avg_reward']:.2f}, steps: {steps}")
                
                continue

            elif policy_name == "noise_distilled":
                print("Creating noise distiller policies based on final stage policies...")
                if not self.policy_snapshots["final"]:
                    print("No final stage policies available. Skipping finalnoisedistiller stage.")
                    continue

                for epoch in range(config["epochs"]):
                    base_policy_data = random.choice(self.policy_snapshots["final"])
                    print(f"Creating noise distiller for epoch {epoch} based on {base_policy_data['policy_name']}")

                    try:
                        noise_distiller = NoiseDistillerWrapper(
                            self.env, 
                            base_policy_data,  
                            max_steps=max_steps_per_epoch
                        )

                        policy_data = {
                            "stage": "finalnoisedistiller",
                            "epoch": epoch,
                            "policy_name": f"noise_distilled_{base_policy_data['policy_name']}",
                            "state_dict": base_policy_data["state_dict"],
                            "noise_params": {
                                "upper": noise_distiller.noise_upper,
                                "lower": noise_distiller.noise_lower,
                                "decay_steps": noise_distiller.noise_decay_steps
                            }
                        }
                        
                        if "policy_kwargs" in base_policy_data:
                            policy_data["policy_kwargs"] = base_policy_data["policy_kwargs"]
                        
                        if base_policy_data["policy_name"] == "ppo_lstm":
                            lstm_config = self.trainer_configs.get("ppo_lstm", {})
                            policy_data["lstm_hidden_size"] = lstm_config.get("lstm_hidden_size", 32)
                            policy_data["n_lstm_layers"] = lstm_config.get("n_lstm_layers", 2)
                            policy_data["enable_critic_lstm"] = lstm_config.get("enable_critic_lstm", True)
                        
                        self.policy_snapshots["finalnoisedistiller"].append(policy_data)
                        print(f"Created noise distiller policy {epoch} based on {base_policy_data['policy_name']}")
                        
                        state = self.env.reset()
                        if isinstance(state, tuple):
                            state = state[0]
                        rewards = []
                        done = False
                        steps = 0
                        max_steps = min(max_steps_per_epoch, 1000)
                        
                        noise_policy = NoiseDistillerPolicy(
                            None,  
                            self.env, 
                            policy_data["noise_params"]
                        )

                        base_policy_name = base_policy_data["policy_name"]
                        if base_policy_name == "ppo_lstm":
                            lstm_config = {
                                "lstm_hidden_size": policy_data.get("lstm_hidden_size", 32),
                                "n_lstm_layers": policy_data.get("n_lstm_layers", 2),
                                "enable_critic_lstm": policy_data.get("enable_critic_lstm", True)
                            }
                            base_policy = RecurrentPPO(
                                "MlpLstmPolicy",
                                self.env,
                                verbose=0,
                                policy_kwargs=lstm_config
                            )
                        elif base_policy_name == "ppo_mlp":
                            base_policy = PPO(
                                "MlpPolicy",
                                self.env,
                                verbose=0
                            )
                        else:  # sac
                            base_policy = SAC(
                                "MlpPolicy",
                                self.env,
                                verbose=0
                            )

                        base_policy.policy.load_state_dict(base_policy_data["state_dict"])
                        noise_policy.base_policy = base_policy
                        
                        lstm_states = None
                        while not done and steps < max_steps:
                            if base_policy_name == "ppo_lstm":
                                action, lstm_states = noise_policy.predict(
                                    state, 
                                    state=lstm_states,
                                    deterministic=True
                                )
                            else:
                                action, _ = noise_policy.predict(state, deterministic=True)
                                
                            step_result = self.env.step(action)
                            if len(step_result) == 5:
                                next_state, reward, terminated, truncated, info = step_result
                                done = terminated or truncated
                            else:
                                next_state, reward, done, info = step_result
                            rewards.append(reward)
                            state = next_state
                            steps += 1
                        
                        print(f"Noise distiller {epoch} test - avg_reward: {sum(rewards)/len(rewards) if rewards else 0:.2f}, steps: {steps}")
                        
                    except Exception as e:
                        print(f"Error creating noise distiller for epoch {epoch}: {e}")
                        import traceback
                        traceback.print_exc()
                
                continue

            elif policy_name == "best_of_all":
                print(f"Training best-of-all policies for stage: {stage}, epochs: {config['epochs']}")
                for epoch in range(config["epochs"]):
                    print(f"\n{'-'*40}")
                    print(f"Epoch {epoch} in stage {stage}:")
                    print(f"{'-'*40}")
                    results = {}

                    for p_name in self.policies_to_use:
                        try:
                            print(f"Training {p_name.upper()}...")
                            train_result = self.trainers[p_name].train(
                                episodes=max_episodes_per_epoch, 
                                max_steps=max_steps_per_epoch
                            )
                            
                            results[p_name] = train_result
                            
                            avg_return = train_result["avg_return"]
                            episode_count = train_result["episode_count"]
                            total_steps = train_result["total_steps"]
                            total_success = train_result["total_success"]
                            
                            print(f"{p_name.upper()} - Epoch {epoch}: episodes={episode_count}, steps={total_steps}, "
                                f"success_rate={(total_success/episode_count if episode_count > 0 else 0):.2%}, "
                                f"avg_return={avg_return:.2f}")
                        
                        except Exception as e:
                            print(f"Unexpected error for {p_name}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                    if results:  
                        best_policy = max(results.keys(), key=lambda k: results[k]["avg_return"])
                        best_policy_model = self.trainers[best_policy].model
                        state_dict = self.trainers[best_policy].get_state_dict()

                        policy_data = {
                            "stage": stage,
                            "epoch": epoch,
                            "policy_name": best_policy,
                            "state_dict": state_dict,
                            "avg_return": results[best_policy]["avg_return"],
                            "success_rate": results[best_policy]["total_success"] / results[best_policy]["episode_count"] if results[best_policy]["episode_count"] > 0 else 0,
                            "episodes": results[best_policy]["episode_count"],
                            "steps": results[best_policy]["total_steps"]
                        }

                        if best_policy == "ppo_lstm":
                            lstm_config = self.trainer_configs.get("ppo_lstm", {})
                            policy_data["policy_kwargs"] = lstm_config.get("policy_kwargs", {})
                            policy_data["lstm_hidden_size"] = lstm_config.get("lstm_hidden_size", 32)
                            policy_data["n_lstm_layers"] = lstm_config.get("n_lstm_layers", 2)
                            policy_data["enable_critic_lstm"] = lstm_config.get("enable_critic_lstm", True)
                        else:
                            policy_kwargs = getattr(best_policy_model, 'policy_kwargs', {})
                            if policy_kwargs:
                                policy_data["policy_kwargs"] = policy_kwargs

                        self.policy_snapshots[stage].append(policy_data)
                        print(f"Epoch {epoch} best policy: {best_policy.upper()} with avg_return: {results[best_policy]['avg_return']:.2f}")

                        print(f"Saved policy configuration:")
                        for k, v in policy_data.items():
                            if k != "state_dict":  
                                print(f"  {k}: {v}")
                    else:
                        print("No policies successfully completed training in this epoch")

            else:
                print(f"Unknown policy type: {policy_name}")

        total_snapshots = sum(len(snapshots) for snapshots in self.policy_snapshots.values())
        if total_snapshots == 0:
            print("Warning: No policy snapshots were created during training.")
            return False
        
        print(f"\nTraining completed successfully. Generated {total_snapshots} policy snapshots across all stages.")

        for stage, snapshots in self.policy_snapshots.items():
            if snapshots:
                policies_by_type = {}
                for snap in snapshots:
                    policy_type = snap["policy_name"]
                    if policy_type not in policies_by_type:
                        policies_by_type[policy_type] = 0
                    policies_by_type[policy_type] += 1
                
                print(f"Stage {stage}: {len(snapshots)} snapshots - {policies_by_type}")
        
        return True

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        behavior_policies = {}
        reference_policies = {}
        
        for stage in self.stages.keys():
            if len(self.policy_snapshots[stage]) > 0:
                behavior_policies[stage] = self.policy_snapshots[stage]  

        if len(self.policy_snapshots["final"]) > 0:
            reference_policies["final"] = self.policy_snapshots["final"][-1]
        
        save_data = {
            "mode": self.mode,  
            "task_config": {    
                "mode": self.mode,
                "ndim": self.task.ndim if hasattr(self.task, 'ndim') else None,
                "state_dim": self.task.state_dim if hasattr(self.task, 'state_dim') else None,
                "action_dim": self.task.action_dim if hasattr(self.task, 'action_dim') else None
            },
            "env_info": getattr(self, 'env_info', {}),  
            "trainer_configs": getattr(self, 'trainer_configs', {}),  
            "behavior_policies": behavior_policies,
            "reference_policies": reference_policies
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Coach saved to {path}")
        # print(f"Saved {sum(len(policies) for policies in behavior_policies.values())} behavior policies")
        # print(f"Saved {len(reference_policies)} reference policies")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save the trained coaches")
    parser.add_argument("--mode", type=str, required=False, choices=["static", "dynamic", "universal"], help="Mode for task sampler.")
    parser.add_argument("--state_dim", type=int, default=256, help="state dimension")
    parser.add_argument("--action_dim", type=int, default=256, help="action dimension")
    parser.add_argument("--ndim", type=int, default=8, help="ndim for task sampler")
    parser.add_argument("--max_steps", type=int, default=4000, help="maximum steps per epoch")
    parser.add_argument("--max_episodes", type=int, default=20, help="maximum episodes per epoch")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--policy", type=str, nargs='+', choices=["sac", "ppo_mlp", "ppo_lstm"], 
                        help="Specify which policies to use. If not specified, all policies will be used.")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.mode is not None:
        modes = [args.mode]
    else:
        modes = ["static", "dynamic", "universal"]
    
    for mode in modes:
        print(f"\nTraining coach for mode: {mode} ")
        
        training_successful = False
        max_attempts = 100  
        attempt = 0
        
        while not training_successful and attempt < max_attempts:
            attempt += 1
            print(f"Attempt {attempt} for mode {mode}")
            
            env = gym.make("anymdp-v2-visualizer")
            task = AnyMDPv2TaskSampler(
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                seed=args.seed
            )
            env.set_task(task)
            
            coach = RLCoach(env, args.n_epochs, mode=mode, seed=args.seed, task=task, policies_to_use=args.policy)
            
            # Check environment validity before training
            env_valid = coach.check_env_validity(num_steps=10)
            if not env_valid:
                print("This environment is invalid because RANDOM and RL policies produce similar rewards. Rebuilding env and task...")
                continue
                
            # If we reach here, the environment is valid, so proceed with training
            training_successful = coach.train_policies(args.max_steps, args.max_episodes)
            if not training_successful:
                print("Training unsuccessful. Rebuilding env and task...")
            else:
                print("Task available. Training completed successfully.")
        
        if not training_successful:
            print(f"After attempting {max_attempts} times, mode {mode} fails to generate available task. Skipping this mode.")
            continue
        
        save_path = os.path.join(args.save_dir, f"coach_{mode}.pkl")
        coach.save(save_path)