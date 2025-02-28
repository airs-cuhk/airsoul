import argparse
import pickle
import numpy
import random
import os
from stable_baselines3 import SAC,PPO
from sb3_contrib import RecurrentPPO
import torch.nn as nn
from l3c.anymdpv2 import AnyMDPv2TaskSampler
from l3c.anymdpv2 import AnyMDPEnv
from noise_distiller import NoiseDistillerWrapper, NoiseDistillerPolicy
class RLCoach:
    def __init__(self, env, n_epochs, mode, seed=None):
        self.env = env
        self.seed = seed
        self.n_epochs = n_epochs
        self.mode = mode

        self.policies = {
            "random": lambda x: env.action_space.sample(),
            "sac": SAC(  
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
            ),
            "ppo_mlp": PPO(
                "MlpPolicy",      
                env,
                verbose=1,
                learning_rate=3e-4,
                batch_size=64,
                gamma=0.99,
            ),
            "ppo_lstm": RecurrentPPO(
                "MlpLstmPolicy",      
                env,
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
        }
        
        self.stages = {
            "random": {"epochs": 2, "policy": "random"},
            "early": {"epochs": 3, "policy": "best_of_all"},   
            "middle": {"epochs": 3, "policy": "best_of_all"},  
            "final": {"epochs": 3, "policy": "best_of_all"},
            "finalnoisedistiller": {"epochs": 3, "policy": "noise_distilled"}
        }
        
        self.policy_snapshots = {stage: [] for stage in self.stages.keys()}
        
    def check_env_validity(self, num_steps=10):
        """
        Check if the environment is valid by running 10 steps of RANDOM and PPO_MLP policies
        and comparing their total rewards.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        print("Checking environment validity...")
        
        # Run random policy for num_steps
        state, info = self.env.reset()
        random_rewards = []
        for _ in range(num_steps):
            action = self.policies["random"](state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            random_rewards.append(reward)
            if terminated or truncated:
                break
            state = next_state
        
        # Reset and run PPO_MLP policy for num_steps
        state, info = self.env.reset()
        ppo_mlp_rewards = []
        for _ in range(num_steps):
            action, _ = self.policies["ppo_mlp"].predict(state, deterministic=False)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            ppo_mlp_rewards.append(reward)
            if terminated or truncated:
                break
            state = next_state
        
        # Compare total rewards
        random_total = sum(random_rewards)
        ppo_mlp_total = sum(ppo_mlp_rewards)
        
        if random_total == ppo_mlp_total:
            print(f"Environment invalid - both policies earned the same reward: {random_total}")
            return False
        
        print(f"Environment valid - different rewards: Random={random_total}, PPO_MLP={ppo_mlp_total}")
        return True
                
    def train_policies(self, max_steps_per_epoch, max_episodes_per_epoch):
        print("Training policies by stages...")
        for stage, config in self.stages.items():
            print(f"\nTraining stage: {stage}")
            policy_name = config["policy"]
            
            # 随机策略处理
            if policy_name == "random":
                for epoch in range(config["epochs"]):
                    state, info = self.env.reset()
                    rewards = []
                    done = False

                    while not done:
                        action = self.policies["random"](state)
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        rewards.append(reward)
                        done = terminated or truncated
                        state = next_state

                for epoch in range(config["epochs"]):
                    self.policy_snapshots[stage].append({
                        "stage": stage,
                        "epoch": epoch,
                        "policy_name": "random",
                        "state_dict": None
                    })
                continue

            elif policy_name == "noise_distilled":
                print("Creating noise distiller policies based on final stage policies...")
                if not self.policy_snapshots["final"]:
                    print("No final stage policies available. Skipping finalnoisedistiller stage.")
                    continue

                for epoch in range(config["epochs"]):
                    base_policy_data = random.choice(self.policy_snapshots["final"])

                    noise_distiller = NoiseDistillerWrapper(
                        self.env, 
                        base_policy_data["state_dict"],
                        max_steps=max_steps_per_epoch
                    )

                    self.policy_snapshots["finalnoisedistiller"].append({
                        "stage": "finalnoisedistiller",
                        "epoch": epoch,
                        "policy_name": f"noise_distilled_{base_policy_data['policy_name']}",
                        "state_dict": base_policy_data["state_dict"],
                        "noise_params": {
                            "upper": noise_distiller.noise_upper,
                            "lower": noise_distiller.noise_lower,
                            "decay_steps": noise_distiller.noise_decay_steps
                        }
                    })
                    print(f"Created noise distiller policy {epoch} based on {base_policy_data['policy_name']}")
                continue

            elif policy_name == "best_of_all":
                for epoch in range(config["epochs"]):
                    print(f"\nEpoch {epoch} in stage {stage}:")
                    results = {}

                    for p_name in ["sac", "ppo_lstm", "ppo_mlp"]:
                        episode_count = 0
                        total_steps = 0
                        total_success = 0
                        episode_returns = []
                        
                        while episode_count < max_episodes_per_epoch and total_steps < max_steps_per_epoch:
                            state, info = self.env.reset()
                            done = False
                            episode_reward = 0
                            while not done:
                                action, _ = self.policies[p_name].predict(state, deterministic=False)
                                next_state, reward, terminated, truncated, info = self.env.step(action)
                                done = terminated or truncated
                                if p_name == "sac":
                                    self.policies[p_name].replay_buffer.add(
                                        obs=state,
                                        action=action,
                                        reward=reward,
                                        next_obs=next_state,
                                        done=done
                                    )
                                episode_reward += reward
                                total_steps += 1
                                state = next_state
                            success = terminated and reward > 0
                            total_success += int(success)
                            episode_returns.append(episode_reward)
                            episode_count += 1
                        
                        self.policies[p_name].train()
                        
                        avg_return = numpy.mean(episode_returns) if episode_returns else 0
                        results[p_name] = {
                            "avg_return": avg_return,
                            "episode_count": episode_count,
                            "total_steps": total_steps,
                            "total_success": total_success,
                            "episode_returns": episode_returns
                        }
                        print(f"{p_name.upper()} - Epoch {epoch}: episodes={episode_count}, steps={total_steps}, "
                              f"success_rate={(total_success/episode_count if episode_count > 0 else 0):.2%}, "
                              f"avg_return={avg_return:.2f}")

                    best_policy = max(results.keys(), key=lambda k: results[k]["avg_return"])
                    state_dict = self.policies[best_policy].policy.state_dict().copy()

                    self.policy_snapshots[stage].append({
                        "stage": stage,
                        "epoch": epoch,
                        "policy_name": best_policy,
                        "state_dict": state_dict
                    })
                    print(f"Epoch {epoch} best policy: {best_policy.upper()} with avg_return: {results[best_policy]['avg_return']:.2f}")
            
            else:
                print(f"Unknown policy type: {policy_name}")

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
                "ndim": self.env.task.ndim,
                "state_dim": self.env.task.state_dim,
                "action_dim": self.env.task.action_dim
            },
            "behavior_policies": behavior_policies,
            "reference_policies": reference_policies
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True,
                        help="directory to save the trained coaches")
    parser.add_argument("--mode", type=str, required=False, choices=["static", "dynamic", "universal"],
                        help="Mode for task sampler.")
    parser.add_argument("--state_dim", type=int, default=256,
                        help="state dimension")
    parser.add_argument("--action_dim", type=int, default=256,
                        help="action dimension")
    parser.add_argument("--ndim", type=int, default=8,
                        help="ndim for task sampler")
    parser.add_argument("--max_steps", type=int, default=4000,
                        help="maximum steps per epoch")
    parser.add_argument("--max_episodes", type=int, default=20,
                        help="maximum episodes per epoch")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
    
    if args.mode is not None:
        modes = [args.mode]
    else:
        modes = ["static", "dynamic", "universal"]
    
    for mode in modes:
        print(f"\nTraining coach for mode: {mode} ")
        
        training_successful = False
        max_attempts = 10  
        attempt = 0
        
        while not training_successful and attempt < max_attempts:
            attempt += 1
            print(f"Attempt {attempt} for mode {mode}")
            
            env = AnyMDPEnv(max_steps=args.max_steps)
            task = AnyMDPv2TaskSampler(
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                ndim=args.ndim,
                mode=mode,
                seed=args.seed
            )
            env.set_task(task)
            
            coach = RLCoach(env, args.n_epochs, mode=mode, seed=args.seed)
            
            # Check environment validity before training
            env_valid = coach.check_env_validity(num_steps=10)
            if not env_valid:
                print("This environment is invalid because RANDOM and PPO_MLP policies produce the same total reward. Rebuilding env and task...")
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