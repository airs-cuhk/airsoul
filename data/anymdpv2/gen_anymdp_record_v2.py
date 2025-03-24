import argparse
import os
import numpy as np
import random
import torch
import multiprocessing
from l3c.anymdpv2 import AnyMDPv2TaskSampler, AnyMDPEnv
from tag_vocab import tag_mapping_id
from stable_baselines3 import SAC, PPO
from sb3_contrib import RecurrentPPO
import pickle
from policy_trainer.noise_distiller import NoiseDistillerWrapper, NoiseDistillerPolicy
import gym
from policy_trainer.sac_trainer import SACTrainer
from policy_trainer.ppo_mlp_trainer import PPO_MLP_Trainer
from policy_trainer.ppo_lstm_trainer import PPO_LSTM_Trainer
import gc

def create_directory(path):
    os.makedirs(path, exist_ok=True)

class DataGenerator:
    def __init__(self, coach_path, mode, state_dim, action_dim, ndim, max_steps, seed=None, policies_to_use=None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if mode is None:
            self.mode = random.choice(["static", "dynamic", "universal"])
        else:
            self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ndim = ndim
        self.max_steps = max_steps

        self.env = gym.make("anymdp-v2-visualizer")
        self.task = AnyMDPv2TaskSampler(
            state_dim=state_dim,
            action_dim=action_dim, 
            ndim=ndim,
            mode=mode,
            seed=seed,
            verbose=False
        )
        self.env.set_task(self.task)

        if policies_to_use is None:
            policies_to_use = ["sac", "ppo_mlp", "ppo_lstm"]
        self.policies_to_use = policies_to_use

        self.policies = {
            "random": lambda x: self.env.action_space.sample()
        }

        if "sac" in policies_to_use:
            self.policies["sac"] = SACTrainer(self.env, seed).model
            
        if "ppo_mlp" in policies_to_use:
            self.policies["ppo_mlp"] = PPO_MLP_Trainer(self.env, seed).model
            
        if "ppo_lstm" in policies_to_use:
            self.policies["ppo_lstm"] = PPO_LSTM_Trainer(self.env, seed).model

        if os.path.isdir(coach_path):
            coach_dir = coach_path
        else:
            coach_dir = os.path.dirname(coach_path)
        
        if not coach_dir:
            coach_dir = "."
            
        coach_file = os.path.join(coach_dir, f"coach_{self.mode}.pkl")
        print(f"Looking for coach file at: {coach_file}")
        
        if not os.path.exists(coach_file):
            raise ValueError(f"No coach found for mode {self.mode} at path: {coach_file}")

        with open(coach_file, 'rb') as f:
            data = pickle.load(f)

        if data["mode"] != self.mode:
            raise ValueError(
                f"Coach mode ({data['mode']}) does not match environment mode ({self.mode})"
            )
        
        self.behavior_policies = data["behavior_policies"]
        self.reference_policies = data["reference_policies"]
        self.task_config = data["task_config"]

        self.env_info = data.get("env_info", {})
        self.trainer_configs = data.get("trainer_configs", {})

        print(f"Loaded coach with the following configuration:")
        print(f"  Mode: {self.mode}")
        print(f"  Task config: {self.task_config}")
        if self.env_info:
            print(f"  Environment info: {self.env_info}")
        if self.trainer_configs:
            print(f"  Trainer configs: {self.trainer_configs}")

        self.mask_all_tag_prob = 0.15
        self.mask_epoch_tag_prob = 0.15

        def create_stage_policy(stage_policies):
            def stage_policy(state, lstm_states=None):
                policy_data = random.choice(stage_policies)
                if policy_data["policy_name"] == "random":
                    return self.env.action_space.sample(), None
                        
                elif "noise_distilled_" in policy_data["policy_name"]:
                    base_policy_name = policy_data["policy_name"].replace("noise_distilled_", "")

                    if base_policy_name == "ppo_lstm":
                        lstm_hidden_size = policy_data.get("lstm_hidden_size", 32)
                        n_lstm_layers = policy_data.get("n_lstm_layers", 2)
                        enable_critic_lstm = policy_data.get("enable_critic_lstm", True)
                        
                        policy_kwargs = {
                            "lstm_hidden_size": lstm_hidden_size,
                            "n_lstm_layers": n_lstm_layers,
                            "enable_critic_lstm": enable_critic_lstm
                        }
                        
                        base_policy = RecurrentPPO(
                            "MlpLstmPolicy",
                            self.env,
                            verbose=0,
                            policy_kwargs=policy_kwargs
                        )
                    elif base_policy_name == "ppo_mlp":
                        base_policy = PPO(
                            "MlpPolicy",
                            self.env,
                            verbose=0
                        )
                    else:  
                        base_policy = SAC(
                            "MlpPolicy",
                            self.env,
                            verbose=0
                        )
                    
                    try:
                        base_policy.policy.load_state_dict(policy_data["state_dict"])
                    except Exception as e:
                        print(f"Error loading state dict for {policy_data['policy_name']}: {e}")
                        print("Policy kwargs:", policy_kwargs if 'policy_kwargs' in locals() else "Not defined")
                        return self.env.action_space.sample(), None
                    
                    noise_policy = NoiseDistillerPolicy(
                        base_policy, 
                        self.env, 
                        policy_data["noise_params"]
                    )
                    
                    if base_policy_name == "ppo_lstm":
                        return noise_policy.predict(state, state=lstm_states, deterministic=True)
                    return noise_policy.predict(state, deterministic=True)
                        
                else:
                    if policy_data["policy_name"] == "ppo_lstm":
                        lstm_hidden_size = policy_data.get("lstm_hidden_size", 32)
                        n_lstm_layers = policy_data.get("n_lstm_layers", 2)
                        enable_critic_lstm = policy_data.get("enable_critic_lstm", True)
                        
                        policy_kwargs = {
                            "lstm_hidden_size": lstm_hidden_size,
                            "n_lstm_layers": n_lstm_layers,
                            "enable_critic_lstm": enable_critic_lstm
                        }
                        
                        try:
                            policy = RecurrentPPO(
                                "MlpLstmPolicy",
                                self.env,
                                verbose=0,
                                policy_kwargs=policy_kwargs
                            )
                            policy.policy.load_state_dict(policy_data["state_dict"])
                            return policy.predict(state, state=lstm_states, deterministic=True)
                        except Exception as e:
                            print(f"Error creating RecurrentPPO: {e}")
                            return self.env.action_space.sample(), None
                            
                    elif policy_data["policy_name"] == "ppo_mlp":
                        policy = PPO(
                            "MlpPolicy",
                            self.env,
                            verbose=0
                        )
                        policy.policy.load_state_dict(policy_data["state_dict"])
                        return policy.predict(state, deterministic=True)
                            
                    else:  # sac
                        policy = SAC(
                            "MlpPolicy",
                            self.env,
                            verbose=0
                        )
                        policy.policy.load_state_dict(policy_data["state_dict"])
                        return policy.predict(state, deterministic=True)
                            
            return stage_policy

        self.stages = ["random", "early", "middle", "final", "finalnoisedistiller"]
        
        self.behavior_dict = [
            (create_stage_policy(self.behavior_policies["random"]), 0.10),
            (create_stage_policy(self.behavior_policies["early"]), 0.10),
            (create_stage_policy(self.behavior_policies["middle"]), 0.10),
            (create_stage_policy(self.behavior_policies["final"]), 0.35),
            (create_stage_policy(self.behavior_policies["finalnoisedistiller"]), 0.35),
        ]

        self.reference_dict = [
            (create_stage_policy([self.reference_policies["final"]]), 1.0)    
        ]
        
        self.blist, bprob = zip(*self.behavior_dict)
        self.rlist, rprob = zip(*self.reference_dict)
        
        self.bprob = np.cumsum(bprob)
        self.bprob /= self.bprob[-1]
        self.rprob = np.cumsum(rprob)
        self.rprob /= self.rprob[-1]
    
    def reset_env_and_task(self):
        if hasattr(self, 'env'):
            self.env.close()
            del self.env
        if hasattr(self, 'task'):
            del self.task

        import gc
        gc.collect()
        print("Reinitializing environment and task...")
        self.env = gym.make("anymdp-v2-visualizer")
        self.task = AnyMDPv2TaskSampler(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            ndim=self.ndim,
            mode=self.mode,
            seed=self.seed,
            verbose=False
        )
        self.env.set_task(self.task)
        
        # Reinitialize PPO_MLP policy for the new environment
        self.ppo_lstm = RecurrentPPO(
            "MlpLstmPolicy",      
            self.env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99,
        )
    
    def load_policy(self, policy_data):
        if policy_data["policy_name"] == "random":
            return lambda x: self.env.action_space.sample(), None
            
        elif "noise_distilled_" in policy_data["policy_name"]:
            base_policy_name = policy_data["policy_name"].replace("noise_distilled_", "")
            base_policy = self.create_base_policy(base_policy_name)
            base_policy.policy.load_state_dict(policy_data["state_dict"])
            
            noise_policy = NoiseDistillerPolicy(
                base_policy, 
                self.env, 
                policy_data["noise_params"]
            )
            return lambda x: noise_policy.predict(x, deterministic=True)
            
        else:
            policy = self.create_base_policy(policy_data["policy_name"])
            policy.policy.load_state_dict(policy_data["state_dict"])
            return lambda x: policy.predict(x, deterministic=True)

    def create_base_policy(self, policy_name):
        if policy_name == "ppo_lstm":
            return RecurrentPPO(
                        "MlpLstmPolicy", 
                        self.env, 
                        verbose=0,
                        policy_kwargs={
                            "lstm_hidden_size": 32,
                            "n_lstm_layers": 2,
                            "enable_critic_lstm": True
                        }
                    )
        elif policy_name == "ppo_mlp":
            return PPO("MlpPolicy", self.env, verbose=0)
        else:  # sac
            return SAC("MlpPolicy", self.env, verbose=0)

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

    def sample_behavior_policy(self):
        return self.blist[np.searchsorted(self.bprob, random.random())]
    
    def sample_reference_policy(self):
        return self.rlist[np.searchsorted(self.rprob, random.random())]
    
    def generate_data(self, epoch_id, max_steps):
        # Validate environment before generating data
        valid_env = False
        max_attempts = 100
        attempts = 0
        
        while not valid_env and attempts < max_attempts:
            attempts += 1
            print(f"Environment validation attempt {attempts}/{max_attempts}")
            valid_env = self.check_env_validity(num_steps=10)
            if not valid_env:
                print("Invalid environment detected. Recreating environment and task...")
                self.reset_env_and_task()
        
        if not valid_env:
            print(f"Failed to create a valid environment after {max_attempts} attempts. Aborting data generation.")
            return None
        
        print("Valid environment confirmed. Proceeding with data generation...")
        
        all_data = {
            "states": [],
            "actions_behavior": [],
            "actions_label": [],
            "rewards": [],
            "prompts": [],
            "tags": []
        }
        
        mask_all_tag = (random.random() < self.mask_all_tag_prob)
        mask_epoch_tag = (random.random() < self.mask_epoch_tag_prob)
        
        steps = 0
        total_reward = 0
        
        init_state = self.env.reset()
        if isinstance(init_state, tuple):
            init_state = init_state[0]
        
        state_shape = init_state.shape if isinstance(init_state, np.ndarray) else (self.state_dim,)
        action_shape = (self.action_dim,)
        
        print(f"State shape: {state_shape}, Action shape: {action_shape}")
        
        while steps < max_steps:
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
                    
            if not isinstance(state, np.ndarray) or state.shape != state_shape:
                state = np.reshape(state, state_shape) if hasattr(state, 'size') else np.zeros(state_shape)

            behavior_idx = np.searchsorted(self.bprob, random.random())
            behavior_policy = self.blist[behavior_idx]

            current_stage = self.stages[behavior_idx] if behavior_idx < len(self.stages) else "unknown"
            
            print(f"Using {current_stage} policy")
            
            done = False
            lstm_states = None  
            
            while not done and steps < max_steps:
                if isinstance(state, np.ndarray) and np.isnan(state).any():
                    print(f"Warning: NaN values in state at step {steps}, replacing with zeros")
                    state = np.zeros(state_shape)
                    
                if "ppo_lstm" in current_stage or "lstm" in current_stage.lower():
                    behavior_action, lstm_states = behavior_policy(state, lstm_states=lstm_states)
                else:
                    behavior_action, _ = behavior_policy(state)
                        
                reference_action, _ = self.sample_reference_policy()(state)
                    
                if not isinstance(behavior_action, np.ndarray) or behavior_action.shape != action_shape:
                    behavior_action = np.reshape(behavior_action, action_shape) if hasattr(behavior_action, 'size') else np.zeros(action_shape)
                    
                if not isinstance(reference_action, np.ndarray) or reference_action.shape != action_shape:
                    reference_action = np.reshape(reference_action, action_shape) if hasattr(reference_action, 'size') else np.zeros(action_shape)
                    
                if np.isnan(behavior_action).any():
                    print(f"Warning: NaN values in behavior_action at step {steps}, replacing with zeros")
                    behavior_action = np.zeros(action_shape)
                    
                if np.isnan(reference_action).any():
                    print(f"Warning: NaN values in reference_action at step {steps}, replacing with zeros")
                    reference_action = np.zeros(action_shape)
                    
                step_result = self.env.step(behavior_action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result
                    
                if np.isnan(reward):
                    print(f"Warning: NaN reward at step {steps}, replacing with 0.0")
                    reward = 0.0
                    
                if not isinstance(next_state, np.ndarray) or next_state.shape != state_shape:
                    next_state = np.reshape(next_state, state_shape) if hasattr(next_state, 'size') else np.zeros(state_shape)
                    
                if mask_all_tag or mask_epoch_tag:
                    tag = tag_mapping_id['unknown']
                else:
                    tag = tag_mapping_id.get(current_stage, tag_mapping_id['unknown'])
                    
                prompt = tag_mapping_id.get(current_stage, tag_mapping_id['unknown'])
                    
                all_data["states"].append(state.copy())  
                all_data["actions_behavior"].append(behavior_action.copy())
                all_data["actions_label"].append(reference_action.copy())
                all_data["rewards"].append(float(reward))  
                all_data["prompts"].append(int(prompt))  
                all_data["tags"].append(int(tag))  
                    
                total_reward += reward
                steps += 1
                state = next_state
                    
                if done:
                    if not isinstance(next_state, np.ndarray) or next_state.shape != state_shape:
                        next_state = np.reshape(next_state, state_shape) if hasattr(next_state, 'size') else np.zeros(state_shape)
                        
                    all_data["states"].append(next_state.copy())
                    all_data["actions_behavior"].append(np.zeros(action_shape))
                    all_data["actions_label"].append(np.zeros(action_shape))
                    all_data["rewards"].append(0.0)
                    all_data["prompts"].append(int(tag_mapping_id['unknown']))
                    all_data["tags"].append(int(tag_mapping_id['unknown']))
                
            mask_epoch_tag = (random.random() < self.mask_epoch_tag_prob)
            
        print(f"Finished epoch {epoch_id:06d}: total reward = {total_reward:.6f}, steps = {steps}")
        
        lengths = {k: len(v) for k, v in all_data.items()}
        if len(set(lengths.values())) > 1:
            print(f"Warning: inconsistent lengths in data: {lengths}")
            min_length = min(lengths.values())
            for k in all_data:
                all_data[k] = all_data[k][:min_length]
            print(f"Truncated all arrays to length {min_length}")
        
        processed_data = {}
        
        try:
            states_array = np.array([s.reshape(state_shape) if hasattr(s, 'reshape') else np.zeros(state_shape) for s in all_data["states"]])
            processed_data["states"] = states_array
            processed_data["actions_behavior"] = np.array([a.reshape(action_shape) if hasattr(a, 'reshape') else np.zeros(action_shape) for a in all_data["actions_behavior"]])
            processed_data["actions_label"] = np.array([a.reshape(action_shape) if hasattr(a, 'reshape') else np.zeros(action_shape) for a in all_data["actions_label"]])
            processed_data["rewards"] = np.array(all_data["rewards"], dtype=np.float32)
            processed_data["prompts"] = np.array(all_data["prompts"], dtype=np.int32)
            processed_data["tags"] = np.array(all_data["tags"], dtype=np.int32)
            
            final_lengths = {k: len(v) for k, v in processed_data.items()}
            assert len(set(final_lengths.values())) == 1, f"Processed data has inconsistent lengths: {final_lengths}"
            
        except Exception as e:
            print(f"Error processing data: {e}")
            for k, v in all_data.items():
                if len(v) > 0:
                    print(f"{k}: type={type(v[0])}")
                    if hasattr(v[0], 'shape'):
                        print(f"  shape={v[0].shape}")
                    elif hasattr(v[0], '__len__'):
                        print(f"  len={len(v[0])}")
                    else:
                        print(f"  value={v[0]}")
                
            return None
        
        return processed_data

def dump_anymdp(path_name, coach_path, max_steps, epoch_range, mode, ndim, state_dim, action_dim, seed=None):
    generator = DataGenerator(
        coach_path=coach_path,
        mode=mode,
        state_dim=state_dim,
        action_dim=action_dim,
        ndim=ndim,
        max_steps=max_steps,
        seed=seed
    )

    for epoch_id in epoch_range:
        results = generator.generate_data(epoch_id, max_steps)
        
        # Skip if data generation failed
        if results is None:
            print(f"Skipping epoch {epoch_id} due to failed data generation")
            continue
            
        file_path = f'{path_name}/record-{epoch_id:06d}'
        try:
            create_directory(file_path)
            print(f"Saving data for epoch {epoch_id} to {file_path}")

            np.save(f"{file_path}/observations.npy", results["states"])
            np.save(f"{file_path}/actions_behavior.npy", results["actions_behavior"])
            np.save(f"{file_path}/actions_label.npy", results["actions_label"])
            np.save(f"{file_path}/rewards.npy", results["rewards"])
            np.save(f"{file_path}/prompts.npy", results["prompts"])
            np.save(f"{file_path}/tags.npy", results["tags"])
            
            print(f"Successfully saved all data for epoch {epoch_id}")
            
        except Exception as e:
            print(f"Error saving data for epoch {epoch_id}: {e}")
            
        del results
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./anymdp_data/", help="Output directory")
    parser.add_argument("--coach_path", type=str, required=True, help="Path to the trained coach")
    parser.add_argument("--max_steps", type=int, default=4000, help="Maximum steps per epoch")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--start_index", type=int, default=0, help="Starting id for record numbering")
    parser.add_argument("--workers", type=int, default=4, help="Number of multiprocessing workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--mode", type=str, required=False, choices=["static", "dynamic", "universal"], help="Mode for task sampler")
    parser.add_argument("--state_dim", type=int, default=256, help="State dimension")
    parser.add_argument("--action_dim", type=int, default=256, help="Action dimension")
    parser.add_argument("--ndim", type=int, default=8, help="ndim for task sampler")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for each worker")
    
    args = parser.parse_args()

    recommended_workers = min(
        args.workers,
        max(1, (args.epochs + args.batch_size - 1) // args.batch_size)
    )
    
    print(f"Using {recommended_workers} workers (requested: {args.workers})")

    epochs_per_worker = []
    remaining_epochs = args.epochs
    current_index = args.start_index
    
    while remaining_epochs > 0:
        batch = min(args.batch_size, remaining_epochs)
        epochs_per_worker.append((current_index, current_index + batch))
        current_index += batch
        remaining_epochs -= batch

    processes = []
for batch_start in range(0, args.epochs, args.batch_size * recommended_workers):
    for worker_id in range(recommended_workers):
        start_idx = batch_start + worker_id * args.batch_size
        end_idx = min(start_idx + args.batch_size, args.epochs)
        if start_idx >= args.epochs:
            break
            
        process = multiprocessing.Process(
            target=dump_anymdp,
            args=(
                args.output_path,
                args.coach_path,
                args.max_steps,
                range(start_idx, end_idx),
                args.mode,
                args.ndim,
                args.state_dim,
                args.action_dim,
                args.seed
            )
        )
        processes.append(process)
        process.start()
        
    # 等待当前批次的进程完成
    for process in processes:
        process.join()
    processes = []