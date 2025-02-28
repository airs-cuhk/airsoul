import argparse
import os
import numpy
import random
import torch
import multiprocessing
from l3c.anymdpv2 import AnyMDPv2TaskSampler, AnyMDPEnv
from tag_vocab import tag_mapping_id
from stable_baselines3 import SAC, PPO
from sb3_contrib import RecurrentPPO
import pickle
from noise_distiller import NoiseDistillerWrapper, NoiseDistillerPolicy

def create_directory(path):
    os.makedirs(path, exist_ok=True)

class DataGenerator:
    def __init__(self, coach_path, mode, state_dim, action_dim, ndim, seed=None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)

        # Save parameters for later reinitialization
        self.mode = mode
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ndim = ndim

        # Create environment and task
        self.env = AnyMDPEnv()
        self.task = AnyMDPv2TaskSampler(
            state_dim=state_dim,
            action_dim=action_dim, 
            ndim=ndim,
            mode=mode,
            seed=seed,
            verbose=False
        )
        self.env.set_task(self.task)

        # Initialize PPO_MLP policy for env validation
        self.ppo_mlp = PPO(
            "MlpPolicy",      
            self.env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99,
        )

        # Load coach from file
        coach_dir = os.path.dirname(coach_path)
        coach_file = os.path.join(coach_dir, f"coach_{self.mode}.pkl")
        if not os.path.exists(coach_file):
            raise ValueError(f"No coach found for mode {self.mode}")

        with open(coach_file, 'rb') as f:
            data = pickle.load(f)

        if data["mode"] != self.mode:
            raise ValueError(
                f"Coach mode ({data['mode']}) does not match environment mode ({self.mode})"
            )
        
        self.behavior_policies = data["behavior_policies"]
        self.reference_policies = data["reference_policies"]
        self.task_config = data["task_config"]

        self.mask_all_tag_prob = 0.15
        self.mask_epoch_tag_prob = 0.15
        
        def create_stage_policy(stage_policies):
            def stage_policy(state):
                policy_data = random.choice(stage_policies)
                if policy_data["policy_name"] == "random":
                    return self.env.action_space.sample(), None
                elif "noise_distilled_" in policy_data["policy_name"]:
                    base_policy_name = policy_data["policy_name"].replace("noise_distilled_", "")
                    
                    if base_policy_name == "ppo_lstm":
                        base_policy = RecurrentPPO(
                            "MlpLstmPolicy",
                            self.env,
                            verbose=0
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
                    
                    base_policy.policy.load_state_dict(policy_data["state_dict"])
                    
                    noise_policy = NoiseDistillerPolicy(
                        base_policy, 
                        self.env, 
                        policy_data["noise_params"]
                    )
                    
                    return noise_policy.predict(state, deterministic=True)[0], None
                else:
                    if policy_data["policy_name"] == "ppo_lstm":
                        policy = RecurrentPPO(
                            "MlpLstmPolicy",
                            self.env,
                            verbose=0
                        )
                    elif policy_data["policy_name"] == "ppo_mlp":
                        policy = PPO(
                            "MlpPolicy",
                            self.env,
                            verbose=0
                        )
                    else:  
                        policy = SAC(
                            "MlpPolicy",
                            self.env,
                            verbose=0
                        )
                    
                    policy.policy.load_state_dict(policy_data["state_dict"])
                    return policy.predict(state, deterministic=True)[0], None
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
        
        self.bprob = numpy.cumsum(bprob)
        self.bprob /= self.bprob[-1]
        self.rprob = numpy.cumsum(rprob)
        self.rprob /= self.rprob[-1]
    
    def reset_env_and_task(self):
        print("Reinitializing environment and task...")
        self.env = AnyMDPEnv()
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
        self.ppo_mlp = PPO(
            "MlpPolicy",      
            self.env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99,
        )
    
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
            action = self.env.action_space.sample()  # Random policy
            next_state, reward, terminated, truncated, info = self.env.step(action)
            random_rewards.append(reward)
            if terminated or truncated:
                break
            state = next_state
        
        # Reset and run PPO_MLP policy for num_steps
        state, info = self.env.reset()
        ppo_mlp_rewards = []
        for _ in range(num_steps):
            action, _ = self.ppo_mlp.predict(state, deterministic=False)
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

    def sample_behavior_policy(self):
        return self.blist[numpy.searchsorted(self.bprob, random.random())]
    
    def sample_reference_policy(self):
        return self.rlist[numpy.searchsorted(self.rprob, random.random())]
    
    def generate_data(self, epoch_id, max_steps):
        # Validate environment before generating data
        valid_env = False
        max_attempts = 10
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
        while steps < max_steps:
            state, _ = self.env.reset()

            behavior_idx = numpy.searchsorted(self.bprob, random.random())
            behavior_policy = self.blist[behavior_idx]

            current_stage = self.stages[behavior_idx] if behavior_idx < len(self.stages) else "unknown"
            
            print(f"Using {current_stage} policy")
            
            done = False
            while not done and steps < max_steps:
                behavior_action, _ = behavior_policy(state)
                reference_action, _ = self.sample_reference_policy()(state)
                
                next_state, reward, terminated, truncated, info = self.env.step(behavior_action)
                done = terminated or truncated
                
                if mask_all_tag or mask_epoch_tag:
                    tag = tag_mapping_id['unknown']
                else:
                    tag = tag_mapping_id.get(current_stage, tag_mapping_id['unknown'])
                
                prompt = tag_mapping_id.get(current_stage, tag_mapping_id['unknown'])
                
                all_data["states"].append(state)
                all_data["actions_behavior"].append(behavior_action)
                all_data["actions_label"].append(reference_action)
                all_data["rewards"].append(reward)
                all_data["prompts"].append(prompt)
                all_data["tags"].append(tag)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    all_data["states"].append(next_state)
                    all_data["actions_behavior"].append(0)
                    all_data["actions_label"].append(0)
                    all_data["rewards"].append(0.0)
                    all_data["prompts"].append(tag_mapping_id['unknown'])
                    all_data["tags"].append(tag_mapping_id['unknown'])
            
            mask_epoch_tag = (random.random() < self.mask_epoch_tag_prob)
        
        print(f"Finished epoch {epoch_id:06d}: total reward = {total_reward:.6f}, steps = {steps}")
        return {k: numpy.array(v) for k, v in all_data.items()}

def dump_anymdp(path_name, coach_path, max_steps, epoch_range, mode, ndim, state_dim, action_dim, seed=None):
    generator = DataGenerator(
        coach_path=coach_path,
        mode=mode,
        state_dim=state_dim,
        action_dim=action_dim,
        ndim=ndim,
        seed=seed
    )
    
    for epoch_id in epoch_range:
        results = generator.generate_data(epoch_id, max_steps)
        
        # Skip if data generation failed
        if results is None:
            print(f"Skipping epoch {epoch_id} due to failed data generation")
            continue
            
        file_path = f'{path_name}/record-{epoch_id:06d}'
        create_directory(file_path)
        
        numpy.save(f"{file_path}/observations.npy", results["states"])
        numpy.save(f"{file_path}/actions_behavior.npy", results["actions_behavior"])
        numpy.save(f"{file_path}/actions_label.npy", results["actions_label"])
        numpy.save(f"{file_path}/rewards.npy", results["rewards"])
        numpy.save(f"{file_path}/prompts.npy", results["prompts"])
        numpy.save(f"{file_path}/tags.npy", results["tags"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./anymdp_data/", help="Output directory")
    parser.add_argument("--coach_path", type=str, required=True, help="Path to the trained coach")
    parser.add_argument("--max_steps", type=int, default=4000, help="Maximum steps per epoch")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--start_index", type=int, default=0, help="Starting id for record numbering")
    parser.add_argument("--workers", type=int, default=4, help="Number of multiprocessing workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--mode", type=str, required=True, choices=["static", "dynamic", "universal"], help="Mode for task sampler")
    parser.add_argument("--state_dim", type=int, default=256, help="State dimension")
    parser.add_argument("--action_dim", type=int, default=256, help="Action dimension")
    parser.add_argument("--ndim", type=int, default=8, help="ndim for task sampler")
    
    args = parser.parse_args()

    worker_splits = args.epochs / args.workers + 1.0e-6
    processes = []
    n_b_t = args.start_index
    
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)
        
        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(
            target=dump_anymdp,
            args=(
                args.output_path,
                args.coach_path,
                args.max_steps,
                range(n_b, n_e),
                args.mode,
                args.ndim,
                args.state_dim,
                args.action_dim,
                args.seed
            )
        )
        processes.append(process)
        process.start()
        
        n_b_t = n_e_t
    
    for process in processes:
        process.join()