import argparse
import pickle
import random
import os
import numpy as np
import gym
from stable_baselines3 import SAC, PPO
from sb3_contrib import RecurrentPPO
from l3c.anymdpv2 import AnyMDPv2TaskSampler, AnyMDPEnv
from policy_trainer.sac_trainer import SACTrainer
from policy_trainer.ppo_mlp_trainer import PPO_MLP_Trainer
from policy_trainer.ppo_lstm_trainer import PPO_LSTM_Trainer
import gc

def check_task_validity(task, num_steps=10, policies_to_use=None, seed=None):
    print("Checking task validity...")

    env = gym.make("anymdp-v2-visualizer")
    env.set_task(task)
    
    if policies_to_use is None:
        policies_to_use = ["sac", "ppo_mlp", "ppo_lstm"]
    
    policies = {
        "random": lambda x: env.action_space.sample()
    }
    
    if "sac" in policies_to_use:
        policies["sac"] = SACTrainer(env, seed).model
        
    if "ppo_mlp" in policies_to_use:
        policies["ppo_mlp"] = PPO_MLP_Trainer(env, seed).model
        
    if "ppo_lstm" in policies_to_use:
        policies["ppo_lstm"] = PPO_LSTM_Trainer(env, seed).model

    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    random_rewards = []
    for _ in range(num_steps):
        action = policies["random"](state)
        step_result = env.step(action)
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
    if "ppo_lstm" in policies_to_use:
        compare_policy = "ppo_lstm"
    elif "ppo_mlp" in policies_to_use:
        compare_policy = "ppo_mlp"
    elif "sac" in policies_to_use:
        compare_policy = "sac"
    else:
        print("No RL policies available for validation.")
        env.close()
        return True
        
    print(f"Using {compare_policy.upper()} for task validation")

    state = env.reset()
    if isinstance(state, tuple):  
        state = state[0]
    policy_rewards = []
    lstm_states = None
    
    for _ in range(num_steps):
        if compare_policy == "ppo_lstm":
            action, lstm_states = policies[compare_policy].predict(
                state, 
                state=lstm_states,  
                deterministic=False
            )
        else:
            action, _ = policies[compare_policy].predict(state, deterministic=False)
            
        step_result = env.step(action)
        if len(step_result) == 5: 
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  
            next_state, reward, done, info = step_result
        policy_rewards.append(reward)
        if done:
            break
        state = next_state

    random_total = sum(random_rewards)
    policy_total = sum(policy_rewards)

    env.close()
    del policies
    gc.collect()

    if policy_total - random_total <= max(3.0 * np.std(random_rewards), 1e-3):
        print(f"Task invalid: no significant improvements for RL")
        print(f"Random reward: {random_total}, {compare_policy.upper()} reward: {policy_total}")
        return False
    
    print(f"Task valid - Random={random_total}, {compare_policy.upper()}={policy_total}")
    return True

if __name__ == "__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", type=int, default=256, help="State dimension")
    parser.add_argument("--action_dim", type=int, default=256, help="Action dimension")
    parser.add_argument("--ndim", type=int, default=8, help="ndim for task sampler")
    parser.add_argument("--mode", type=str, default="static", 
                        choices=["static", "dynamic", "universal"], 
                        help="Mode for task sampler")
    parser.add_argument("--task_number", type=int, default=1, help="Number of tasks to generate")
    parser.add_argument("--output_path", type=str, required=True, help="Output file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max_attempts", type=int, default=500, help="Maximum attempts to generate valid tasks")
    parser.add_argument("--skip_validation", action="store_true", help="Skip task validation")
    parser.add_argument("--validation_steps", type=int, default=10, help="Number of steps for validation")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Generate tasks
    tasks = []
    task_count = 0
    attempt_count = 0
    
    print(f"Attempting to generate {args.task_number} valid tasks...")
    
    while task_count < args.task_number and attempt_count < args.max_attempts:
        attempt_count += 1
        current_seed = args.seed + attempt_count if args.seed is not None else None
        
        print(f"Task generation attempt {attempt_count}/{args.max_attempts}, seed: {current_seed}")
        
        task = AnyMDPv2TaskSampler(
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            ndim=args.ndim,
            mode=args.mode,
            seed=current_seed,
            verbose=False
        )

        if args.skip_validation or check_task_validity(task, num_steps=args.validation_steps, seed=current_seed):
            task_count += 1
            tasks.append(task)
            print(f"Successfully generated task {task_count}/{args.task_number}")
        else:
            print(f"Rejected invalid task, continuing search...")

        gc.collect()
    
    if task_count < args.task_number:
        print(f"Warning: Only generated {task_count} valid tasks out of {args.task_number} requested after {args.max_attempts} attempts")
    
    # Prepare output file path
    if not args.output_path.endswith('.pkl'):
        output_file = args.output_path + ".pkl"
    else:
        output_file = args.output_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Writing {len(tasks)} tasks to {output_file} with the following configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  State dimension: {args.state_dim}")
    print(f"  Action dimension: {args.action_dim}")
    print(f"  ndim: {args.ndim}")
    
    # Save tasks to file
    with open(output_file, 'wb') as fw:
        pickle.dump(tasks, fw)
    
    print(f"Successfully saved {len(tasks)} tasks to {output_file}")