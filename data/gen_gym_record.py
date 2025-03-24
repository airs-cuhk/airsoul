import argparse
import os
import multiprocessing
import numpy as np
import re

import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from stable_baselines3 import DQN, A2C, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from airsoul.utils import DiscreteEnvWrapper

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def extract_state_space_dimensions(env_name, name="pendulum"):
        pattern = rf'^{name}(\d+)x(\d+)$'
        match = re.match(pattern, env_name)
        if not match:
            raise ValueError(f"Invalid environment name format: {env_name}. Expected format: '{name}<digit>x<digit>'.")
        state_space_dim1 = int(match.group(1))
        state_space_dim2 = int(match.group(2))

        return state_space_dim1, state_space_dim2

def create_env(args):
    if(args.env_name.lower().find("lake") >= 0):
        if args.random_env:
            env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True)
            return env
        else:
            env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
            return env
    if(args.env_name.lower().find("cliff") >= 0):
        env = gym.make('CliffWalking-v0')
        env = DiscreteEnvWrapper(env=env,
                                env_name=args.env_name.lower(),
                                action_space=args.action_clip,
                                state_space_dim1=48,
                                state_space_dim2=1)
        return env
    elif(args.env_name.lower() == "lander"):
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        return env
    elif(args.env_name.lower().find("pendulum") >= 0):
        env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)
        if args.map_env_to_discrete:
            state_space_dim1, state_space_dim2 = extract_state_space_dimensions(args.env_name.lower(), "pendulum")
            env = DiscreteEnvWrapper(env=env,
                                    env_name=args.env_name.lower(),
                                    action_space=args.action_clip,
                                    state_space_dim1=state_space_dim1,
                                    state_space_dim2=state_space_dim2)
        return env
    elif(args.env_name.lower().find("mountaincar") >= 0):
        env = gym.make("MountainCar-v0", render_mode="rgb_array")
        if args.map_env_to_discrete:
            state_space_dim1, state_space_dim2 = extract_state_space_dimensions(args.env_name.lower(), "mountaincar")
            env = DiscreteEnvWrapper(env=env,
                                    env_name=args.env_name.lower(),
                                    action_space=args.action_clip,
                                    state_space_dim1=state_space_dim1,
                                    state_space_dim2=state_space_dim2,
                                    reward_shaping=True)
        return env
    else:
        raise ValueError("Unknown env name: {}".format(args.env_name))

def train_model(env, model_name, n_total_timesteps, save_path):
    # For more RL alg, please refer to https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
    model_classes = {'dqn': DQN, 'a2c': A2C, 'td3': TD3, 'ppo': PPO}
    if model_name.lower() not in model_classes:
        raise ValueError("Unknown policy type: {}".format(model_name))
    
    model = model_classes[model_name.lower()]('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=int(n_total_timesteps))
    file_path = f'{save_path}/model/{model_name.lower()}'
    create_directory(file_path)
    model.save(file_path)
    
    return model

def load_model(model_name, save_path, env):
    model_classes = {'dqn': DQN, 'a24': A2C, 'td3': TD3, 'ppo': PPO}
    if model_name.lower() not in model_classes:
        raise ValueError("Unknown policy type: {}".format(model_name))
    
    return model_classes[model_name.lower()].load(f'{save_path}/model/{model_name.lower()}.zip', env=env)

def reset_env(env, args):
    if args.env_name.lower().find("pendulum") >= 0:
        state, *_ = env.reset(seed=123, options={"low": -0.7, "high": 0.5})
    elif args.env_name.lower().find("mountaincar") >= 0:
        state, *_ = env.reset(seed=123, options={"x_init": np.pi/2, "y_init": 0.5})
    else:
        state, *_ = env.reset()
    return state
def produce_data(args, worker_id, shared_list, seg_len):
    # Create environment
    env = create_env(args)
    model = load_model(args.policy_name, args.save_path, env)

    state_list = []
    act_list = []
    reward_list = []
    trail_reward_list = []
    tag_list = []
    prompt_list = []

    task_count = 0
    success_count = 0
    total_action_count = 0
    step = 0
    while step < seg_len:
        trail_reward = 0
        state = reset_env(env, args)  # Reset environment
        done = False
        step_trail_start = step
        while not done:
            action, _ = model.predict(state)  # Select action
            action = np.squeeze(action).item()
            next_state, reward, terminated, truncated, *_ = env.step(action)  # Execute action
            
            if args.env_name.lower().find("anymdp") >=0:
                done = terminated
            else:
                if terminated or truncated:
                    done = True
            # Reward shaping
            if args.env_name.lower().find("mountaincar") >=0 and terminated:
                shaped_reward = 1.0
            elif args.env_name.lower().find("lake") >=0 and done and reward < 0.5:
                shaped_reward = -1.0
            elif args.env_name.lower().find("pendulum") >=0:
                shaped_reward = max(reward/30 + 0.1, -0.1)
            elif args.env_name.lower().find("cliff") >=0:
                if done:
                    if terminated:
                        shaped_reward = 1.0
                    else:
                        shaped_reward = -1.0
                elif step - step_trail_start + 1 > args.n_max_steps:
                    shaped_reward = -1.0
                else:
                    shaped_reward = -0.03

            # Record state, action, and reward
            state_list.append(state)  # Append current state
            
            act_list.append(action)    # Append action taken

            reward_list.append(shaped_reward) # Append reward received

            tag_list.append(args.tag)

            prompt_list.append(3)
            
            trail_reward += reward
            state = next_state
            step += 1
            if step - step_trail_start > args.n_max_steps:
                done = True
            if done:
                state_list.append(next_state)  # Append next state
                act_list.append(np.array(args.action_done))    # Append action flag
                reward_list.append(np.array(args.reward_done)) # Append reward zero
                tag_list.append(7)
                prompt_list.append(7)
                trail_reward_list.append(trail_reward) # Append trail reward
            
        task_count += 1
        if(args.env_name.lower() == "lake"):
            if trail_reward > 0:
                success_count += 1
                total_action_count += step - step_trail_start
        elif(args.env_name.lower() == "lander"):
            if trail_reward > 200:
                success_count += 1
                total_action_count += step - step_trail_start
        elif(args.env_name.lower() == "mountaincar"):
            if terminated:
                success_count += 1
                total_action_count += step - step_trail_start
        elif(args.env_name.lower() == "cliff"):
            if reward > 0:
                success_count += 1
                total_action_count += step - step_trail_start

    if(success_count>0):
      print(f"Worker {worker_id}: average action count when success = {total_action_count/success_count}, success rate = {success_count/task_count}")

    result = {
        "states": np.squeeze(np.array(state_list)),
        "prompts": np.squeeze(np.array(prompt_list)),
        "tags": np.squeeze(np.array(tag_list)),
        "actions": np.squeeze(np.array(act_list)),
        "rewards": np.squeeze(np.array(reward_list)),
        "trail_reward": np.squeeze(np.array(trail_reward_list))
    }
    env.close()
    shared_list.append(result)
    return

def generate_records(args, task_id):
    env = create_env(args)
    if not args.enable_load_model or args.random_env:
        train_model(env, args.policy_name, args.n_total_timesteps, args.save_path)

    worker_splits = args.n_seq_len / args.n_workers + 1.0e-6
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    processes = []

    for worker_id in range(args.n_workers):

        multiprocessing.set_start_method('spawn', force=True)
        process = multiprocessing.Process(target=produce_data, 
                                          args=(args, worker_id, shared_list, worker_splits))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    results = list(shared_list)
    merged_result = {}
    for result in results:
        if "states" not in merged_result:
            merged_result["states"] = result["states"]
            merged_result["prompts"] = result["prompts"]
            merged_result["tags"] = result["tags"]
            merged_result["actions"] = result["actions"]
            merged_result["rewards"] = result["rewards"]
            merged_result["trail_reward"] = result["trail_reward"]
            continue            
        merged_result["states"] = np.concatenate((merged_result["states"], result["states"]))
        merged_result["prompts"] = np.concatenate((merged_result["prompts"], result["prompts"]))
        merged_result["tags"] = np.concatenate((merged_result["tags"], result["tags"]))
        merged_result["actions"] = np.concatenate((merged_result["actions"], result["actions"]))
        merged_result["rewards"] = np.concatenate((merged_result["rewards"], result["rewards"]))
        merged_result["trail_reward"] = np.concatenate((merged_result["trail_reward"], result["trail_reward"]))
    merged_result["states"] = merged_result["states"][:args.n_seq_len]
    merged_result["prompts"] = merged_result["prompts"][:args.n_seq_len]
    merged_result["tags"] = merged_result["tags"][:args.n_seq_len]
    merged_result["actions"] = merged_result["actions"][:args.n_seq_len]
    if args.env_name.lower() == "lander":
        merged_result["rewards"] = np.multiply(merged_result["rewards"][:args.n_seq_len], 0.01)
    else:
        merged_result["rewards"] = merged_result["rewards"][:args.n_seq_len]

    file_path = f'{args.save_path}/data/record-{task_id:06d}'
    create_directory(file_path)
    np.save(f"{file_path}/observations.npy", merged_result["states"])  # Save state data
    np.save(f"{file_path}/prompts.npy", merged_result["prompts"])  # Save prompt data
    np.save(f"{file_path}/tags.npy", merged_result["tags"])  # Save tag data
    np.save(f"{file_path}/actions_behavior.npy", merged_result["actions"])  # Save action data
    np.save(f"{file_path}/rewards.npy", merged_result["rewards"])  # Save reward data
    np.save(f"{file_path}/actions_label.npy", merged_result["actions"])  # Save fake label data.
    average_trail_reward = np.average(merged_result["trail_reward"])
    with open(f"{file_path}/trail_reward.txt", 'w') as file:
        file.write(f"Average trail reward = {average_trail_reward}\n")
        file.write("Trail rewards:\n")
        for reward in merged_result["trail_reward"]:
            file.write(f"{reward}\n")




if __name__ == "__main__":
    # Use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Q-learning agent in a gym environment.")
    parser.add_argument('--env_name' , type=str, default='LAKE', help="The name of the gym environment")
    parser.add_argument('--save_path', type=str, required=True, help='The path to save the training data (without file extension).')
    parser.add_argument('--policy_name', choices=['DQN', 'A2C', 'TD3', 'PPO'], default='DQN', help="Policy Type")
    parser.add_argument('--n_total_timesteps', type=int, default=200000, help='Total number of epochs for training.')
    parser.add_argument('--n_task', type=int, default=1000, help='Total number of task for generating.')
    parser.add_argument('--n_seq_len', type=int, default=100, help='Maximum number of actions per epoch.')
    parser.add_argument('--n_max_steps', type=int, default=200, help='Max step per trail.')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers for training.')
    parser.add_argument('--enable_load_model', type=str, default="False", help='Whether to load a pre-trained model.')
    parser.add_argument('--random_env', type=str, default="False", help='Random env.')
    parser.add_argument('--tag', type=int, default=3, help='Tag. 3 for opt with gramma 0.994. 4 for opt with noise. 5 for q-learning with noise. 6 for random.')
    parser.add_argument('--action_done', type=int, default=5, help='Action when done.')
    parser.add_argument('--reward_done', type=float, default=0.0, help='Reward when left.')
    parser.add_argument('--map_env_to_discrete', type=str, default=0, help='Map env to discrete state.')
    parser.add_argument('--action_clip', type=int, default=5, help='Discrete env action_clip.')

    args = parser.parse_args()
    args.enable_load_model = args.enable_load_model.lower() == "true"
    args.random_env = args.random_env.lower() == "true"
    args.map_env_to_discrete = args.map_env_to_discrete.lower() == "true"

    for task_id in range(args.n_task):
        generate_records(args, task_id)

    print("Finish generation.")