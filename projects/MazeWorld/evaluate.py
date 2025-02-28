import random
import argparse
import sys
import os
import torch
import numpy
import gym
import l3c.mazeworld
import cv2
import pickle
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from collections import namedtuple
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset

from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent
from l3c.mazeworld.envs.maze_task import MazeTaskManager

TaskConfig = namedtuple("TaskConfig", ["start", "cell_landmarks", "cell_walls", "cell_texts", 
    "cell_size", "wall_height", "agent_height", "initial_life", "max_life",
    "step_reward", "goal_reward", "landmarks_rewards", "landmarks_coordinates", "landmarks_refresh_interval", "commands_sequence"])

from airsoul.models import E2EObjNavSA
from airsoul.utils import create_folder, VideoWriter
from airsoul.utils import Configure
from airsoul.utils import img_pro, custom_load_model


def postprocess_image(img, cell_size, scale_factor, actions):
    action_texts = ["Stop", "Turn-Left", "Turn Right", "Backward", "Forward"]
    def action_text(action):
        return "Actions:" + action_texts[action]
    w, h, c = img.shape
    W = w * scale_factor
    H = h * scale_factor
    img_out = cv2.resize(img, (H, W))
    offset = cell_size * scale_factor
    for i, action in enumerate(actions):
        p_x = int(offset * (i + 0.80))
        p_y = int(offset * 0.90)
        text = action_text(action)
        img_out = cv2.putText(img_out, text, (p_x, p_y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return img_out


def string_mean_var(mean, var):
    string=""
    for i in range(mean.shape[0]):
        string += f'{mean[i]}\t{var[i]}\n'
    return string

def reward_smoothing(rewards, kernel_size=192):
    def gaussian_kernel(size=kernel_size, sigma=kernel_size // 6):
        """ Returns a 1D Gaussian kernel array """
        size = int(size) // 2
        x = numpy.linspace(-size, size, 2*size+1)
        gauss = numpy.exp(-0.5 * (x / sigma) ** 2)
        return gauss / gauss.sum()
    # +0.01 and -0.01 to avoid boundary effects
    data_convolved = numpy.convolve(numpy.array(rewards) + 0.01, gaussian_kernel(), mode='same') - 0.01
    return data_convolved

def model_epoch(maze_env, task, model, policy_config, device, max_step, 
        autoregressive_steps=[], autoregressive_length=10, 
        video_writer=None, video_text=True):
    # Example training loop
    maze_env.set_task(task)
    obs = maze_env.reset()
    obs_arr = [obs]
    pred_obs_arr = []
    act_arr = []

    rew_arr = []
    acc_rew_arr = []

    agent = SmartSLAMAgent(maze_env=maze_env, render=False)
    next_rew_gt = 0.0

    done = False
    step = 1
    cache = None
    act_tor = None
    last_obs = obs
    temperature = policy_config.T_ini
    wm_loss = dict()
    model.init_mem()
    while not done:
        if(step not in autoregressive_steps):
            n_step = 1
        else:
            n_step = autoregressive_length
        step += n_step
        if(step >= max_step):
            break

        temperature = max(temperature * (1.0 - policy_config.T_dec), policy_config.T_min)
        pred_obss, pred_acts, cache = model.inference_step_by_step(obs_arr, act_arr, temperature, step, device, n_step=n_step, cache=cache)

        obs_arr = []
        # The next step (t+1) has already been cached, thus we only need the decisions from t+2
        act_arr = pred_acts[1:]
        for act in pred_acts:
            next_obs_gt, next_rew_gt, done, _ = maze_env.step(act)
            obs_arr.append(next_obs_gt)
            rew_arr.append(next_rew_gt)
            if(len(acc_rew_arr) < 1):
                acc_rew_arr.append(max(0.0, next_rew_gt + 0.01))
            else:
                acc_rew_arr.append(acc_rew_arr[-1] + max(0.0, next_rew_gt + 0.01))

        if(n_step > 1):
            if(step not in wm_loss):
                wm_loss[step] = numpy.zeros((autoregressive_length,), dtype=numpy.float32)
            for idx, (wm_img, gt_img) in enumerate(zip(pred_obss, obs_arr)):
                wm_loss[step][idx] += numpy.mean((img_pro(wm_img) - img_pro(gt_img)) ** 2)

        if(video_writer is not None):
            img_pred = numpy.concatenate([last_obs] + pred_obss, axis=0)
            img_gt = numpy.concatenate([last_obs] + obs_arr, axis=0)
            img_syn = numpy.transpose(numpy.concatenate((img_gt, img_pred), axis=1), (1, 0, 2))
            if(video_text):
                img_syn = postprocess_image(img_syn, 128, 3, pred_acts)
            
            video_writer.add_image(img_syn)
        
        print("Step: %d, Reward: %f, Reward Summary: %f" % (step, next_rew_gt, numpy.sum(rew_arr)))
        last_obs = obs_arr[-1]

        sys.stdout.flush()
    return reward_smoothing(rew_arr), acc_rew_arr, wm_loss

def random_epoch(maze_env, task):
    # Example training loop
    maze_env.set_task(task)
    observation = maze_env.reset()

    done = False
    step = 0
    rew_arr = []
    acc_rew_arr = []
    reward = 0
    while not done:
        step += 1
        action = random.randint(0, 4)
        observation, reward, done, _ = maze_env.step(action)
        rew_arr.append(reward)
        if(len(acc_rew_arr) < 1):
            acc_rew_arr.append(max(0.0, reward + 0.01))
        else:
            acc_rew_arr.append(acc_rew_arr[-1] + max(0.0, reward + 0.01))
    return reward_smoothing(rew_arr), acc_rew_arr

def agent_epoch(maze_env, task, mem_kr):
    # Example training loop
    maze_env.set_task(task)
    observation = maze_env.reset()
    agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=mem_kr)

    done = False
    step = 0
    rew_arr = []
    acc_rew_arr = []
    reward = 0
    while not done:
        step += 1
        action = agent.step(observation, reward)
        observation, reward, done, _ = maze_env.step(action)
        rew_arr.append(reward)
        if(len(acc_rew_arr) < 1):
            acc_rew_arr.append(max(0.0, reward + 0.01))
        else:
            acc_rew_arr.append(acc_rew_arr[-1] + max(0.0, reward + 0.01))
    return reward_smoothing(rew_arr), acc_rew_arr

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', type=str, help="YAML configuration file")
    parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    config = Configure()
    config.from_yaml(args.configuration)

    # Get the dictionary of attributes
    if args.configs:
        for pair in args.configs:
            key, value = pair.split('=')
            config.set_value(key, value)
            print(f"Rewriting configurations from args: {key} to {value}")
    print("Final configuration:\n", config)
    demo_config = config.demo_config
    os.environ['MASTER_PORT'] = demo_config.master_port        # Example port, choose an available port

    if(demo_config.task_file is not None):
        print(f"Reading tasks from {demo_config.task_file}...")
        tasks = []
        with open(demo_config.task_file, 'rb') as fr:
            task_dicts = pickle.load(fr)
            #print(task_dicts)
            for task_dict in task_dicts:
                task = MazeTaskManager.TaskConfig(**task_dict)
                tasks.append(task)
        create_folder(f'{demo_config.output}')
    else:
        raise Exception("Must set 'demo_config.task_file'")

    run_model = demo_config.run_model
    run_rule = demo_config.run_rule
    run_random = demo_config.run_random

    if(run_model):
        model = E2EObjNavSA(config.model_config)
        use_gpu = torch.cuda.is_available()
        if(use_gpu):
            device = torch.device(f'cuda:0')
        else:
            device = torch.device('cpu')

        load_model_path = demo_config.model_config.load_model_path
        load_model_parameter_blacklist = demo_config.model_config.load_model_parameter_blacklist
        model = custom_load_model(DP(model), f'{load_model_path}/model.pth', black_list=load_model_parameter_blacklist, strict_check=True)

        model = model.module.to(device)

        reward_model = []
        acc_reward_model = []

    if(run_rule):
        reward_agent = []
        acc_reward_agent = []

    if(run_random):
        reward_random = []
        acc_reward_random = []

    maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=demo_config.time_step, task_type="NAVIGATION", resolution=(128, 128))

    # Go across all tasks and perform evaluation
    task_id_rewards = []
    wm_losses = dict()
    for idx, task in enumerate(tasks):
        if(run_model):
            n_step = demo_config.model_config.autoregressive_length
            img_size = 384
            if(demo_config.write_video < 1):
                video_writer = None
                create_folder(f'{demo_config.output}/{idx}')
            else:
                video_writer = VideoWriter(f'{demo_config.output}/{idx}/', "demo", window_size=((n_step + 1) * img_size, 2 * img_size))
            rewards, acc_rewards, wm_loss = model_epoch(maze_env, task, model, demo_config.model_config.policy, device, demo_config.time_step,
                    video_writer=video_writer, 
                    video_text=True, 
                    autoregressive_steps=demo_config.model_config.autoregressive_steps, 
                    autoregressive_length=demo_config.model_config.autoregressive_length)
            reward_model.append(rewards)
            acc_reward_model.append(acc_rewards)
            for key in wm_loss:
                if key not in wm_losses:
                    wm_losses[key] = []
                wm_losses[key].append(wm_loss[key])
            task_id_rewards.append((idx, acc_rewards[-1]))
            maze_env.save_trajectory(f'{demo_config.output}/{idx}/traj_model_agent.jpg')
            if(video_writer is not None):
                video_writer.clear()

        if(run_rule):
            rewards, acc_rewards = agent_epoch(maze_env, task, demo_config.rule_config.mem_kr)
            reward_agent.append(rewards)
            acc_reward_agent.append(acc_rewards)
            create_folder(f'{demo_config.output}/{idx}')
            maze_env.save_trajectory(f'{demo_config.output}/{idx}/traj_rule_agent.jpg')

        if(run_random):
            rewards, acc_rewards = random_epoch(maze_env, task)
            reward_random.append(rewards)
            acc_reward_random.append(acc_rewards)
            create_folder(f'{demo_config.output}/{idx}')
            maze_env.save_trajectory(f'{demo_config.output}/{idx}/traj_random_agent.jpg')

    if(run_model):
        reward_model = numpy.array(reward_model, dtype=numpy.float64)
        rewards_mean = numpy.mean(reward_model, axis=0)
        rewards_std = numpy.std(reward_model, axis=0)
        acc_rewards_mean = numpy.mean(acc_reward_model, axis=0)
        acc_rewards_std = numpy.std(acc_reward_model, axis=0)
        wml_stat = dict()
        for key in wm_losses:
            wml = numpy.array(wm_losses[key], dtype=numpy.float64)
            wml_mean = numpy.mean(wml, axis=0)
            wml_std = numpy.std(wml, axis=0)
            wml_stat[key] = (wml_mean, wml_std)

        string_model = string_mean_var(rewards_mean, rewards_std)
        with open(f'{demo_config.output}/reward_model_agent.txt', 'w') as f_model:
            f_model.write(string_model)
        acc_string_model = string_mean_var(acc_rewards_mean, acc_rewards_std)
        with open(f'{demo_config.output}/acc_reward_model_agent.txt', 'w') as f_model:
            f_model.write(acc_string_model)
        with open(f'{demo_config.output}/task_reward_model_agent.txt', 'w') as f_model:
            for idx, r in task_id_rewards:
                f_model.write(f"{idx}\t{r}")
        for key in wml_stat:
            with open(f'{demo_config.output}/wm_loss_{key}.txt', 'w') as f_model:
                wm_string_model = string_mean_var(*wml_stat[key])
                f_model.write(wm_string_model)

    if(run_rule):
        reward_agent = numpy.array(reward_agent, dtype=numpy.float64)
        rewards_mean = numpy.mean(reward_agent, axis=0)
        rewards_std = numpy.std(reward_agent, axis=0)
        acc_rewards_mean = numpy.mean(acc_reward_agent, axis=0)
        acc_rewards_std = numpy.std(acc_reward_agent, axis=0)
        string_agent = string_mean_var(rewards_mean, rewards_std)
        with open(f'{demo_config.output}/reward_rule_agent.txt', 'w') as f_model:
            f_model.write(string_agent)
        string_agent = string_mean_var(acc_rewards_mean, acc_rewards_std)
        with open(f'{demo_config.output}/acc_reward_rule_agent.txt', 'w') as f_model:
            f_model.write(string_agent)

    if(run_random):
        reward_random = numpy.array(reward_random, dtype=numpy.float64)
        rewards_mean = numpy.mean(reward_random, axis=0)
        rewards_std = numpy.std(reward_random, axis=0)
        acc_rewards_mean = numpy.mean(acc_reward_random, axis=0)
        acc_rewards_std = numpy.std(acc_reward_random, axis=0)
        string_random = string_mean_var(rewards_mean, rewards_std)
        with open(f'{demo_config.output}/reward_rule_random.txt', 'w') as f_model:
            f_model.write(string_random)
        string_random = string_mean_var(acc_rewards_mean, acc_rewards_std)
        with open(f'{demo_config.output}/acc_reward_rule_random.txt', 'w') as f_model:
            f_model.write(string_random)
