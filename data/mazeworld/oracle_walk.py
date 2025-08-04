#!/usr/bin/env python
# coding=utf8
# File: dump_maze.py
import gym
import sys
import os
import random
import time
import numpy
import argparse
import multiprocessing
import pickle
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler, Resampler
from xenoverse.mazeworld.agents import OracleAgent
from xenoverse.mazeworld.agents import SmartSLAMAgent
import numpy as np

current_folder = os.path.dirname(os.path.abspath(__file__))
if current_folder not in sys.path:
    sys.path.append(current_folder)
from maze_behavior_solver import MazeNoisyExpertAgent

def run_maze_epoch(
        maze_env,
        max_steps,
        record_points):
    # Must intialize agent after reset
    # expert_agent = SmartSLAMAgent(maze_env=maze_env, render=False, memory_keep_ratio=memory_keep_ratio)
    # print("Sampled behavior agent:", expert_agent)
    # print("Memory keep ratio:", expert_agent.memory_keep_ratio)
     # actions id: 0 - 15 
    oracle_agent = OracleAgent(maze_env=maze_env, render=False)


    done=False
    observation, information = maze_env.reset()
    sum_reward = 0
    reward = 0
    observation_list = [observation]
    cmd_list = [information["command"]]
    bact_id_list = []
    lact_id_list = []
    bact_val_list = []
    lact_val_list = []
    bact_type_list = []
    bev_list = []
    reward_list = []
    step = 0
    L_task = []
    start_step = -1
    while not done:
        if step in record_points:
            last_command = information["command"]
            maze_env.refresh_command()
            current_command = maze_env.maze_core.get_command()
            print(f"current command is {current_command}, and last is : {last_command}")
            start_step = step
        lact_id = oracle_agent.step(observation, reward)
        obs, reward, done, info = maze_env.step(lact_id)

        if reward != 0 and start_step != -1:
            l_task = step - start_step
            L_task.append(l_task)
            print(f"The length of {start_step} to finish is: {l_task}")
            start_step = -1
        if step - start_step > 500 and start_step != -1:
            print(f"The {start_step} fails")
            start_step = -1
            L_task.append(-1)
        observation_list.append(obs)
        observation = obs
        reward_list.append(reward)
        cmd_list.append(info["command"])
        sum_reward += reward
        step += 1

    print("Finish running, sum reward = %f, steps = %d\n"%(sum_reward, len(observation_list)-1))
    return {
            "rewards": numpy.array(reward_list, dtype=numpy.float32),
            "commands": numpy.array(cmd_list, dtype=numpy.int32),
            "L_task": numpy.array(L_task, dtype=numpy.int32),
            }

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_maze(work_id, path_name, traj_folders, memory_keep_ratio, max_steps):
    # Tasks in Sequence: Number of tasks sampled for each sequence: settings for continual learning
    for idx in range(len(traj_folders)):
        traj_folder =  traj_folders[idx]
        task_path = os.path.join(traj_folder, 'task.pkl')
        task = pickle.load(open(task_path, 'rb'))

        record_points = [1, 50]
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        maze_env.set_task(task)
        results = run_maze_epoch(
                maze_env,
                600,
                record_points)

        file_path = traj_folders[idx]

        # create_directory(file_path)
        if not os.path.exists(file_path):
            print("not exist: %s" % file_path)
            assert False
        print("Saving trajectory to %s" % file_path)
        maze_env.save_trajectory(os.path.join(file_path, f"trajectory.png"))
        re_path = os.path.join(file_path, f"oracle_rewards.npy")
        numpy.save(re_path, results["rewards"])
        print("Saved rewards to %s" % re_path)
        cmd_path = os.path.join(file_path, f"oracle_commands.npy")
        numpy.save(cmd_path, results["commands"])
        print("Saved commands to %s" % cmd_path)
        L_task_path = os.path.join(file_path, f"Ltask1.npy")
        numpy.save(L_task_path, results["L_task"])
        print("Saved L_task to %s" % L_task_path)

        record_points = [100, 200, 300, 1000, 1100, 1200, 8900, 9000, 9100]
        maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps, resolution=(128, 128))
        maze_env.set_task(task)
        results = run_maze_epoch(
                maze_env,
                max_steps,
                record_points)

        file_path = traj_folders[idx]

        # create_directory(file_path)
        if not os.path.exists(file_path):
            print("not exist: %s" % file_path)
            assert False
        print("Saving trajectory to %s" % file_path)
        maze_env.save_trajectory(os.path.join(file_path, f"oracle_trajectory.png"))
        re_path = os.path.join(file_path, f"oracle_rewards.npy")
        numpy.save(re_path, results["rewards"])
        print("Saved rewards to %s" % re_path)
        cmd_path = os.path.join(file_path, f"oracle_commands.npy")
        numpy.save(cmd_path, results["commands"])
        print("Saved commands to %s" % cmd_path)
        L_task_path = os.path.join(file_path, f"Ltask.npy")
        numpy.save(L_task_path, results["L_task"])
        print("Saved L_task to %s" % L_task_path)        



        print("Finish processing %s" % traj_folder)
        print("------------------------------------------------------------")

def getTasksFolders(data_root):
    import os
    import pickle
    traj_folders = []
    for sub_folder in os.listdir(data_root):
        traj_folders.append(os.path.join(data_root, sub_folder))
    print('Number of trajectories: ', len(traj_folders))
    print(traj_folders)
    return traj_folders


if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./maze_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--task_root", type=str, help="choose task source to generate the trajectory. FILE: tasks sample from existing file; NEW: create new tasks")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--start_index", type=int, default=0, help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--memory_keep_ratio", type=float, default=0.25, help="memory keep ratio, default:0.25")
    args = parser.parse_args()

    if args.task_root is None:
        raise ValueError("Please provide the task root directory")

    traj_folders = getTasksFolders(args.task_root)
    print("Number of tasks: ", len(traj_folders))
    print("Memory keep ratio: ", args.memory_keep_ratio)
    worker_splits = (len(traj_folders) + 1) / args.workers + 1.0e-6
    processes = []
    n_b_t = args.start_index
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = min(int(n_e_t), len(traj_folders))
        if n_b >= len(traj_folders):
            break

        print("start processes generating %04d to %04d" % (n_b, n_e))
        print("the folder to process is: ", traj_folders[n_b:n_e])
        process = multiprocessing.Process(target=dump_maze, 
                args=(worker_id, args.output_path, traj_folders[n_b:n_e], args.memory_keep_ratio,
                args.max_steps))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
