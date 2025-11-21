# pip install gym[classic_control] stable-baselines3[extra]
import sys
import os
import random
import time
import numpy as np
import argparse
import multiprocessing
from multiprocessing import Process, Queue
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from xenoverse import linds
from xenoverse.linds import LinearDSSamplerRandomDim, dump_linds_task
from xenoverse.linds.solver import LTISystemMPC
from xenoverse.utils import RandomFourier

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def queue_to_list(q):
    temp_list = []
    while not q.empty():
        temp_list.append(q.get())
    return temp_list

def sample_task(queue):
    task = LinearDSSamplerRandomDim()
    queue.put(task)

class NoiseScheduler:
    def __init__(self, seq_length):
        self.max_beta = random.random()
        self.max_sigma = random.uniform(-0.10, 1.0)
        self.step_scheduler_beta = RandomFourier(max_steps=seq_length, max_order=8, max_item=4)
        self.step_scheduler_sigma = RandomFourier(max_steps=seq_length, max_order=8, max_item=4)

    def disturb(self, action, step):
        beta = np.max(self.step_scheduler_beta(step), 0.0) * self.max_beta
        sigma = np.abs(self.step_scheduler_sigma(step)) * self.max_sigma
        return np.sqrt(1 - beta) * action + np.sqrt(beta) * np.random.normal(0, 1, size=action.shape) * sigma

def dump_lti_record(
    task,
    file_path,
    seq_length
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    env = gym.make('linear-dynamics-v0-visualizer')
    env.set_task(task)

    solver = LTISystemMPC(env, K=50, gamma=0.99)

    obs_history = []
    act_history = []
    label_history = []
    reward_history = []
    cmd_history = []
    reset_history = []

    obs, info = env.reset()
    x_current = env._state
    cmd = info["command"]
    ns = NoiseScheduler(seq_length)
    
    for t in range(seq_length):
        #action = env.action_space.sample()
        action = solver.solve(x_current, cmd)
        action_noisy = ns.disturb(action, t)
        obs, reward, terminated, truncated, info = env.step(action_noisy)
        cmd_history.append(info["command"])
        obs_history.append(obs)
        act_history.append(action_noisy)
        reset_history.append(terminated or truncated)
        label_history.append(action)
        reward_history.append(reward)

        if terminated or truncated:
            obs, info = env.reset()
            x_current = env._state
            cmd = info["command"]
            cmd_history.append(info["command"])
            obs_history.append(obs)
            act_history.append(action * 0.0)
            reset_history.append(True)
            label_history.append(action * 0.0)
            reward_history.append(0.0)

    arr_obs = np.array(obs_history, dtype=np.float32)
    arr_bactions = np.array(act_history, dtype=np.int32)
    arr_lactions = np.array(label_history, dtype=np.int32)
    arr_rewards = np.array(reward_history, dtype=np.float32)
    arr_cmd = np.array(cmd_history, dtype=np.float32)
    arr_reset = np.array(reset_history, dtype=bool)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    np.save(file_path + "observations.npy", arr_obs)
    np.save(file_path + "actions_behavior.npy", arr_bactions)
    np.save(file_path + "actions_label.npy", arr_lactions)
    np.save(file_path + "rewards.npy", arr_rewards)
    np.save(file_path + "commands.npy", arr_cmd)
    np.save(file_path + "resets.npy", arr_reset)
    dump_linds_task("%s/task_info.json" % (file_path), task)


def dump_multi_records(
    rank_id,
    task_queue,
    task_ids,
    output_path,
    seq_length):

    task_number = len(task_queue)

    for task_id in task_ids:
        task = task_queue[task_id % task_number]
        file_path = "%s/record_%04d/" % (output_path.rstrip('/'), task_id)
        dump_lti_record(
            task,
            file_path,
            seq_length,
        )
        print("Worker %d finished task %d, data saved to %s" % (rank_id, task_id, file_path))

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./lti_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--seq_length", type=int, default=1000, help="max steps, default:10000")
    parser.add_argument("--task_number", type=int, default=8, help="number of tasks to generate")
    parser.add_argument("--start_index", type=int, default=0, help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    args = parser.parse_args()

    # Task Generation
    task_queue = Queue()
    print("Generating Tasks At First...")
    max_task_number = args.task_number

    if(max_task_number > 0):
        task_workers = min(args.workers, max_task_number)
        worker_splits = int((max_task_number - 1) // task_workers + 1)
        processes = []
        n_b_t = 0
        for worker_id in range(task_workers):
            n_e_t = min(n_b_t + worker_splits, max_task_number)
            n_b = int(n_b_t)
            n_e = int(n_e_t)
            if(n_e_t - n_b_t < 1):
                break
            print("start processes generating tasks %04d to %04d" % (n_b, n_e))
            process = multiprocessing.Process(target=sample_task,
                    args=(task_queue,))
            processes.append(process)
            process.start()
            n_b_t = n_e_t
        for process in processes:
            process.join()
    print("Task Generation Finished.")

    task_queue = queue_to_list(task_queue)
    task_number = len(task_queue)

    print("Start Data Generation...")
    # Data Generation
    n_workers = args.workers
    worker_tasks = int((task_number - 1) // n_workers + 1)

    print("Total data number: %d, each worker will generate %d tasks" % (
            task_number, worker_tasks))
    processes = []
    n_b_t = args.start_index
    for worker_id in range(n_workers):
        n_e_t = min(n_b_t + worker_tasks, task_number)
        n_b = int(n_b_t)
        n_e = int(n_e_t)
        if(n_e_t - n_b_t < 1):
            break

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_multi_records,
                args=(worker_id, args.workers,
                        task_queue,
                        range(n_b, n_e),
                        args.output_path,
                        args.seq_length))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join()