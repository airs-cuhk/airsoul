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
from queue import Empty
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_cartpole

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def is_task_in_range(task, gravity_scope=(8.0, 12.0),
                        masscart_scope=(0.8, 1.2),
                        masspole_scope=(0.08, 0.12),
                        length_scope=(0.4, 0.6)):
    if(task["gravity"] > gravity_scope[1] or task["gravity"] < gravity_scope[0]):
        return False
    if(task["masscart"] > masscart_scope[1] or task["masscart"] < masscart_scope[0]):
        return False
    if(task["masspole"] > masspole_scope[1] or task["masspole"] < masspole_scope[0]):
        return False
    if(task["length"] > length_scope[1] or task["length"] < length_scope[0]):
        return False
    return True

def queue_to_list(q):
    temp_list = []
    while q.qsize() > 0:
        try:
            item = q.get(timeout=1)
            temp_list.append(item)
            print(len(temp_list), q.qsize())
        except Exception as e:
            print(e)
    return temp_list

def sample_task(task_num, gravity_scope, masscart_scope, masspole_scope, length_scope, remove_scope=None):
    task_list = []
    for tid in range(task_num):
        task = sample_cartpole(gravity_scope=gravity_scope,
                                masscart_scope=masscart_scope,
                                masspole_scope=masspole_scope,
                                length_scope=length_scope)
        if(remove_scope is not None):
            while is_task_in_range(task, **remove_scope):
                task = sample_cartpole(gravity_scope=gravity_scope,
                                    masscart_scope=masscart_scope,
                                    masspole_scope=masspole_scope,
                                    length_scope=length_scope)
        task_list.append(task)
    return task_list

def dump_cartpole_record(
    task,
    file_path,
    seq_number,
    seq_length
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    env = gym.make('random-cartpole-v0', frameskip=2)
    env.set_task(task)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=0
    )

    model.learn(total_timesteps=40000)
    arr_obs = []
    arr_bactions = []
    arr_lactions = []
    arr_rewards = []

    for _ in range(seq_number):
        seq_obs = []
        seq_bactions = []
        seq_lactions = []
        seq_rewards = []

        env = gym.make('random-cartpole-v0', frameskip=2)
        env.set_task(task)
        obs, info = env.reset()
        exp_ratio = random.uniform(0.20, 0.50)
        seq_obs.append(obs)
        iteration = 0
        while iteration < seq_length:
            laction, _ = model.predict(obs, deterministic=True)
            if(random.random() > exp_ratio):
                baction = laction
            else:
                baction = env.action_space.sample()
            env.render()
            obs, reward, terminated, truncated, info = env.step(baction)
            seq_obs.append(obs)
            seq_bactions.append(baction)
            seq_lactions.append(laction)
            seq_rewards.append(reward)
            iteration += 1
            if (terminated or truncated) and iteration < seq_length:
                # for intra-sequence resets, use smaller reset bounds
                # resample environment
                env = gym.make('random-cartpole-v0', frameskip=2, reset_bounds_scale=0.05)
                env.set_task(task)
                # resample exp_ratio
                exp_ratio = random.uniform(0.20, 1.0)

                obs, info = env.reset()
                seq_obs.append(obs)
                seq_bactions.append(2)
                seq_lactions.append(0)
                seq_rewards.append(reward)
                iteration += 1


        arr_bactions.append(seq_bactions)
        arr_lactions.append(seq_lactions)
        arr_rewards.append(seq_rewards)
        arr_obs.append(seq_obs)
    arr_obs = np.array(arr_obs, dtype=np.float32)
    arr_bactions = np.array(arr_bactions, dtype=np.int32)
    arr_lactions = np.array(arr_lactions, dtype=np.int32)
    arr_rewards = np.array(arr_rewards, dtype=np.float32)

    env.close()
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    np.save(file_path + "observations.npy", arr_obs)
    np.save(file_path + "actions_behavior.npy", arr_bactions)
    np.save(file_path + "actions_label.npy", arr_lactions)
    np.save(file_path + "rewards.npy", arr_rewards)
    with open(file_path + "task_hyper_para.txt", "w") as f:
        for key, value in task.items():
            f.write(f"{key}\t{value}\t")
        f.write("\n")

def dump_multi_records(
    rank_id,
    world_size,
    task_queue,
    output_path,
    task_ids,
    seq_number,
    seq_length):

    file_size = 256

    task_number = len(task_queue)

    for task_id in task_ids:
        task = task_queue[task_id % task_number]
        file_path = "%s/record_%04d/" % (output_path.rstrip('/'), task_id)
        dump_cartpole_record(
            task,
            file_path,
            seq_number,
            seq_length,
        )
        print("Worker %d finished task %d, data saved to %s" % (rank_id, task_id, file_path))
    return

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./cartpole_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--seq_length", type=int, default=200, help="max steps, default:200")
    parser.add_argument("--offpolicy_labeling", type=int, default=0, help="enable offpolicy labeling (DAgger), default:False")
    parser.add_argument("--task_number", type=int, default=8, help="task number, default:8")
    parser.add_argument("--start_index", type=int, default=0, help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--task_seq_number", type=int, default=100, help="sequence number per task, default:100")
    parser.add_argument("--gravity_scope", type=float, nargs=2, default=(2.0, 16.0), help="gravity scope for the cartpole task")
    parser.add_argument("--masscart_scope", type=float, nargs=2, default=[0.5, 2.0], help="mass cart scope for the cartpole task")
    parser.add_argument("--masspole_scope", type=float, nargs=2, default=[0.05, 0.20], help="mass pole scope for the cartpole task")
    parser.add_argument("--length_scope", type=float, nargs=2, default=[0.20, 1.0], help="length scope for the cartpole task")
    parser.add_argument("--remove_scope", type=bool, default=False, help="avoid selection from that scope")
    parser.add_argument("--origin_in_scope", type=bool, default=True, help="whether make sure the original task is in the scope, default:True")
    args = parser.parse_args()

    if(args.remove_scope):
        remove_scope = {"gravity_scope":(8.0, 12.0),
                        "masscart_scope":(0.8, 1.2),
                        "masspole_scope":(0.08, 0.12),
                        "length_scope":(0.4, 0.6)}
    else:
        remove_scope = None

    gravity_scope=args.gravity_scope
    masscart_scope=args.masscart_scope
    masspole_scope=args.masspole_scope
    length_scope=args.length_scope

    # Task Generation
    task_queue = []
    print("Generating Tasks At First...")
    left_task_number = args.task_number
    if(args.origin_in_scope):
        # Make sure the original task is in the training set
        task_queue.append({
            "gravity": 9.8,
            "masscart": 1.0,
            "masspole": 0.1,
            "length": 0.5
        })
        left_task_number -= 1
        print("Original task added to the task queue.")

    task_queue.extend(sample_task(left_task_number,
                        gravity_scope,
                        masscart_scope,
                        masspole_scope,
                        length_scope,
                        remove_scope))

    task_number = len(task_queue)
    print(f"Task Generation Finished, acquiring {task_number} tasks (expect: {args.task_number})")


    print("Start Data Generation...")
    # Data Generation
    data_workers = args.workers
    data_number = task_number * args.task_seq_number
    worker_splits = int((data_number - 1) // data_workers + 1)
    seq_number = min(args.task_seq_number, 512) # max file_size
    worker_taskids = (data_number - 1) // (seq_number * data_workers) + 1
    total_taskids = worker_taskids * data_workers

    print("Total data number: %d, each worker will generate %d directories, \
            with each directory containing %d sequences." % (
            data_number, worker_taskids, seq_number))
    processes = []
    n_b_t = args.start_index
    for worker_id in range(data_workers):
        n_e_t = min(n_b_t + worker_taskids, total_taskids + args.start_index)
        n_b = int(n_b_t)
        n_e = int(n_e_t)
        if(n_e - n_b < 1):
            break

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_multi_records,
                args=(worker_id, args.workers,
                        task_queue,
                        args.output_path,
                        range(n_b, n_e),
                        seq_number,
                        args.seq_length))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join()
