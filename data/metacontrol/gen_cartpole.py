# 环境准备（先执行安装命令）
# pip install gym[classic_control] stable-baselines3[extra]
import sys
import os
import random
import time
import torch
import numpy as np
import argparse
import multiprocessing
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_cartpole

def dump_cartpole_record(
    file_path,
    seq_number=500,
    seq_length=200,
    gravity_scope=(1.0, 15.0),
    masscart_scope=[0.5, 2.0],
    masspole_scope=[0.05, 0.20],
    length_scope=[0.25, 1.0]
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    env = gym.make('random-cartpole-v0', render_mode='rgb_array')
    env.set_task(sample_cartpole(gravity_scope=gravity_scope,
                                masscart_scope=masscart_scope,
                                masspole_scope=masspole_scope,
                                length_scope=length_scope))

    model = PPO(
        policy="MlpPolicy", 
        env=env,
        n_steps=2048,
        batch_size=16,
        gamma=0.99,
        verbose=1
    )
 
    model.learn(total_timesteps=40960)
    arr_obs = []
    arr_bactions = []
    arr_lactions = []
    arr_rewards = []
    print("finish coach training")

    for _ in range(seq_number):
        seq_obs = []
        seq_bactions = []
        seq_lactions = []
        seq_rewards = []

        obs, info = env.reset()
        seq_obs.append(obs)
        iteration = 0
        while iteration < seq_length:
            if(random.random() < 0.10):
                baction = env.action_space.sample()
            else:
                baction, _ = model.predict(obs, deterministic=False)
            laction, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(baction)
            seq_obs.append(obs)
            seq_bactions.append(baction)
            seq_lactions.append(laction)
            seq_rewards.append(reward)
            iteration += 1
            if (terminated or truncated) and iteration < seq_length:
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

def dump_multi_records(
    rank_id, 
    world_size, 
    output_path, 
    task_ids,
    seq_number=500,
    seq_length=200,
    gravity_scope=(1.0, 15.0),
    masscart_scope=[0.5, 2.0],
    masspole_scope=[0.05, 0.20],
    length_scope=[0.25, 1.0]):
    for task_id in task_ids:
        file_path = "%s/record_%04d/" % (output_path.strip('/'), task_id)
        dump_cartpole_record(
            file_path,
            seq_number=seq_number,
            seq_length=seq_length,
            gravity_scope=gravity_scope,
            masscart_scope=masscart_scope,
            masspole_scope=masspole_scope,
            length_scope=length_scope
        )
        print("Worker %d finished task %d, data saved to %s" % (rank_id, task_id, file_path))

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./cartpole_data/", help="output directory, the data would be stored as output_path/record-xxxx.npy")
    parser.add_argument("--seq_length", type=int, default=200, help="max steps, default:500")
    parser.add_argument("--offpolicy_labeling", type=int, default=0, help="enable offpolicy labeling (DAgger), default:False")
    parser.add_argument("--seq_number", type=int, default=100, help="sequence number, default:500")
    parser.add_argument("--task_number", type=int, default=8, help="task number, default:8")
    parser.add_argument("--start_index", type=int, default=0, help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--gravity_scope", type=float, nargs=2, default=(9.8, 9.8), help="gravity scope for the cartpole task")
    parser.add_argument("--masscart_scope", type=float, nargs=2, default=[1.0, 1.0], help="mass cart scope for the cartpole task")
    parser.add_argument("--masspole_scope", type=float, nargs=2, default=[0.1, 0.1], help="mass pole scope for the cartpole task")
    parser.add_argument("--length_scope", type=float, nargs=2, default=[0.5, 0.5], help="length scope for the cartpole task")
    args = parser.parse_args()

    gravity_scope=args.gravity_scope
    masscart_scope=args.masscart_scope
    masspole_scope=args.masspole_scope
    length_scope=args.length_scope
    # Data Generation
    worker_splits = args.task_number / args.workers + 1.0e-6
    processes = []
    n_b_t = args.start_index
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)

        print("start processes generating %04d to %04d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_multi_records, 
                args=(worker_id, args.workers, 
                        args.output_path, 
                        range(n_b, n_e), 
                        args.seq_number,
                        args.seq_length,
                        gravity_scope,
                        masscart_scope,
                        masspole_scope,
                        length_scope))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
