# pip install gym[classic_control] stable-baselines3[extra]
import sys
import os
import random
import time
import numpy
import argparse
import multiprocessing
from multiprocessing import Process, Queue, Lock
import gymnasium as gym
import torch
from queue import Empty
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_cartpole


def evaluate_diff(task, obs, act):
    """
    This evaluates the performance of the GT world model of cartpole (hyper-parametered by task)
    on the trajectory of  (obs, act)
    """
    env = gym.make('random-cartpole-v0', frameskip=2)
    env.set_task(task)

    bszo, len_obs, _ = numpy.shape(obs)
    bsza, len_act = numpy.shape(act)
    assert bszo == bsza

    len_eff = min(len_obs - 1, len_act)

    tv = 0
    for i in range(bszo):
        for j in range(len_eff):
            env.state = obs[i][j]
            if(act[i][j] > 1):
                nobs, _ = env.reset()
            else:
                nobs, _, te, tr, _ = env.step(act[i][j])
            err = nobs - obs[i][j + 1]
            tv += numpy.mean(err ** 2)
    return tv / len_act / bszo

def read_evaluate_diff(task_file, obs_file, act_file):
    obs = numpy.load(obs_file)
    act = numpy.load(act_file)
    task = dict()
    with open(task_file, "r") as f_in:
        task_para = f_in.readlines()[0].strip().split("\t")
        for i in range(len(task_para)//2):
            task[task_para[i*2]] = float(task_para[i*2 + 1])
    return evaluate_diff(task, obs, act)


def find_name_in_directory(root_dir, target_name):
    """
    检查文件夹及其子文件夹中是否存在指定名称

    :param root_dir: 根目录路径
    :param target_name: 要查找的名称（文件或文件夹）
    :return: 找到返回完整路径，未找到返回 None
    """
    # 首先检查根目录
    if target_name in os.listdir(root_dir):
        yield os.path.join(root_dir, target_name), root_dir
    else:
        # 递归检查所有子文件夹
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if target_name in dirnames or target_name in filenames:
                yield os.path.join(dirpath, target_name), dirpath

def split_list(lst, n):
    """
    将列表lst尽可能均等地分成n份
    
    参数:
        lst: 要分割的列表
        n: 要分成的份数
    
    返回:
        包含n个子列表的列表
    """
    # 计算每份的基础大小和余数
    avg = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    
    for i in range(n):
        # 前remainder份多分配一个元素
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    
    return result

def evaluate_multiple(output, lock, dataname, obs_file, act_file, tasks):
    for taskname, task in tasks:
        diff = read_evaluate_diff(task, obs_file, act_file)
        with lock:
            with open(output, 'a') as f:  # 追加模式
                f.write(f"{taskname}\t{dataname}\t{diff}\n")
                f.flush()  # 确保立即写入磁盘
        print(f"...Finish calculating {taskname}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./dist_estimate", help="output path")
    parser.add_argument("--task_para", type=str, help="task parameter directory")
    parser.add_argument("--data", type=str, help="trajectory direction")
    parser.add_argument("--workers", type=str, default=32, help="number of workers")
    args = parser.parse_args()
    lock = Lock()
    
    obs_file = os.path.join(args.data, "observations.npy")
    act_file = os.path.join(args.data, "actions_behavior.npy")
    tasks = list()

    for task_file, name in find_name_in_directory(args.task_para, "task_hyper_para.txt"):
        tasks.append((name, task_file))

    tasks_splits = split_list(tasks, args.workers)

    # Data Generation
    processes = []
    for worker_id in range(args.workers):
        process = multiprocessing.Process(target=evaluate_multiple,
                args=(args.output, lock, args.data,
                        obs_file, act_file,
                        tasks_splits[worker_id]))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
