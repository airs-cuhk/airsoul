#!/usr/bin/env python
# coding=utf8
# File: gen_darkroom_record.py
import gym
import sys
import os
import time
import numpy
import argparse
import multiprocessing
import pickle
import random as rnd
from numpy import random
import xenoverse
from packaging import version
assert version.parse(xenoverse.__version__) >= version.parse('0.2.1.19')
from airsoul.utils import tag_vocabulary, tag_mapping_gamma, tag_mapping_id

from xenoverse.anymdp import AnyMDPTaskSampler
from xenoverse.anymdp import AnyMDPSolverOpt
from xenoverse.utils import pseudo_random_seed

current_folder = os.path.dirname(os.path.abspath(__file__))
if current_folder not in sys.path:
    sys.path.append(current_folder)
from anymdp_behavior_solver import AnyPolicySolver, AnyMDPOptNoiseDistiller, AnyMDPOTSOpter, AnyMDPQNoiseDistiller, AnyMDPOTSNoiseDistiller, AnyMDPOpter


def darkroom_to_anymdp_task(dim, goal, horizon):
    state_space = dim * dim
    action_space = 5
    
    state_mapping = numpy.array([(i//dim, i%dim) for i in range(state_space)])
    
    transition_matrix = numpy.zeros((state_space, action_space, state_space))
    
    for s in range(state_space):
        row, col = s // dim, s % dim
        
        # action 0: move right 
        next_row = min(row + 1, dim - 1)
        next_s = next_row * dim + col
        transition_matrix[s, 0, next_s] = 1.0
        
        # action 1: move left 
        next_row = max(row - 1, 0)
        next_s = next_row * dim + col
        transition_matrix[s, 1, next_s] = 1.0
        
        # action 2: move up 
        next_col = min(col + 1, dim - 1)
        next_s = row * dim + next_col
        transition_matrix[s, 2, next_s] = 1.0
        
        # action 3: move down
        next_col = max(col - 1, 0)
        next_s = row * dim + next_col
        transition_matrix[s, 3, next_s] = 1.0
        
        # action 4: stay
        transition_matrix[s, 4, s] = 1.0

    reward_matrix = numpy.zeros((state_space, action_space, state_space))
    goal_state = goal[0] * dim + goal[1]

    for s in range(state_space):
        for a in range(action_space):
            for next_s in range(state_space):
                if next_s == goal_state:
                    reward_matrix[s, a, next_s] = 1.0
    
    reset_triggers = numpy.zeros(state_space)
    reset_triggers[goal_state] = 1.0
    
    reset_states = numpy.zeros(state_space)
    reset_states[0] = 1.0
    
    reward_noise = numpy.zeros_like(reward_matrix)
    
    task = {
        "state_space": state_space,
        "action_space": action_space,
        "state_mapping": state_mapping,
        "transition": transition_matrix,
        "reward": reward_matrix,
        "reward_noise": reward_noise,
        "reward_noise_type": "normal",
        "reset_triggers": reset_triggers,
        "reset_states": reset_states,
        "max_steps": horizon
    }
    
    return task


class DarkroomEnvAdapter(gym.Env):
    def __init__(self, dim, goal, horizon, max_steps):
        self.dim = dim
        self.goal = goal
        self.horizon = horizon
        self.max_steps = max_steps
        
        self.task = darkroom_to_anymdp_task(dim, goal, horizon)
        
        self.observation_space = type('obj', (object,), {'n': self.task['state_space']})
        self.action_space = type('obj', (object,), {'n': self.task['action_space']})
        
        self._state = None
        self.steps = 0
        self.need_reset = True

    def set_task(self, task=None):
        self.need_reset = True
        return
    
    def reset(self):
        self.steps = 0
        self.need_reset = False
        
        self._state = numpy.random.choice(len(self.task["state_mapping"]),
                                       replace=True,
                                       p=self.task["reset_states"])
        
        return self._state, {"steps": self.steps}
    
    def step(self, action):
        if self.need_reset:
            raise Exception("Must reset before doing any actions")
            
        transition_gt = self.task["transition"][self._state, action]
        next_state = random.choice(len(self.task["state_mapping"]), p=transition_gt)
        
        reward_gt = self.task["reward"][self._state, action, next_state]
        
        self.steps += 1
        self._state = next_state
        
        done = (self.steps >= self.max_steps or self.task["reset_triggers"][self._state])
        if done:
            self.need_reset = True
            
        info = {
            "steps": self.steps,
            "reward_gt": reward_gt,
            "transition_gt": transition_gt
        }
        
        return next_state, reward_gt, done, info
    
    @property
    def state(self):
        return self._state


def run_epoch(
        epoch_id,
        env,
        max_steps,
        offpolicy_labeling=True,
        ):
    # Must intialize agent after reset
    steps = 0
    
    # Steps to reset the 
    nstate = env.observation_space.n
    naction = env.action_space.n

    # Referrence Policiess
    solveropt0 = AnyMDPOpter(0, env)    #gamma = 0.0
    solveropt1 = AnyMDPOpter(1, env)    #gamma = 0.5
    solveropt2 = AnyMDPOpter(2, env)    #gamma = 0.93
    solveropt3 = AnyMDPOpter(3, env)    #gamma = 0.994

    # List of Behavior Policies
    solverneg = AnyPolicySolver(env)
    solverots = AnyMDPOTSNoiseDistiller(env, max_steps=max_steps)
    solverq = AnyMDPQNoiseDistiller(env, max_steps=max_steps)
    solverotsopt0 = AnyMDPOTSOpter(env, solver_opt=solveropt0, max_steps=max_steps)
    solverotsopt1 = AnyMDPOTSOpter(env, solver_opt=solveropt1, max_steps=max_steps)
    solverotsopt2 = AnyMDPOTSOpter(env, solver_opt=solveropt2, max_steps=max_steps)
    solverotsopt3 = AnyMDPOTSOpter(env, solver_opt=solveropt3, max_steps=max_steps)
    solveroptnoise2 = AnyMDPOptNoiseDistiller(env, opt_solver=solveropt2)
    solveroptnoise3 = AnyMDPOptNoiseDistiller(env, opt_solver=solveropt3)

    # Data Generation Strategy
    behavior_dict = [(solverneg, 0.10),     #rnd, 6
                     (solverots, 0.10),     #rand, 6; exp1, 4;
                     (solverq,   0.10),     #rand, 6; exp2, 5
                     (solverotsopt0, 0.10), #rnd, 6; or opt0, 0; or exp1, 4
                     (solverotsopt1, 0.10), #rnd, 6; or opt1, 1; or exp1, 4
                     (solverotsopt2, 0.10), #rnd, 6; or opt2, 2; or exp1, 4
                     (solverotsopt3, 0.10), #rnd, 6; or opt3, 3; or exp1, 4
                     (solveroptnoise2, 0.10), #rnd, 6; or opt2, 2; or exp2, 5
                     (solveroptnoise3, 0.10), #rnd, 6; or opt3, 3; or exp2, 5
                     (solveropt1, 0.02),    #opt1, 1
                     (solveropt2, 0.03),    #opt2, 2
                     (solveropt3, 0.05)]    #opt3, 3
    reference_dict = [(solveropt0, 0.10),   #opt0, 0
                      (solveropt1, 0.10),   #opt1, 1
                      (solveropt2, 0.20),   #opt2, 2
                      (solveropt3, 0.60)]   #opt3, 3
    
    # Policy Sampler
    blist, bprob = zip(*behavior_dict)
    rlist, rprob = zip(*reference_dict)

    bprob = numpy.cumsum(bprob)
    bprob /= bprob[-1]
    rprob = numpy.cumsum(rprob)
    rprob /= rprob[-1]

    def sample_behavior():
        return blist[numpy.searchsorted(bprob, random.random())]
    
    def sample_reference():
        return rlist[numpy.searchsorted(rprob, random.random())]

    state, info = env.reset()

    ppl_sum = []
    mse_sum = []

    bsolver = sample_behavior()
    rsolver = sample_reference()

    mask_all_tag_prob = 0.15
    mask_epoch_tag_prob = 0.15

    need_resample_b = (random.random() < 0.85)
    resample_freq_b = 0.20
    need_resample_r = (random.random() < 0.75)
    resample_freq_r = 0.20

    mask_all_tag = (random.random() < mask_all_tag_prob) # 15% probability to mask all tags
    mask_epoch_tag = (random.random() < mask_epoch_tag_prob) # 15% probability to mask all tags

    # Data Storage
    state_list = list()
    lact_list = list()
    bact_list = list()
    reward_list = list()
    prompt_list = list()
    tag_list = list()

    while steps <= max_steps:
        if(offpolicy_labeling):
            bact, bact_type = bsolver.policy(state)
            lact, prompt = rsolver.policy(state)
            if(need_resample_r and resample_freq_r > random.random()):
                rsolver = sample_reference()
        else:
            bact, bact_type = solverotsopt3.policy(state)
            lact = bact
            prompt = bact_type

        next_state, reward, done, info = env.step(bact)
        if(mask_all_tag or mask_epoch_tag):
            bact_type = tag_mapping_id['unk']

        ppl = -numpy.log(info["transition_gt"][next_state])
        mse = (reward - info["reward_gt"]) ** 2
        ppl_sum.append(ppl)
        mse_sum.append(mse)

        for solver, _ in behavior_dict:
            solver.learner(state, bact, next_state, reward, done)

        state_list.append(state)
        bact_list.append(bact)
        lact_list.append(lact)
        reward_list.append(reward)
        tag_list.append(bact_type)
        prompt_list.append(prompt)

        if(done): # If done, push the next state, but add a dummy action
            state_list.append(next_state)
            bact_list.append(naction)
            lact_list.append(naction)
            reward_list.append(0.0)
            tag_list.append(tag_mapping_id['unk'])
            prompt_list.append(tag_mapping_id['unk'])

            steps += 1
            next_state, info = env.reset()
            if(need_resample_b and resample_freq_b > random.random()):
                bsolver = sample_behavior()
            mask_epoch_tag = (random.random() < mask_epoch_tag_prob)

        state = next_state
        steps += 1

    print("Finish running %06d, sum reward: %f, steps: %d, gt_transition_ppl: %f, gt_reward_mse: %f"%(
            epoch_id, numpy.sum(reward_list), len(state_list)-1, numpy.mean(ppl_sum), numpy.mean(mse_sum)))

    return {
            "states": numpy.array(state_list, dtype=numpy.uint32),
            "prompts": numpy.array(prompt_list, dtype=numpy.uint32),
            "tags": numpy.array(tag_list, dtype=numpy.uint32),
            "actions_behavior": numpy.array(bact_list, dtype=numpy.uint32),
            "rewards": numpy.array(reward_list, dtype=numpy.float32),
            "actions_label": numpy.array(lact_list, dtype=numpy.uint32),
            }

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def dump_darkroom(work_id, world_work, path_name, epoch_ids, dim, max_steps, 
                  offpolicy_labeling=True):
    goals = numpy.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
    numpy.random.RandomState(seed=42).shuffle(goals)
    
    for idx in epoch_ids:
        goal_idx = (work_id + idx * world_work) % len(goals)
        goal = goals[goal_idx]
        
        horizon = dim * 3  
        env = DarkroomEnvAdapter(dim, goal, horizon, max_steps)
        
        results = run_epoch(idx, env, max_steps, offpolicy_labeling=offpolicy_labeling)
        
        file_path = f'{path_name}/record-{idx:06d}'
        create_directory(file_path)
        
        numpy.save("%s/observations.npy" % file_path, results["states"])
        numpy.save("%s/prompts.npy" % file_path, results["prompts"])
        numpy.save("%s/tags.npy" % file_path, results["tags"])
        numpy.save("%s/actions_behavior.npy" % file_path, results["actions_behavior"])
        numpy.save("%s/rewards.npy" % file_path, results["rewards"])
        numpy.save("%s/actions_label.npy" % file_path, results["actions_label"])
        
        with open("%s/task_info.pkl" % file_path, 'wb') as f:
            task_info = {
                "dim": dim,
                "goal": goal,
                "horizon": horizon
            }
            pickle.dump(task_info, f)
        
        print(f"Saved data for dim={dim}, goal={goal} to {file_path}")


if __name__=="__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./darkroom_data/", help="output directory")
    parser.add_argument("--dim", type=int, default=8, help="dimension of the darkroom grid, default:8")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--offpolicy_labeling", type=int, default=1, help="enable offpolicy labeling (DAgger), default:True")
    parser.add_argument("--epochs", type=int, default=64, help="number of epochs/environments to generate, default:64")
    parser.add_argument("--start_index", type=int, default=0, help="start id of the record number")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
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
            target=dump_darkroom, 
            args=(
                worker_id, 
                args.workers, 
                args.output_path, 
                range(n_b, n_e), 
                args.dim,
                args.max_steps, 
                (args.offpolicy_labeling > 0)
            )
        )
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join()