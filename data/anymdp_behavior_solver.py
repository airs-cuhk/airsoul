#!/usr/bin/env python
# coding=utf8
# File: dump_maze.py
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
from airsoul.utils import tag_vocabulary, tag_mapping_gamma, tag_mapping_id
from l3c.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ
from l3c.utils import pseudo_random_seed


class AnyPolicySolver(object):
    """
    Sample Any Policy for AnyMDP
    """
    def __init__(self, env):
        if(not env.task_set):
            raise Exception("AnyMDPEnv is not initialized by 'set_task', must call set_task first")
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        ent_1 = numpy.random.exponential(2.0)
        ent_2 = numpy.random.exponential(1.0e-5)
        self.policy_matrix = numpy.random.normal(size=(self.n_states, self.n_actions), scale=ent_1)
        self.policy_matrix = numpy.exp(self.policy_matrix)
        self.policy_matrix /= numpy.sum(self.policy_matrix, axis=1, keepdims=True)
        self.policy_transfer = numpy.eye(self.n_actions, self.n_actions) + numpy.random.normal(scale=ent_2, size=(self.n_actions, self.n_actions))
        self.policy_transfer = numpy.clip(self.policy_transfer, 0, 1)
        self.policy_transfer = self.policy_transfer / numpy.sum(self.policy_transfer, axis=1, keepdims=True)

    def learner(self, *args, **kwargs):
        self.policy_matrix = numpy.matmul(self.policy_matrix, self.policy_transfer)

    def policy(self, state):
        return numpy.random.choice(self.n_actions, size=1, p=self.policy_matrix[state])[0], tag_mapping_id['rnd']
    
class AnyMDPOptNoiseDistiller(object):
    def __init__(self, env, opt_solver=None, noise_start=1.0, noise_decay=None):
        self.noise = noise_start
        self.nstate = env.observation_space.n
        self.naction = env.action_space.n
        self.opt_solver = opt_solver
        self.noise_decay = random.uniform(0.0, 1.0 / (self.nstate * self.naction))
    
    def learner(self, *args, **kwargs):
        self.noise -= self.noise_decay

    def policy(self, state):
        if(self.opt_solver is None or random.random() < self.noise):
            action = random.randint(0, self.naction - 1)
            act_type = tag_mapping_id['rnd']
        else:
            action, act_type = self.opt_solver.policy(state)
        return action, act_type

class AnyMDPOTSNoiseDistiller(AnyMDPSolverOTS):
    def __init__(self, env, 
                 max_steps=16000):
        if(random.random() < 0.5):
            c = 0.005
            alpha = 0.01
            gamma = random.uniform(0.90, 0.99)
            self.noise = 0.0
        else:
            c = 0.005 * random.exponential(1.0)
            alpha = 0.01 * random.exponential(1.0)
            max_steps = random.uniform(100, max_steps)
            gamma = 0.99
            self.noise = random.uniform(0.1, 0.5)
        super().__init__(env, 
                        gamma=gamma,
                        c=c,
                        alpha=alpha,
                        max_steps=max_steps)
        self.nstate = env.observation_space.n
        self.naction = env.action_space.n
        self.noise_decay = random.uniform(0.0, 0.10 / (self.nstate * self.naction))
    
    def learner(self, *args, **kwargs):
        super().learner(*args, **kwargs)
        self.noise -= self.noise_decay

    def policy(self, state):
        if(random.random() < self.noise):
            action = random.randint(0, self.naction - 1)
            act_type = tag_mapping_id['rnd']
        else:
            action = super().policy(state)
            act_type = tag_mapping_id['exp1']
        return action, act_type
    
class AnyMDPQNoiseDistiller(AnyMDPSolverQ):
    def __init__(self, env, 
                 max_steps=16000):
        if(random.random() < 0.5):
            c = 0.005
            alpha = 0.01
            gamma = random.uniform(0.90, 0.99)
            self.noise = 0.0
        else:
            c = 0.005 * random.exponential(1.0)
            alpha = 0.01 * random.exponential(1.0)
            max_steps = random.uniform(100, max_steps)
            gamma = 0.99
            self.noise = random.uniform(0.1, 0.5)
        super().__init__(env, 
                        gamma=gamma,
                        c=c,
                        alpha=alpha,
                        max_steps=max_steps)
        self.nstate = env.observation_space.n
        self.naction = env.action_space.n
        self.noise_decay = random.uniform(0.0, 0.10 / (self.nstate * self.naction))
    
    def learner(self, *args, **kwargs):
        super().learner(*args, **kwargs)
        self.noise -= self.noise_decay

    def policy(self, state):
        if(random.random() < self.noise):
            action = random.randint(0, self.naction - 1)
            act_type = tag_mapping_id['rnd']
        else:
            action = super().policy(state)
            act_type = tag_mapping_id['exp2']
        return action, act_type
    
class AnyMDPOTSOpter(AnyMDPSolverOTS):
    def __init__(self, env, solver_opt=None,
                 max_steps=16000):
        super().__init__(env, 
                        gamma=0.99,
                        c=0.005,
                        alpha=0.01,
                        max_steps=max_steps)
        self.nstate = env.observation_space.n
        self.naction = env.action_space.n
        self.solver_opt = solver_opt
        self.noise = random.uniform(0.0, 1.0)
        self.noise_decay = random.uniform(0.0, 0.10 / (self.nstate * self.naction))
        self.noise_end = random.uniform(-0.5, 0.5)
        self.opt = random.uniform(-2.0, -0.5)
        self.opt_end = random.uniform(0.0, 0.20)
        self.opt_inc = random.uniform(0.0, 0.10 / (self.nstate * self.naction))

    def learner(self, *args, **kwargs):
        super().learner(*args, **kwargs)
        self.noise -= self.noise_decay
        self.noise = max(self.noise_end, self.noise)
        self.opt += self.opt_inc
        self.opt = min(self.opt_end, self.opt)

    def policy(self, state):
        if(random.random() < self.noise):
            action = random.randint(0, self.naction - 1)
            act_type = tag_mapping_id['rnd']
        elif(random.random() < self.opt and self.solver_opt is not None):
            action, act_type = self.solver_opt.policy(state)
        else:
            action = super().policy(state)
            act_type = tag_mapping_id['exp1']
        return action, act_type


class AnyMDPOpter(AnyMDPSolverOpt):
    def __init__(self, i, *args, **kwargs):
        self.tag = f'opt{i}'
        self.prompts = tag_mapping_id[self.tag]
        super().__init__(*args, **kwargs, gamma=tag_mapping_gamma[self.tag])
    
    def learner(self, *args, **kwargs):
        pass

    def policy(self, state):
        return super().policy(state), int(self.prompts)
