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
from l3c.mazeworld.agents import SmartSLAMAgent

    
class MazeNoisyExpertAgent(SmartSLAMAgent):
    def __init__(self, 
                 noise_initial_range=[0.5, 1.5],
                 noise_final_range=[0.0, 0.5],
                 memory_keep_prob_range=[0.05, 1.0],
                 horizon=1000, # Noise decay factor
                 **kwargs):
        super().__init__(**kwargs)
        self.noise_initial = random.uniform(*noise_initial_range)
        self.noise_final = random.uniform(*noise_final_range)
        self.memory_keep_ratio = random.uniform(*memory_keep_prob_range)
        self.steps = 0
        self.horizon = horizon
        self.noise_array = numpy.linspace(self.noise_initial, self.noise_final, self.horizon)

    
    def step(self, observation, r):
        act = super().step(observation, r)
        self.steps = min(self.steps + 1, self.horizon)
        if(random.random() < self.noise_array[self.steps - 1]):
            return self.maze_env.action_space.sample(), 'rnd' # Random action
        else:
            return act, f'exp_{self.memory_keep_ratio}'
        
    def __repr__(self):
        return f"MazeNoisyExpertAgent with noise {self.noise_initial:.2f} to {self.noise_final:.2f} and expert_{self.memory_keep_ratio:.2f}"