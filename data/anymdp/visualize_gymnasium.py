import gymnasium as gym
import xenoverse
import numpy
import multiprocessing
import argparse
import sys
from copy import deepcopy
sys.path.append("../../projects/OmniRL")
from gym_env_wapper import DiscreteEnvWrapper

from xenoverse.anymdp import AnyMDPTaskSampler
from xenoverse.anymdp import AnyMDPSolverOpt, AnyMDPSolverMBRL, AnyMDPSolverQ
from xenoverse.utils import pseudo_random_seed
from xenoverse.anymdp.visualizer import anymdp_task_visualizer
from xenoverse.anymdp.anymdp_env import map_transition_reward
from xenoverse.anymdp.solver import update_value_matrix

def make_env():
    env = DiscreteEnvWrapper(gym.make("Pendulum-v1", g=9.81), 
            'pendulum', 
            action_space=5, 
            state_space_dim1=12, 
            state_space_dim2=5, 
            reward_shaping = False, 
            skip_frame=0)
    return env

def test_AnyMDP_task(max_epochs=10000,
                     gamma=0.99,
                     exploration=10.0):
    env = make_env()
    epoch_rews_q =[]
    epoch_steps_q = []

    solver = AnyMDPSolverMBRL(env, gamma=gamma, c=exploration)
    for epoch in range(max_epochs):
        last_obs, info = env.reset()
        epoch_rew = 0
        epoch_step = 0
        terminated, truncated = False, False
        solver.set_reset_states(last_obs)
        while not terminated and not truncated:
            act = solver.policy(last_obs)
            obs, rew, terminated, truncated, info = env.step(act)
            solver.learner(last_obs, act, obs, rew, terminated, truncated)
            epoch_rew += rew
            epoch_step += 1
            last_obs = obs
            if(epoch_step > 200):
                truncated = True
        epoch_rews_q.append(epoch_rew)
        epoch_steps_q.append(epoch_step)

    task = dict()
    task["ns"] = env.observation_space.n
    task["na"] = env.action_space.n
    task["transition"] = solver.t_mat
    task["reward"] = solver.r_mat
    task["s_e"] = solver.s_e
    task["s_0"] = solver.s_0
    task["s_0_prob"] = solver.s_0_prob
    task["state_mapping"] = list(range(solver.ns))

    return task

def rearrange_states(task, K=5):
    trans_ss = numpy.sum(task["transition"], axis=1)
    ra_task = deepcopy(task)

    s_map = []
    for s in task["s_0"]:
        s_map.append(s)

    vm = numpy.zeros((task["ns"], task["na"]), dtype='float32')
    vm = update_value_matrix(task["transition"], task["reward"], 0.99, vm)
    vsm = numpy.max(vm, axis=-1)
    print(task["s_e"])

    while len(s_map) < task["ns"]:
        s_trans_sum = []
        for s in range(len(trans_ss)):
            if(s in s_map):
                continue
            p2s = numpy.mean(trans_ss[s_map, [s for _ in range(len(s_map))]], axis=0)
            if(p2s > 1.0e-6):
                s_trans_sum.append((s, vsm[s], p2s))
        s_sorted_trans = sorted(s_trans_sum, key=lambda x:x[2], reverse=True)
        s_sorted_trans = sorted(s_sorted_trans[:K], key=lambda x:x[1], reverse=False)
        s_map.append(s_sorted_trans[0][0])

    # make the goal last
    for s in task["s_e"]:
        if(numpy.sum(task["reward"][:, :, s] > 0) and s_map.index(s) > task['ns'] // 2 and s_map.index(s) != task["ns"] - 1): # mv the goal to the end
            s_map[-1], s_map[s_map.index(s)] = s_map[s_map.index(s)], s_map[-1]

    s_map_inv = list(range(task["ns"]))
    for i, s in enumerate(s_map):
        s_map_inv[s] = i

    ra_task["transition"] *= 0.0
    ra_task["reward"] *= 0.0
    ra_task["transition"], ra_task["reward"] = map_transition_reward(
                        task["transition"], 
                        task["reward"], 
                        ra_task["transition"], 
                        ra_task["reward"], 
                        s_map_inv)
    ra_task["state_mapping"] = s_map
    
    ra_task["s_0"] = []
    ra_task["s_e"] = []
    for s in task["s_0"]:
        ra_task["s_0"].append(s_map_inv[s])
    for s in task["s_e"]:
        ra_task["s_e"].append(s_map_inv[s])

    return ra_task

if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./gymnasium", help="Output result path")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=0.10)
    parser.add_argument("--trans_top_K", type=int, default=5)
    args = parser.parse_args()

    task = test_AnyMDP_task(max_epochs=5000,
                     gamma=0.99,
                     exploration=5.0)
    task = rearrange_states(task, args.trans_top_K)
    
    anymdp_task_visualizer(task, 
                    need_lengends=False, 
                    need_ticks=False,
                    show_gui=True, 
                    file_path=args.output_path)