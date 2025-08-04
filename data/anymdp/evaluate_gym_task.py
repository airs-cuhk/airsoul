import gymnasium as gym
import xenoverse
import numpy
import multiprocessing
import argparse

from xenoverse.utils import pseudo_random_seed
from airsoul.utils import TabularQ

def test_gym_task(result, ns=16, na=4, 
                     max_epochs_rnd=200, max_epochs_q=5000, 
                     sub_sample=100, gamma=0.9):
    env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    epoch_rews_rnd = []
    epoch_rews_q =[]
    epoch_steps_q = []

    for epoch in range(max_epochs_rnd):
        obs, info = env.reset()
        epoch_rew = 0
        done = False
        while not done:
            act = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(act)
            epoch_rew += rew
            done = terminated or truncated
        epoch_rews_rnd.append(epoch_rew)

    rnd_perf = numpy.mean(epoch_rews_rnd)

    solver_q = TabularQ(ns, na, gamma=gamma)
    for epoch in range(max_epochs_q):
        last_obs, info = env.reset()
        epoch_rew = 0
        done = False
        epoch_step = 0
        while not done:
            act = solver_q.policy(last_obs)
            obs, rew, terminated, truncated, info = env.step(act)
            solver_q.learner(last_obs, act, obs, rew, terminated)
            epoch_rew += rew
            epoch_step += 1
            last_obs = obs
            done = terminated or truncated

        epoch_rews_q.append(epoch_rew)
        epoch_steps_q.append(epoch_step)

    steps = numpy.shape(epoch_rews_q)[0] // sub_sample
    eff_size = steps * sub_sample
    epoch_rews_q= numpy.reshape(epoch_rews_q[:eff_size], (-1, sub_sample))
    normalized_steps = numpy.reshape(numpy.array(epoch_steps_q)[:eff_size], (-1, sub_sample))
    result.put((numpy.mean(epoch_rews_q, axis=-1), numpy.mean(normalized_steps, axis=-1), rnd_perf))
            
if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./res.txt", help="Output result path")
    parser.add_argument("--state_num", type=int, default=16, help="state num, default:128")
    parser.add_argument("--action_num", type=int, default=4, help="action num, default:5")
    parser.add_argument("--min_state_space", type=int, default=16, help="minimum state dim in task, default:8")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--max_epochs", type=int, default=10000, help="multiple epochs:default:1000")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--sub_sample", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    # Data Generation
    processes = []
    res = multiprocessing.Manager().Queue()
    for worker_id in range(args.workers):
        process = multiprocessing.Process(target=test_gym_task, 
                args=(res,
                      int(args.state_num), 
                      int(args.action_num),
                      200, 
                      args.max_epochs, 
                      args.sub_sample,
                      args.gamma))
        processes.append(process)
        process.start()

    for process in processes:
        process.join() 
    
    scores = []
    steps = []
    rnds = []
    while not res.empty():
        score, step, rnd = res.get()
        scores.append(score)
        steps.append(step)
        rnds.append(rnd)

    lower_bound = numpy.mean(rnds)
    scores = numpy.array(scores)
    print(scores.shape)
    higher_bound = numpy.mean(numpy.max(scores[:, -50:], axis=-1))
    scores = (scores - lower_bound) / (higher_bound - lower_bound)

    s_mean = numpy.mean(scores, axis=0)
    s2_mean = numpy.mean(scores**2, axis=0)
    std = numpy.sqrt(s2_mean - s_mean**2)
    
    steps = numpy.array(steps)
    sp_mean = numpy.mean(steps, axis=0)
    sp2_mean = numpy.mean(steps**2, axis=0)
    pstd = numpy.sqrt(sp2_mean - sp_mean**2)

    deltas = numpy.array(rnds)
    d_mean = numpy.mean(deltas)
    d2_mean = numpy.mean(deltas**2)
    dstd = numpy.sqrt(d2_mean - d_mean**2)

    fout = open(args.output_path, "w")
    for i in range(s_mean.shape[0]):
        fout.write("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % 
              ((i+1)*args.sub_sample, s_mean[i], std[i], sp_mean[i], pstd[i], d_mean, dstd))
    fout.close()