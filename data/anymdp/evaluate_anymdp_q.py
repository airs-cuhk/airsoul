import gymnasium as gym
import xenoverse
import numpy
import multiprocessing
import argparse
import numpy

from copy import deepcopy
from xenoverse.anymdp import AnyMDPTaskSampler
from xenoverse.anymdp import AnyMDPSolverOpt, AnyMDPSolverMBRL, AnyMDPSolverQ
from xenoverse.anymdp.test_utils import train
from xenoverse.utils import pseudo_random_seed

sub_sample=100

# for comparison
def resample_task(task, t=True, r=True):
    new_task = deepcopy(task)
    if(t):
        new_task["transition"] = numpy.clip(numpy.random.normal(size=task["transition"].shape), 0.0, None)
        new_task["transition"] = new_task["transition"] / numpy.sum(new_task["transition"], axis=-1, keepdims=True)
    if(r):
        new_task["reward"] = numpy.random.normal(size=task["reward"].shape)
    return new_task

def test_AnyMDP_task(worker_id, 
                     result, 
                     ns, 
                     na, 
                     max_epochs):
    gamma=0.99
    lr=0.20

    env = gym.make("anymdp-v0")
    task = AnyMDPTaskSampler(ns, na, verbose=True)
    #task = resample_task(task)
    env.set_task(task)

    opt_score, _, opt_steps = train(env, max_epochs=200, gamma=gamma, solver_type='opt')
    rnd_score, _, rnd_steps = train(env, max_epochs=200, gamma=gamma, solver_type='random')

    opt_score= numpy.mean(opt_score)
    rnd_score= numpy.mean(rnd_score)

    if(opt_score - rnd_score < 0.01):
        print("[Trivial task], skip")
        return
    
    def normalize(score):
        return numpy.clip((score - rnd_score) / (opt_score - rnd_score), 0.0, 1.0)
    
    q_train_score, q_test_score, q_steps = train(env, max_epochs=max_epochs, gamma=gamma, solver_type='q', lr=lr, test_interval=sub_sample)
    #q_train_score = normalize(q_train_score)
    q_test_score = normalize(q_test_score)
    q_steps = numpy.cumsum(q_steps) * sub_sample

    result.put((q_test_score, q_steps, opt_score - rnd_score))
    print(f"finish task： {worker_id}")
            
if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./res.txt", help="Output result path")
    parser.add_argument("--state_num", type=int, default=16, help="state num, default:16")
    parser.add_argument("--action_num", type=int, default=5, help="action num, default:5")
    parser.add_argument("--max_epochs", type=int, default=10000, help="multiple epochs:default:10000")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--learning_rate", type=float, default=0.01)

    args = parser.parse_args()

    # Data Generation
    processes = []
    res = multiprocessing.Manager().Queue()
    for worker_id in range(args.workers):
        process = multiprocessing.Process(target=test_AnyMDP_task, 
                args=(worker_id,
                      res,
                      int(args.state_num), 
                      int(args.action_num),
                      args.max_epochs))
        processes.append(process)
        process.start()

    for process in processes:
        process.join() 
    
    scores = []
    steps = []
    deltas = []
    while not res.empty():
        score, step, delta = res.get()
        scores.append(score)
        steps.append(step)
        deltas.append(delta)

    scores = numpy.array(scores)
    s_mean = numpy.mean(scores, axis=0)
    s2_mean = numpy.mean(scores**2, axis=0)
    std = numpy.sqrt(s2_mean - s_mean**2)
    conf = 2.0 * std / numpy.sqrt(scores.shape[0])
    print(f"Number of valid tasks: {scores.shape[0]}")

    steps = numpy.array(steps)
    sp_mean = numpy.mean(steps, axis=0)
    sp2_mean = numpy.mean(steps**2, axis=0)
    pstd = numpy.sqrt(sp2_mean - sp_mean**2)

    deltas = numpy.array(deltas)
    d_mean = numpy.mean(deltas)
    d2_mean = numpy.mean(deltas**2)
    dstd = numpy.sqrt(d2_mean - d_mean**2)

    fout = open(args.output_path, "w")
    for i in range(s_mean.shape[0]):
        fout.write("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % 
              ((i+1)*sub_sample, s_mean[i], conf[i], sp_mean[i], pstd[i], d_mean, dstd))
    fout.close()
