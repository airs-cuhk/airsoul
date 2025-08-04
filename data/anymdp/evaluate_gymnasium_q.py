import gymnasium as gym
import xenoverse
import numpy
import multiprocessing
import argparse
import sys
sys.path.append("../../projects/OmniRL")
from gym_env_wapper import DiscreteEnvWrapper

from xenoverse.anymdp.test_utils import train
from xenoverse.anymdp import AnyMDPTaskSampler
from xenoverse.anymdp import AnyMDPSolverOpt, AnyMDPSolverQ
from xenoverse.utils import pseudo_random_seed

sub_sample=100

def make_env(env_id):
    if(env_id.lower().find('pendulum1') > -1):
        env = DiscreteEnvWrapper(gym.make("Pendulum-v1", g=9.81), 
                'pendulum', 
                action_space=5, 
                state_space_dim1=12, 
                state_space_dim2=5, 
                reward_shaping = False, 
                skip_frame=0)
        env_opt_reward = -385
    if(env_id.lower().find('pendulum2') > -1):
        env = DiscreteEnvWrapper(gym.make("Pendulum-v1", g=5.0), 
                'pendulum', 
                action_space=5, 
                state_space_dim1=12, 
                state_space_dim2=5, 
                reward_shaping = False, 
                skip_frame=0)
        env_opt_reward = -200
    if(env_id.lower().find('pendulum3') > -1):
        env = DiscreteEnvWrapper(gym.make("Pendulum-v1", g=1.0), 
                'pendulum', 
                action_space=5, 
                state_space_dim1=12, 
                state_space_dim2=5, 
                reward_shaping = False, 
                skip_frame=0)
        env_opt_reward = -40
    if(env_id.lower().find('lake1') > -1):
        env = DiscreteEnvWrapper(gym.make("FrozenLake-v1", is_slippery=True), 
                'pendulum', 
                action_space=4, 
                state_space_dim1=4, 
                state_space_dim2=4, 
                reward_shaping = False, 
                skip_frame=0)
        env_opt_reward = 0.833
    if(env_id.lower().find('lake2') > -1):
        env = DiscreteEnvWrapper(gym.make("FrozenLake-v1", is_slippery=False), 
                'pendulum', 
                action_space=5, 
                state_space_dim1=12, 
                state_space_dim2=5, 
                reward_shaping = False, 
                skip_frame=0)
        env_opt_reward = 1.0
    return env, env_opt_reward

def test_task(worker_id, 
                    env_id, 
                     result, 
                     max_epochs):
    gamma=0.99
    lr=0.20

    env, opt_score = make_env(env_id)

    rnd_score, _, rnd_steps = train(env, max_epochs=200, gamma=0.99, solver_type='random')

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
    parser.add_argument("--env", type=str, default="pendulum1", help="env id: pendulum1, pendulum2, pendulum3, lake1, lake2")
    parser.add_argument("--max_epochs", type=int, default=10000, help="multiple epochs:default:10000")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.20)

    args = parser.parse_args()

    # Data Generation
    processes = []
    res = multiprocessing.Manager().Queue()
    for worker_id in range(args.workers):
        process = multiprocessing.Process(target=test_task, 
                args=(worker_id,
                      args.env,
                      res,
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