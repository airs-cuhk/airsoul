import argparse
import pickle
import random
from l3c.anymdp import AnyMDPTaskSampler



if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_states", type=int, default=128, help="number of states")
    parser.add_argument("--n_actions", type=int, default=5, help="number of actions")
    parser.add_argument("--task_number", type=int, default=1, help="multiple epochs:default:1")
    parser.add_argument("--output_path", type=str, help="output file name")
    args = parser.parse_args()

    #used for dump tasks only
    tasks = []
    for idx in range(args.task_number):
        task = AnyMDPTaskSampler(args.n_states, args.n_actions)
        tasks.append(task)
    if(args.output_path.find('.pkl') < 0):
        output_file = args.output_path + ".pkl"
    else:
        output_file = args.output_path
    print(f"Writing tasks to {output_file} with following configuration")
    with open(output_file, 'wb') as fw:
        pickle.dump(tasks, fw)
