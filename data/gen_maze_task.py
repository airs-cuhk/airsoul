import argparse
import pickle
import random
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler, Resampler
from l3c.mazeworld.agents import SmartSLAMAgent


if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="9,15", help="a list of scales separated with comma to randomly choose")
    parser.add_argument("--landmarks", type=str, default="6,10", help="landmarks:a list of number of landmarks")
    parser.add_argument("--task_number", type=int, default=1, help="multiple epochs:default:1")
    parser.add_argument("--output_path", type=str, help="output file name")
    args = parser.parse_args()

    #used for dump tasks only
    tasks = []
    scales = list(map(int, args.scale.split(',')))
    landmarks = list(map(int, args.landmarks.split(',')))
    for idx in range(args.task_number):
        task = MazeTaskSampler(n_range=scales,
                landmarks_number_range=landmarks,
                commands_sequence=10000,
                verbose=True)
        tasks.append(task)
    if(args.output_path.find('.pkl') < 0):
        output_file = args.output_path + ".pkl"
    else:
        output_file = args.output_path
    print(f"Writing tasks to {output_file} with following configuration")
    with open(output_file, 'wb') as fw:
        pickle.dump(tasks, fw)
