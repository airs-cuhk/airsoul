import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0, 1, 2, 3, 4, 5, 6, 7" #
import sys

custom_paths = [
    '/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev',
    '/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/airsoul'
]
for path in custom_paths[::-1]:
    sys.path.insert(0, path)

from airsoul.models import E2EObjNavSA
from airsoul.utils import Runner
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# append the parent's parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from maze_epoch import MazeEpochVAE, MazeEpochCausal, MazeEpochCausalSplit

if __name__ == "__main__":
    runner=Runner()
    print(f"Visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    runner.start(E2EObjNavSA, [MazeEpochCausalSplit], [MazeEpochCausalSplit])
