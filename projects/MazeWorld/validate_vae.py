import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
import sys
from airsoul.models import E2EObjNavSA
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MazeEpochVAE

if __name__ == "__main__":
    runner=Runner()
    runner.start(E2EObjNavSA, [], [MazeEpochVAE], 'validate')
