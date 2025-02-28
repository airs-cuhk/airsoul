import os
import sys
from airsoul.models import E2EObjNavSA
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MazeEpochVAE, MazeEpochCausal

if __name__ == "__main__":
    runner=Runner()
    runner.start(E2EObjNavSA, [MazeEpochVAE, MazeEpochCausal], [MazeEpochVAE, MazeEpochCausal])
