import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7" # "4, 5, 6, 7"
import sys
from airsoul.models import E2EObjNavSA
from airsoul.utils import Runner
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MazeEpochVAE, MazeEpochCausal

if __name__ == "__main__":
    
    torch.cuda.empty_cache()
    runner=Runner()
    print(f"Visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    runner.start(E2EObjNavSA, [MazeEpochCausal], [MazeEpochCausal])
    # runner.start(E2EObjNavSA, [MazeEpochVAE], [MazeEpochVAE])
    # runner.start(E2EObjNavSA, [MazeEpochVAE, MazeEpochCausal], [MazeEpochVAE, MazeEpochCausal])
