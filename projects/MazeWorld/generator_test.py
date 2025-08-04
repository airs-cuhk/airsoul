import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"  # Set the visible GPUs for the process
import sys
from airsoul.models import E2EObjNavSA
from airsoul.utils import GeneratorRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import ARgeneral_generator, general_generator

if __name__ == "__main__":
    print("Start generator test")
    print(f"Visible GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    runner=GeneratorRunner()
    runner.start(E2EObjNavSA, general_generator)