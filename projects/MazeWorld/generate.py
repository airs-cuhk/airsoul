import os
import sys
from airsoul.models import E2EObjNavSA
from airsoul.utils import GeneratorRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MAZEGenerator

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    runner=GeneratorRunner()
    runner.start(E2EObjNavSA, MAZEGenerator)