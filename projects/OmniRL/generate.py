import os
import sys
from airsoul.models import OmniRL
from airsoul.utils import GeneratorRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omnirl_epoch import OmniRLGenerator, MultiAgentGenerator

if __name__ == "__main__":
    runner=GeneratorRunner()
    runner.start(OmniRL, OmniRLGenerator)