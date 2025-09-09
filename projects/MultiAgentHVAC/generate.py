import os
import sys
from airsoul.models import OmniRL_MultiAgent
from airsoul.utils import GeneratorRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hvac_epoch import HVACGenerator

if __name__ == "__main__":
    runner=GeneratorRunner()
    runner.start(OmniRL_MultiAgent, HVACGenerator)