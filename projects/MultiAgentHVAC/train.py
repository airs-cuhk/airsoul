import os
import sys
from airsoul.models import OmniRL_MultiAgent
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hvac_epoch import HVACEpoch

if __name__ == "__main__":
    runner=Runner()
    runner.start(OmniRL_MultiAgent, HVACEpoch, HVACEpoch)