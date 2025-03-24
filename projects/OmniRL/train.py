import os
import sys
from airsoul.models import OmniRL
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omnirl_epoch import OmniRLEpoch

if __name__ == "__main__":
    runner=Runner()
    runner.start(OmniRL, OmniRLEpoch, OmniRLEpoch)