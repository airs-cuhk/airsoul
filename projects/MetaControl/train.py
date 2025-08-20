import os
import sys
from airsoul.models import OmniRL, MLPDecision
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omnicontrol_epoch import OmniControlEpoch

if __name__ == "__main__":
    runner=Runner()
    runner.start(OmniRL, OmniControlEpoch, OmniControlEpoch)
#   runner.start(MLPDecision, OmniControlEpoch, OmniControlEpoch)