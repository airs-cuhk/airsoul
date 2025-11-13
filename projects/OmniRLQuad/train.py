import os
import sys
import sys

custom_paths = [
    '/home/wangfan/airsoul',
    '/home/wangfan/airsoul/airsoul'
]
for path in custom_paths[::-1]:
    sys.path.insert(0, path)

from airsoul.models import OmniRL
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from omnirl_epoch import OmniRLEpoch

import airsoul
print(airsoul.__file__)

if __name__ == "__main__":
    runner=Runner()
    runner.start(OmniRL, OmniRLEpoch, OmniRLEpoch)
