import os
import sys
from airsoul.models import LanguageModel
from airsoul.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lm_epoch import LMEpoch

if __name__ == "__main__":
    runner=Runner()
    runner.start(LanguageModel, LMEpoch, LMEpoch)