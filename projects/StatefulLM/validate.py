import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

custom_paths = [
    '/goosefsx/91mst04h/airs/qxg/czy/airsoul',
    '/goosefsx/91mst04h/airs/qxg/czy/airsoul/airsoul'
]
for path in custom_paths[::-1]:
    sys.path.insert(0, path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
os.environ['NCCL_TIMEOUT'] = '3600'


# # CUDA_LAUNCH_BLOCKING=1
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# # Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

from airsoul.models import StatefulLM
from airsoul.utils import Runner

from lm_epoch import LMEpoch

if __name__ == "__main__":
    runner = Runner()
    runner.start(StatefulLM, [], LMEpoch, extra_info='validate')
