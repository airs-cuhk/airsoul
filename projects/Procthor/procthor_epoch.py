import os
import sys
sys.path.append("/pfs/pfs-r36Cge/zsj/yuxuan/airsoul-diffusion/projects/Procthor")
sys.path.append("/pfs/pfs-r36Cge/zsj/yuxuan/airsoul-diffusion/projects/Procthor/third_party/spoc_robot_training")

os.environ["OBJAVERSE_DATA_BASE_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_assets"
os.environ["OBJAVERSE_HOUSES_BASE_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses"
os.environ["OBJAVERSE_DATA_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_assets/2023_07_28"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/2023_07_28"


import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from PIL import Image
# from airsoul.dataloader import segment_iterator
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal
from airsoul.utils import custom_load_model, noam_scheduler, LinearScheduler
from airsoul.utils import Configure, DistStatistics, rewards2go
from airsoul.utils import EpochManager, GeneratorBase
from airsoul.utils import noam_scheduler, LinearScheduler
from airsoul.utils import weighted_loss, img_pro, img_post
from airsoul.dataloader.prefetch_dataloader import PrefetchDataLoader
from airsoul.dataloader import ProcthorDataSet, MazeDataSet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import ai2thor.fifo_server

import warnings
from ai2thor.platform import CloudRendering

from third_party.spoc_robot_training.utils.data_generation_utils.navigation_utils import \
    is_any_object_sufficiently_visible_and_in_center_frame, get_room_id_from_location

from third_party.spoc_robot_training.utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS, \
    AGENT_ROTATION_DEG, AGENT_MOVEMENT_CONSTANT

from expert.demo_test_grid import Expert, initialize_expert_controller, save_data_json
from expert.action_maze import shortest_path_to_actions, plot_grid_map, determine_turn_actions_
import time

import copy
from typing import Dict, Optional, Sequence, List, Tuple, Iterable, Literal
def _is_goal_in_range(controller) -> bool:
    return any(
        obj
        for obj in controller.last_event.metadata["objects"]
        if obj["visible"] and obj["objectType"] == controller.task_info["object_type"]
    )

def horiz_dist(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Horizontal (XZ-plane) Euclidean distance."""
    import math
    return math.hypot(p1["x"] - p2["x"], p1["z"] - p2["z"])

def if_close_to_goal(evt, goal_pos, dist_thresh=0.2):
    if horiz_dist(evt.metadata["agent"]["position"], goal_pos) <= dist_thresh:
        return True
    return False

def get_position_rotions(agent_location):
    """Get positions and rotations from a list of positions."""
    return [
        agent_location['position']['x'], agent_location['position']['y'], agent_location['position']['z'],
        agent_location['rotation']['x'], agent_location['rotation']['y'], agent_location['rotation']['z']
    ]



ASSETS_VERSION = "2023_07_28"
OBJAVERSE_DATA_DIR = os.path.abspath(
    os.environ.get("OBJAVERSE_DATA_DIR", os.path.expanduser("~/.objathor-assets"))
)

if not os.path.basename(OBJAVERSE_DATA_DIR) == ASSETS_VERSION:
    OBJAVERSE_DATA_DIR = os.path.join(OBJAVERSE_DATA_DIR, ASSETS_VERSION)

OBJAVERSE_ASSETS_DIR = os.environ.get(
    "OBJAVERSE_ASSETS_DIR", os.path.join(OBJAVERSE_DATA_DIR, "assets")
)
OBJAVERSE_ANNOTATIONS_PATH = os.environ.get(
    "OBJAVERSE_ANNOTATIONS_PATH", os.path.join(OBJAVERSE_DATA_DIR, "annotations.json.gz")
)
OBJAVERSE_HOUSES_DIR = os.environ.get("OBJAVERSE_HOUSES_DIR")

for var_name in ["OBJAVERSE_ASSETS_DIR", "OBJAVERSE_ANNOTATIONS_PATH", "OBJAVERSE_HOUSES_DIR"]:
    if locals()[var_name] is None:
        warnings.warn(f"{var_name} is not set.")
    else:
        locals()[var_name] = os.path.abspath(locals()[var_name])

if OBJAVERSE_HOUSES_DIR is None:
    warnings.warn("`OBJAVERSE_HOUSES_DIR` is not set.")
else:
    OBJAVERSE_HOUSES_DIR = os.path.abspath(OBJAVERSE_HOUSES_DIR)

# print(
#     f"Using"
#     f" '{OBJAVERSE_ASSETS_DIR}' for objaverse assets,"
#     f" '{OBJAVERSE_ANNOTATIONS_PATH}' for objaverse annotations,"
#     f" '{OBJAVERSE_HOUSES_DIR}' for procthor-objaverse houses."
# )

try:
    from ai2thor.hooks.procedural_asset_hook import (
        ProceduralAssetHookRunner,
        get_all_asset_ids_recursively,
        create_assets_if_not_exist,
    )
except ImportError:
    raise ImportError(
        "Cannot import `ProceduralAssetHookRunner`. Please install the appropriate version of ai2thor:\n"
        f"```\npip install --extra-index-url https://ai2thor-pypi.allenai.org"
        f" ai2thor==0+5e43486351ac6339c399c199e601c9dd18daecc3\n```"
    )



ASSETS_VERSION = "2023_07_28"
OBJAVERSE_DATA_DIR = os.path.abspath(
    os.environ.get("OBJAVERSE_DATA_DIR", os.path.expanduser("~/.objathor-assets"))
)

if not os.path.basename(OBJAVERSE_DATA_DIR) == ASSETS_VERSION:
    OBJAVERSE_DATA_DIR = os.path.join(OBJAVERSE_DATA_DIR, ASSETS_VERSION)

OBJAVERSE_ASSETS_DIR = os.environ.get(
    "OBJAVERSE_ASSETS_DIR", os.path.join(OBJAVERSE_DATA_DIR, "assets")
)
OBJAVERSE_ANNOTATIONS_PATH = os.environ.get(
    "OBJAVERSE_ANNOTATIONS_PATH", os.path.join(OBJAVERSE_DATA_DIR, "annotations.json.gz")
)
OBJAVERSE_HOUSES_DIR = os.environ.get("OBJAVERSE_HOUSES_DIR")

for var_name in ["OBJAVERSE_ASSETS_DIR", "OBJAVERSE_ANNOTATIONS_PATH", "OBJAVERSE_HOUSES_DIR"]:
    if locals()[var_name] is None:
        warnings.warn(f"{var_name} is not set.")
    else:
        locals()[var_name] = os.path.abspath(locals()[var_name])

if OBJAVERSE_HOUSES_DIR is None:
    warnings.warn("`OBJAVERSE_HOUSES_DIR` is not set.")
else:
    OBJAVERSE_HOUSES_DIR = os.path.abspath(OBJAVERSE_HOUSES_DIR)

# print(
#     f"Using"
#     f" '{OBJAVERSE_ASSETS_DIR}' for objaverse assets,"
#     f" '{OBJAVERSE_ANNOTATIONS_PATH}' for objaverse annotations,"
#     f" '{OBJAVERSE_HOUSES_DIR}' for procthor-objaverse houses."
# )

try:
    from ai2thor.hooks.procedural_asset_hook import (
        ProceduralAssetHookRunner,
        get_all_asset_ids_recursively,
        create_assets_if_not_exist,
    )
except ImportError:
    raise ImportError(
        "Cannot import `ProceduralAssetHookRunner`. Please install the appropriate version of ai2thor:\n"
        f"```\npip install --extra-index-url https://ai2thor-pypi.allenai.org"
        f" ai2thor==0+5e43486351ac6339c399c199e601c9dd18daecc3\n```"
    )


# class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.last_asset_id_set = set()

#     def Initialize(self, action, controller):
#         if self.asset_limit > 0:
#             return controller.step(
#                 action="DeleteLRUFromProceduralCache", assetLimit=self.asset_limit
#             )

#     def CreateHouse(self, action, controller):
#         house = action["house"]
#         asset_ids = get_all_asset_ids_recursively(house["objects"], [])
#         asset_ids_set = set(asset_ids)
#         if not asset_ids_set.issubset(self.last_asset_id_set):
#             controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)
#             self.last_asset_id_set = set(asset_ids)

#         return create_assets_if_not_exist(
#             controller=controller,
#             asset_ids=asset_ids,
#             asset_directory=self.asset_directory,
#             asset_symlink=self.asset_symlink,
#             stop_if_fail=self.stop_if_fail,
#         )

# AGENT_MOVEMENT_CONSTANT = 0.2
# INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 128, 128
# MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER = (
#     0.8673349051766235  # Computed with fixed arm agent, should have pairity with real
# )
# INTEL_VERTICAL_FOV = 59
# STRETCH_COMMIT_ID = "5e43486351ac6339c399c199e601c9dd18daecc3" # from stretch_initialization_utils.py
# MAXIMUM_SERVER_TIMEOUT = 2000  # default : 100 Need to increase this for cloudrendering
# OBJAVERSE_ASSETS_DIR = os.environ.get(
#     "OBJAVERSE_ASSETS_DIR", os.path.join(OBJAVERSE_DATA_DIR, "assets")
# )
# _ACTION_HOOK_RUNNER = ProceduralAssetHookRunnerResetOnNewHouse(
#     asset_directory=OBJAVERSE_ASSETS_DIR, asset_symlink=True, verbose=True, asset_limit=200
# )



# STRETCH_ENV_ARGS = dict(
#     gridSize=AGENT_MOVEMENT_CONSTANT* 0.75,  # Intentionally make this smaller than AGENT_MOVEMENT_CONSTANT to improve fidelity
#     width=INTEL_CAMERA_WIDTH,
#     height=INTEL_CAMERA_HEIGHT,
#     visibilityDistance=MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER,
#     visibilityScheme="Distance",
#     fieldOfView=INTEL_VERTICAL_FOV,
#     server_class=ai2thor.fifo_server.FifoServer,
#     useMassThreshold=True,
#     massThreshold=10,
#     autoSimulation=False,
#     autoSyncTransforms=True,
#     renderInstanceSegmentation=True,
#     agentMode="stretch",
#     renderDepthImage=False, # SAVE_DEPTH = False
#     cameraNearPlane=0.01,  # VERY VERY IMPORTANT
#     branch=None,  # IMPORTANT do not use branch
#     commit_id=STRETCH_COMMIT_ID,
#     server_timeout=MAXIMUM_SERVER_TIMEOUT,
#     snapToGrid=False,
#     fastActionEmit=True,
#     action_hook_runner=_ACTION_HOOK_RUNNER,
#     platform = CloudRendering,
#     gpu_device = 0,
#     use_quick_navi_action=True
#     # antiAliasing="smaa", # We can get nicer looking videos if we turn on antiAliasing and change the quality
#     # quality="Ultra",
# )


# ACTION_MAP = {0: "ms", 1: "rs", 2: "ls", 5: "r", 6: "l", 11: "m", 15: " ", 16: "end", 17: " "}
# ACTION_MAP = {0: "ms", 1: "rs", 2: "ls", 5: "r", 6: "l", 11: "m", 16: "end", 17:" "}
ACTION_MAP = { 0: "ms", 1: "rs", 2: "ls", 5: "r", 6: "l", 11: "m", 15: "b", 16: "end", 17: " "}
# ACTION_ID = {v: k for k, v in ACTION.items()}

# ACTION_CONTINUOUS = { # TO update
#     0: (0, 0.25),
#     1: (np.deg2rad(9), 0),
#     2: (np.deg2rad(-9), 0),
#     5: (np.deg2rad(36), 0),
#     6: (np.deg2rad(-36), 0),
#     11:(0, 0.5),
#     15:(0, -0.25),
#     16:(0,0),
#     17:(0,0)
# }
REVER_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}



class learned_from_label_interactive_trajectory_Procthor(GeneratorBase):

    def epoch_end(self, epoch_id):
        pass
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        for key in kwargs:
            setattr(self, key, kwargs[key])
            print(f"{key}: {kwargs[key]}")
        self.output_root = self.config.output_root
        self.learning_steps = self.config.learning_steps 
        self.test_steps = self.config.test_steps
        self.test_points = self.config.test_points


        self.in_context_len = self.config.in_context_len
        self.end_position = self.config.end_position
        self.start_position = self.config.start_position
        self.record_interval = self.config.record_interval
        self.record_points = [i for i in range(self.start_position, self.end_position, self.record_interval)]
        
        
        if self.config.has_attr("max_maze"):
            self.max_maze = self.config.max_maze
        else:
            self.max_maze = None
        # if self.output_root is not None:
        #     if not os.path.exists(self.output_root):
        #         os.makedirs(self.output_root)
        #         print(f"Created output folder {self.output_root}")
        # if self.output_root is None:
        #     assert False, "output_root is required for general_generator"
        print(f"output_root: {self.output_root}")

    def preprocess(self):
        
        # self.dataloader = PrefetchDataLoader(
        #     ProcthorDataSet(self.config.data_path, 
        #                     self.learning_steps+10, 
        #                     verbose=self.main,
        #                     max_maze = self.max_maze, 
        #                     folder_verbose=True),
        #     batch_size=1, # TODO 
        #     rank=self.rank,
        #     world_size=self.world_size
        #     )

        self.dataloader = PrefetchDataLoader(
            MazeDataSet(self.config.data_path, 
                            self.learning_steps+10, 
                            verbose=self.main,
                            max_maze = self.max_maze, 
                            folder_verbose=True),
            batch_size=1, # TODO 
            rank=self.rank,
            world_size=self.world_size
            )
        if self.output_root is not None:
            if not os.path.exists(self.output_root):
                os.makedirs(self.output_root)
                print(f"Created output root {self.output_root}")
            if self.config.data_path[-1] == "/":
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-2])
            else:
                output_folder_path = os.path.join(self.output_root, self.config.data_path.split("/")[-1])
            print(f"output folder path: {output_folder_path}")
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                print(f"Created output folder {output_folder_path}")
            self.output_folder_path = output_folder_path
        # print(f"saving in {self.output_folder_path}")
        print(f"Preprocessed dataloader with {len(self.dataloader)} batches")



    def exploration(self, env, max_steps, model):
        pass

    class ExploreHistory: # TODO put it to other place
        def __init__(self, agent_name, env_info, env_type, keys = ["obs", "action", "reward", "command"]):
            self.history = {}
            self.time = 0
            
            self.env_info = env_info
            self.env_type = env_type
            self.agent_name = agent_name
            for key in keys:
                self.history[key] = []
        def update(self, values):
            keys = self.history.keys()
            for key in keys:
                if key in values.keys():
                    self.history[key].append(values[key])
                else:
                    self.history[key].append(None)
            self.time += 1
        def get(self, key):
            return self.history[key]
        def get_all(self):
            return self.history
        def add_key(self, key):
            if key in self.history.keys():
                return False
            self.history[key] = []
            for i in range(self.time):
                self.history[key].append(None)

            return True
        def clear(self):
            keys = self.history.keys
            self.history = {}
            self.time = 0
            for key in keys:
                self.history[key] = []
        def __len__(self):
            return self.time
        def __str__(self):
            return f"ExploreHistory of {self.env_name} with {self.env_type}, totally {self.time} steps"    


    def __call__(self, epoch_id, rank):
        import gym
        import pickle
        import cv2

        max_steps = 11000
        learning_steps = self.learning_steps
        test_steps = self.test_steps
        n_range = (15,16)
        print(f"------start with learning steps {learning_steps}------------")

        data_root = "/pfs/pfs-r36Cge/qxg/program/l3cprocthor/projects/Procthor/expert/test_root"
        stretch_env_args = STRETCH_ENV_ARGS

        def process_obs(raw_obs): # Expert_controller.navigation_camera
            _o = Image.fromarray(raw_obs).resize((128, 128))
            _o = np.array(_o)
            _o = np.transpose(_o, (1, 0, 2)) 
            _o = np.rot90(_o, k=2, axes=(0, 1))
            _o = np.transpose(_o, (2, 0, 1)) # (H, W, C) to (C, H, W)
            # print(f"processed obs shape: {_o.shape}")
            return _o


        Expert_controller = None 
        for batch_id, (batch_data, folder_path) in enumerate(self.dataloader):
            folder_path = folder_path[0] if isinstance(folder_path, list) or isinstance(folder_path, tuple) else folder_path
            folder_name = folder_path.split("/")[-1]

            cmd_arr, obs_arr, behavior_actid_arr, label_actid_arr, behavior_act_arr, label_act_arr, rew_arr = batch_data
            obs_arr = obs_arr.permute(0, 1, 4, 2, 3) # (B, T, H, W, C) to (B, T, C, H, W)
            # states = obs_arr.contiguous()
            commands = cmd_arr.contiguous()
            actions = behavior_actid_arr.contiguous()
            rewards = rew_arr.contiguous()

            print(f"batch_id: {batch_id} processing {folder_name} with {len(batch_data)} data of shape of {obs_arr.shape}")
            assert obs_arr.shape[1] == actions.shape[1] + 1, f"states shape: {obs_arr.shape}, actions shape: {actions.shape}"

            done = False
            sum_reward = 0
            
            output_root = self.output_folder_path
            maze_output_folder = os.path.join(output_root, folder_name)
            if not os.path.exists(maze_output_folder):
                os.makedirs(maze_output_folder)
            output_folder = os.path.join(maze_output_folder, self.config.model_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            print(f"output folder: {output_folder}")
            print("-----------------------------")

            # learning from the Oracle agent
            start_step = -1
            reward = 0
            cache = None

            self.model.module.reset()
            maze_history = self.ExploreHistory("OracleLeadsDivLong", folder_name, "maze", keys = ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss", "prediction", "position", "target"])
            command = None
            for step in range(learning_steps):
                if done:
                    print(f"done at step {step}")
                    break
                observation = obs_arr[:, step:step+1]
                action = actions[:, step:step+1]
                command = commands[:, step:step+1]
                pred_obs_list, pred_act_list, cache = self.model.module.generate_states_only(
                                prompts=command,
                                current_observation=observation, 
                                action_trajectory=action,
                                history_observation=None, #states[start:end],
                                history_action=None, #actions[start:end],
                                history_update_memory=False, 
                                autoregression_update_memory=False, # TOTEST
                                cache=cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=True, 
                                need_action=True)
                
                # obs, reward, done, information = maze_env.step(action)
                reward = rewards[:, step:step+1]
                obs = obs_arr[:, step+1:step+2]
                # squ = (obs[0,0] - pred_obs_list[0,0])**2/(255*255)
                # print(squ.shape, obs.shape, pred_obs_list[0,0].shape)
                # print(squ)
                mse_loss = torch.mean((obs[0, 0] - pred_obs_list[0, 0])**2/(255*255))

                last_command = command
                last_observation = observation
                last_action = action

                observation = obs
                sum_reward += reward
                # ["obs", "oracle_action", "agent_action", "reward", "command", "wm_loss"]
                to_update = {
                    # "obs": last_observation,
                    "oracle_action": action,
                    "agent_action": pred_act_list[0, 0], 
                    "reward": reward, 
                    "command": last_command, 
                    "wm_loss": mse_loss,
                    # "prediction": pred_obs_list[0, 0],
                }
                maze_history.update(to_update)
            print(f"sum reward during learning from oracle: {sum_reward}")

            def learn_end_frame(cache):
                end_black = torch.from_numpy(np.zeros(obs_arr[:, 0:1].shape)).float()
                action = 16 # 16:end
                command = torch.from_numpy(np.zeros(768,)).float()
                pred_obs_list, pred_act_list, cache = self.model.module.generate_states_only(
                                prompts=command,
                                current_observation=end_black, 
                                action_trajectory=np.array([action]),
                                history_observation=None, #states[start:end],
                                history_action=None, #actions[start:end],
                                history_update_memory=False, 
                                autoregression_update_memory=False, # TOTEST
                                cache=cache,
                                single_batch=True,
                                history_single_step=False,
                                future_single_step=False,
                                raw_images=True,
                                need_numpy=True, 
                                need_action=True)
                return None # cache
            
            
            
            if cache != None:
                cache = learn_end_frame(cache)
            
            # After learning the last frame, which is a black frame, we can start testing the model
            # We should first set the houses and reset the environment, and teleport the agent to the start position
            house_path = os.path.join(folder_path, "house.npy")
            house = np.load(house_path, allow_pickle=True).item()
            house_id = batch_id
            env_args = copy.deepcopy(STRETCH_ENV_ARGS)
            # env_args['gpu_device'] = self.device
            if Expert_controller is None:
                Expert_controller: Expert = initialize_expert_controller(5, env_args, house, house_id,
                                                             root=f"./test_root/{house_id:06d}",)
            # Expert_controller.reset(house)

            try:
                Expert_controller.reset(house)
            except Exception as e:
                print(f"Error resetting Expert_controller: {e}")
                continue

            self.reachable_points = Expert_controller.get_reachable_positions(0.5) # AGENT_MOVEMENT_CONSTANT_S = 0.25
            # save the reachable points
            reachable_points_path = os.path.join(maze_output_folder, "reachable_points.npy")
            if not os.path.exists(reachable_points_path):
                np.save(reachable_points_path, self.reachable_points)
                print(f"Saved reachable points to {reachable_points_path}")
            def get_random_target_seg_from_folder(folder_path, targets):
                targets_id = [target['objectId'] for target in targets]
                targets_map = {target['objectId']: target for target in targets}
                if folder_path.endswith("/"):
                    folder_path = folder_path[:-1]
                task_name = folder_path.split("/")[-2] + "/" + folder_path.split("/")[-1]
                if 'val' in folder_path:
                    original_root = "/pfs/pfs-r36Cge/qxg/datasets/procthor/0716-train-10000/train/"
                    # task_name = folder_path.split("/")[-2] + "/" + folder_path.split("/")[-1].split("_")[0] + "/" + folder_path.split("/")[-1].split("_")[1]
                    # "/pfs/pfs-r36Cge/qxg/datasets/procthor/0617-trajectory/train/"
                else:
                    original_root = "/pfs/pfs-r36Cge/qxg/datasets/procthor/0716-train-10000/train/"
                    # "/pfs/pfs-r36Cge/qxg/datasets/procthor/trajectory_2/train/"
                original_path = os.path.join(original_root, task_name)
                # print(original_path)
                obj_seg_map = {}
                for file_name in os.listdir(original_path):
                    # if target['objectId'] in file_name:
                    file_path = os.path.join(original_path, file_name, "metadata/object_seg.jpg")
                    if os.path.exists(file_path):
                        obj_id = "|".join(file_name.split("|")[4:])
                        if obj_id not in targets_id:
                            continue
                        obj_seg = np.array(Image.open(file_path).resize([16, 16]))
                        obj_seg_map[obj_id] = obj_seg
                if len(obj_seg_map) == 0:
                    print(f"No object segmentation found in {original_path}")
                    return None, None
                rnd_obj_seg = str(np.random.choice(list(obj_seg_map.keys())))
                # print(f"From {task_name} choice a new random object segmentation: {rnd_obj_seg}")
                return targets_map[rnd_obj_seg], obj_seg_map[rnd_obj_seg].reshape(-1).astype(np.float32) # (768,)


            def refresh_command(Expert_controller, reachable_points, trajectory_id):
                # Get a random command from the reachable points
                targets = Expert_controller.full_objects()
                target, command = get_random_target_seg_from_folder(folder_path, targets)
                assert command.shape == (768,), f"command shape: {command.shape}"

                attempts = 0      
                flag = False
                while attempts < 10:
                    start_position = reachable_points[np.random.randint(len(reachable_points))]
                    if Expert_controller.teleport_agent(position=start_position):
                        flag = True
                        break
                Expert_controller.reset_task(target,mode=2, trajectory_id=f"{trajectory_id:06d}")
                return target, command, flag

            def if_close_to_goal(evt, goal_pos, dist_thresh=0.2):
                if horiz_dist(evt.metadata["agent"]["position"], goal_pos) <= dist_thresh:
                    return True
                return False

            def find_object_id_by_type(Expert_controller, object_type):
                targets = Expert_controller.full_objects()
                obj_ids = []
                for target in targets:
                    if target['objectType'] == object_type:
                        obj_ids.append(target['objectId'])
                if len(obj_ids) == 0:
                    return None
                return obj_ids

            import tqdm
            K_step = 1
            start_step = -1
            # (H, W, C) to (C, H, W)
            sum_reward = 0
            
            test_points = self.test_points #[100, 1000, 9000]
            print(f"test points: {test_points}")

            # TODO init the obs cmd
            target, command, flag = refresh_command(Expert_controller, self.reachable_points, batch_id) # To start a new command to record
            observation = process_obs(Expert_controller.navigation_camera).copy() # Expert_controller.navigation_camera
            
            traj_obj_length = 0
            N_command = 1 # number of commands updated
            N_success = 0 # number of successful commands
            N_fail = 0 # number of failed commands
            N_action_fail = 0 # number of failed actions
            action_retry = 0 # number of action retries
            self.temp_scheduler = LinearScheduler(self.config.temp_scheduler, 
                    self.config.temp_value)
            
            for step in range(test_steps):
                traj_obj_length += 1
                if done:
                    print(f"done at step {step}")
                    break
                last_command = command
                if step in test_points:
                    target, command, flag = refresh_command(Expert_controller, self.reachable_points, batch_id) # To start a new command to record
                    N_command += 1
                    cache = learn_end_frame(cache)
                    obs = process_obs(Expert_controller.navigation_camera).copy()

                white_id = [0, 1, 2, 5, 6, 11, 15]
                mask = [0 for i in range(17)] # it is the action space, 17 actions, while....procthor cannot cover all of them
                reverse_mapping = {
                    "rs": 2, "ls": 1, "r": 6, "l": 5,
                }
                #  {0: "ms", 1: "rs", 2: "ls", 5: "r", 6: "l", 11: "m", 16: "end", 17:" "}
                for i in white_id:
                    mask[i] = 1
                # if the last action is rotation, then we should not rotate back again
                if maze_history.get('agent_action') is not None and len(maze_history.get('agent_action')) > 0: 
                    if maze_history.get('agent_action')[-1] in reverse_mapping.keys():
                        mask[reverse_mapping[maze_history.get('agent_action')[-1]]] = 0
                        if len(maze_history.get('agent_action')) > 4:
                            if maze_history.get('agent_action')[-2] == maze_history.get('agent_action')[-1] and maze_history.get('agent_action')[-3] == maze_history.get('agent_action')[-1]:
                                mask[REVER_ACTION_MAP[maze_history.get('agent_action')[-1]]] = 0 # if the last two actions are the same, then we should not repeat it
                    
                mask = np.array(mask, dtype=np.float32) # (17,)
                last_cache = cache.copy() if cache is not None else None
                if last_cache is None:
                    print("last_cache is None, initializing cache??? in step", step)

                # Try to execute the action with the model
                agent_event = None
                for i in range(17): # 17 is the maximum number of actions to try
                    if (mask == 0).all():
                        action_retry += i + 1
                        print("All actions are black, refresh the command and reset the cache")
                        break
                    pred_obs, action_id, cache, act_distribution = self.model.module.policy(command, observation, 
                                                                                            cache=last_cache, update_memory = False, need_cache = True, 
                                                                                            mask=mask, need_distribution=True, 
                                                                                            temperature=self.temp_scheduler()) # TODO the update to deal with the failure
                    action_id = action_id[0, 0]
                    mask[action_id] = 0 # set the action to black, so that it will not be selected again
                    action = ACTION_MAP[action_id]
                    agent_event = Expert_controller.agent_step(action)
                    if agent_event.metadata["lastActionSuccess"]:
                        action_retry += i + 1
                        break

                self.temp_scheduler.step()
                if not agent_event.metadata["lastActionSuccess"]:
                    print(agent_event.metadata["errorMessage"])
                    reward = -1 # if fail, reward = -1
                    # refresh the command and let the model learn a black action
                    target, command, flag = refresh_command(Expert_controller, self.reachable_points, batch_id) # To start a new command to record
                    cache = learn_end_frame(cache)
                    N_command += 1
                    N_action_fail += 1
                    traj_obj_length = 0
                else: # if_close_to_goal(agent_event, ) and 
                    obj_ids = find_object_id_by_type(Expert_controller, target['objectType'])
                    if is_any_object_sufficiently_visible_and_in_center_frame(Expert_controller, [target['objectId']], absolute_min_pixels=30):
                        reward = 1
                        target, command, flag = refresh_command(Expert_controller, self.reachable_points, batch_id) # To start a new command to record
                        N_command += 1
                        N_success += 1
                        traj_obj_length = 0
                        print(f"Reached the target {target['objectId']} with reward {reward}")
                        cache = learn_end_frame(cache)
                    elif traj_obj_length > 500:
                        reward = 0
                        target, command, flag = refresh_command(Expert_controller, self.reachable_points, batch_id) # To start a new command to record
                        traj_obj_length = 0
                        cache = learn_end_frame(cache)
                        N_command += 1
                        N_fail += 1
                    else:
                        reward = 0
                
                # obs = np.transpose(np.array(Image.fromarray(Expert_controller.navigation_camera).resize((128, 128))), (2, 0, 1))
                obs = process_obs(Expert_controller.navigation_camera).copy()
                # TODO: Update the command. 1. if task done --- sucess or fail give a reward. 2. if not done, then keep the last command and give a reward 0
                obs = np.array(obs, dtype=np.uint8)
                mse_loss = np.mean((obs - pred_obs[0, 0])**2/(255*255))

                last_observation = observation
                observation = obs
                sum_reward += reward
                position = Expert_controller.get_position_rotions(Expert_controller.get_current_agent_full_pose())
                to_update = {
                    "obs": last_observation, # (C, H, W)
                    "agent_action": action, 
                    "reward": reward, 
                    "command": last_command, 
                    "wm_loss": mse_loss,
                    "prediction": pred_obs[0, 0],
                    "position": position, 
                    "target": target['objectId'],
                }
                maze_history.update(to_update)
            
            N_total = N_fail + N_success
            if N_total == 0:
                N_total = 1
            if N_command == 0:
                N_command = 1
            print(f"Model total Reward: {sum_reward} with total steps {maze_history.__len__()} Success {N_success} / {N_command} commands ({N_success/N_command*100:.2f}%) Fail Action {N_action_fail} / {N_command} commands ({N_action_fail/N_command*100:.2f}%) Fail Rate {N_fail/(N_total)*100:.2f}% Action Retry {action_retry}")
            print("Traj wm loss: ", np.mean(np.array(maze_history.get("wm_loss"))))
            # save maze_history to pkl
            pickle.dump(maze_history.get_all(), open(os.path.join(maze_output_folder, "maze_history.pkl"), "wb"))
            print(f"Saved maze_history to {os.path.join(maze_output_folder, 'maze_history.pkl')}")
            # maze_env.save_trajectory(os.path.join(maze_output_folder, f"trajectory.png"))
            # print(f"Saved trajectory to", os.path.join(maze_output_folder, f"trajectory.png"))

        Expert_controller.stop()


    

