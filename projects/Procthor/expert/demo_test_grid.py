import sys
import os, copy

# sys.path.append("/pfs/pfs-r36Cge/qxg/program/l3cprocthor/projects/Procthor")
# sys.path.append("/pfs/pfs-r36Cge/qxg/program/l3cprocthor/projects/Procthor/third_party/spoc_robot_training")

sys.path.append("/pfs/pfs-r36Cge/zsj/yuxuan/airsoul-diffusion/projects/Procthor")
sys.path.append("/pfs/pfs-r36Cge/zsj/yuxuan/airsoul-diffusion/projects/Procthor/third_party/spoc_robot_training")
# export OBJAVERSE_DATA_BASE_DIR="/pfs/pfs-r36Cge/libo/procthor/objaverse_assets"
# export OBJAVERSE_HOUSES_BASE_DIR="/pfs/pfs-r36Cge/libo/procthor/objaverse_houses"
# export OBJAVERSE_DATA_DIR="/pfs/pfs-r36Cge/libo/procthor/objaverse_assets/2023_07_28"
# export OBJAVERSE_HOUSES_DIR="/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/2023_07_28"
# export PYTHONPATH="./"

os.environ["OBJAVERSE_DATA_BASE_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_assets"
os.environ["OBJAVERSE_HOUSES_BASE_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses"
os.environ["OBJAVERSE_DATA_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_assets/2023_07_28"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/2023_07_28"

# os.environ["OBJAVERSE_DATA_BASE_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_assets"
# os.environ["OBJAVERSE_HOUSES_BASE_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses"
# os.environ["OBJAVERSE_DATA_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_assets/2023_07_28"
# os.environ["OBJAVERSE_HOUSES_DIR"] = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/houses_2023_07_28"


import matplotlib.pyplot as plt
import shutil
import time
import gc
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import heapq

from tqdm import tqdm
from multiprocessing import Pool

from collections import deque

from third_party.spoc_robot_training.environment.stretch_controller import StretchController
from third_party.spoc_robot_training.utils.data_generation_utils.navigation_utils import \
    is_any_object_sufficiently_visible_and_in_center_frame, get_room_id_from_location
from third_party.spoc_robot_training.utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS, \
    AGENT_ROTATION_DEG, AGENT_MOVEMENT_CONSTANT

# from expert.actions import shortest_path_to_actions, plot_grid_map, determine_turn_actions_
# from expert.actions_qu import shortest_path_to_actions, plot_grid_map, determine_turn_actions_
from expert.action_maze import shortest_path_to_actions, plot_grid_map, determine_turn_actions_

from expert.functions import (l2_distance,
                              load_procthor_houses,
                              save_data_json,
                              check_agent_position,
                              save_img_from_frame,
                              positions2path, save_as_txt,
                              save_navigation_frame, save_as_numpy, save_data_h5py, save_as_video,
                              get_rooms_lens,
                              build_dir,
                              select_rooms)

from expert.a_star import test_save_grid

# Constants
import warnings

warnings.filterwarnings("ignore")

MEDIUM_RANGE = (0.2, 0.6)  #
TARGET_RANGE = 1  # All targets
DEBUG = True
MAX_FRAME = 10000  #
MAX_AGENT_OBJECT_DISTANCE = 0.5 # 1  #
MAX_RANDOM_AGENT_POSITION_ATTEMPTS = 200  #
MAX_ATTEMPTS = 200  # 与前面有啥区别

ABSOLUTE_MIN_PIXELS = 50
ALNATIVE_ACTIONS = ['b', 'm', 'ms', 'l', 'r', 'ls', 'rs']
REFER_ACTION_LEN = 11  # include behavior action
SPLIT = "train"

all_count_1 = 0
fail_count_1 = 0
all_count_3 = 0
fail_count_3 = 0

# 显卡与编号可能不对应
GPU_MAP = {
    0: 0,
    1: 2,
    2: 3,
    3: 6,
    4: 7,
    5: 5,
    6: 1,
    7: 4
}

# multi-task
GPU_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
GPU_PROCESSES_PER_DEVICE = 16  # 再多可能会崩
DATASET_DIR = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-21-Oracle-data"
# house_ids = range(11251, 150000) # 150k

# # debug
# GPU_LIST = [0]
# GPU_PROCESSES_PER_DEVICE = 2  # 12
# DATASET_DIR = "/pfs/mt-r36Cge/qxg/datasets/procthor/trajectory_temp"

RouteDeviationThreshold = 0.25  # 0.2
NUM_PROCESSES = GPU_PROCESSES_PER_DEVICE * len(GPU_LIST)


class Expert(StretchController):
    def __init__(self, house, id=0, root=f"{DATASET_DIR}/{SPLIT}", **kwargs):
        """Initialize the Expert controller."""
        super().__init__(**kwargs)
        self.id = f"{id:06d}"
        self.successful_objects = []
        self.root = root
        self.root_dir = build_dir(f"{self.root}/{self.id}")
        self.status = []  # save as h5py

        # 保存前一天成功保存序列的终点，作为下一次尝试的起点
        self.pre_end_pose = None

        os.makedirs(f"{self.root}/{self.id}", exist_ok=True)
        save_data_json(house, f"{self.root}/{self.id}/house.json")

        # self.reachable_positions = self.get_reachable_positions()

    def reset(self, scene):
        reset_event = super().reset(scene)
        self.reachable_positions = self.get_reachable_positions()
        self.room_num = len(scene['rooms'])
        return reset_event

    def reset_task(self, target, mode, trajectory_id=f"{0:06d}"):
        """Reset the task."""
        self.target = target
        self.target_id = target['objectId']
        self.mode = mode
        self.tag = None
        self.trajectory_id = trajectory_id
        self.init_agent_pose = self.get_current_agent_full_pose()
        self.actions = [' ']
        self.len_explore = 0
        self.refer_actions = [' ']
        self.len_actions = 0
        self.positions = [self.get_position_rotions(self.init_agent_pose)]
        self.reference = ['init']  # choice ['expert','random']
        self.nav_frames = deque([self.navigation_camera], maxlen=MAX_FRAME)  # Navigation frame buffer
        self.last_actions_success = [True]  # ['start',True,False]

        # Create directories
        # self.build_object_dir(self.target,trajectory_id)

    def reachable_positions2grid(self, save_path, resolution=0.151):
        xs = [rp["x"] for rp in self.reachable_positions]
        zs = [rp["z"] for rp in self.reachable_positions]
        # print(self.reachable_positions)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(xs, zs)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
        ax.set_title(f"Reachable Positions in the Scene (house_id:{self.id})")
        ax.set_aspect("equal")
        save_fig_dir = os.path.join(save_path, f"{self.id}_reachable_positions.jpg")
        fig.savefig(save_fig_dir)
        plt.close(fig)

        # 确定网格的范围和分辨率
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
        # resolution = 0.151  # 每个网格单元的大小，根据需要调整(约0.15)

        # 计算网格的尺寸
        grid_width = int((x_max - x_min) / resolution) + 1
        grid_height = int((z_max - z_min) / resolution) + 1

        # 初始化网格，默认值为 1（不可达）
        grid_map = np.ones((grid_height, grid_width), dtype=float)

        # 将 reachable_positions 映射到网格
        for x, z in zip(xs, zs):
            # 计算网格索引
            x_idx = int((x - x_min) / resolution)
            z_idx = int((z - z_min) / resolution)
            # 标记为 0（可达）
            grid_map[z_idx, x_idx] = 0

        grid_map = np.pad(grid_map, pad_width=1, mode='constant', constant_values=1)

        # 可视化网格地图
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(grid_map, cmap='gray_r', origin='lower', extent=[x_min, x_max, z_min, z_max])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$z$")
        ax.set_aspect("equal")
        ax.set_title(f"Grid Map of Reachable Positions (house_id:{self.id})")
        save_fig_dir = os.path.join(save_path, f"{self.id}_positions_grid.jpg")
        fig.savefig(save_fig_dir)
        # plt.show()
        plt.close(fig)

        print(f"grid map保存至{save_fig_dir}")
        return grid_map

    def build_object_dir(self):
        """Check and create necessary directories."""
        self.object_dir = f"{self.root}/{self.id}/{self.trajectory_id}|{self.mode}|{self.tag}|{self.target['objectType']}|{self.target['objectId']}"
        self.frame_dir = f"{self.root}/{self.id}/{self.trajectory_id}|{self.mode}|{self.tag}|{self.target['objectType']}|{self.target['objectId']}/frame"
        self.metadata_dir = f"{self.root}/{self.id}/{self.trajectory_id}|{self.mode}|{self.tag}|{self.target['objectType']}|{self.target['objectId']}/metadata"

        for dir in [self.object_dir, self.frame_dir, self.metadata_dir]:
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=True)

    def target_types(self, datapath="/pfs/pfs-r36Cge/qxg/program/l3cprocthor/projects/Procthor/expert/domain/procthor.yaml"):
        """Load target object types from a YAML file."""
        import yaml
        with open(datapath, 'r') as file:
            data = yaml.safe_load(file)
        return data.get('target_object_types', [])

    def get_objects_(self):
        """Get objects from the environment."""
        with self.include_object_metadata_context():
            return list(self.controller.last_event.metadata["objects"])

    # split the reachable points
    def split_reachable_area(self, mode='split'):
        """Split the reachable area into difficulty levels."""
        obj_pos_from_obj_id = self.get_obj_pos_from_obj_id(self.target_id)
        length_from_positions = [(l2_distance(self.get_obj_pos_from_obj_id(self.target_id), pos), pos) for pos in
                                 self.reachable_positions]
        # if len(length_from_positions)<10:
        #     print(f"{self.id}length-from-positions",len(length_from_positions))
        length_from_positions.sort(key=lambda x: x[0])
        if mode == "all":
            return {
                "easy": length_from_positions,
                "medium": length_from_positions,
                "hard": length_from_positions
            }
        return {
            "easy": length_from_positions[
                    : int(len(length_from_positions) * MEDIUM_RANGE[0])
                    ],
            "medium": length_from_positions[
                      int(len(length_from_positions) * MEDIUM_RANGE[0]): int(
                          len(length_from_positions) * MEDIUM_RANGE[1]
                      )
                      ],
            "hard": length_from_positions[
                    int(len(length_from_positions) * MEDIUM_RANGE[1]):
                    ],
        }

    def in_the_same_room(self):
        object_room_id, _ = self.get_objects_room_id_and_type(self.target_id)
        room_id = get_room_id_from_location(self.room_poly_map, self.get_current_agent_full_pose()["position"])
        # print(f"object_room_id:{object_room_id},room_id:{room_id}")
        return object_room_id == room_id

    def random_agent_position(self, objectid, difficulty=random.choice(["easy", "medium", "hard"])):
        """Set a random agent position."""
        old_agent_location = self.get_current_agent_full_pose()
        self.difficulty_spots = self.split_reachable_area(objectid)[difficulty]
        position = self.difficulty_spots[random.randint(0, len(self.difficulty_spots) - 1)][1]

        rot_y = random.choice(list(range(0, 360, int(AGENT_ROTATION_DEG))))

        event = self.controller.step(
            action="Teleport",
            position=position,
            rotation=dict(x=0, y=rot_y, z=0),
            standing=old_agent_location["isStanding"],
        )
        return event.metadata["lastActionSuccess"]

    def teleport_agent(self, agent_pose=None, teleport_horizon=False, position=None):
        """Teleport agent to a given pose."""
        if position is not None:
            old_agent_location = self.get_current_agent_full_pose()
            rot_y = random.choice(list(range(0, 360, int(AGENT_ROTATION_DEG))))
            event = self.controller.step(
                action="Teleport",
                position=position,
                rotation=dict(x=0, y=rot_y, z=0),
                standing=old_agent_location["isStanding"],
            )
            return event.metadata["lastActionSuccess"]

        action_params = {
            'action': 'Teleport',
            'position': agent_pose['position'],
            'rotation': agent_pose['rotation'],
            'standing': agent_pose['isStanding']
        }
        if teleport_horizon:
            action_params['horizon'] = agent_pose["cameraHorizon"]

        return self.controller.step(**action_params).metadata["lastActionSuccess"]

    def full_objects(self):
        """Get a random selection of PART objects."""
        objects = self.get_filtered_objects()
        selected_objects = random.sample(objects, int(len(objects) * TARGET_RANGE))
        return selected_objects

    def get_filtered_objects(self):
        """Get filtered objects based on target types."""
        return [o for o in self.get_objects_() if o["objectType"] in self.target_types()]

    def get_actions_from_shortest_path(self):
        """Convert shortest path to actions."""
        shortest_path = self.get_shortest_path_to_object(self.target_id, initial_rotation=self.get_current_agent_full_pose()['rotation'] )
        return shortest_path_to_actions(
            shortest_path, self.get_current_agent_full_pose()
        )

    def get_agent_distance_to_object(self):
        """Get the distance between the agent and an object."""
        object_info = self.get_object(self.target_id)
        agent_info = self.get_current_agent_full_pose()
        return math.hypot(object_info["position"]["x"] - agent_info["position"]["x"],
                          object_info["position"]["z"] - agent_info["position"]["z"])

    def perform_action_with_retry(self, action, alternative_actions, max_retries=3):

        retries = 0
        # # 理论上不需要，每次retry结果一样(fast:1)
        agent_event = self.agent_step(action)
        if agent_event.metadata["lastActionSuccess"]:
            return True, action
        # while retries < max_retries:
        #     agent_event = self.agent_step(action)
        #     if agent_event.metadata["lastActionSuccess"]:
        #         return True, action
        #     retries += 1

        # if action == "m":
        #     agent_event = self.agent_step('b')
        #     if agent_event.metadata["lastActionSuccess"]:
        #         return True, 'b'


        # import pdb;pdb.set_trace()
        # 如果系统给的action失败，遍历尝试别的action
        # for alt_action in alternative_actions:
        #     # Changed
        #     # if alt_action == 'b' and random.random() < 0.8:  # Reduce the probability of action 'b'
        #         # continue
        #     # retries = 0
        #     # while retries < max_retries:
        #     #     agent_event = self.agent_step(alt_action)
        #     #     if agent_event.metadata["lastActionSuccess"]:
        #     #         return True, alt_action
        #     #     retries += 1
        #     agent_event = self.agent_step(alt_action)
        #     if agent_event.metadata["lastActionSuccess"]:
        #         return True, alt_action
        # import pdb;pdb.set_trace()
                # print(f"Alternative action {alt_action} failed. Retrying ({retries}/{max_retries})...")
        return False, None

    def is_agent_position_same_shortest_path_start(self, shortest_path, threshold=0.05):
        agent_position = self.get_current_agent_full_pose()["position"]
        start_position = shortest_path[0]
        return l2_distance(agent_position, start_position) < threshold

    def get_position_rotions(self, agent_location):
        """Get positions and rotations from a list of positions."""
        return [
            agent_location['position']['x'], agent_location['position']['y'], agent_location['position']['z'],
            agent_location['rotation']['x'], agent_location['rotation']['y'], agent_location['rotation']['z']
        ]

    def update_state(self, action='', full_last_agent_location=None, reference='expert', explore=False):
        """Update the state of the environment."""

        self.actions.append(action)
        self.nav_frames.append(self.navigation_camera)
        self.positions.append(self.get_position_rotions(
            self.get_current_agent_full_pose() if full_last_agent_location is None else full_last_agent_location))
        self.reference.append(reference)

        if explore:
            self.update_refer_actions(explore=True, curr_action=action)

    def update_refer_actions(self, explore=False, curr_action=None):
        if explore:
            shortest_path = self.get_shortest_path_to_object(self.target_id, 
                                                            initial_rotation=self.get_current_agent_full_pose()['rotation'], 
                                                            attempt_path_improvement=True)
            if shortest_path is None:
                # import pdb;pdb.set_trace()
                # self.get_shortest_path_to_object(self.target_id, attempt_path_improvement=True)
                raise Exception("2:Failed to retrieve the shortest path. The object is not reachable.")
                # else:
            #     print("2: Success to retrieve the shortest path")
            if not self.is_agent_position_same_shortest_path_start(shortest_path):
                shortest_path.insert(0, self.get_current_agent_full_pose()["position"])
            actions = shortest_path_to_actions(shortest_path, self.get_current_agent_full_pose())
            # sublist = [curr_action] + actions[0: REFER_ACTION_LEN - 1]
            sublist = actions[0: REFER_ACTION_LEN]

            self.refer_actions.append(sublist)
        else:
            a = []
            for i in range(len(self.actions) - self.len_explore):
                sublist = self.actions[i + self.len_explore:i + REFER_ACTION_LEN + self.len_explore]
                # 如果长度不足 k，用 None 补位
                if len(sublist) < REFER_ACTION_LEN:
                    sublist += [' '] * (REFER_ACTION_LEN - len(sublist))
                a.append(sublist)
            self.refer_actions.extend(a)

    def seg_object_from_last_frame(self):
        object_seg = self.get_segmentation_mask_of_object(self.target_id, which_camera='nav')
        mask_expanded = object_seg[:, :, np.newaxis]
        masked_image = np.where(mask_expanded, self.nav_frames[-1], 0)
        rows = np.any(mask_expanded, axis=(1, 2))
        cols = np.any(mask_expanded, axis=(0, 2))
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # 裁剪目标区域
        cropped_image = masked_image[y_min:y_max + 1, x_min:x_max + 1]

        # 计算目标区域的高度和宽度
        h, w = cropped_image.shape[:2]

        # 确定扩充后的正方形边长
        max_side = max(h, w)

        # 计算需要填充的像素数
        pad_h = (max_side - h) // 2
        pad_w = (max_side - w) // 2

        # 扩充为正方形，短边两侧补0
        padded_image = np.pad(cropped_image,
                              ((pad_h, max_side - h - pad_h),
                               (pad_w, max_side - w - pad_w),
                               (0, 0)),
                              mode='constant', constant_values=0)
        return padded_image

    def save_collected_data(self, error_debug=None):
        print("saving!!!!")

        """Save agent's collected data into both JSON and HDF5 formats."""
        self.build_object_dir()
        self.successful_objects.append(f"{self.target['objectType']}|{self.target_id}")

        if error_debug is not None:
            with open(f"{self.object_dir}/error.txt", "a") as f:
                f.write("%s\n" % error_debug)

        # updata refer_actions
        self.update_refer_actions()

        # Save data to JSON files
        save_data_json(self.successful_objects, f"{self.root_dir}/successful_objects.json")

        save_data_json(self.init_agent_pose, f"{self.metadata_dir}/agent.json")
        save_data_json(self.target, f"{self.metadata_dir}/object.json")
        save_data_json(self.positions, f"{self.metadata_dir}/positions.json")
        save_data_json(self.actions, f"{self.metadata_dir}/actions.json")
        save_data_json(self.reference, f"{self.metadata_dir}/reference.json")
        save_data_json(self.refer_actions, f"{self.metadata_dir}/actions_refer.json")
        self.len_actions = len(self.actions)

        # print("self.actions:", len(self.actions))
        # print("self.refer_actions:", len(self.refer_actions))

        # # Save navigation frames to a directory
        # save_navigation_frame(self.nav_frames, self.frame_dir)
        # save_as_video(self.nav_frames,self.metadata_dir)

        # save the object_seg image from the last frame
        masked_image = self.seg_object_from_last_frame()
        save_img_from_frame(masked_image, save_path=f"{self.metadata_dir}/object_seg.jpg")

        # save_img_from_frame(masked_image, save_path=f"{self.metadata_dir}/object_seg.jpg", save_size=(16, 16))
        save_as_numpy(self.nav_frames, self.metadata_dir)
        # Generate a top-down view of the agent's path and save it as an image
        shortest_path = positions2path(self.positions)
        top_down_path_frame = self.get_top_down_path_view(shortest_path)
        save_img_from_frame(top_down_path_frame, save_path=f"{self.metadata_dir}/test_top_down_along_path.jpg")

        # Save the robot's path as a plot
        plot_grid_map(shortest_path, self.actions, self.init_agent_pose, step_length=AGENT_MOVEMENT_CONSTANT, save_path=self.metadata_dir, reachable_positions=self.reachable_positions)

        self.pre_end_pose = self.get_current_agent_full_pose()

    def plot_path(self, shortest_path, previous_actions, attempts):
        """Plot the grid map with the current path and actions."""
        plot_grid_map(shortest_path, previous_actions, self.get_current_agent_full_pose(),
                      step_length=AGENT_MOVEMENT_CONSTANT, save_path=self.metadata_dir)

    # def verify_room_get_actions_from_shortest_path(self):
    #     # 先判断agent所在房间的id1，再判断目标物体的房间id2，然后在所有的房间id中选取一个id3，让agent去id3

    #     self.tag = 1
    #     now_room_id = get_room_id_from_location(self.room_poly_map, self.get_current_agent_full_pose()["position"])
    #     object_room_id, _ = self.get_objects_room_id_and_type(self.target_id)
    #     target_room_id = now_room_id

    #     while target_room_id == now_room_id or target_room_id == object_room_id:
    #         target_room_id = random.choice(list(self.room_poly_map.keys()))
    #     # print(f"target_room_id:{target_room_id},now_room_id:{now_room_id},object_room_id:{object_room_id}")

    #     attempts = 0

    #     # 每一轮的区别：起点不同
    #     while attempts < MAX_ATTEMPTS:  #

    #         shortest_path = self.get_shortest_path_to_room(target_room_id)  # TODO:CHECK

    #         if shortest_path is None:
    #             raise Exception("Failed to retrieve the shortest path to room {target_room_id}.")

    #             # api返回的路径不包括agent初始点
    #         # agent的position坐标可能不在生成的最短路径坐标（grid坐标，0.25/0.15）上
    #         if not self.is_agent_position_same_shortest_path_start(shortest_path):
    #             shortest_path.insert(0, self.get_current_agent_full_pose()["position"])

    #         actions = shortest_path_to_actions(shortest_path, self.get_current_agent_full_pose())
    #         # self.plot_path( shortest_path, previous_actions, attempts)#debug

    #         for action in actions:

    #             if action == "end":
    #                 return True

    #             previous_agent_location = self.get_current_agent_full_pose()
    #             action_is_success, action = self.perform_action_with_retry(action, alternative_actions=ALNATIVE_ACTIONS)

    #             if not action_is_success:
    #                 raise Exception(f"Action {action} and all alternatives failed during the exploration.")

    #             reference = "exploration"

    #             full_last_agent_location = self.get_current_agent_full_pose()  #

    #             if target_room_id == get_room_id_from_location(self.room_poly_map,
    #                                                            self.get_current_agent_full_pose()["position"]):
    #                 # print("objec is in the same room")
    #                 self.update_state(action=action, full_last_agent_location=full_last_agent_location,
    #                                   reference=reference, explore=True)
    #                 return True

    #             # 如果agent偏离最短路径，执行新一轮attempt
    #             if not check_agent_position(shortest_path, full_last_agent_location, threshold=RouteDeviationThreshold):
    #                 self.teleport_agent(agent_pose=previous_agent_location)
    #                 break

    #             self.update_state(action=action, full_last_agent_location=full_last_agent_location, reference=reference,
    #                               explore=True)

    #         attempts += 1

    #     raise Exception("Max attempts reached. room is still not visible.")

    def get_agent_distance_to_point(self, point):
        agent_info = self.get_current_agent_full_pose()
        return math.hypot(point[0] - agent_info["position"]["x"],
                          point[1] - agent_info["position"]["z"])

    def verify_n_room_get_actions_from_shortest_path(self):
        """ If there're n rooms in this houses, randomly choose(0 ~ n-1) rooms for exploration """

        # TIMER 003
        now_room_id = get_room_id_from_location(self.room_poly_map, self.get_current_agent_full_pose()["position"])
        object_room_id, _ = self.get_objects_room_id_and_type(self.target_id)
        room_id_list = list(self.room_poly_map.keys())
        # target_room_id = now_room_id
        # target_room_id = object_room_id #debug
        print("in verify_n_room_get_actions_from_shortest_path")
        if now_room_id == object_room_id:
            target_room_ids = [now_room_id]
        else:
            target_room_ids = select_rooms(room_id_list, now_room_id, object_room_id)
            # target_room_ids = []
        # self.tag = (len(target_room_ids) + 1) //  2
        self.tag = len(target_room_ids)
        # target_room_ids = []
        # if len(target_room_ids) == 0:
        #     return 
        # if  len(target_room_ids) > 1:
        #     choice_n = random.randint(1, len(target_room_ids))
        #     target_room_ids_ = random.sample(target_room_ids, choice_n)
        # else:
        #     target_room_ids_ = target_room_ids
        
        for target_room_id in target_room_ids:
            # print('start point and obj in the same room: target_room_id:', target_room_id)
            attempts = 0
            replan = 0
            path_retry = 0
            # 每一轮的区别：起点不同
            while attempts < MAX_ATTEMPTS:  #
                previous_shortest_path = self.get_shortest_path_to_room(target_room_id, 
                                                                                        # initial_rotation=self.get_current_agent_full_pose()['rotation'], 
                                                                                        )  # TODO:CHECK
                # history_positions = [self.get_current_agent_full_pose()['position']]
                # origin_rotation = self.get_current_agent_full_pose()['rotation']
                if previous_shortest_path is None:
                    if path_retry > 2:
                    # import pdb;pdb.set_trace()
                        raise Exception(f"Failed to retrieve the shortest path to room {target_room_id} in inner loop.")
                    path_retry += 1
                    print("Failed to retrieve the shortest path in outer loop. retry")
                    continue

                    # api返回的路径不包括agent初始点
                # agent的position坐标可能不在生成的最短路径坐标（grid坐标，0.25/0.15）上
                if not self.is_agent_position_same_shortest_path_start(previous_shortest_path):
                    previous_shortest_path.insert(0, self.get_current_agent_full_pose()["position"])

                previous_actions = shortest_path_to_actions(previous_shortest_path, self.get_current_agent_full_pose())
                # self.plot_path( previous_shortest_path, previous_actions, attempts)#debug

                # print("previous_actions:", previous_actions)
                # count = 0
                for action in previous_actions:

                    if action == "end":
                        # self.plot_current_traj(previous_shortest_path,  self.positions, self.positions[0][3:])
                        # import pdb;pdb.set_trace()

                        return True

                    # previous_agent_location = self.get_current_agent_full_pose()
                    # action_is_success, action = self.perform_action_with_retry(action,
                    #                                                            alternative_actions=ALNATIVE_ACTIONS)
                    previous_action = action
                    action_is_success, action = self.perform_action_with_retry(action,
                                                                               alternative_actions=[])
                    # history_positions.append(self.get_current_agent_full_pose()['position'])
                    reference = "exploration"
                    full_last_agent_location = self.get_current_agent_full_pose()

                    if previous_action != action:
                        # reference = "random"
                        if  action_is_success:

                            self.update_state(action=action, full_last_agent_location=full_last_agent_location,
                                        reference=reference, explore=True)
                            print(f"Action {previous_action} and all alternatives failed in outer loop. replanning path")
                            break
                        # else:
                        #     raise Exception(f"Action {previous_action} and all alternatives failed during exploration in inner loop.")

                    # else:
                    #     reference = "expert"
                    if not action_is_success:
                        # if replan > 2:
                        raise Exception(f"Action {previous_action} and all alternatives failed during exploration in inner loop.")
                        # replan += 1
                        # break
                        # raise Exception(f"Action {previous_action} and all alternatives failed during exploration in inner loop.")


                    # full_last_agent_location = self.get_current_agent_full_pose()  #

                    if target_room_id == get_room_id_from_location(self.room_poly_map,
                                                                   self.get_current_agent_full_pose()["position"]) and \
                        self.get_agent_distance_to_point([previous_shortest_path[-1]['x'], previous_shortest_path[-1]['z']]) <= MAX_AGENT_OBJECT_DISTANCE: #0.5:

                        # self.get_agent_distance_to_point(previous_shortest_path[-1]) <= MAX_AGENT_OBJECT_DISTANCE:

                        # print("objec is in the same room")
                       
                        # self.plot_current_traj(previous_shortest_path,  self.positions, self.positions[0][3:])
                        # import pdb;pdb.set_trace()
                        self.update_state(action=action, full_last_agent_location=full_last_agent_location,
                                          reference=reference, explore=True)
                        return True

                    # 如果agent偏离最短路径，执行新一轮attempt
                    if not check_agent_position(previous_shortest_path, full_last_agent_location,
                                                threshold=RouteDeviationThreshold):
                        # self.teleport_agent(agent_pose=previous_agent_location)
                        break
                    # count += 1
                    # print(count)
                    # if count == 26:
                    #     import pdb;pdb.set_trace()

                    self.update_state(action=action, full_last_agent_location=full_last_agent_location,
                                    reference=reference, explore=True)
       

                attempts += 1
            # self.reset_task(target=self.target, mode=2, trajectory_id=self.target_id)
        raise Exception("All rooms are still not visible. in inner loop")

    def verify_get_actions_from_shortest_path(self, exploration_room=False):
        # 给定position，给定object，单次任务

        # TIMER 002
        # import pdb;pdb.set_trace()
        if exploration_room:
            self.verify_n_room_get_actions_from_shortest_path()
            # self.verify_room_get_actions_from_shortest_path()
        else:
            self.tag = 0

        self.len_explore = len(self.actions)

        attempts = 0
        replan = 0
        path_retry = 0
        # 每一轮的区别：起点不同
        while attempts < MAX_ATTEMPTS: # TODO
            # 每次循环开始时，先检查 agent 是否已经足够靠近目标对象，并且在同一个房间内
            if self.get_agent_distance_to_object() <= MAX_AGENT_OBJECT_DISTANCE and self.in_the_same_room():
                # self.plot_current_traj(shortest_path,  self.positions, self.positions[0][3:])

                # import pdb;pdb.set_trace()

                self.execute_final_action()
                self.update_state(action='end')
                # if len(self.actions) < 5:
                #     raise Exception("Less than 5 actions were executed in outer loop.")

                self.save_collected_data()
                # 提前结束
                return self.status

            shortest_path = self.get_shortest_path_to_object(self.target_id, 
                                                            initial_rotation=self.get_current_agent_full_pose()['rotation'], 
                                                            attempt_path_improvement=True) 
            # history_positions = [self.get_current_agent_full_pose()['position']]
            # origin_rotation = self.get_current_agent_full_pose()['rotation']

            if shortest_path is None:
                if path_retry > 2:
                    raise Exception("1:Failed to retrieve the shortest path. The object is not reachable.in outer loop")
                path_retry += 1
                print("Failed to retrieve the shortest path in outer loop. retry")
                continue

            # api返回的路径不包括agent初始点
            # agent的position坐标可能不在生成的最短路径坐标（grid坐标，0.25/0.15）上
            if not self.is_agent_position_same_shortest_path_start(shortest_path):
                shortest_path.insert(0, self.get_current_agent_full_pose()["position"])

            # 将最短路径转换为一系列actions
            actions = shortest_path_to_actions(shortest_path, self.get_current_agent_full_pose())

            # self.plot_path( shortest_path, previous_actions, attempts)#debug

            # 遍历action
            for action in actions:

                # 终点，保存结果
                if action == "end":
                    # import pdb;pdb.set_trace()
                    # self.plot_current_traj(shortest_path,  self.positions, self.positions[0][3:])

                    self.execute_final_action()
                    self.update_state(action='end')  ##
                    ## Changed
                    # if len(self.actions) < 5:
                    #     raise Exception("Less than 5 actions were executed.")

                    self.save_collected_data()
                    return self.status

                previous_action = action
                # previous_agent_location = self.get_current_agent_full_pose()
                # print("beforce retry:", self.get_current_agent_full_pose()['position'])
                # 碰撞测试
                # action_is_success, action = self.perform_action_with_retry(action, alternative_actions=ALNATIVE_ACTIONS)
                action_is_success, action = self.perform_action_with_retry(action, alternative_actions=ALNATIVE_ACTIONS)

                # print("after retry:", self.get_current_agent_full_pose()['position'])
                # history_positions.append(self.get_current_agent_full_pose()['position'])
                full_last_agent_location = self.get_current_agent_full_pose()

                if previous_action != action:
                    reference = "random"
                    if  action_is_success:
                        # if replan > 2:
                        #     raise Exception(f"Action {previous_action} and all alternatives failed during exploration in inner loop.")
                        # replan += 1
                        self.update_state(action=action, full_last_agent_location=full_last_agent_location,
                                      reference=reference)
                        print(f"Action {previous_action} and all alternatives failed in outer loop. replanning path")
                        break
                    # else:
                    #     raise Exception(f"Action {action} and all alternatives failed in outer loop.")
                else:
                    reference = "expert"

                if not action_is_success:
                    # if replan > 2:
                    raise Exception(f"Action {previous_action} and all alternatives failed during exploration in inner loop.")
                    # replan += 1
                    # break
                # if not action_is_success:
                #     raise Exception(f"Action {action} and all alternatives failed in outer loop.")

                    



                # full_last_agent_location = self.get_current_agent_full_pose()

                if self.get_agent_distance_to_object() <= MAX_AGENT_OBJECT_DISTANCE and self.in_the_same_room():
                    # print("objec is in the same room")
                    self.update_state(action=action, full_last_agent_location=full_last_agent_location,
                                      reference=reference)
                    # self.plot_current_traj(shortest_path, self.positions, self.positions[0][3:])
                    
                    # import pdb;pdb.set_trace()
                    
                    self.execute_final_action()
                    self.update_state(action='end')

                    # if len(self.actions) < 5:
                    #     raise Exception("Less than 5 actions were executed.")

                    self.save_collected_data()

                    return self.status

                # 如果agent偏离最短路径，执行新一轮attempt
                if not check_agent_position(shortest_path, full_last_agent_location, threshold=RouteDeviationThreshold):
                    # self.teleport_agent(agent_pose=previous_agent_location)
                    # self.plot_current_traj(actions[:-1], shortest_path, history_positions, origin_rotation)
                    # self.plot_current_traj(shortest_path, history_positions, origin_rotation)

                    # import pdb;pdb.set_trace()

                    # check_agent_position(shortest_path, full_last_agent_location, threshold=RouteDeviationThreshold)
                    break

                self.update_state(action=action, full_last_agent_location=full_last_agent_location, reference=reference)

            attempts += 1
        raise Exception("Max attempts reached. Object is still not visible in outer loop.")

    def simulate_traj(self, actions,
                    start_pos=(0.0, 0.0),            # (x, z)
                    dx=0, dz=0): # 朝向向量
        """
        根据动作序列计算轨迹，返回 [(x, z, heading)] 列表
        - actions: ['l','m','ls','ms', ...]
        - start_pos: (x, z) 起点坐标

        """
        heading = math.degrees(math.atan2(dx, dz)) % 360
        ROT_DEG = {'l':  30, 'r': -30,
                'ls':  6, 'rs': -6}
        STEP_LEN = {'m': 0.2, 'ms': 0.1}
        
        x, z = start_pos
        traj = [(x, z, heading)]  # 初始位置记录
        
        for act in actions:
            if act in STEP_LEN:
                rad = math.radians(heading)
                step = STEP_LEN[act]
                x += step * math.sin(rad)
                z += step * math.cos(rad)
            
            elif act in ROT_DEG:
                heading = (heading + ROT_DEG[act]) % 360
            
            else:
                raise ValueError(f"未知动作: {act}")
            
            traj.append((x, z, heading))
        
        return traj

    def plot_current_traj(self, shortest_path, history_positions, ori_rotation):

        # action_traj = self.simulate_traj(actions, start_pos=[history_positions[0]['x'], history_positions[0]['z']], dx=ori_rotation['x'], dz=ori_rotation['z'])
        
        x1 = [p['x'] for p in shortest_path]
        z1 = [p['z'] for p in shortest_path]
        x2 = [p[0] for p in history_positions]
        z2 = [p[2] for p in history_positions]
        # x3 = [p[0] for p in action_traj]
        # z3 = [p[2] for p in action_traj]
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(x1, z1, marker='o', color='blue', label='shortest_path')
        plt.plot(x2, z2, marker='s', color='red', label='trajectory')
        # plt.plot(x3, z3, marker='x', color='yellow', label='plan_trajectory')


        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('2D Projection of Points (x, z)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        # 保存图片为 PNG 文件
        out_path = os.path.join("tmp_traj", f"{self.id}_{self.target_id}_trajectory_plot.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        # plt.show()  # 不在 terminal 下使用

        print(f"图像已保存为 {out_path}")

    # 控制一个 agent 转动视角，尝试将指定物体 (object_id) 放置在画面中心且保证其足够可见
    def execute_final_action(self):

        # 判断目标物体是否已经足够可见且位于画面中心
        if not is_any_object_sufficiently_visible_and_in_center_frame(
                self, [self.target_id], absolute_min_pixels=ABSOLUTE_MIN_PIXELS
        ):
            #  如果物体不在画面中心，计算 agent 当前朝向与目标物体之间的角度差
            angle_diff = self.get_agent_alignment_to_object(self.target_id)

            # 根据计算出的角度差，确定 agent 需要执行的旋转动作序列
            actions_ = determine_turn_actions_(angle_diff)
            # import pdb;pdb.set_trace()
            # 遍历每一个旋转动作
            # print("actions_:", actions_)
            for i, action in enumerate(actions_):
                
                #  执行当前旋转动作 action，并获取执行结果 agent_event
                agent_event = self.agent_step(action)

                # 检查动作是否执行成功。如果执行失败 (例如撞墙或其他限制)，则抛出异常
                if not agent_event.metadata["lastActionSuccess"]:
                    raise Exception(f"Action {i} {action} failed.")

                # 更新 agent 的状态，这可能包括更新 agent 的位置、朝向
                self.update_state(action=action)

                # 再次检查目标物体是否已经足够可见且位于画面中心。
                # 如果满足条件，则 break 跳出循环，不再执行后续的旋转动作
                if is_any_object_sufficiently_visible_and_in_center_frame(
                        self, [self.target_id], absolute_min_pixels=ABSOLUTE_MIN_PIXELS
                ):
                    break

        # 在所有旋转动作执行完毕后，再次检查目标物体是否满足条件
        if not is_any_object_sufficiently_visible_and_in_center_frame(
                self, [self.target_id], absolute_min_pixels=ABSOLUTE_MIN_PIXELS
        ):
            raise Exception("Object is still not visible.")
# end


def initialize_expert_controller(max_init_attempts=5, stretch_env_args=None, house=None, house_id=0,
                                 root=f"{DATASET_DIR}/{SPLIT}"):
    def remove_objtype_from_house(house):
        if house is None or "objects" not in house:
            return None
        import json
        objtype2remove = ["garbagebag", "dogbed", "Cart"]
        objectType_dict_path = "/pfs/pfs-r36Cge/qxg/program/l3cprocthor/projects/Procthor/expert/domain/objectType_dict.json"
        with open(objectType_dict_path, "r") as f:
            objectType_dict = json.load(f)
        for k in objectType_dict.keys():
            if "Obja" in k:
                objtype2remove.append(k)
        for obj in house["objects"]:
            if obj["objectType"] in objtype2remove:
                house["objects"].remove(obj)
        return house
    """Attempt to initialize the Expert controller with retries."""
    init_attempts = 0
    stretch_env_args = stretch_env_args or {}

    # multi-task
    # stretch_env_args['gpu_device'] = house_id%8  # Assign a GPU device based on house_id
    stretch_env_args['gpu_device'] = GPU_MAP[GPU_LIST[house_id % len(GPU_LIST)]]

    while init_attempts < max_init_attempts:
        try:
            Expert_controller = Expert(house=house, id=house_id, root=root, **stretch_env_args)
            scene = remove_objtype_from_house(house) # houses[house_id]
            if scene == None:
                return None
            Expert_controller.reset(scene)
            return Expert_controller  # Successfully initialized, return controller
        except TimeoutError as e:
            init_attempts += 1
            time.sleep(0.1)  # Wait before retrying

    return None


def filter_repeat_objects(Expert_controller: Expert, target):
    """Filter out repeated objects."""
    if len(Expert_controller.successful_objects) == 0:
        return False
    for successful_object in Expert_controller.successful_objects:
        if target['objectType'] == successful_object.split('|')[0]:
            return True
    return False




def process_house(args):
    """Process a house."""
    global all_count_1, all_count_3, fail_count_1, fail_count_3
    success_room = []
    house, house_id = args  # Unpack the arguments
    # print("house:", house_id)
    
    # 筛选1：house的room_num > 3
    # if get_rooms_lens(house) <= 3:
    #     print(f"House {house_id} has less than 3 rooms.")
    #     return
    group = house_id // 1000
    Expert_controller: Expert = initialize_expert_controller(5, STRETCH_ENV_ARGS, house, house_id,
                                                             root=f"{DATASET_DIR}/{SPLIT}/{group:03d}/")

    if Expert_controller is None:
        return

    # test_save_grid(Expert_controller, save_path="/home/libo/program/l3cprocthor/projects/Procthor/expert/save_grid")

    trajectory_id = 0
    trajectory_len = 0

    reach_positions = Expert_controller.reachable_positions

    if reach_positions is None:
        Expert_controller.stop()
        return

    # # 筛选2：reachable positions >= 100
    # if len(reach_positions) < 100:
    #     Expert_controller.stop()
    #     print(f"House {house_id} has less than 100 reachable positions.")
    #     return

    targets = Expert_controller.full_objects()

    # # 1. 前后两个target的初始position一致
    # random.shuffle(reach_positions)  # 起始位置随机化
    # for position in reach_positions:
    #     random.shuffle(targets)  # 打乱目标 (targets) 的顺序,使机器人的目标随机化
    #     if trajectory_len > 2_100:  # 如果轨迹总长度 (trajectory_len) 超过 2100，则跳出循环(why)
    #         break
    #     for target in targets:
    #         all_count_1 +=1

    #         try:
    #             Expert_controller.teleport_agent(position=position) # 与3的关键区别
    #             Expert_controller.reset_task(target, mode=1, trajectory_id=f"{trajectory_id:06d}")
    #             status = Expert_controller.verify_get_actions_from_shortest_path()
    #             trajectory_len += Expert_controller.len_actions
    #             trajectory_id +=1

    #             # gc.collect()  # 手动调用垃圾回收器，释放内存

    #         except Exception as e:
    #             # debug
    #             print(e)
    #             trajectory_id += 1
    #             Expert_controller.save_collected_data(e)
    #             print("next!")

    #             # 传送到上一条序列的终点，保持连续性
    #             Expert_controller.teleport_agent(position=Expert_controller.pre_end_pose)
    #             # gc.collect()

    # 2. 前后两个target的初始position一致 + 中间去别的房间转一圈
    # random.shuffle(reach_positions) # TIMER 001
    Nf = 0
    Ns = 0
    # import pdb;pdb.set_trace()
    pos_target_pairs = [[position, target] for position in reach_positions for target in targets]
    random.shuffle(pos_target_pairs)
    for pairs in pos_target_pairs:
        if trajectory_len >= 10000:
            break
        position, target = pairs
        print(f"house_id:{house_id},target:{target['name']},trajectory_len:{trajectory_len}")

        try:
            Expert_controller.teleport_agent(position=position)
            Expert_controller.reset_task(target,mode=2, trajectory_id=f"{trajectory_id:06d}")
            status = Expert_controller.verify_get_actions_from_shortest_path(exploration_room=False) # TODO open it after testing
            trajectory_len += Expert_controller.len_actions
            trajectory_id +=1
            # gc.collect()
            Ns += 1

        except Exception as e:
            print(f"House {house_id} object {target['objectType'],target['objectId']} failed: {e}")
            Nf += 1

    # for position in reach_positions:
    #     random.shuffle(targets)
    #     if trajectory_len > 10:
    #         break
    #     for target in targets:
    #         # position = {'x': 13.800000190734863, 'y': 0.9009921550750732, 'z': 7.75}
    #         # position = {'x': 5.400000095367432, 'y': 0.9009921550750732, 'z': 3.8499999046325684}
    #         # target = {'name': 'chair-diningtable|8|2|1', 'position': {'x': 11.199691772460938, 'y': -0.0011418461799621582, 'z': 7.37462043762207}, 'rotation': {'x': -0.0, 'y': 100.40934753417969, 'z': -0.0}, 'visible': False, 'isInteractable': False, 'receptacle': True, 'toggleable': False, 'isToggled': False, 'breakable': False, 'isBroken': False, 'canFillWithLiquid': False, 'isFilledWithLiquid': False, 'fillLiquid': None, 'dirtyable': False, 'isDirty': False, 'canBeUsedUp': False, 'isUsedUp': False, 'cookable': False, 'isCooked': False, 'temperature': 'RoomTemp', 'isHeatSource': False, 'isColdSource': False, 'sliceable': False, 'isSliced': False, 'openable': False, 'isOpen': False, 'openness': 0.0, 'pickupable': False, 'isPickedUp': False, 'moveable': True, 'mass': 14.999999046325684, 'salientMaterials': ['Metal', 'Fabric'], 'receptacleObjectIds': ['ObjaHandbag|8|70'], 'distance': 6.69846248626709, 'objectType': 'Chair', 'objectId': 'chair-diningtable|8|2|1', 'assetId': 'Chair_313_1', 'parentReceptacles': ['Floor'], 'controlledObjects': None, 'isMoving': False, 'axisAlignedBoundingBox': {'cornerPoints': [[11.546333312988281, 0.848613977432251, 7.753922462463379], [11.546333312988281, 0.848613977432251, 7.003195762634277], [11.546333312988281, 2.8014183044433594e-06, 7.753922462463379], [11.546333312988281, 2.8014183044433594e-06, 7.003195762634277], [10.79800796508789, 0.848613977432251, 7.753922462463379], [10.79800796508789, 0.848613977432251, 7.003195762634277], [10.79800796508789, 2.8014183044433594e-06, 7.753922462463379], [10.79800796508789, 2.8014183044433594e-06, 7.003195762634277]], 'center': {'x': 11.172170639038086, 'y': 0.4243083894252777, 'z': 7.378559112548828}, 'size': {'x': 0.7483253479003906, 'y': 0.8486111760139465, 'z': 0.7507266998291016}}, 'objectOrientedBoundingBox': {'cornerPoints': [[11.429740905761719, 2.950429916381836e-06, 7.003196716308594], [11.546332359313965, 2.950429916381836e-06, 7.637870788574219], [10.91459846496582, 2.950429916381836e-06, 7.753922462463379], [10.798007011413574, 2.950429916381836e-06, 7.119247913360596], [11.429740905761719, 0.848613440990448, 7.003196716308594], [11.546332359313965, 0.848613440990448, 7.637870788574219], [10.91459846496582, 0.848613440990448, 7.753922462463379], [10.798007011413574, 0.848613440990448, 7.119247913360596]]}}
    #         # position = {'x': 6.0, 'y': 0.9009921550750732, 'z': 11.5}
    #         # target = {'name': 'Mug|4|7', 'position': {'x': 3.9430975914001465, 'y': 0.5525742173194885, 'z': 14.29828929901123}, 'rotation': {'x': 0.0, 'y': 180.0, 'z': 0.0}, 'visible': False, 'isInteractable': False, 'receptacle': True, 'toggleable': False, 'isToggled': False, 'breakable': True, 'isBroken': False, 'canFillWithLiquid': True, 'isFilledWithLiquid': False, 'fillLiquid': None, 'dirtyable': True, 'isDirty': False, 'canBeUsedUp': False, 'isUsedUp': False, 'cookable': False, 'isCooked': False, 'temperature': 'RoomTemp', 'isHeatSource': False, 'isColdSource': False, 'sliceable': False, 'isSliced': False, 'openable': False, 'isOpen': False, 'openness': 0.0, 'pickupable': True, 'isPickedUp': False, 'moveable': False, 'mass': 1.0, 'salientMaterials': ['Ceramic'], 'receptacleObjectIds': [], 'distance': 3.4903676509857178, 'objectType': 'Mug', 'objectId': 'Mug|4|7', 'assetId': 'Mug_1', 'parentReceptacles': ['Floor', 'ObjaBench|4|5'], 'controlledObjects': None, 'isMoving': False, 'axisAlignedBoundingBox': {'cornerPoints': [[4.017430782318115, 0.6562117338180542, 14.34969711303711], [4.017430782318115, 0.6562117338180542, 14.247182846069336], [4.017430782318115, 0.5525742769241333, 14.34969711303711], [4.017430782318115, 0.5525742769241333, 14.247182846069336], [3.891842842102051, 0.6562117338180542, 14.34969711303711], [3.891842842102051, 0.6562117338180542, 14.247182846069336], [3.891842842102051, 0.5525742769241333, 14.34969711303711], [3.891842842102051, 0.5525742769241333, 14.247182846069336]], 'center': {'x': 3.954636812210083, 'y': 0.6043930053710938, 'z': 14.298439979553223}, 'size': {'x': 0.12558794021606445, 'y': 0.10363751649856567, 'z': 0.10251426696777344}}, 'objectOrientedBoundingBox': {'cornerPoints': [[3.891840934753418, 0.5525741577148438, 14.247182846069336], [4.017428398132324, 0.5525741577148438, 14.247182846069336], [4.017428398132324, 0.5525741577148438, 14.349696159362793], [3.891840934753418, 0.5525741577148438, 14.349696159362793], [3.891840934753418, 0.6562091112136841, 14.247182846069336], [4.017428398132324, 0.6562091112136841, 14.247182846069336], [4.017428398132324, 0.6562091112136841, 14.349696159362793], [3.891840934753418, 0.6562091112136841, 14.349696159362793]]}}
    #         try:
    #             Expert_controller.teleport_agent(position=position)
    #             Expert_controller.reset_task(target,mode=2, trajectory_id=f"{trajectory_id:06d}")
    #             status = Expert_controller.verify_get_actions_from_shortest_path(exploration_room=True) # TODO open it after testing
    #             trajectory_len += Expert_controller.len_actions
    #             trajectory_id +=1
    #             # gc.collect()
    #             Ns += 1

    #         except Exception as e:
    #             print(f"House {house_id} object {target['objectType'],target['objectId']} failed: {e}")
    #             Nf += 1
                # print(f"House {house_id} object {target['objectType'],target['objectId']} failed: {e}")
                # gc.collect()

     # # 3. 前后两个target位置连续
    # random.shuffle(reach_positions)
    # for position in reach_positions:
    #     if trajectory_len > 2_100:
    #         break
    #     Expert_controller.teleport_agent(position=position)
    #     random.shuffle(targets)
    #     for target in targets:
    #         all_count_3 +=1
    #         try:
    #             Expert_controller.reset_task(target, mode=3, trajectory_id=f"{trajectory_id:06d}")
    #             status = Expert_controller.verify_get_actions_from_shortest_path()
    #             trajectory_len += Expert_controller.len_actions
    #             trajectory_id +=1
    #             # gc.collect()

    #         except Exception as e:
    #             # print(e)
    #             # print(f"House {house_id} object {target['objectType'],target['objectId']} failed: {e}")
    #     fail_count_3 +=1
    #     # gc.collect()

    # # print("all_count:", all_count_3)
    # # print("fail_count:", fail_count_3)
    # # print(f"fail_rate_3:{(fail_count_3/all_count_3):.2f}")

    # # 4. 前后两个target位置连续 + 去别的房间转一圈
    # random.shuffle(reach_positions)
    # for position in reach_positions:
    #     random.shuffle(targets)
    #     if trajectory_len > 2_100:
    #         break
    #     Expert_controller.teleport_agent(position=position)b  
    #     for target in targets:
    #         try:
    #             Expert_controller.reset_task(target, mode=4, trajectory_id=f"{trajectory_id:06d}")
    #             status = Expert_controller.verify_get_actions_from_shortest_path(exploration_room=True)
    #             trajectory_len += Expert_controller.len_actions
    #             trajectory_id +=1
    #             # gc.collect()

    #         except Exception as e:
    #             # print(f"House {house_id} object {target['objectType'],target['objectId']} failed: {e}")
    #             # gc.collect()


    # # 5. 全流程连续
    # random.shuffle(reach_positions)
    # Expert_controller.teleport_agent(position=reach_positions[0])
    # Expert_controller.pre_end_pose = Expert_controller.get_current_agent_full_pose()
    # for i in range(2_000):
    #     random.shuffle(targets)
    #     for target in targets:
    #         if trajectory_len > 2_100:
    #             break

    #         # start_time = time.time()
    #         try:

    #             Expert_controller.reset_task(target, mode=4, trajectory_id=f"{trajectory_id:06d}")
    #             status = Expert_controller.verify_get_actions_from_shortest_path(exploration_room=True)
    #             trajectory_len += Expert_controller.len_actions
    #             trajectory_id += 1

    #             # end_time = time.time()
    #             # print("Success! Time:", end_time - start_time)


    #             # gc.collect()

    #         except Exception as e:
    #             # print(f"House {house_id} object {target['objectType'],target['objectId']} failed: {e}")
        
    #             # # 【debug】
    #             # end_time = time.time()
    #             # print("Fail! Time:", end_time - start_time)
    #             # print(e)
    #             # trajectory_id += 1
    #             # Expert_controller.save_collected_data(e)
    #             # print("next!")

    #             # 传送到上一条序列的终点，保持连续性
    #             Expert_controller.teleport_agent(position=Expert_controller.pre_end_pose)
    #             # gc.collect()
    Expert_controller.stop()
    if trajectory_len > 1_0000:
        print(f"House {house_id} completed. Trajectories: {trajectory_id}, Length: {trajectory_len}")
        success_room.append(f"{house_id:06d}+{trajectory_len:06d}")
        save_as_txt(success_room, f"{Expert_controller.root}")
    
    if Ns + Nf > 0:
        print(f"House {house_id} processing finished. Sucess ratio: {Ns/(Ns+Nf):.2f}")
        # gc.collect()





if __name__ == "__main__":
    # enable_remote_debug(65532)
    # houses = list(load_procthor_houses()[2])
    # house_ids = range(1000, 1100)
    # house_ids = range(1072, 1073)
    # house_ids = range(3646, 150000) # 150k
    # house_ids = range(11251, 150000)
    # house_ids = range(13000, 13030) # training set


    houses = load_procthor_houses(mode="val")
    # from prior import load_dataset_local
    import torch
    torch.multiprocessing.set_start_method("spawn", force=True)   
    # import time
    # house_id = 12033
    # s = time.time()
    # process_house((houses[house_id], house_id))
    # print(time.time()-s)
    # import pdb;pdb.set_trace()

    # houses = load_dataset_local("/pfs/pfs-r36Cge/qxg/program/l3cprocthor/data_collection-master/task1")['val'] 
    # shutil.rmtree(f"/pfs/mt-epYhpB/libo/datasets/{SPLIT}", ignore_errors=True)
    house_ids = range(1000,2000)
    print("Load procthor house finish!") 
    # NUM_PROCESSES = 1
    with Pool(processes=NUM_PROCESSES) as pool:
        for _ in tqdm(pool.imap_unordered(process_house, [(houses[house_id], house_id) for house_id in house_ids]),
                      total=len(house_ids)):
            pass
    print("All houses completed.")
