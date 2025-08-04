import json,os,shutil
import prior
import math
from matplotlib import pyplot as plt
from expert.actions_qu import sample_from_shortest_path
from environment.stretch_controller import StretchController
from utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio
import h5py
import numpy as np
import random

def get_rooms_lens(house):
    if house is None:
        return 0
    return len(house['rooms'])

def select_rooms(room_list, target_room_id, now_room_id):
    L = len(room_list)
    selected_rooms = []
    
    if target_room_id == now_room_id:
        return selected_rooms
    else:
        # 随机选择 0 到 L-2 的数字 n
        n = random.randint(0, L - 2)
        # 从列表中随机选择 n 个不含 target_room_id 和 now_room_id 的 room_id
        candidates = [room for room in room_list if room != target_room_id and room != now_room_id]
        selected_rooms = random.sample(candidates, min(n, len(candidates)))
    
    return selected_rooms

def build_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    return dir

def l2_distance(a, b, ignore_y: bool = True) -> float:
    """Return the L2 distance between two points."""
    dx = a["x"] - b["x"]
    dz = a["z"] - b["z"]
    
    if ignore_y:
        return (dx ** 2 + dz ** 2) ** 0.5
    
    dy = a["y"] - b["y"]
    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

# python -m scripts.download_objaverse_houses --save_dir /home/fd_chen/in_context_RL/spoc-robot-training/data/objaverse_houses --subset train

def load_procthor_houses(mode ="small"):
    """Load the Procthor houses dataset."""
    print("mode:", mode)
    # print(OBJAVERSE_HOUSES_DIR)
    # OBJAVERSE_HOUSES_DIR = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/2023_07_28" # "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/houses_2023_07_28"
    # OBJAVERSE_HOUSES_DIR = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/houses_2023_07_28"

    OBJAVERSE_HOUSES_DIR = os.environ.get("OBJAVERSE_HOUSES_DIR")
    if not os.path.exists(OBJAVERSE_HOUSES_DIR):
        raise FileNotFoundError(f"OBJAVERSE_HOUSES_DIR does not exist: {OBJAVERSE_HOUSES_DIR}")
    else:
        print(f"OBJAVERSE_HOUSES_DIR: {OBJAVERSE_HOUSES_DIR}")
    if mode =="small":
        return prior.load_dataset(
            dataset="spoc-data", 
            entity="spoc-robot", 
            revision="houses-test-val",
            )["val"]
    elif mode == "train":
        # OBJAVERSE_HOUSES_DIR = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/2023_07_28"
        max_houses_per_split = {"train": 150000, "val": 15000, "test": 0}
        train_houses = prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="local-objaverse-procthor-houses",
            path_to_splits=None,
            split_to_path={
                k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                for k in ["train", "val", "test"]
            },
            max_houses_per_split=max_houses_per_split,
        )
        print(train_houses)
        return train_houses['train']
    elif mode == "val":
        # OBJAVERSE_HOUSES_DIR = "/pfs/pfs-r36Cge/libo/procthor/objaverse_houses/houses_2023_07_28"
        print("Loading validation houses...")
        max_houses_per_split = {"train": 150000, "val": 15000, "test": 0}
        train_houses = prior.load_dataset(
            dataset="spoc-data",
            entity="spoc-robot",
            revision="local-objaverse-procthor-houses",
            path_to_splits=None,
            split_to_path={
                k: os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                for k in ["train", "val", "test"]
            },
            max_houses_per_split=max_houses_per_split,
        )
        print(train_houses)
        return train_houses['val']
    
def save_data_json(data, save_path, should_save=True):
    """Save data to a file if should_save is True."""
    if should_save:
        try:
            with open(save_path, "w") as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            raise(f"Error saving data to {save_path}: {e}")

def check_agent_position(shortest_path, agent_position, threshold):
    """Check if the agent's position is within a certain threshold of the shortest path."""
    simple_points = sample_from_shortest_path(shortest_path)
    distances = [math.hypot(agent_position["position"]["x"] - point[0], agent_position["position"]["z"] - point[1]) for point in simple_points]
    return min(distances) <= threshold

def save_img_from_frame(frame, save_path="./generator_data/test.jpg", save_size=None):
    """Save an image from a frame."""
    try:
        if save_size is not None:
            pil_image = Image.fromarray(frame)
            resized_image = pil_image.resize(save_size)
            plt.imsave(save_path, resized_image)
        else:
            plt.imsave(save_path, frame)
    except Exception as e:
        raise(f"Error saving image to {save_path}: {e}")

def positions2path(positions):
    """Convert positions to a path format."""
    return [{'x': point[0], 'y': point[1], 'z': point[2]} for point in positions]

def debug_info(DEBUG, Expert_controller: StretchController):
    """Print debug information if DEBUG is True."""
    if DEBUG:
        full_info = Expert_controller.get_current_agent_full_pose()
        info_names = ['name', 'position', 'rotation', 'cameraHorizon', 'isStanding', 'inHighFrictionArea']
        for info in info_names:
            print(f"{info}: {full_info[info]}")
        print("=" * 40)

from concurrent.futures import ThreadPoolExecutor

def save_navigation_frame(nav_frames, frame_dir, async_save=True,vedio = True):
    """Save navigation frames to a directory, optionally asynchronously."""
    
    def async_save_image(file_path, nav_frame):
        """Helper function to save an image asynchronously."""
        plt.imsave(file_path, nav_frame)
    
    if async_save:
        executor = ThreadPoolExecutor(max_workers=4)  # 设置最大线程数
        for i, nav_frame in enumerate(nav_frames):
            file_path = f"{frame_dir}/{i}.png"
            executor.submit(async_save_image, file_path, nav_frame)
    else:
        for i, nav_frame in enumerate(nav_frames):
            plt.imsave(f"{frame_dir}/{i}.png", nav_frame)

def save_as_video(nav_frames,save_dir):
    video_path = f"{save_dir}/navigation_video.mp4"
    imageio.mimwrite(video_path, nav_frames, fps=30)

def save_as_numpy(nav_frames, save_dir):
    # Get the height and width of the first frame
    try:
        height, width = nav_frames[0].shape[:2]
        if height != 128 or width != 128:
            # Resize frames to 128x128 using PIL
            nav_frames = [np.array(Image.fromarray(frame).resize((128, 128))) for frame in nav_frames]
        np.save(f"{save_dir}/navigation.npy", nav_frames)
    except Exception as e:
        print(f"Error saving navigation frames as numpy array: {e}")

def save_data_h5py(data_dict, file_path):
    """
    Save multiple datasets in a single HDF5 file.

    :param data_dict: A dictionary where keys are dataset names and values are the data to save.
    :param file_path: The path to the HDF5 file.
    """
    with h5py.File(file_path, 'w') as f:
        for dataset_name, data in data_dict.items():
            f.create_dataset(dataset_name, data=np.array(data))

def save_as_txt(data: list, save_dir):
    with open(f"{save_dir}/success_room.txt", "a") as f:
        for item in data:
            f.write("%s\n" % item)