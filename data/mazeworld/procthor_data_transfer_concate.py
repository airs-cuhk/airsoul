import os
import sys
import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from numpy import random
from torch.utils.data import DataLoader, Dataset
import json

# action id to action name

# ACTION = {0: " ", 1: "m", 2: "b", 3: "l", 4: "r", 5: "ls", 6: "rs", 7: "end"}
# ACTION_CONTINUOUS = {
#     0: (0, 0),
#     1: (0.25, 0),
#     2: (-0.25, 0),
#     3: (0, np.deg2rad(-30)),
#     4: (0, np.deg2rad(30)),
#     5: (0, np.deg2rad(-6)),
#     6: (0, np.deg2rad(6)),
#     7: (0, 0),
# }

# ACTION = { 0: "ms", 1: "rs", 2: "ls", 5: "r", 6: "l", 11: "m", 16: "end", 17:" "}
# ACTION_ID = {v: k for k, v in ACTION.items()}

ACTION = { 0: "ms", 1: "rs", 2: "ls", 5: "r", 6: "l", 11: "m", 15: "b", 16: "end", 17: " "}
ACTION_ID = {v: k for k, v in ACTION.items()}

ACTION_CONTINUOUS = { # TO update
    0: (0, 0.25),
    1: (np.deg2rad(9), 0),
    2: (np.deg2rad(-9), 0),
    5: (np.deg2rad(36), 0),
    6: (np.deg2rad(-36), 0),
    11:(0, 0.5),
    15:(0, -0.25),
    16:(0,0),
    17:(0,0)
}


def get_house_spec(house):
    house_spec = len(house["rooms"])
    return house_spec


def read_txt(
    file_path # "/home/libo/program/wordmodel/libo/datasets/train/success_room.txt",
):
    houses = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if int(line.strip().split("+")[-1]) > 2_000:
                houses.append(line.strip().split("+")[0])
    houses = sorted(houses)
    # print("houses",houses)
    return houses


def aim_houses(
    successful_houses=None,
    # successful_houses=read_txt(file_path="/home/libo/program/wordmodel/libo/datasets/train/success_room.txt"),
    root=None #"/home/libo/program/wordmodel/libo/datasets/train",
):
    aim_houses_dir = []
    for house in successful_houses:
        house_dir = os.path.join(root, house.strip())
        aim_houses_dir.append(house_dir)
    return aim_houses_dir


def change_bev(bev_path, size=(128, 128)):
    bev = Image.open(bev_path)
    if size is not None:
        bev = bev.resize(size)
    bev = np.array(bev)
    return bev

def change_seg(seg_path, size=(16, 16)):
    bev = Image.open(seg_path)
    if size is not None:
        bev = bev.resize(size)
    bev = np.array(bev)
    return bev


def change_action(action_path):  # id and continuous value

    _actions_behavior_val, _actions_behavior_id = [], []

    with open(action_path, "r") as f:
        _actions_behavior = json.load(f)  # str

    for i in range(len(_actions_behavior)):
        if i==0:
            continue
        _actions_behavior_id.append(ACTION_ID[_actions_behavior[i]])  # id
        _actions_behavior_val.append(
            ACTION_CONTINUOUS[ACTION_ID[_actions_behavior[i]]]
        )  # continuous value

   

    return _actions_behavior_val, _actions_behavior_id


def change_action_refer(refer_action_path):
    _actions_label_val, _actions_label_id = [], []

    with open(refer_action_path, "r") as f:
        _actions_label = json.load(f)  # str

    refer_len = len(_actions_label[-1])

    for i in range(len(_actions_label)):
        if i==0:
            continue
        _action_label_val, _action_label_id = [], []

        if len(_actions_label[i]) != refer_len:
            _actions_label += [' '] * (refer_len - len(_actions_label[i]))
            
        for j in range(refer_len):
            if j >= len(_actions_label[i]):
                _actions_label[i].append(' ')
            _action_label_id.append(ACTION_ID[_actions_label[i][j]])  # id
            _action_label_val.append(
                ACTION_CONTINUOUS[ACTION_ID[_actions_label[i][j]]]
            )  # continuous value

        if refer_len != 11:
            print(refer_action_path)
            exit()
        

        _actions_label_val.append(np.array(_action_label_val))
        _actions_label_id.append(np.array(_action_label_id))
        
    return _actions_label_val, _actions_label_id



def change_agent(agent_path):

    with open(agent_path, "r") as f:
        agent = json.load(f)

    # position = agent.get("position", {})
    # rotation = agent.get("rotation", {})
    camera_horizon = agent.get("cameraHorizon", None)
    is_standing = agent.get("isStanding", None)
    return {
        # "position": position,
        # "rotation": rotation,
        "cameraHorizon": camera_horizon,
        "isStanding": is_standing,
    }


def change_object(object_path):
    with open(object_path, "r") as f:
        objects = json.load(f)
    return objects


def change_position(position_path):

    with open(position_path, "r") as f:
        position = json.load(f)
    return position


def chane_houses(house_path):
    with open(house_path, "r") as f:
        houses = json.load(f)
    return houses


def one_target2maze(target_dir):
    _tag = int(target_dir.split('|')[2])

    _observations = np.load(target_dir + "/metadata/navigation.npy")
    _actions_behavior_val, _actions_behavior_id = change_action(
        target_dir + "/metadata/actions.json"
    )

    _actions_label_val, _actions_label_id = change_action_refer(
        target_dir + "/metadata/actions_refer.json"
    )


    _position = change_position(
        target_dir + "/metadata/positions.json"
    )  # add the position of the agent

    # _BEVs = change_bev(
    #     target_dir + "/metadata/test_top_down_along_path.jpg"
    # )  # TODO: bev is top down along path

    _seg_obj = change_seg(
        target_dir + "/metadata/object_seg.jpg"
    )

    # # cut the last obs 最后两个一样，删掉最后一个
    observations = _observations[:-1]  # s->a->s->a->s->a_end

    actions_behavior_id, actions_behavior_val = (
        _actions_behavior_id,
        _actions_behavior_val,
    )

    actions_label_val, actions_label_id = (
        _actions_label_val,
        _actions_label_id,
    )

    # reward: [0,0,0,...,1,0]
    rewards = [0] * len(observations)
    rewards[-1] = 1 # ?ToFix

    # tag
    tags = [_tag] * len(observations)

    # TODO:Check
    seg_objs = [_seg_obj.copy() for _ in range(len(observations))]
    
    position = _position[:-1]  # cut the last position
    agent = change_agent(target_dir + "/metadata/agent.json")
    agent = [agent] * len(observations)

    target = change_object(target_dir + "/metadata/object.json")

    # 全黑
    observations = np.concatenate((observations, [np.zeros(observations.shape[1:])]), axis=0)
    seg_objs.append(np.zeros(seg_objs[0].shape))
    # BEVs.append(np.zeros(BEVs[0].shape))

    # 末尾添加action_end
    actions_behavior_val.append(actions_behavior_val[-1])
    actions_behavior_id.append(actions_behavior_id[-1])
    actions_label_id.append(actions_label_id[-1])
    actions_label_val.append(actions_label_val[-1])
    

    # 
    # agent
    # position
    # target

    tags.append(0)
    rewards.append(0)
    target = [target] * len(observations)
    # check if the lengths match
    assert len(observations) == len(actions_behavior_id), f"Subsequence Length mismatch: {len(observations)} != {len(actions_behavior_id)}"
    assert len(observations) == len(actions_behavior_val), f"Subsequence Length mismatch: {len(observations)} != {len(actions_behavior_val)}"
    assert len(observations) == len(actions_label_val), f"Subsequence Length mismatch: {len(observations)} != {len(actions_label_val)}"
    assert len(actions_behavior_id) == len(actions_label_id), f"Subsequence Length mismatch: {len(actions_behavior_id)} != {len(actions_label_id)}"
    return (
        observations,
        actions_behavior_id,
        actions_behavior_val,
        actions_label_id,
        actions_label_val,
        rewards,
        # BEVs,
        agent,
        position,
        target,
        tags,
        seg_objs
    )


def house_target2maze(
    house_dir,
    save=True,
    save_dir= None,
    # save_dir= None #"/home/libo/program/wordmodel/libo/for_train_word_model",
):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        # print(f"Create save directory: {save_dir}")
    
    def get_house_id_as_save_dir(house_dir=house_dir, save_dir=save_dir):
        house_id = house_dir.split("/")[-1]
        save_dir = os.path.join(save_dir, house_id)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    


    # house_spec = get_house_spec(_house)

    _observations = []
    _actions_behavior_id = []
    _actions_behavior_val = []
    _actions_label_id = []
    _actions_label_val = []
    _rewards = []
    _position = []

    _agent = []
    _target = []
    # _BEVs = []
    _command_objs = []
    _tags = []

    _house = chane_houses(house_dir + "/house.json")
    for target in os.listdir(house_dir):
        target_dir = os.path.join(house_dir, target)
        # print(f"Processing target: {target}")
        if os.path.isdir(target_dir):
            try:
                (
                    observations,
                    actions_behavior_id,
                    actions_behavior_val,
                    actions_label_id,
                    actions_label_val,
                    rewards,
                    # BEVs,
                    agent,
                    position,
                    target,
                    tags,
                    seg_objs
                ) = one_target2maze(target_dir)
            except Exception as e:
                # print(f"Error processing target {target_dir}: {e}")
                continue

            _observations.extend(
                observations
            )  # TODO: check if this is correct match the length of the observation
            _actions_behavior_id.extend(
                actions_behavior_id
            )  # TODO: check if this is correct match the length of the observation
            _actions_behavior_val.extend(
                actions_behavior_val
            )  # TODO: check if this is correct  match the length of the observation
            _actions_label_id.extend(
                actions_label_id
            )  # TODO: check if this is correct match the length of the observation
            _actions_label_val.extend(
                actions_label_val
            )  # TODO: check if this is correct match the length of the observation
            _rewards.extend(
                rewards
            )  # TODO: check if this is correct match the length of the observation

            # _BEVs.extend(BEVs)  # Convert BEVs to numpy array before appending


            _position.extend(
                position
            )  # TODO: check if this is correct match the length of the observation
            _agent.extend(agent)
            _target.extend(target)

            _command_objs.extend(seg_objs)
            _tags.extend(tags)
    

    assert len(_observations) == len(_actions_behavior_id) == len(_actions_behavior_val) == len(_actions_label_id) == len(_actions_label_val) == len(_rewards) == len(_command_objs) == len(_tags), \
        f"Length mismatch: {_observations}, {_actions_behavior_id}, {_actions_behavior_val}, {_actions_label_id}, {_actions_label_val}, {_rewards}, {_position}, {_agent}, {_target}, {_command_objs}, {_tags}"    
    if len(_observations) <= 10000:
        print(f"Warning: The length of the observations is less than 10000, which may not be sufficient for training. Length: {len(_observations)}")
        return None
    if save:
        save_dir = get_house_id_as_save_dir(house_dir)
        np.save(os.path.join(save_dir, "observations.npy"), np.array(_observations))

        np.save(
            os.path.join(save_dir, "actions_behavior_id.npy"),
            np.array(_actions_behavior_id),
        )
        np.save(
            os.path.join(save_dir, "actions_behavior_val.npy"),
            np.array(_actions_behavior_val),
        )
        np.save(
            os.path.join(save_dir, "actions_label_id.npy"), np.array(_actions_label_id)
        )
        np.save(
            os.path.join(save_dir, "actions_label_val.npy"),
            np.array(_actions_label_val),
        )

        np.save(os.path.join(save_dir, "rewards.npy"), np.array(_rewards))
        # np.save(os.path.join(save_dir, "BEVs.npy"), np.array(_BEVs))

        np.save(os.path.join(save_dir, "positions.npy"), np.array(_position))
        np.save(
            os.path.join(save_dir, "target.npy"),
            np.array(_target, dtype=object),
            allow_pickle=True,
        )
        np.save(
            os.path.join(save_dir, "agent.npy"),
            np.array(_agent, dtype=object),
            allow_pickle=True,
        )
        np.save(
            os.path.join(save_dir, "house.npy"),
            np.array(_house, dtype=object),
            allow_pickle=True,
        )

        np.save(os.path.join(save_dir, "commands.npy"), np.array(_command_objs))
        np.save(
            os.path.join(save_dir, "actions_behavior_prior.npy"),
            np.array(_tags)
        )

    return (
        _observations,
        _actions_behavior_id,
        _actions_behavior_val,
        _actions_label_id,
        _actions_label_val,
        _rewards,
        # _BEVs,
        _agent,
        _position,
        _target,
        _house,
    )


def get_all_house_target2maze(
    houses_dir=None,#"/home/libo/program/wordmodel/libo/datasets/train", \
    # houses_dir="/home/libo/program/wordmodel/libo/datasets/train", \
    save_dir=None,
    timestep=10_000
):  
    print(f"houses_dir: {houses_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Create save directory: {save_dir}")
    subdirs = sorted([d for d in os.listdir(houses_dir) if os.path.isdir(os.path.join(houses_dir, d))])
    
    for i, subdir in enumerate(subdirs):
        sub_houses_dir = os.path.join(houses_dir, subdir)
        house_ids = read_txt(file_path= os.path.join(sub_houses_dir,"success_room.txt"))
        info = {}
        for house in tqdm(house_ids, desc=f"Processing houses in {subdir}"):
            # show the house in the tqdm
            house_dir = os.path.join(sub_houses_dir, house)
            if os.path.isdir(house_dir):
                (
                    _observations,
                    _actions_behavior_id,
                    _actions_behavior_val,
                    _actions_label_id,
                    actions_label_val,
                    _rewards,
                    # _BEVs,
                    _agent,
                    _position,
                    _target,
                    _house,
                ) = house_target2maze(house_dir, save_dir=os.path.join(save_dir, subdir))
    return



def enable_remote_debug(port=None):
    try:
        import debugpy

        if port is None:
            ENV_DEBUG_PORT = os.environ.get("DEBUG_PORT")
            port = int(ENV_DEBUG_PORT) if ENV_DEBUG_PORT else 5678
            if not ENV_DEBUG_PORT:
                print(
                    "Set env DEBUG_PORT can change default port. (Linux example cmd: export DEBUG_PORT=5678)"
                )
        address = ("0.0.0.0", port)
        debugpy.listen(address)
        print("### Wait Remote Debug (port:" + str(port) + ") ###")
        debugpy.wait_for_client()
        print("### Connected Remote Debug ###")
    except BaseException as e:
        print("enable_remote_debug err:", e)
        return False
    else:
        return True


import os
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_house(house, sub_houses_dir, save_subdir):
    """处理单个 house 的函数，用于多进程调用"""
    house_dir = os.path.join(sub_houses_dir, house)
    try:
        if os.path.isdir(house_dir):
            (
                _observations,
                _actions_behavior_id,
                _actions_behavior_val,
                _actions_label_id,
                actions_label_val,
                _rewards,
                _agent,
                _position,
                _target,
                _house,
            ) = house_target2maze(house_dir, save_dir=save_subdir)
    except Exception as e:
        print(f"Error processing house {house}: {e}")
        return

def process_subdir(subdir, houses_dir, save_dir):
    """处理单个 subdir 的函数，用于多进程调用"""
    sub_houses_dir = os.path.join(houses_dir, subdir)
    save_subdir = os.path.join(save_dir, subdir)
    os.makedirs(save_subdir, exist_ok=True)
    try:
        house_ids = read_txt(file_path=os.path.join(sub_houses_dir, "success_room.txt"))
    except FileNotFoundError:
        print(f"File not found: {os.path.join(sub_houses_dir, 'success_room.txt')}")
        return
    
    # 使用进程池处理每个 house
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm(
            pool.imap(
                partial(process_house, sub_houses_dir=sub_houses_dir, save_subdir=save_subdir),
                house_ids
            ),
            total=len(house_ids),
            desc=f"Processing houses in {subdir}"
        ))

def get_all_house_target2maze_mutiprocess(
    houses_dir=None,
    save_dir=None,
    timestep=10_000,
    num_processes= None
):
    print(f"houses_dir: {houses_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Create save directory: {save_dir}")
    
    subdirs = sorted([d for d in os.listdir(houses_dir) if os.path.isdir(os.path.join(houses_dir, d))])
    # 顺序处理每个 subdir
    for subdir in subdirs:
        process_subdir(subdir, houses_dir, save_dir)
    # # 使用进程池处理每个 subdir
    # with multiprocessing.Pool(processes=num_processes or multiprocessing.cpu_count()) as pool:
    #     pool.map(
    #         partial(process_subdir, houses_dir=houses_dir, save_dir=save_dir),
    #         subdirs
    #     )

# Test Maze Data Set
if __name__ == "__main__":
    # from the inital datasets
    # get_all_house_target2maze(houses_dir="/home/libo/program/wordmodel/libo/datasets/train")
    # get_all_house_target2maze(houses_dir="/home/libo/program/wordmodel/libo/datasets_test/train",
    #                           save_dir = "/home/libo/program/wordmodel/qxg/datasets/procthor")

    # get_all_house_target2maze(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/trajectory/train",
    #                           save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-07-val_trajectory_wm")

    # get_all_house_target2maze(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/0617-trajectory/train/",
    #                           save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-07-val_trajectory_wm")

    # get_all_house_target2maze(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/0714-train-10000/train",
    #                           save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-17-train_oracle_trajectory_wm")
    # # "/pfs/pfs-r36Cge/qxg/datasets/procthor/0716-train-10000"
    # get_all_house_target2maze(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/0716-train-10000/train", # TODO
    #                           save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-21-train_trajectory")

    # get_all_house_target2maze(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/07-21-Oracle-data/train/", 
    #                           save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-22-oracle_validation_trajectory")

    get_all_house_target2maze_mutiprocess(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/07-21-Oracle-data/train/", 
                              save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-22-oracle_validation_trajectory")
    # get_all_house_target2maze_mutiprocess(houses_dir="/pfs/pfs-r36Cge/qxg/datasets/procthor/0716-train-10000/train/", 
    #                           save_dir = "/pfs/pfs-r36Cge/qxg/datasets/procthor/07-22-expert_train_trajectory")
    # enable_remote_debug()


    exit()
       
