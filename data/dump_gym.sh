#!/bin/bash

# Run the gen_gym_record.py script

ENV_NAME="LAKE" # Name of the gym environment, i.e. 'LAKE', 'CLIFF', 'LANDER', 'PENDULUM8x8', 'MOUNTAINCAR8x8', default is LAKE,
SAVE_PATH="./gym_data" # Path to save the training data (without file extension)
POLICY_NAME="PPO" # Policy type, default is DQN, can be DQN, TD3, A2C, or PPO. Note, TD3 is for continous problem.
N_TOTAL_TIMESTEPS=20000 # Total number of epochs for training, default is 200000
N_TASK=1 # Total number of tasks for generating, default is 1. If env isn't random, set 1.
N_SEQ_LEN=4000 # Maximum number of actions per sequence, default is 4000.
N_MAX_STEPS=200 # Maximum number of steps per trail, for PENDULUM & MOUNTAINCAR default is 200.
N_WORKERS=10 # Number of parallel workers for segment data generation, default is 10
ENABLE_LOAD_MODEL="False" # Whether to load a pre-trained model, default is False
RANDOM_ENV="False" # Whether to use random environment, default is False
TAG=3 # Tag. 3 for opt with gramma 0.994. 4 for opt with noise. 5 for q-learning with noise.
ACTION_DONE=5 # Action when env return done, default is action_dim of env.
REWARD_DONE=0.0 # Reward when env return done, default is 0.0
MAP_ENV_TO_DISCRETE="True" # Whether to map the env to discrete space, default is True
ACTION_CLIP=5 # Action discrete space, default is 5.

# Run the gen_gym_record.py script
export CUDA_VISIBLE_DEVICES=0
python gen_gym_record.py \
    --env_name $ENV_NAME \
    --save_path $SAVE_PATH \
    --policy_name $POLICY_NAME \
    --n_total_timesteps $N_TOTAL_TIMESTEPS \
    --n_task $N_TASK \
    --n_seq_len $N_SEQ_LEN \
    --n_max_steps $N_MAX_STEPS \
    --n_workers $N_WORKERS \
    --enable_load_model $ENABLE_LOAD_MODEL \
    --random_env $RANDOM_ENV \
    --tag $TAG \
    --action_done $ACTION_DONE \
    --reward_done $REWARD_DONE \
    --map_env_to_discrete $MAP_ENV_TO_DISCRETE \
    --action_clip $ACTION_CLIP