# AIRSoul : Towards the Next Generation Foundation Model for Embodied Agents
Embodied AI faces key challenges in generalization and adaptation to different environments, cross-embodiments, and variant tasks. We believe in-weight learning (IWL) and scaling laws alone can not efficiently solve these problems. 

AIRSoul is towarding building general-purpose in-context learning (ICL) agents featured with the following characteristics:
- Generalized ICL: Ability to address novel tasks with reinforcement learning, imitation learning, and self-supervised learning.
- Long-horizon ICL: Ability to complex tasks requiring a huge number of steps at minimum.
- Continual Learning in ICL: Learning and switching among a wide range of tasks without catastrophic forgetting.

# Directory Structure
- [projects](./projects): implementations of model training and validating for different benchmarks and projects.
    - [MetaLM](./projects/MetaLM) foundation model for [Xenoverse MetaLM](https://github.com/FutureAGI/Xenoverse/tree/main/xenoverse/metalang)
    - [MazeWorld](./projects/MazeWorld) foundation model for [Xenoverse MazeWorld](https://github.com/FutureAGI/Xenoverse/tree/main/xenoverse/mazeworld)
    - [OmniRL](./projects/OmniRL) foundation model for [Xenoverse AnyMDP](https://github.com/FutureAGI/Xenoverse/tree/main/xenoverse/anymdp)

- `data`: For general-purpose learning to learn, we generate the datasets by procedurally sampling tasks and use expert's demonstration for imitation learning and self-supervised learning. This directory contains the scripts to generate unlimited datasets for training.

- `airsoul`: contains the building blocks and utils of different models
    - `modules`: contains the basic blocks
    - `utils`: contains the utils for building networks, training, and evaluation
    - `models`: contains higher-level models built from basic blocks
    - `dataloader`: contains the dataloader for different tasks

# Training and Evaluating

## Install Requirements
To train a model run
```bash
pip install airsoul
```

## Generate Datasets

check the [data](./data) directory to generate datasets for training and evaluation.

## Start Training

### Reconfigure the Config File

Basically you need to modify the configuration file to start the training. The config file basically need to contain three major parts:
- `log_config`: configuration of the log directory, including the path to save the checkpoints and tensorboard logs
- `model_config`: configuration of the model structure and hyperparameters
- `train_config`: configuration of the training process, including learning rate, batch size, etc.
- `test_config`: configuration of the evaluation process, including the dataset to be evaluated

### Start Training

To train a model run
```bash
cd ./projects/PROJECT_NAME/
python train.py config.yaml
```

You might also overwrite the config file with command line arguments with ```--config```
```bash
python train.py config.yaml --configs key1=value1 key2=value2 ...
```

### Validating with static dataset
```bash
python validate.py config.yaml --configs key1=value1 key2=value2 ...
```

### Validating with interaction
The repo is under active development.
Feel free to submit a pull request.
