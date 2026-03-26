# AIRSoul : Towards the General-Purpose Foundation Model for Interactive and Embodied Agents
Embodied AI encounters significant challenges in generalizing across varying environments, embodiments, and tasks. AIRSoul does not attempt to overcome these challenges through zero-shot generalization or parametric memory. Instead, we address these tasks using gradient-free black-box learning: Multi-Paradigm In-Context Learning. AIRSoul aims to construct a general-purpose learning machine with the following characteristics:

- Scalable ICL Incentivization: Incentivizing ICL in a training efficient manner.
- Multi-Paradigm ICL: The capability to tackle novel tasks by integrating reinforcement learning, imitation learning, self-supervised learning, and other learning methods within a unified model.
- Long-Context ICL: The ability to learn highly complex novel tasks that require a substantial number of steps with minimal effort.
- In-Context Continual Learning (ICCL): The potential for continual learning and even lifelong learning within context.

# Directory Structure
- [projects](./projects): implementations of model training and validating for different benchmarks and projects.
    - [MetaLM](./projects/MetaLM) foundation model for [Xenoverse MetaLM](https://github.com/FutureAGI/Xenoverse/tree/main/xenoverse/metalang)
    - [MazeWorld](./projects/MazeWorld) foundation model for [Xenoverse MazeWorld](https://github.com/FutureAGI/Xenoverse/tree/main/xenoverse/mazeworld)
    - [OmniRL](./projects/OmniRL) foundation model for [Xenoverse AnyMDP](https://github.com/FutureAGI/Xenoverse/tree/main/xenoverse/anymdp)

- `data`: Contains the scripts to generate mega-datasets for training.

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
