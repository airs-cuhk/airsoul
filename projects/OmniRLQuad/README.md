OmniRL (Omnipotent-Reinforcement-Leanring) is an in-context reinforcement learning framework meta-trained on the large-scale dataset of Markov Decision Processes (MDPs).

## Features of OmniRL
- **Generalized In-context Learning**: OmniRL can learn a novel MDP task by in-context learning using imitation learning, reinforce learning, and offline-RL.
- **Long-horizon In-Context Learning**: Can reasoning over trajectories as long as 1 million steps.
- **Highly Generalizable**: OmniRL can generalize to unseen MDPs and environments, including Cliff, Lake, MountainCar, Pendulum etc.

## Performances

A sketch of the structure of OmniRL
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRL_Structure.gif" alt="OmniRL Sketch" style="width: 320px;">
</div>

OmniRL is completely meta-trained on synthetic MDPs ([AnyMDP](https://github.com/FutureAGI/L3C/tree/main/l3c/anymdp)) and can generalize to various environments.
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/AnyMDP_Visualization.png" alt="OmniRL Train" style="width: 320px;">
</div>

Demonstration of OmniRL on various environments:
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo1.gif" alt="OmniRL Demo" style="height: 100px;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo2.gif" alt="OmniRL Demo" style="height: 100px;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo3.gif" alt="OmniRL Demo" style="height: 100px;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo4.gif" alt="OmniRL Demo" style="height: 100px;">
</div>

Performance of OmniRL on Lake benchmarks and its capability to ICL from random trajectory:
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRL_Figure.png" alt="OmniRL Performance" style="width: 320px;">
</div>

## Play with OmniRL

A trained stage-3 version of OmniRL can be downloaded from [here](https://pan.baidu.com/s/1y_a1I0xFQLt6ZQlc2RQp4g?pwd=vg5k).

You main play the model to interact with any discrete-space environments by 

```bash
python generator.py config_xxx.yaml
```

## Configuration

The `config.yaml` file contains all the necessary configuration for running OmniRL. Each configuration item is composed of multiple keys and sub-keys, can be over-written by commandline arguments. For instance, 
```yaml
model_config:
    state_encode:
        input_type: "Discrete"
```
can be over-written by commandline arguments as follows:
```bash
python train.py config.yaml --model_config.state_encode.input_type="Continuous"
```
Below we explain key configuration items in detail.

### General Configuration

- **run_name**:  # Names airsoul will use to discrimate the run from the others in the logs

- **master_port**: # A port used for connecting to the master node

- **load_model_path**: # Set to none in a cold start, or set to a path to load the model from a checkpoint

### Log Configuration

Specify the log path and whether to use tensorboard.

### Model Configuration (model_config)

Configuration for the overall model architecture and components, including encoders, decoders, and causal blocks. It defines the structure and behavior of the model during training and inference.

- **max_position_loss_weighting**: Defines the maximum sequence length that the model can handle.
- **context_warmup**: specify a increasing loss weighting with the context length, as shown in Appendices of [EPRNN](https://arxiv.org/pdf/2109.03554).
- **rsa_type**: Specifies how states, actions, rewards, prompts are encoded. Options include `sa`, `sar`, `psar`, `star` etc. OmniRL uses `star` by default.
- **causal_block**:  Options include `Transformer`, `GSA`, `GLA`, `MAMBA`, and `RWKV6`. OmniRL automatically use causal masks for `Transformer` and `RWKV6`, and employ a chunk-wise forward and backward pass. E.g., `Transformer` is automatically set to sliding window attention mode by setting train_config.seg_len.
- **state_encode**, **state_decode**, **action_encode**, ...: specify the encoder and decoder for states, actions, rewards etc.


### Training Configuration (train_config)

Settings for training the model.

- **seq_len**: Specify the sequence length loaded into the memory when training.
- **seg_len**: Specify the segment length used in chunk-wise forward and backward pass.
- **lr**, **lr_decay_interval**, **lr_start_step**: OmniRL apply noam decay with the warmup step specified by `lr_decay_interval`, use `lr_start_step` in cases of warm start.

### Test Configuration (test_config)

specify the configurations used for valiations between episodes or during static testing.

### Evaluation Configuration (generator_config)

specify the configurations for auto-regressive interaction with the environment during dynamic evaluation.

- **env**: #  anymdp32x5 / lake4x4 / cliff / mountaincar12x5 / pendulum12x5 / switch(multi-agent)
- **task_file**: # Use pre-defined tasks (pickle file generated by gen_xxx_task.py), only valid for AnyMDP task evaluation
- **action_clip**: # clip the action to fit the environment, e.g., if action space in OmniRL = 6, and action_clip = 4, action 4, 5 will be mapped into 0,1 in the environments
- **decoding_strategy**: # specify **T_ini**, **T_fin**, **T_step**, and **decay_type**, the temporature of decoding will decay from **T_ini** to **T_fin** after **T_step**

**Parameters not explicitly shown above may retain their default values as per the recommended configuration. Custom adjustments are available when aligned with specific application requirements.**
