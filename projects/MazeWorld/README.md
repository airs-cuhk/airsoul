A baseline State-Action Causal Modeling for [Maze World](https://github.com/FutureAGI/L3C/tree/main/l3c/mazeworld)

<div style="width: 960; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-1.jpg" alt="Keyboard Demo">
</div>

## Model Structure
<div style="width: 960; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Causal_Modeling.jpg" alt="Keyboard Demo">
</div>

## Training
Reconfigure the settings in `config.yaml`, start training with the following command.
```bash
export CUDA_VISIBLE_DEVICES=...
python train.py config.yaml --configs key1=value1 key2=value2 ...
```

## Validation
```bash
export CUDA_VISIBLE_DEVICES=...
python validate.py config.yaml --configs key1=value1 key2=value2 ...
```

## This Directory is Under Construction