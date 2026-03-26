#!/bin/bash
pkill -f "train.py /goosefsx/91mst04h/airs/qxg/czy/airsoul/train_config/02-15-train-rwkv_3b-token_180b-fp32.yaml" || true
pkill -f "torchrun" || true
killall -9 python || true
killall -9 /goosefsx/91mst04h/airs/zsj/yuxuan/statefull_LM/bin/python || true
sleep 3

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0  # 使用相同网段
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT=6000000

# take a input as node_rank and nnodes
node_rank=$1
nnodes=$2

# export NCCL_DEBUG_SUBSYS=init,net,graph,env,tuning
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_FILE=/pfs/pfs-D9GUPM/log/15-server-log/nccl_log.%h.%p 

# 40: 10.0.0.42



# 执行前，先 echo 完整指令
echo "torchrun --nnodes=$nnodes --node_rank=$node_rank --nproc_per_node=8 \
--master_addr="10.0.0.30" --master_port=29500 \
./projects/StatefulLM/train.py /goosefsx/91mst04h/airs/qxg/czy/airsoul/train_config/02-15-train-rwkv_3b-token_180b-fp32.yaml"

# 启动节点
torchrun --nnodes=$nnodes --node_rank=$node_rank --nproc_per_node=8 \
--master_addr="10.0.0.30" --master_port=29500 \
./projects/StatefulLM/train.py /goosefsx/91mst04h/airs/qxg/czy/airsoul/train_config/02-15-train-rwkv_3b-token_180b-fp32.yaml
