#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0  # 使用相同网段
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT=6000000

# take a input as node_rank and nnodes
GPUS_PER_NODE=$1
NNODES=$2
RANK=$3
MASTER=$4
PORT=$5
config_path=$6
OUTPUT_FILE=$7


unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

pkill -f "./projects/StatefulLM/train.py $config_path" || true
pkill -f "torchrun" || true
killall -9 python || true
killall -9 /goosefsx/91mst04h/airs/zsj/yuxuan/statefull_LM_pt/bin/python || true
sleep 3



# 执行前，先 echo 完整指令
echo "torchrun --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
--master_addr="$MASTER" --master_port=$PORT \
./projects/StatefulLM/train.py $config_path > $OUTPUT_FILE 2>&1 &"


# 启动节点
nohup torchrun --nnodes=$NNODES --node_rank=$RANK --nproc_per_node=$GPUS_PER_NODE \
--master_addr="$MASTER" --master_port=$PORT \
./projects/StatefulLM/train.py $config_path > $OUTPUT_FILE 2>&1 &
