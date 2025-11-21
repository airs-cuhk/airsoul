#!/bin/bash
pkill -f "train.py config.yaml" || true
pkill -f "torchrun" || true
sleep 3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=eth0  # 使用相同网段
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT=600000

# export NCCL_DEBUG_SUBSYS=init,net,graph,env,tuning
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_FILE=/pfs/pfs-D9GUPM/log/15-server-log/nccl_log.%h.%p 


# 启动主节点
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
--master_addr="192.168.16.11" --master_port=29500 \
train.py config_test.yaml > multi_machine.log 2>&1
