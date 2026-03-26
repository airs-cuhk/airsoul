export CUDA_VISIBLE_DEVICES=0,1
export NCCL_SOCKET_IFNAME=xxxxx  # 替换为实际的网络接口名称
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO  
# 启动主节点
torchrun --nnodes=x --node_rank=x --nproc_per_node=x \
  --master_addr="xx.xx.xxx.xxx" --master_port=xxxxx \
  train.py config.yaml > run.log 2>&1
