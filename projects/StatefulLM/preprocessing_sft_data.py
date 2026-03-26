import os
import glob
import numpy as np
import argparse
from multiprocessing import Process
# from megatron.core.datasets.indexed_dataset import IndexedDataset
from transformers import AutoTokenizer
import json
import tqdm
import numpy as np
import random
import torch

# 1. 加载数据集函数
def load_json_dataset(path):
    """加载JSON数据集文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

# 2. 初始化tokenizer
def initialize_tokenizer(model_name_or_path="/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/hf_checkpoints"):
    """
    初始化tokenizer
    
    Args:
        model_name_or_path: 预训练模型名称或路径
        
    Returns:
        tokenizer实例
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # 设置padding token（如果不存在）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
            
        return tokenizer
    except Exception as e:
        print(f"初始化tokenizer失败: {e}")
        return None

# 3. 数据处理函数
def prepare_dataset_for_finetuning(data, tokenizer, max_length=None):
    """
    准备数据集用于微调
    
    Args:
        data: 原始数据列表
        tokenizer: transformers tokenizer
        max_length: 最大序列长度
        
    Returns:
        处理后的数据集
    """
    processed_data = []
    
    # for item in tqdm.tqdm(data):
    for item in data:
        # 构建输入文本（组合instruction和input）
        if 'input' in item and item['input']:
            input_text = f"{item['instruction']}\n{item['input']}\n"
        else:
            input_text = item['instruction']
        
        # 目标文本
        target_text = f"{item['output']}\n{item['answer']}" # item.get('output', '') or item.get('answer', '')

        
        # tokenize输入
        input_encoding = tokenizer(
            input_text,
            truncation=True if max_length else False,
            # padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # tokenize目标
        target_encoding = tokenizer(
            target_text,
            truncation=True if max_length else False,
            # padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        processed_item = {
            'input_ids': torch.cat([input_encoding['input_ids'].squeeze(0), target_encoding['input_ids'].squeeze(0), torch.tensor([tokenizer.eos_token_id])]),  # 输入和目标拼接，并添加EOS
            'mask': torch.cat([torch.zeros_like(input_encoding['input_ids'].squeeze(0)), torch.ones_like(target_encoding['input_ids'].squeeze(0)), torch.ones(1, dtype=torch.long)])  # 输入部分为1，目标部分为0
        }
        
        processed_data.append(processed_item)
    
    return processed_data

def sft_json_dataset(json_path, model_name="/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/hf_checkpoints", max_seq_length=None, verbose=True):
    # dataset_path = "/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/benchmark/data/commonsense_170k.json"
    # model_name = "/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/hf_checkpoints"  # 可替换为其他模型，如"gpt2", "facebook/opt-125m"等
    # max_seq_length = None
    dataset_path = json_path
    if verbose:
        # 加载数据
        print("加载数据集...")
    raw_data = load_json_dataset(dataset_path)
    if raw_data is None:
        return
    
    if verbose:
        print(f"数据集大小: {len(raw_data)}")
    
        # 检查数据结构
        if len(raw_data) > 0:
            print(f"数据示例: {raw_data[0]}")
            print(f"数据键: {raw_data[0].keys()}")
    
    # 初始化tokenizer
    if verbose:
        print(f"\n初始化tokenizer: {model_name}")
    tokenizer = initialize_tokenizer(model_name)
    if tokenizer is None:
        return
    if verbose:
        print(f"Tokenizer特殊token: {tokenizer.special_tokens_map}")
        # 处理数据
        print("\n处理数据集...")
    processed_data = prepare_dataset_for_finetuning(
        data=raw_data[:],  # 示例：只处理前10条
        tokenizer=tokenizer,
        max_length=max_seq_length
    )
    if verbose:
        print(f"处理后的数据条数: {len(processed_data)}")
        if len(processed_data) > 0:
            print(f"处理后数据示例: {processed_data[0].keys()}")
            print(f"input_ids形状: {processed_data[0]['input_ids'].shape}")
            print(f"mask: {processed_data[0]['mask'].shape}")
            print(processed_data[0])
    return tokenizer, processed_data

# -----------------------------------------------------------------------------
# 核心处理逻辑：读取 -> 切割 src/tgt -> 堆叠 -> 存为 (N, 2, seq_len)
# -----------------------------------------------------------------------------
def worker_func(worker_id, file_prefixes, output_dir, seq_length, samples_per_file, files_per_subdir, pad_id=0):
    """
    输出格式：
    .npy 文件，Shape 为 (Batch_Size, 2, seq_length)
    - Channel 0: Source (输入)
    - Channel 1: Target (标签)
    Dtype: int64
    """

    buffer = []
    chunk_counter = 0
    # 我们仍然需要读取 seq_len + 1 个 token 才能切分出错位的 src 和 tgt
    read_len = seq_length + 1
    token_stream = []
    Total_padding_list = []
    def save_buffer_to_disk(current_buffer, current_chunk_idx):
        # 1. 计算子目录索引 (例如: 0, 1, 2...)
        subdir_idx = current_chunk_idx // files_per_subdir
        
        # 2. 拼接子目录路径 (output_dir/0, output_dir/1...)
        subdir_path = os.path.join(output_dir, str(subdir_idx))
        
        # 3. 创建子目录 (exist_ok=True 避免多进程冲突)
        os.makedirs(subdir_path, exist_ok=True)
        
        # 4. 生成最终文件路径
        # 文件名包含 worker_id，确保不同进程不会写入同一个文件
        filename = f"chunk_{worker_id}_{current_chunk_idx}.npy"
        save_path = os.path.join(subdir_path, filename)
        
        # 5. 保存
        data_array = np.stack(current_buffer)
        np.save(save_path, data_array)
        
        if worker_id == 0:
            print(f"[Worker {worker_id}] Saved {save_path} "
                  f"(Shape: {data_array.shape}, Subdir: {subdir_idx})")
    
    if isinstance(file_prefixes, str):
        file_prefixes = [file_prefixes]
    for prefix in file_prefixes:
        try:
            # ds = IndexedDataset(prefix, multimodal=False)
            tokenizer, ds = sft_json_dataset(prefix)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading {prefix}: {e}")
            continue
        # make sure for all data in ds, len(data['mask']) - sum(data['mask']) < read_len
        assert max([len(data['mask']) for data in ds]) < read_len, "Must longer than that, otherwise data loss"
        
        token_stream = []
        mask_stream = []
        
        # shuffle the dataset ds
        rnd_index = np.random.permutation(len(ds))
        # for i in range(len(ds)):
        for i in rnd_index:
            try:
                processed_item = ds[i]
                doc = processed_item['input_ids']
                mask = processed_item['mask']
            except Exception:
                continue
            
            # if len(token_stream) == 0:
            #     token_stream = list(doc)
            # else:
            #     token_stream.extend(doc)
            
            # if worker_id == 0:
            #     print(len(doc))
            
            # 当流中的数据足够切分时
            
            # if len(token_stream) + 
            if len(token_stream) + len(doc) >= read_len:
                # 0. 将 token_stream padding 到 read_len
                # print(len(token_stream), len(mask_stream))
                if len(token_stream) < read_len:
                    Total_padding_list.append(read_len - len(token_stream))
                    token_stream = token_stream + [tokenizer.pad_token_id] * (read_len - len(token_stream))
                    mask_stream = mask_stream + [0] * (read_len - len(mask_stream))
                    full_seq = token_stream[:read_len]
                    full_mask = mask_stream[:read_len]
                    token_stream = []
                    mask_stream = []
                else:
                    raise NotImplementedError("token_stream is too long, but not enough for read_len")
                
                    # full_seq = token_stream[:read_len]
                    # token_stream = token_stream[seq_length:] # 滑动窗口或截断，这里按截断处理
                
                # 2. 切分 src 和 tgt
                # src: 0 到 -1
                # tgt: 1 到 结尾
                src_seq = full_seq[:-1]
                tgt_seq = full_seq[1:]
                mask_seq = full_mask[:-1]
                
                # 3. 堆叠成一对 (3, seq_len)
                # 强制转为 int64
                pair = np.stack([src_seq, tgt_seq, mask_seq]).astype(np.int64)
                buffer.append(pair)
                # print(f"Pair shape: {pair.shape}")
                # 4. 存盘逻辑
                if len(buffer) >= samples_per_file:
                    save_buffer_to_disk(buffer, chunk_counter)
                    buffer = []
                    chunk_counter += 1
            
            token_stream.extend(doc)
            mask_stream.extend(mask)

        del ds
    # save chunk if it is not empty
    if len(buffer) > 0:
        save_buffer_to_disk(buffer, chunk_counter)
        if worker_id == 0:
            print(f"[Worker {worker_id}] Saved chunk tail.")
        buffer = []
        chunk_counter += 1
    print(f"Worker {worker_id} Total_padding: {sum(Total_padding_list)}, length: {len(Total_padding_list)}, and Total chunk: {chunk_counter}")

    # discard the last chunk if it is shorter than samples_per_file
    # if len(token_stream) > 0:
    #     save_path = os.path.join(output_dir, f"chunk_{worker_id}_{chunk_counter}_last.npy")
    #     pad_len = read_len - len(token_stream)
    #     padded = token_stream + [pad_id] * pad_len
    #     src = padded[:-1]
    #     tgt = padded[1:]
    #     buffer.append(np.stack([src, tgt]))
    #     data_array = np.stack(buffer)
    #     np.save(save_path, data_array)
    #     print(f"[Worker {worker_id}] Saved last chunk {save_path} (Shape: {data_array.shape})")


# -----------------------------------------------------------------------------
# 主控制流程
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--samples_per_file", type=int, default=500)
    parser.add_argument("--files_per_subdir", type=int, default=100, help="每个子目录存放多少个文件")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    processes = []

    print(f"开始处理... ")
    print(f"配置: Shape=(N, 3, {args.seq_length}), 每个文件样本数={args.samples_per_file}")
    print(f"分桶策略: 每 {args.files_per_subdir} 个文件存入一个子文件夹 (0/, 1/, ...)")
    
    for i in range(args.num_workers):
        sub_prefixes = "/goosefsx/91mst04h/airs/qxg/czy/airsoul_dev/benchmark/data/commonsense_170k.json"
            
        p = Process(target=worker_func, args=(
            i, sub_prefixes, args.output_dir, 
            args.seq_length, args.samples_per_file, args.files_per_subdir,
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("预处理完成！")

if __name__ == "__main__":
    main()