import os
import glob
import numpy as np
import argparse
from multiprocessing import Process
from megatron.core.datasets.indexed_dataset import IndexedDataset

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

    for prefix in file_prefixes:
        try:
            ds = IndexedDataset(prefix, multimodal=False)
        except Exception as e:
            print(f"[Worker {worker_id}] Error loading {prefix}: {e}")
            continue

        for i in range(len(ds)):
            try:
                doc = ds[i]
            except Exception:
                continue
            
            if len(token_stream) == 0:
                token_stream = list(doc)
            else:
                token_stream.extend(doc)
            if worker_id == 0:
                print(len(doc))
            # 当流中的数据足够切分时
            while len(token_stream) >= read_len:
                # 1. 取出完整的 seq_len + 1
                full_seq = token_stream[:read_len]
                token_stream = token_stream[seq_length:] # 滑动窗口或截断，这里按截断处理
                
                # 2. 切分 src 和 tgt
                # src: 0 到 -1
                # tgt: 1 到 结尾
                src_seq = full_seq[:-1]
                tgt_seq = full_seq[1:]
                
                # 3. 堆叠成一对 (2, seq_len)
                # 强制转为 int64
                pair = np.stack([src_seq, tgt_seq]).astype(np.int64)
                buffer.append(pair)
                
                # 4. 存盘逻辑
                if len(buffer) >= samples_per_file:
                    # save_path = os.path.join(output_dir, f"chunk_{worker_id}_{chunk_counter}.npy")
                    # data_array = np.stack(buffer)      
                    # np.save(save_path, data_array)
                    save_buffer_to_disk(buffer, chunk_counter)
                    # if worker_id == 0:
                    #     # 打印 shape 以确认: 应为 (N, 2, 1024)
                    #     print(f"[Worker {worker_id}] Saved {save_path} (Shape: {data_array.shape}, Dtype: {data_array.dtype})")
                    
                    buffer = []
                    chunk_counter += 1

        del ds
    # save chunk if it is not empty
    if len(buffer) > 0:
        # save_path = os.path.join(
        #     output_dir, f"chunk_{worker_id}_{chunk_counter}.npy"
        # )
        # data_array = np.stack(buffer)
        # np.save(save_path, data_array)
        save_buffer_to_disk(buffer, chunk_counter)
        # if worker_id == 0:
        #     print(f"[Worker {worker_id}] Saved chunk tail "
        #         f"(Shape: {data_array.shape})")
        if worker_id == 0:
            print(f"[Worker {worker_id}] Saved chunk tail.")
        buffer = []
        chunk_counter += 1

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
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--samples_per_file", type=int, default=2000)
    parser.add_argument("--files_per_subdir", type=int, default=100, help="每个子目录存放多少个文件")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prefixes = set()
    # 兼容性处理：有些环境 glob 出来的路径分隔符可能不一致
    for f in glob.glob(os.path.join(args.input_dir, "*.idx")):
        prefix = f[:-4]
        # 简单校验对应的 bin 文件是否存在
        if os.path.exists(prefix + ".bin"):
            prefixes.add(prefix)
    
    all_prefixes = sorted(list(prefixes))
    print(f"找到 {len(all_prefixes)} 个原始数据文件。")

    chunk_size = int(np.ceil(len(all_prefixes) / args.num_workers))
    processes = []

    print(f"开始处理... ")
    print(f"配置: Shape=(N, 2, {args.seq_length}), 每个文件样本数={args.samples_per_file}")
    print(f"分桶策略: 每 {args.files_per_subdir} 个文件存入一个子文件夹 (0/, 1/, ...)")
    
    for i in range(args.num_workers):
        sub_prefixes = all_prefixes[i * chunk_size : (i + 1) * chunk_size]
        if not sub_prefixes:
            continue
            
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