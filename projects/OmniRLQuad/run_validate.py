import os
import glob
import subprocess
import sys
import time
import logging
import torch
import gc
import psutil
from ruamel.yaml import YAML
import re

def modify_config(config_path, new_model_path, new_log_dir, output_config_path):
    # 使用 ruamel.yaml 保留原始格式
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    # 读取原始配置文件
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    
    # 只修改指定的字段
    config['load_model_path'] = new_model_path
    # config['log_config']['tensorboard_log'] = new_log_dir

# 修改 log_config 中的 tensorboard_log
    if 'log_config' not in config:
        logging.warning(f"配置文件 {config_path} 的 log_config 不存在，已初始化为空字典")
        config['log_config'] = {}
    config['log_config']['tensorboard_log'] = new_log_dir
    
    # 修改 test_config 中 datasets 的 log_dir（仅在存在且格式正确时）
    if 'test_config' in config and 'datasets' in config['test_config'] and isinstance(config['test_config']['datasets'], list):
        for dataset in config['test_config']['datasets']:
            if 'log_dir' in dataset:
                dataset['log_dir'] = new_log_dir
                logging.info(f"更新数据集 {dataset.get('name', '未知')} 的 log_dir 为 {new_log_dir}")
            else:
                logging.info(f"数据集 {dataset.get('name', '未知')} 缺少 log_dir 字段，已添加")
                dataset['log_dir'] = new_log_dir
    
    # 保存修改后的配置文件，保留原始格式
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f)

def main():
    # 配置
    checkpoint_dir = "/mnt/data/cassel/omnirl/16-task-ckpt/fast-ckpt"
    config_path = "config_test.yaml"
    base_log_dir = "/mnt/data/cassel/omnirl/checkpoint_log/debug"  # TensorBoard 基础目录
    
    # 创建基础日志目录
    os.makedirs(base_log_dir, exist_ok=True)
    
    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # 查找所有 .pth 文件
    pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not pth_files:
        print(f"在 {checkpoint_dir} 中未找到 .pth 文件")
        return
    
    # 依次处理每个 .pth 文件
    for pth_file in pth_files:
        print(f"正在处理检查点: {pth_file}")
        
        # 获取检查点文件名（不含扩展名）
        base_name = os.path.basename(pth_file).replace('.pth', '')

        # 使用正则表达式提取数字部分
        match = re.search(r'(\D*)(\d+)$', base_name)
        if match:
            prefix, number = match.groups()
            # 将数字补0到6位
            formatted_number = f"{int(number):06d}"
            run_name = f"{prefix}{formatted_number}"
        else:
            # 如果没有匹配到数字，直接使用原始名称
            run_name = base_name
            logging.warning(f"文件名 {base_name} 未匹配到数字部分，使用原始名称")

        # 为本次运行创建唯一的配置文件
        temp_config = f"temp_config_{run_name}.yaml"

        # 设置唯一的 TensorBoard 日志目录
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)

        # 修改配置文件，保留原始格式
        modify_config(config_path, pth_file, log_dir, temp_config)
        
        # 运行 validate.py
        try:
            cmd = [
                sys.executable,
                "validate.py",
                temp_config  # 假设 Runner 读取 sys.argv[1] 作为配置文件
            ]
            subprocess.run(cmd, check=True)
            print(f"完成 {pth_file} 的验证")
        except subprocess.CalledProcessError as e:
            print(f"运行 {pth_file} 验证时出错: {e}")
        finally:
            # 清理临时配置文件
            if os.path.exists(temp_config):
                os.remove(temp_config)

            # 清理 GPU 内存和垃圾
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)  # 短暂延迟，确保资源释放
            logging.info(f"完成资源清理: {pth_file}")

if __name__ == "__main__":
    main()
