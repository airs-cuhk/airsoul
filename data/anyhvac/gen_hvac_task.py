import os
import numpy as np
import random
import pickle 
import queue
import time
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiscreteAction, HVACEnvDiffAction
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID


def validate_env_with_pid(env, task, max_steps=20000):
    """
    Verify whether the environment can run to max_steps with PID policy.
    
    Return:
        bool: pass/fail
    """
    env.set_task(task)
    n_sensors = len(env.sensors)
    n_coolers = len(env.coolers)

    obs, info = env.reset()
    agent = HVACSolverGTPID(env)
    terminated, truncated = False, False
    step_count = 0
    
    while (not terminated) and (not truncated) and (step_count < max_steps):
        action = 1 - agent.policy(obs[:n_sensors])[n_coolers:]
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if step_count % 1000 == 0:
            print(f"PID test progress: {100*step_count/max_steps} %")
    
    success = (step_count >= max_steps and not terminated and not truncated)
    print(f"{'✓ PID test pass' if success else '✗ PID test fail'} (Total {step_count} steps.)")
    return success

def validate_env_with_max(env, task, max_steps=20000):
    """
    Verify whether the environment can run to max_steps with Max power policy.
    
    Return:
        bool: pass/fail
    """
    env.set_task(task)
    obs, info = env.reset()
    terminated, truncated = False, False
    step_count = 0
    
    while (not terminated) and (not truncated) and (step_count < max_steps):
        action = env.sample_action(mode="max")
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if step_count % 1000 == 0:
            print(f"Max power test progress: {100*step_count/max_steps} %")
    
    success = (step_count >= max_steps and not terminated and not truncated)
    print(f"{'✓ Max power test pass' if success else '✗ Max power test fail'} (Total {step_count} steps.)")
    return success

def sample_env_with_validation(
    save_path="./save_path",
    max_steps=20000,
    num_valid_envs=5,
    use_diff_action=True,
    verbose=True      
):
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_count = 0
    attempts = 0
    max_attempts = 20 * num_valid_envs
    task_config_path = os.path.join(save_path , "/hvac_task_{}.pkl")
    while valid_count < num_valid_envs and attempts < max_attempts:
        attempts += 1
        current_config_path = task_config_path.format(valid_count)
        task = HVACTaskSampler()
        if use_diff_action:
            env = HVACEnvDiffAction(verbose=verbose)
        else:
            env = HVACEnvDiscreteAction(verbose=verbose)
        pid_pass = validate_env_with_pid(env, task, max_steps)
        max_power_pass = validate_env_with_max(env, task, max_steps)

        if pid_pass and max_power_pass:
            valid_count += 1
            with open(current_config_path, "wb") as f:
                pickle.dump(task, f)

    print(f"\n========================================")
    print(f"Sample {valid_count} valid tasks.")

def sample_env_with_validation_parallel(
    save_path="./hvac_tasks",
    max_steps=20000,
    num_valid_envs=5,
    use_diff_action=True,
    verbose=True,
    max_workers=None
):
    """
    并行生成并验证 HVAC 环境配置
    
    参数:
        save_path (str): 保存有效配置的目录
        max_steps (int): 环境验证的最大步数
        num_valid_envs (int): 需要生成的有效环境数量
        use_diff_action (bool): 是否使用差分动作环境
        verbose (bool): 是否显示详细输出
        max_workers (int): 最大并行工作进程数
        
    返回:
        int: 成功生成的有效环境数量
    """
    # 创建保存目录
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置并行工作进程数
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"Starting HVAC environment sampling with {max_workers} workers")
    print(f"Target: {num_valid_envs} valid environments")
    print(f"Using {'differential' if use_diff_action else 'discrete'} action environment")
    
    # 创建管理器
    manager = multiprocessing.Manager()
    
    # 创建任务队列和结果队列
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    lock = manager.Lock()

    # 创建共享计数器
    next_task_id = manager.Value('i', 0)  # 下一个可用的任务ID
    valid_count = manager.Value('i', 0)   # 当前有效的环境数量
    
    # 初始化任务队列 - 放入初始任务
    for _ in range(max_workers * 2):  # 初始放入足够任务保持工作进程忙碌
        with lock:  # 使用锁保护共享变量
            task_id = next_task_id.value
            next_task_id.value += 1
        task_queue.put(task_id)
    
    # 创建并启动工作进程
    processes = []
    for i in range(max_workers):
        p = multiprocessing.Process(
            target=worker_process,
            args=(task_queue, result_queue, save_dir, max_steps, 
                use_diff_action, verbose, next_task_id, valid_count, lock)  # 添加lock参数
        )
        p.daemon = True
        p.start()
        processes.append(p)
    
    # 创建进度条
    progress = tqdm(total=num_valid_envs, desc="Generating valid environments")
    
    # 主进程监控结果
    try:
        while valid_count.value < num_valid_envs:
            # 检查是否有新结果
            if not result_queue.empty():
                task_id, success = result_queue.get()
                if success:
                    progress.update(1)
                    if verbose:
                        print(f"✓ Task {task_id} validated and saved")
                else:
                    if verbose:
                        print(f"✗ Task {task_id} validation failed")
            
            # 避免忙等待
            time.sleep(0.1)
            
            # 检查工作进程是否全部退出（异常情况）
            if all(not p.is_alive() for p in processes):
                print("All worker processes exited unexpectedly")
                break
    
    except KeyboardInterrupt:
        print("\nSampling interrupted by user")
    
    finally:
        # 通知工作进程退出
        for _ in range(max_workers):
            task_queue.put(None)  # 发送退出信号
        
        # 等待工作进程结束
        for p in processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        
        progress.close()
    
    # 打印最终结果
    final_count = valid_count.value
    print(f"\nSuccessfully generated {final_count}/{num_valid_envs} valid HVAC environments")
    return final_count

def worker_process(
    task_queue, result_queue, save_dir, max_steps, 
    use_diff_action, verbose, next_task_id, valid_count, lock
):
    """工作进程：持续尝试生成有效环境，直到任务队列为空"""
    pid = os.getpid()
    seed = int(time.time() * 1000) % (2**32 - 1) + pid
    random.seed(seed)
    np.random.seed(seed)
    if verbose:
        print(f"Worker {pid} started with seed {seed}")
    
    while True:
        try:
            # 获取新任务ID，设置超时避免永久阻塞
            task_id = task_queue.get(timeout=30)
            
            # 检查退出信号
            if task_id is None:
                if verbose:
                    print(f"Worker {pid}: Received exit signal")
                break
            
            if verbose:
                print(f"Worker {pid} processing task {task_id}")
            
            # 尝试生成环境
            task = HVACTaskSampler()
            if use_diff_action:
                env = HVACEnvDiffAction(verbose=verbose)
            else:
                env = HVACEnvDiscreteAction(verbose=verbose)
            
            # 验证环境
            pid_pass = validate_env_with_pid(env, task, max_steps)
            # max_power_pass = validate_env_with_max(env, task, max_steps)
            
            if pid_pass: #and max_power_pass:
                # 保存有效配置
                task_config_path = save_dir / f"hvac_task_{task_id}.pkl"
                with open(task_config_path, "wb") as f:
                    pickle.dump(task, f)
                
                # 通知主进程成功
                result_queue.put((task_id, True))
                
                # 更新有效计数器
                with lock:
                    valid_count.value += 1
            else:
                # 通知主进程失败
                result_queue.put((task_id, False))
                
                # 生成新的任务ID用于重试
                with lock:
                    new_task_id = next_task_id.value
                    next_task_id.value += 1
                
                # 将新任务放入队列
                task_queue.put(new_task_id)
                if verbose:
                    print(f"Worker {pid}: Enqueued new task {new_task_id}")
        
        except queue.Empty:
            if verbose:
                print(f"Worker {pid}: Queue empty, exiting")
            break
        
        except Exception as e:
            print(f"Worker {pid}: Error processing task - {str(e)}")
            # 生成新的任务ID用于重试
            with lock:
                new_task_id = next_task_id.value
                next_task_id.value += 1
            task_queue.put(new_task_id)
    
    if verbose:
        print(f"Worker {pid} exiting")

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='HVAC Environment Sampler')
    parser.add_argument('--save_path', type=str, default="./hvac_tasks", 
                        help='Directory to save task configurations')
    parser.add_argument('--max_steps', type=int, default=20000,
                        help='Maximum steps for environment validation')
    parser.add_argument('--num_envs', type=int, default=10,
                        help='Number of valid environments to sample')
    parser.add_argument('--use_diff_action', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), 
                        default=False, help='Use differential action environment (true/false)')
    parser.add_argument('--verbose', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), 
                        default=False, help='Enable verbose output (true/false)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    valid_count = sample_env_with_validation_parallel(
        save_path=args.save_path,
        max_steps=args.max_steps,
        num_valid_envs=args.num_envs,
        use_diff_action=args.use_diff_action,
        verbose=args.verbose,
        max_workers=args.workers
    )
    
    duration = time.time() - start_time
    print(f"\nTotal time: {duration:.2f} seconds")
    print(f"Average time per task: {duration/max(1, valid_count):.2f} seconds")
    print(f"Successfully generated {valid_count} valid HVAC environments")
