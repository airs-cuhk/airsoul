import os
import pickle
import shutil
import multiprocessing
import concurrent
import time
import argparse
from pathlib import Path
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiffAction
from rl_trainer_hvac import HVACRLTrainer
import gymnasium as gym

def train_task(task_file, model_save_root, n_envs, total_steps, algorithm, reward_modes, device, verbose):
    try:
        with open(task_file, "rb") as f:
            task = pickle.load(f)
        
        task_name = Path(task_file).stem
        task_save_dir = Path(model_save_root) / task_name
        task_save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(task_file, task_save_dir / f"{task_name}.pkl")
        
        for reward_mode in reward_modes:

            def make_env():
                env = HVACEnvDiffAction(reward_mode=reward_mode)
                env.set_task(task)
                env.set_random_start_t(True)
                return env
            
            model_save_path = task_save_dir / f"{algorithm}_reward_mode_{reward_mode}.zip"
            log_file_path = task_save_dir / f"{algorithm}_reward_mode_{reward_mode}.log"
            
            trainer = HVACRLTrainer(
                env_maker=make_env,
                n_envs=n_envs,
                vec_env_type="subproc",
                algorithm=algorithm,
                stage_steps=100,
                vec_env_args={"start_method": "spawn"},
                verbose=verbose,
                device=device,
                log_path=log_file_path
            )
            
            trainer.train(total_steps=total_steps)
            
            trainer.save_model(model_save_path)
            print(f"Model saved: {model_save_path}")
        return True, task_file, None
    except Exception as e:
        return False, task_file, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HVAC RL Training')
    parser.add_argument('--task_files_dir', type=str, default="./task_files",
                        help='Directory containing task files')
    parser.add_argument('--model_save_root', type=str, default="./rl_models",
                        help='Root directory for saving models')
    parser.add_argument('--n_envs_per_task', type=int, default=64,
                        help='Number of parallel environments per task')
    parser.add_argument('--total_steps', type=int, default=2000000,
                        help='Total training steps per reward mode')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers for task processing')
    parser.add_argument('--algorithm', type=str, default="sac",
                        choices=["sac", "rppo", "ppo"],
                        help='RL algorithm to use (sac, rppo, ppo)')
    parser.add_argument('--reward_modes', type=str, default="0,1,2",
                        help='Comma-separated list of reward modes to train')
    parser.add_argument('--device', type=str, default="gpu",
                        choices=["gpu", "cpu"],
                        help='Device to use for training (gpu or cpu)')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity level (0 for minimal, 1 for more)')
    
    args = parser.parse_args()
    
    reward_modes = [int(mode.strip()) for mode in args.reward_modes.split(",")]
    
    task_files = sorted([os.path.join(args.task_files_dir, f) 
                  for f in os.listdir(args.task_files_dir) 
                  if f.endswith(".pkl")])
    
    print(f"Found {len(task_files)} task files")
    print(f"Training configuration:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Reward modes: {reward_modes}")
    print(f"  Device: {args.device}")
    print(f"  Environments per task: {args.n_envs_per_task}")
    print(f"  Total steps per mode: {args.total_steps}")
    print(f"  Parallel workers: {args.num_workers}")
    
    ctx = multiprocessing.get_context('spawn')
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=ctx
    ) as executor:
        futures = {}
        
        # 提交所有任务
        for task_file in task_files:
            future = executor.submit(
                train_task, 
                task_file, 
                args.model_save_root, 
                args.n_envs_per_task, 
                args.total_steps, 
                args.algorithm, 
                reward_modes, 
                args.device, 
                args.verbose
            )
            futures[future] = task_file
        
        failed = 0
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            task_file = futures[future]
            try:
                success, task_file, error_msg = future.result()
                if success:
                    completed += 1
                    print(f"✅ Task completed: {task_file} ({completed}/{len(task_files)})")
                else:
                    failed += 1
                    print(f"❌ Task failed: {task_file} ({completed + failed}/{len(task_files)}), error msg: {error_msg}")
            except Exception as e:
                failed += 1
                print(f"❌ Task failed: {task_file} - {str(e)} ({completed + failed}/{len(task_files)})")
    
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f} seconds")
    print(f"Success: {completed}, Failed: {failed}, Total: {len(task_files)}")
