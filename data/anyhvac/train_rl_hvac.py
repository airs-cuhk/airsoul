import os
import pickle
import shutil
import multiprocessing
import concurrent
import time
import torch
import argparse
from pathlib import Path
from datetime import datetime
from glob import glob
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiffAction
from rl_trainer_hvac import HVACRLTrainer
import gymnasium as gym
from numpy import random as rnd

def train_task(task_file, model_save_root, n_envs, total_steps, algorithm, reward_mode, n_ckpts, post_training, device, verbose):
    try:
        torch.set_num_threads(n_envs)
        with open(task_file, "rb") as f:
            task = pickle.load(f)
        
        task_name = Path(task_file).stem
        task_save_dir = Path(model_save_root) / task_name
        task_save_dir.mkdir(parents=True, exist_ok=True)
        source_path = Path(task_file).resolve()
        target_path = (task_save_dir / f"{task_name}.pkl").resolve()
        if source_path != target_path:
            shutil.copy(source_path, target_path)

        def make_env():
            with open(task_file, "rb") as f:
                task = pickle.load(f)
                
            env = HVACEnvDiffAction(reward_mode=reward_mode)
            env.set_task(task,discretize_rl_action_space=True,add_action_cost=True,too_cold_limit=False)
            env.set_random_start_t(True)
            overheat_no_reset = rnd.uniform(0.0, 1.0) > 0.5
            env.set_overheat_no_termiated_training_only(overheat_no_reset)
            return env
        
        model_save_path = task_save_dir / f"{algorithm}_reward_mode_{reward_mode}.zip"
        log_file_path = task_save_dir / f"{algorithm}_reward_mode_{reward_mode}.log"
        
        ckpts_save_path = task_save_dir / f"{algorithm}_reward_mode_{reward_mode}_ckpts"
        ckpts_save_path.mkdir(parents=True, exist_ok=True)

        trainer = HVACRLTrainer(
            env_maker=make_env,
            n_envs=n_envs,
            vec_env_type="subproc",
            algorithm=algorithm,
            stage_steps=100,
            vec_env_args={"start_method": "spawn"},
            verbose=verbose,
            device=device,
            log_path=log_file_path,
            n_ckpts=n_ckpts,
            ckpts_save_path=ckpts_save_path
        )

        if post_training and model_save_path.exists():
            backup_date = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_backup_path = task_save_dir / f"{algorithm}_reward_mode_{reward_mode}_backup_{backup_date}.zip"
            same_dir = model_save_path.parent.resolve() == model_backup_path.parent.resolve()
            if same_dir:
                shutil.move(str(model_save_path), str(model_backup_path))
            trainer.load_model(model_backup_path)
        
        trainer.train(total_steps=total_steps)
        
        trainer.save_model(model_save_path)
        print(f"Model saved: {model_save_path}")
        return True, task_file, None
    except Exception as e:
        import traceback
        import sys
        
        print(f"\n{'='*80}")
        print(f"❌ Task FAILED: {task_file}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"{'='*80}")
        
        # 打印完整的堆栈跟踪
        print("Full traceback:")
        traceback.print_exc(file=sys.stdout)
        
        print(f"{'='*80}\n")
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
    parser.add_argument('--n_ckpts', type=int, default=30,
                        help='Number of checkpoints to save per coach')
    parser.add_argument('--post_training', action='store_true',
                    help='Enable post-training evaluation or processing after training completes')

    
    args = parser.parse_args()
    
    reward_modes = [int(mode.strip()) for mode in args.reward_modes.split(",")]
    
    if args.post_training:
        task_files = sorted(glob(os.path.join(args.task_files_dir, "*", "*.pkl")))
    else:
        task_files = sorted(glob(os.path.join(args.task_files_dir, "*.pkl")))
    
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
            for reward_mode in reward_modes:
                future = executor.submit(
                    train_task, 
                    task_file, 
                    args.model_save_root, 
                    args.n_envs_per_task, 
                    args.total_steps, 
                    args.algorithm,
                    reward_mode, 
                    args.n_ckpts,
                    args.post_training,
                    args.device, 
                    args.verbose
                )
                futures[future] = (task_file, reward_mode)
        
        failed = 0
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            task_file, reward_mode = futures[future]
            try:
                success, _, error_msg = future.result()
                if success:
                    print(f"✅ {task_file} (mode={reward_mode}) completed")
                else:
                    print(f"❌ {task_file} (mode={reward_mode}) failed: {error_msg}")
            except Exception as e:
                failed += 1
                print(f"❌ Task failed: {task_file} - {str(e)} ({completed + failed}/{len(task_files)})")
    
    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.2f} seconds")
    print(f"Success: {completed}, Failed: {failed}, Total: {len(task_files)}")