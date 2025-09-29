python train_rl_hvac.py \
    --task_files_dir "./task_files" \
    --model_save_root "./rl_models" \
    --n_envs_per_task 4 \
    --total_steps 400 \
    --num_workers 3 \
    --algorithm "sac" \
    --reward_modes "0,1,2" \
    --device "cpu" \
    --verbose 0
