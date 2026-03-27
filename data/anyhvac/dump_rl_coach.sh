python train_rl_hvac_old.py \
    --task_files_dir "./task_files" \
    --model_save_root "./rl_models" \
    --n_envs_per_task 1 \
    --total_steps 20000 \
    --num_workers 16 \
    --algorithm "rppo" \
    --reward_modes "0,1,2" \
    --device "cpu" \
    --verbose 0
