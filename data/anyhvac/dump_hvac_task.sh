export PYTHONUNBUFFERED=1
python -u gen_hvac_task.py \
    --save_path ./task_files/0326_test \
    --num_envs 10 \
    --max_steps 2000 \
    --use_diff_action False\
    --mode constant_conservative\
    --verbose False\
    --workers 16
