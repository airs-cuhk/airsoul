python gen_hvac_record.py \
    --input_folders "./rl_models/test/" \
    --output_folder "./rl_models/test/hvac_task_1185/data/" \
    --rl_mode_type "rppo" \
    --use_rrd False \
    --reference_action_deterministic True \
    --max_steps 2000 \
    --use_use_rrd False \ 
    --num_records 100 \
    --num_processes 3
