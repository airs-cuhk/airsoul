python gen_hvac_record.py \
    --input_folder "./rl_models" \
    --output_folder "./data" \
    --rl_mode_type "sac" \
    --max_steps 50 \
    --num_processes 16 \
    --self_regression True
