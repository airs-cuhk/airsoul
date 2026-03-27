export PYTHONPATH="$PYTHONPATH:/home/shaopt/code/develop_branch/airsoul/airsoul"
export PYTHONPATH="$PYTHONPATH:/home/shaopt/code/develop_branch/airsoul"
python multiagent_dataset.py \
    --load_dir /data/ \
    --save_dir /seq_data \
    --time_step 20160 \
    --max_obs_num 10 \
    --max_agent_num 10 \
    --prompt_num 4 \
    --temperature_value_num 320 \
    --temperature_resolution 0.1 \
    --policy_value_num 12 \
    --policy_resolution 0.5 \
    --vocab_size 366 \
    --num_workers 1
