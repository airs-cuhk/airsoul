export PYTHONPATH="$PYTHONPATH:/home/shaopt/code/develop_branch/airsoul/airsoul"
export PYTHONPATH="$PYTHONPATH:/home/shaopt/code/develop_branch/airsoul"
python multiagent_dataset.py \
    --load_dir /pfs/pfs-r36Cge/shaopt/data/HVAC/env1/ \
    --save_dir /pfs/pfs-r36Cge/shaopt/data/HVAC/env1_seq/ \
    --time_step 20160 \
    --max_obs_num 10 \
    --max_agent_num 10 \
    --prompt_num 3 \
    --value_num 300 \
    --resolution 0.1 \
    --vocab_size 331 \
    --num_workers 1
