python gen_lm_data.py \
  --version v2 \
  --vocab_size 16 \
  --embedding_size 16,32 \
  --hidden_size 16,32,64 \
  --n_gram 3 \
  --sequence_length 4096 \
  --file_size 500 \
  --file_number 8 \
  --workers 8 \
  --output_path /root/paddlejob/workspace/env_run/data_wg/wangfan/lm_data/lm_data_demo.simp.16.3
