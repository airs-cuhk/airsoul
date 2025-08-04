if [[ $# -lt 1 ]]; then
	echo "Usage: $0 output_directory task_number"
	exit 1
fi
echo "Output to $1 with $2 tasks"


# Sample Tasks
python gen_metalang.py \
  --sample_type tasks \
  --samples $2 \
  --file_number 1 \
  --n_gram 2 3 4 5 6 \
  --output_path $1

sleep 5
task_file=$1.pkl
mkdir $1_dir

python gen_metalang.py \
  --sample_type sequences \
  --task_file $task_file \
  --sequence_length 4096 \
  --samples 500 \
  --file_number 8 \
  --workers 8 \
  --output_path $1_dir
