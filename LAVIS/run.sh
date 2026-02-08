NODE=$1
GPU_NUM=$2
CONFIG_PATH=$3

export HF_HOME=~/.cache/huggingface

srun -w $NODE --gres=gpu:$GPU_NUM -N 1 python -m torch.distributed.run --nproc_per_node=$GPU_NUM train.py --cfg-path $CONFIG_PATH