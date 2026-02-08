#!/bin/zsh
CFG=$1
NODE=$2

./scripts/init.sh

if [ $# -eq 0]; then
    echo "No arguments provided. Usage: ./train.sh <config_file> <node_name (optional)>"
    exit 1
fi

if [ -z "$NODE" ]; then
    srun python train.py --cfg $CFG
    exit 0
fi

srun -w $NODE python train.py --cfg $CFG