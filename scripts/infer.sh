#!/bin/zsh
CFG=$1
CKPT=$2

./scripts/init.sh
srun python infer.py --cfg $CFG --ckpt $CKPT