#!/bin/bash -e
CUDA_VISIBLE_DEVICES=5 \
    python main.py \
    2>&1 | tee logs/demo.log > /dev/null &

# srun -p ai4science \
#     --job-name=demo \
#     --gres=gpu:1 \
#     --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=4 \