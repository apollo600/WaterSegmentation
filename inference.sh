#!/bin/bash

set -x

python inference.py \
    --model_path /project/train/models/2023-04-28-04-35-49_epoch-100_lr-0.0005_loss-CrossEntropy_optim-Adagrad_best_acc-0.5457.pt \
    --data_root /home/data/ \
    --data_dir 1945/ \
    --log_root /project/train/ \
    --log_dir log/infer/ \
    --mask_root /project/ev_sdk/ \
    --mask_dir ./
