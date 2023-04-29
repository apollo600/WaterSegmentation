#!/bin/bash

set -x

python /project/train/src_repo/DeepLab/inference.py \
    --model_path /project/train/models/deeplab/my/best_epoch_weights.pth \
    --data_root /home/data/ \
    --data_dir 1945/ \
    --log_root /project/train/ \
    --log_dir log/deeplab/infer/ \
    --mask_root /project/ev_sdk/ \
    --mask_dir ./
