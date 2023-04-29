#!/bin/bash

set -x

python UNet/inference.py \
    --model_path /project/train/models/unet/best.pt \
    --data_root /home/data/ \
    --data_dir 1945/ \
    --log_root /project/train/ \
    --log_dir log/unet/infer/ \
    --mask_root /project/ev_sdk/ \
    --mask_dir ./
