#!/bin/bash


# dataset properties
dataset_name=My
num_classes=5

# paths
train_path=/project/train/src_repo/UNet/train.py
dataset_root=/home/data/
dataset_dir=1945/
save_root=/project/train/
save_dir=models/unet/
log_root=/project/train/
log_dir=log/unet/train/

# training methods
loss=CrossEntropy
lr=0.0005
batch_size=10
epoch=100
image_width=512
image_height=512
optimizer=RMSprop

# showing
log_visual=true
use_tqdm=false
only_best=true


if [[ $# -gt 0 ]] ; then
    if [[ $1 == "Kitti" ]] ; then
        # change dataset properties
        dataset_name=$1
        num_classes=34

        # change paths
        dataset_root=/project/train/src_repo
        dataset_dir=data_semantics/
    fi
fi


set -x

python $train_path \
    --dataset $dataset_name \
    --num_classes $num_classes \
    --data_root $dataset_root \
    --data_dir $dataset_dir \
    --save_root $save_root \
    --save_dir $save_dir \
    --log_root $log_root \
    --log_dir $log_dir \
    --loss $loss \
    --lr $lr \
    --batch_size $batch_size \
    --epoch $epoch \
    --image_width $image_width \
    --image_height $image_height \
    --optimizer $optimizer \
    --log_visual $log_visual \
    --use_tqdm $use_tqdm \
    --only_best $only_best
