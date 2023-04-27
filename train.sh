#!/bin/bash


# paths
dataset_root=/home/data/
save_root=/project/train/
dataset_dir=1945/
save_dir=models/

# dataset properties
dataset_name=My
num_classes=5

# training methods
loss=CrossEntropy
lr=0.0005
batch_size=2
epoch=100


if [[ $# -gt 0 ]] ; then
    if [[ $1 == "Kitti" ]] ; then
        # change paths
        dataset_root=/project/train/src_repo
        dataset_dir=data_semantics/

        # change dataset properties
        dataset_name=$1
        num_classes=34
    fi
fi


set -x

python ./train.py \
    --data_root $dataset_root \
    --save_root $save_root \
    --data_dir $dataset_dir \
    --save_dir $save_dir \
    --dataset $dataset_name \
    --num_classes $num_classes \
    --loss $loss \
    --lr $lr \
    --batch_size $batch_size \
    --epoch $epoch