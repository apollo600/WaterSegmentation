#!/bin/bash

res=My
if [ $# -gt 0 ] ; then
    res=$1
fi

python ./train.py \
    --data_root /project/train/src_repo \
    --save_root /project/train/ \
    --dataset $res \
    --loss CrossEntropy \
    --lr 0.0005 \
    --batch_size 2 \
    --epoch 100 \
    --save_dir models \
    --num_classes 5