set -x

python /project/train/src_repo/train.py \
    --dataset Pascal \
    --num_classes 21 \
    --data_root "/project/train/src_repo/VOCdevkit/" \
    --data_dir "VOC2012" \
    --save_root "/project/train/" \
    --save_dir "models/pascal" \
    --log_root "/project/train" \
    --log_dir "log" \
    --lr "5e-4" \
    --image_width 512 \
    --image_height 512 \
    --optimizer "Adam" \
    --log_visual \
    --model Deeplab \
    --backbone Mobilenet \
    --pretrain_model_path "/project/train/models/pascal/" \
    --downsample_factor 16 \
    --init_epoch 14 \
    --freeze_epoch 25 \
    --freeze_batch_size 16 \
    --unfreeze_epoch 50 \
    --unfreeze_batch_size 8 \
    --min_lr 5e-6 \
    --momentum 0.9 \
    --weight_decay 0 \
    --lr_decay_type cos \
    --focal_loss \
    # --class_weights 1 1 10 10 20 1
    # --loss "" \
    # --batch_size \
    # --epoch \
