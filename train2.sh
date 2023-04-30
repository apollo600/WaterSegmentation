set -x

python /project/train/src_repo/train.py \
    --dataset Pascal \
    --num_classes 21 \
    --data_root "/project/train/src_repo/VOCdevkit/" \
    --data_dir "VOC2012" \
    --save_root "/project/train/" \
    --save_dir "models/pascal" \
    --save_period 5 \
    --log_root "/project/train" \
    --log_dir "tensorboard" \
    --lr "5e-4" \
    --image_width 512 \
    --image_height 512 \
    --optimizer "Adam" \
    --log_visual \
    --model Deeplab \
    --backbone Mobilenet \
    --pretrain_model_path "/project/train/src_repo/deeplab_mobilenetv2.pth" \
    --downsample_factor 16 \
    --init_epoch 0 \
    --freeze_epoch 40 \
    --freeze_batch_size 32 \
    --unfreeze_epoch 40 \
    --unfreeze_batch_size 16 \
    --min_lr 5e-6 \
    --momentum 0.9 \
    --weight_decay 0 \
    --lr_decay_type cos \
    --focal_loss \
    --dice_loss \
    --enable_tqdm
    # --class_weights 1 1 10 10 20 1
    # --loss "" \
    # --batch_size \
    # --epoch \
