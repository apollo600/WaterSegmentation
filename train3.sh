set -x

python /project/train/src_repo/train.py \
    --dataset My \
    --num_classes 6 \
    --data_root "/home/data/" \
    --data_dir "1945" \
    --save_root "/project/train/" \
    --save_dir "models/my" \
    --log_root "/project/train" \
    --log_dir "tensorboard" \
    --lr "5e-4" \
    --image_width 512 \
    --image_height 512 \
    --optimizer "Adam" \
    --log_visual \
    --model Deeplab \
    --backbone Mobilenet \
    --resume \
    --pretrain_model_path "/project/train/models/best_epoch_weights.pth" \
    --downsample_factor 16 \
    --init_epoch 35 \
    --freeze_epoch 25 \
    --freeze_batch_size 8 \
    --unfreeze_epoch 50 \
    --unfreeze_batch_size 8 \
    --min_lr 5e-6 \
    --momentum 0.9 \
    --weight_decay 0 \
    --lr_decay_type cos \
    --focal_loss \
    --dice_loss \
    --class_weights 1 2.3697 787.4016 628.9308 270.2703 1.7953
    # --loss "" \
    # --batch_size \
    # --epoch \
