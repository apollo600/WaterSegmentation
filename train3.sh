set -x

python train.py \
    --dataset My \
    --num_classes 6 \
    --data_root "/home/data/" \
    --data_dir "1945" \
    --save_root "/project/train/" \
    --save_dir "models/deeplab" \
    --log_root "/project/train" \
    --log_dir "log" \
    --lr "5e-4" \
    --image_width 512 \
    --image_height 512 \
    --optimizer "Adam" \
    --log_visual \
    --model Deeplab \
    --backbone Mobilenet \
    --pretrain_model_path "/project/train/models/deeplab_mobilenetv2.pth" \
    --downsample_factor 16 \
    --init_epoch 0 \
    --freeze_epoch 7 \
    --freeze_batch_size 8 \
    --unfreeze_epoch 20 \
    --unfreeze_batch_size 8 \
    --min_lr 5e-6 \
    --momentum 0.9 \
    --weight_decay 0 \
    --lr_decay_type cos \
    --focal_loss \
    --class_weights 1 1 50 50 100 1
    # --loss "" \
    # --batch_size \
    # --epoch \