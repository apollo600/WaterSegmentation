set -x

python train.py \
    --dataset Pascal \
    --num_classes 21 \
    --data_root "/project" \
    --data_dir "1945" \
    --save_root "/project/train/" \
    --save_dir "models" \
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
    --freeze_epoch 25 \
    --freeze_batch_size 8 \
    --unfreeze_epoch 50 \
    --unfreeze_batch_size 4 \
    --min_lr 5e-6 \
    --momentum 0.9 \
    --weight_decay 0 \
    --lr_decay_type cos \
    --dice_loss \
    --focal_loss \
    --class_weights 1 1 10 10 20 1
    # --loss "" \
    # --batch_size \
    # --epoch \
