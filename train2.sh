python train.py \
--dataset My \
--num_classes 6 \
--data_root "/home/data/" \
--data_dir "1945" \
--save_root "/project/train/" \
--save_dir "models" \
--log_root "/project/train" \
--log_dir "log" \
# --loss "" \
--lr "5e-4" \
# --batch_size \
# --epoch \
--image_width 512 \
--image_height 512 \
--optimizer "Adam" \
--log_visual \
--model Deeplab \
--backbone Mobilenet \
--pretrain_model_path "/" \

