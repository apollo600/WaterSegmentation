python DeepLab/inference.py ^
    --model_path ./models/deeplab/my/best_epoch_weights.pth ^
    --data_root ./ ^
    --data_dir 1945/ ^
    --log_root ./ ^
    --log_dir log/deeplab/infer/ ^
    --mask_root ./ ^
    --mask_dir ./
