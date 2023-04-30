@echo off


@REM dataset properties
set dataset_name=My
set num_classes=5

@REM paths
set train_path=./UNet/train.py
set dataset_root=./
set dataset_dir=1945/
set save_root=./
set save_dir=models/unet/
set log_root=./
set log_dir=log/unet/train/

@REM training methods
set loss=CrossEntropy
set lr=0.0001
set batch_size=2
set epoch=100
set image_width=512
set image_height=512
set optimizer=RMSprop

@REM showing
set log_visual=true
set use_tqdm=true
set only_best=false


set argc=0
for %%x in (%*) do Set /A argc+=1
if %argc% gtr 0 (
    if "%1" == "Kitti" (
        @REM change dataset properties
        set dataset_name=%1
        set num_classes=34

        @REM change paths
        set dataset_root=./
        set dataset_dir=data_semantics/
    )
)


@echo on

python %train_path% ^
    --dataset %dataset_name% ^
    --num_classes %num_classes% ^
    --data_root %dataset_root% ^
    --data_dir %dataset_dir% ^
    --save_root %save_root% ^
    --save_dir %save_dir% ^
    --log_root %log_root% ^
    --log_dir %log_dir% ^
    --loss %loss% ^
    --lr %lr% ^
    --batch_size %batch_size% ^
    --epoch %epoch% ^
    --image_width %image_width% ^
    --image_height %image_height% ^
    --optimizer %optimizer% ^
    --log_visual %log_visual% ^
    --use_tqdm %use_tqdm% ^
    --only_best %only_best%
