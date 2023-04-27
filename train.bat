@echo off


@REM dataset properties
set dataset_name=My
set num_classes=5

@REM paths
set dataset_root=./
set dataset_dir=1945/
set save_root=./
set save_dir=models/

@REM training methods
set loss=CrossEntropy
set lr=0.0005
set batch_size=2
set epoch=100
set image_width=640
set image_height=640
set optimizer=Adagrad


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

python ./train.py ^
    --dataset %dataset_name% ^
    --num_classes %num_classes% ^
    --data_root %dataset_root% ^
    --data_dir %dataset_dir% ^
    --save_root %save_root% ^
    --save_dir %save_dir% ^
    --loss %loss% ^
    --lr %lr% ^
    --batch_size %batch_size% ^
    --epoch %epoch% ^
    --image_width %image_width% ^
    --image_height %image_height% ^
    --optimizer %optimizer%
