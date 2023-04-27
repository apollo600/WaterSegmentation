@echo off


@REM paths
set dataset_root=./
set save_root=./
set dataset_dir=1945/
set save_dir=models/

@REM dataset properties
set dataset_name=My
set num_classes=5

@REM training methods
set loss=CrossEntropy
set lr=0.0005
set batch_size=2
set epoch=100


set argc=0
for %%x in (%*) do Set /A argc+=1
if %argc% gtr 0 (
    if "%1" == "Kitti" (
        @REM change paths
        set dataset_root=./
        set dataset_dir=data_semantics/

        @REM change dataset properties
        set dataset_name=%1
        set num_classes=34
    )
)


@echo on

python ./train.py ^
    --data_root %dataset_root% ^
    --data_dir %dataset_dir% ^
    --save_root %save_root% ^
    --save_dir %save_dir% ^
    --dataset %dataset_name% ^
    --num_classes %num_classes% ^
    --loss %loss% ^
    --lr %lr% ^
    --batch_size %batch_size% ^
    --epoch %epoch%