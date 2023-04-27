@echo off

set argc=0
for %%x in (%*) do Set /A argc+=1

set res=My
if %argc% gtr 0 (
    set res=%1
)

@echo on

python ./train.py ^
    --data_root . ^
    --save_root . ^
    --dataset %res% ^
    --loss CrossEntropy ^
    --lr 0.0005 ^
    --batch_size 2 ^
    --epoch 100 ^
    --save_dir models ^
    --num_classes 5