# Adagrad

## 1

```
+ python ./train.py --dataset My --num_classes 5 --data_root /home/data/ --data_dir 1945/ --save_root /project/train/ --save_dir models/ --loss CrossEntropy --lr 0.0005 --batch_size 10 --epoch 100 --image_width 384 --image_height 384 --optimizer Adagrad
```

显存占用峰值：14060
time per epoch：15

## 2

```
+ python ./train.py --dataset My --num_classes 5 --data_root /home/data/ --data_dir 1945/ --save_root /project/train/ --save_dir models/ --loss CrossEntropy --lr 0.0005 --batch_size 10 --epoch 100 --image_width 384 --image_height 384 --optimizer AdamW
```

这次增加了 `data.detach` 和 `label.detach`

显存占用峰值：12956
time per eoch：17

## 3

```
+ python ./train.py --dataset My --num_classes 5 --data_root /home/data/ --data_dir 1945/ --save_root /project/train/ --save_dir models/ --loss CrossEntropy --lr 0.0005 --batch_size 10 --epoch 100 --image_width 384 --image_height 384 --optimizer Adagrad
```

显存占用峰值：13058
time per eoch：17
