from tqdm import tqdm
import numpy as np
import torch
import os
from utils import visual
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


def Deeplab_trainer(train_loader: DataLoader, val_loader: DataLoader, train_model: nn.Module, args, optimizer, train_size, val_size):
                                                                                
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#

    # 冻结一定部分
    for param in model.backbone.parameters():
        param.requires_grad = False

    batch_size = args.freeze_batch_size
    Init_lr = args.lr
    Min_lr = args.min_lr

    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 16
    lr_limit_max = 5e-4 if args.optimizer == 'adam' else 1e-1
    lr_limit_min = 3e-4 if args.optimizer == 'adam' else 5e-4
    if args.backbone == "Xception":
        lr_limit_max = 1e-4 if args.optimizer == 'adam' else 1e-1
        lr_limit_min = 1e-4 if args.optimizer == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr,
                      lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr,
                     lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    
    #---------------------------------------#
    #   判断每一个世代的长度
    #---------------------------------------#
    epoch_step      = train_size // batch_size
    epoch_step_val  = val_size // batch_size
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
    
    #----------------------#
    #   记录eval的map曲线
    #----------------------#
    if local_rank == 0:
        eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                        eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback   = None


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
                        # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

