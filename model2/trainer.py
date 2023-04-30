from tqdm import tqdm
import numpy as np
import torch
import os
from utils import visual
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from model2.utils.callbacks import EvalCallback, LossHistory
from model2.loss import Dice_loss, CE_Loss, Focal_Loss
from model2.utils.utils_metrics import f_score
from model2.utils.utils import get_lr
import datetime
from functools import partial
import math


def Deeplab_trainer(train_loader: DataLoader, val_loader: DataLoader, train_model: nn.Module, args, optimizer, train_size, val_size, val_filelist, data_root):
                                                                                                
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#

    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(args.log_root, args.log_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, train_model, input_shape=(args.image_width, args.image_height))
    
    # 冻结主体部分
    for param in train_model.backbone.parameters():
        param.requires_grad = False

    batch_size = args.freeze_batch_size
    Init_lr = args.lr
    Min_lr = args.min_lr
    Init_Epoch = args.init_epoch
    Freeze_Epoch = args.freeze_epoch
    UnFreeze_Epoch = args.unfreeze_epoch
    Unfreeze_batch_size = args.unfreeze_batch_size
    UnFreeze_flag = False

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
        args.lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step = train_size // batch_size
    epoch_step_val = val_size // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # ----------------------#
    #   记录eval的map曲线
    # ----------------------#
    eval_callback = EvalCallback(train_model, (args.image_width, args.image_height), args.num_classes, val_filelist, data_root, 
                                log_dir, cuda=True, 
                                eval_flag=True, period=1)

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # ---------------------------------------#
        #   如果完成了冻结学习部分，则解冻，并设置参数
        # ---------------------------------------#
        if epoch >= Freeze_Epoch and not UnFreeze_flag:
            batch_size = Unfreeze_batch_size

            # -------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            # -------------------------------------------------------------------#
            nbs = 16
            lr_limit_max = 5e-4 if args.optimizer == 'Adam' else 1e-1
            lr_limit_min = 3e-4 if args.optimizer == 'Adam' else 5e-4
            if args.backbone == "Xception":
                lr_limit_max = 1e-4 if args.optimizer == 'Adam' else 1e-1
                lr_limit_min = 1e-4 if args.optimizer == 'Adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr,
                              lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr,
                             lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            # ---------------------------------------#
            #   获得学习率下降的公式
            # ---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(
                args.lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            # 进入解冻模式，并在此解冻
            for param in train_model.backbone.parameters():
                param.requires_grad = True

            epoch_step = train_size // batch_size
            epoch_step_val = val_size // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            UnFreeze_flag = True

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(train_model, loss_history, eval_callback, optimizer, epoch,
                      epoch_step, epoch_step_val, train_loader, val_loader, UnFreeze_Epoch, UnFreeze_flag, 
                      args.class_weights, args.num_classes, save_period=1, save_dir=os.path.join(args.save_root, args.save_dir), args=args)

    loss_history.writer.close()


def fit_one_epoch(train_model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                  train_loader, val_loader, Epoch, unfreeze_flag, cls_weights, num_classes, save_period, save_dir, args):
                                                                
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    state = 'Unfrozen' if unfreeze_flag else 'Frozen'

    print("+ Start Train")

    if args.enable_tqdm:
        pbar = tqdm(total=epoch_step,
                    desc=f'{state} Epoch {epoch + 1}/{Epoch} Training', postfix=dict, mininterval=1)
    else:
        pbar = None
    train_model.train()

    for iteration, batch in enumerate(train_loader):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        imgs = imgs.float()

        with torch.no_grad():
            weights = torch.from_numpy(args.class_weights)
            imgs = imgs.cuda()
            pngs = pngs.cuda()
            labels = labels.cuda()
            weights = weights.cuda()
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()

        #!!! 原代码这里有针对半精度浮点数到选项，这里直接简化掉了
        # ----------------------#
        #   前向传播
        # ----------------------#
        outputs = train_model(imgs)
        # ----------------------#
        #   计算损失
        # ----------------------#
        if args.focal_loss:
            loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

        if args.dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice

        with torch.no_grad():
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

        # ----------------------#
        #   反向传播
        # ----------------------#
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if pbar is not None:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    print('- Finish Train')
    print('+ Start Validate')
    if args.enable_tqdm:
        pbar = tqdm(total=epoch_step_val,
                    desc=f'{state} Epoch {epoch + 1}/{Epoch} Validating', postfix=dict, mininterval=1)
    else:
        pbar = None

    train_model.eval()
    for iteration, batch in enumerate(val_loader):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch  
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            imgs = imgs.float()
            imgs = imgs.cuda()
            pngs = pngs.float()
            pngs = pngs.cuda()
            labels = labels.cuda()
            weights = weights.cuda()

            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = train_model(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if args.focal_loss:
                loss = Focal_Loss(outputs, pngs, weights,
                                  num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if args.dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # -------------------------------#
            #   计算f_score
            # -------------------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if pbar is not None:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    update_best_model_flag, old_miou, new_miou = eval_callback.on_epoch_end(epoch + 1, train_model, args)
    # print(f'{state} Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print(f"===> In {state} Epoch: {epoch + 1} / {Epoch}")
    print('===> Total Loss: %.3f || Val Loss: %.3f ' %
          (total_loss / epoch_step, val_loss / epoch_step_val))

    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
    #     torch.save(train_model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' %
    #                (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
    #     print('Save best model to best_epoch_weights.pth')
    #     torch.save(train_model.state_dict(), os.path.join(
    #         save_dir, "best_epoch_weights.pth"))

    # torch.save(train_model.state_dict(), os.path.join(
    #     save_dir, "last_epoch_weights.pth"))
    # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            #             torch.save(train_model, os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' %
    #                (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    os.makedirs(save_dir, exist_ok=True)

    if (epoch + 1) % args.save_period == 0 or epoch + 1 == Epoch:
        torch.save(train_model, os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' %
                   (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    # if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
    #     print('Save best model to best_epoch_weights.pth')
    #     torch.save(train_model, os.path.join(
    #         save_dir, "best_epoch_weights.pth"))

    if update_best_model_flag and new_miou is not None:
        print(f"+ Update best model {old_miou:.4f} ==> {new_miou:.4f}")
        print('+ Save best model to best_epoch_weights.pth')
        torch.save(train_model, os.path.join(
            save_dir, "best_epoch_weights.pth"))
    else:
        print(f"- Best model stays at {old_miou:.4f}")

    torch.save(train_model, os.path.join(
        save_dir, "last_epoch_weights.pth"))

    os.system(f"ls -alh {save_dir}")


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.3, step_num=10):

    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters /
                                              float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) /
                               (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters,
                       warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):

    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
