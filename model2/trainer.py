from tqdm import tqdm
import numpy as np
import torch
import os
from utils import visual
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


def Deeplab_trainer(train_loader: DataLoader, val_loader: DataLoader, train_model: nn.Module, args):                               
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Interval_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#

    # 冻结一定部分
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    batch_size = args.freeze_batch_size
    