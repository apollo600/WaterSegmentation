import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader

import os
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

from model.UNet import UNet
from model.deeplabv3_plus import DeepLabV3Plus
# from model.loss import FocalLoss
from utils.dataset import MyData
from utils.kitti_dataset import KittiData
from model.trainer import Unet_trainer_one_epoch

from model2.deeplab_v3plus import DeepLab, weights_init
from utils.visual import show_config
from model2.loss import Focal_Loss, Dice_loss, CE_Loss


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--dataset", type=str, default="Kitti", help="dataset to use")
    parser.add_argument("--num_classes", type=int, default="34", help="number of classes, in Deeplab should = num_classes + 1")
    parser.add_argument("--data_root", type=str, default="./", help="data directory root path (where training/ testing/ or *.png is in)")
    parser.add_argument("--data_dir", type=str, default="dataset/", help="directory where data are saved")
    parser.add_argument("--save_root", type=str, default="./", help="save directory root path (where models/ is in)")
    parser.add_argument("--save_dir", type=str, default="models/", help="directory where models are saved")
    parser.add_argument("--log_root", type=str, default="./", help="log directory root path (where logs/ is in)")
    parser.add_argument("--log_dir", type=str, default="log/train/", help="directory where logs are saved")
    parser.add_argument("--loss", type=str, default="CrossEntropy", help="loss function to use")
    parser.add_argument("--lr", type=float, default="0.001", help="initial learning rate")
    parser.add_argument("--batch_size", type=int, default="2", help="size to train each batch")
    parser.add_argument("--epoch", type=int, default="10", help="train epochs")
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer to use")
    parser.add_argument("--log_visual", action="store_true", default=True, help="save visualized picture while training")

    # Add since Deeplabv3+
    parser.add_argument("--model", type=str, default="Deeplab", help="[Unet, Deeplab]")
    # 在创建模型实例时修改 num_classes = 5 + 1 = 6
    parser.add_argument("--backbone", default="Mobilenet", help="使用的主干网络, [mobilenet, xception]")
    parser.add_argument("--pretrain_model_path", default="", help=" 模型的 预训练权重 对不同数据集是通用的，\
                                                                    因为特征是通用的.模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。")
    parser.add_argument("--downsample_factor", type=int, default=16, help="下采样的倍数，越小显存占用越多 [8, 16]")
    parser.add_argument("--init_epoch", type=int, default=0, help=" 模型当前开始的训练世代, 其值可以大于Freeze_Epoch, 如设置: \
                                                                    Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100 \
                                                                    会跳过冻结阶段, 直接从60代开始, 并调整对应的学习率。 \
                                                                    断点续练时使用）")
    parser.add_argument("--freeze_epoch", type=int, default=50, help="模型冻结训练的Freeze_Epoch, (当Freeze_Train=False时失效)")
    parser.add_argument("--freeze_batch_size", type=int, default=8, help="由于 Freeze 阶段使用显存较少，可以适当调大 Batch Size, 但需要注意保持和 Unfreeze 时 batch size 的大小在 1-2 倍之间")
    parser.add_argument("--unfreeze_epoch", type=int, default=100, help="用于修改主干部分，主干的意义是提取特征，这一部分占据显存较大")
    parser.add_argument("--unfreeze_batch_size", type=int, default=4, help="设置的稍微小一点")
    # 注意修改 lr 为 SGD: 7e-3, Adam: 5e-4
    parser.add_argument("--min_lr", type=float, default=5e-6, help="init_lr * 0.01")
    parser.add_argument("--momentum", type=float, default=0.9, help="Used in optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权值衰减，使用 Adam 时建议设置为 0")
    parser.add_argument("--lr_decay_type", type=str, default="cos", help="使用的权值下降方式, [cos, step]")
    parser.add_argument("--dice_loss", action="store_true", help="种类少(几类)时, 设置为True")
    parser.add_argument("--focal_loss", action="store_true", help="防止正负样本不平衡，需要给每个类型样本设置权重")
    parser.add_argument("--class_weights",type=int, nargs='+', help="每个类别的权重，长度和 num_classes 相同")

    
    args = parser.parse_args()

    return args


def train(train_loader: DataLoader, val_loader: DataLoader, train_model: nn.Module, args):
    init_epoch = args.epoch
    init_lr = args.lr

    if args.loss == "Focal":
        criterion = FocalLoss()
    elif args.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError("wrong type of criterion given:", args.loss)
    criterion = criterion.cuda()

    if args.optimizer == "Adam":
        optimizer = optim.Adam(train_model.parameters(), lr=init_lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(train_model.parameters(), init_lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(train_model.parameters(), lr=init_lr)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(train_model.parameters(), lr=init_lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(train_model.parameters(), lr=init_lr)
    else:
        raise RuntimeError("wrong type of optimizer given:", args.optimizer)

    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    if args.save_dir == "":
        log_dir = f"{time_stamp}_epoch-{args.epoch}_lr-{args.lr}_loss-{args.loss}_optim-{args.optimizer}"
    else:
        log_dir = os.path.join(args.save_root, args.save_dir)
    os.makedirs(log_dir, exist_ok=True)

    best_acc = 0

    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 1.5e4 if args.optimizer == "SGD" else 0.5e4
    total_step  = train_size // args.unfreeze_batch_size * args.unfreeze_epoch
    if total_step <= wanted_step:
        if train_size // args.unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (train_size // args.unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(args.optimizer, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(train_size, args.unfreeze_batch_size, args.unfreeze_epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    if args.model == "Deeplab":
        pass
    else:
        Unet_trainer_one_epoch(train_loader, val_loader, train_model, args, criterion, optimizer)


if __name__ == "__main__":
    args = get_parser()

    # Get device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device, os.environ['CUDA_VISIBLE_DEVICES'])

    # Load the data from the folders
    one_hot = False if args.loss == "CrossEntropy" else True
    if args.dataset == "Kitti":
        dataset = KittiData(
            os.path.join(args.data_root, args.data_dir), num_classes=args.num_classes,
            image_width=args.image_width, image_height=args.image_height, one_hot=one_hot, to_torch=True
        )
    elif args.dataset == "My":
        dataset = MyData(
            os.path.join(args.data_root, args.data_dir), num_classes=args.num_classes,
            image_width=args.image_width, image_height=args.image_height, one_hot=one_hot, to_torch=True
        )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, (train_size, val_size))

    r, s = dataset[0]
    print("image shape and label shape:", r.shape, s.shape)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create the model
    print("Loading model to device")
    if args.model == "Unet":
        model = UNet(3, args.num_classes)
    elif args.model == "Deeplab":
        model = DeepLab(num_classes=args.num_classes+1, backbone=args.backbone, downsample_factor=args.downsample_factor, pretrained=False)
        # 初始化大模型中的参数
        weights_init(model, init_type="normal")
        # 加载预训练模型
        model_dict = model.state_dict()
        # 使用 map_location 直接加载到显存
        pretrained_dict = torch.load(args.pretrain_model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 输出加载预训练结果
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    train_model = model.train()
    train_model.cuda()

    # 展示参数
    if args.model == "Deeplab":
        show_config(num_classes=args.num_classes, backbone=args.backbone, pretrain_model_path=args.pretrain_model_path,
        input_shape=(args.image_width, args.image_height), init_epoch=args.init_epoch, freeze_epoch=args.freeze_epoch, 
        unfreeze_epoch=args.unfreeze_epoch, freeze_batch_size=args.freeze_batch_size, unfreeze_batch_size=args.unfreeze_batch_size,
        init_lr=args.lr, min_lr=args.min_lr, optimizer=args.optimizer, momentum=args.momentum, lr_decay_type=args.lr_decay_type, 
        )

    # Train
    print("Start Train")
    train(train_loader, val_loader, train_model, args)
