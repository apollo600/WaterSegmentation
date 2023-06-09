import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader

import os
import sys
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

from model.UNet import UNet
from utils.dataset import MyData
from utils.kitti_dataset import KittiData
from utils import visual


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def get_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--dataset", type=str, default="Kitti", help="dataset to use")
    parser.add_argument("--num_classes", type=int, default="34", help="number of classes")
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
    parser.add_argument("--log_visual", type=str2bool, default=True, help="save visualized picture while training")
    parser.add_argument("--use_tqdm", type=str2bool, default=True)
    parser.add_argument("--only_best", type=str2bool, default=True, help="only save best .pt")

    args = parser.parse_args()

    return args


def train(train_loader: DataLoader, val_loader: DataLoader, train_model: nn.Module, args):
    init_epoch = args.epoch
    init_lr = args.lr

    if args.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError("wrong type of criterion given:", args.loss)
    criterion = criterion.cuda()

    if args.optimizer == "AdamW":
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
    log_dir = os.path.join(args.save_root, args.save_dir)
    os.makedirs(log_dir, exist_ok=True)

    best_acc = 0

    for epoch in range(init_epoch):
        batches = len(train_loader)

        pbar: tqdm = None
        ave_loss = 0
        ave_acc = 0
        if args.use_tqdm:
            pbar = tqdm(total=batches, desc=f"Epoch {epoch+1}/{init_epoch}: ", maxinterval=0.3, ascii=True)
        else:
            sys.stdout.write(f"Epoch {epoch+1}/{init_epoch}\n")

        for iteration, (data, label) in enumerate(train_loader):

            data, label = data.cuda(), label.cuda()
            # label: N, H, W; pred_label: N, C, H, W
            pred_label = train_model(data)

            if args.loss == "CrossEntropy":
                # N, C, H, W => C, N*H*W
                pred_label = pred_label.contiguous().permute(0, 2, 3, 1)
                pred_label = pred_label.reshape(-1, pred_label.size(3))
                # N, C, H, W => C*N*H*W
                label = label.view(-1)
            else:
                pass
            loss = criterion(pred_label, label)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # copy the tensor to host memory first
            t_pred_label = pred_label.cpu().detach().numpy()
            t_label = label.cpu().detach().numpy()
            # get max arg as output label
            t_pred_label = np.argmax(t_pred_label, axis=1)
            if args.loss == "CrossEntropy":
                pass
            elif args.loss == "Focal":
                t_label = np.transpose(t_label, [0, 3, 1, 2]).argmax(axis=1)
            # update accuracy
            acc = np.sum(t_label == t_pred_label) / np.prod(t_label.shape)

            if args.use_tqdm:
                pbar.set_description(f"Epoch {epoch+1}/{init_epoch} loss: {loss:.4f} train_acc: {acc:.4f}")
                pbar.update(1)
            else:
                ave_loss += loss / len(train_loader)
                ave_acc += acc / len(train_loader)
        
        if args.use_tqdm:
            pbar.close()
        else:
            sys.stdout.write(f"ave loss: {loss:.4f}, ave train_acc: {acc:.4f}\n")

        sys.stdout.write("validating\n")
        with torch.no_grad():
            batches = len(val_loader)
            total_acc = 0

            pbar: tqdm = None
            if args.use_tqdm:
                pbar = tqdm(total=batches, maxinterval=0.3, ascii=True)

            for iteration, (data, label) in enumerate(val_loader):

                pred_label = train_model(data.cuda())

                # copy the tensor to host memory first
                t_pred_label = pred_label.cpu().detach().numpy()
                t_label = label.detach().numpy()
                # get max arg as output label
                t_pred_label = np.argmax(t_pred_label, axis=1)
                if args.loss == "CrossEntropy":
                    pass
                elif args.loss == "Focal":
                    t_label = np.transpose(t_label, [0, 3, 1, 2]).argmax(axis=1)
                # update accuracy
                acc = np.sum(t_label == t_pred_label) / np.prod(t_label.shape[1:])
                total_acc += acc

                # visual pictures
                max_picture_shown_each_epoch = 3
                if args.log_visual and iteration < max_picture_shown_each_epoch:
                    log_path = os.path.join(args.log_root, args.log_dir)
                    os.makedirs(log_path, exist_ok=True)
                    visual.visualize(np.squeeze(t_pred_label, axis=0), os.path.join(log_path, f"e{epoch}_i{iteration}_pred.png"))
                    visual.visualize(np.squeeze(t_label, axis=0), os.path.join(log_path, f"e{epoch}_i{iteration}_gt.png"))
                    Image.fromarray(np.squeeze(np.uint8(data.detach().numpy()), axis=0).transpose([1, 2, 0])).save(os.path.join(log_path, f"e{epoch}_i{iteration}_src.png"))

                if args.use_tqdm:
                    pbar.update(1)

            if args.use_tqdm:
                pbar.close()

            total_acc /= batches

            model_path = os.path.join(log_dir, "last.pt")
            torch.save(train_model, model_path)

            if not args.only_best:
                model_path = os.path.join(log_dir, f"{time_stamp}_epoch-{args.epoch}_lr-{args.lr}_loss-{args.loss}_optim-{args.optimizer}_acc-{total_acc:.4f}.pt")
                torch.save(train_model, model_path)
            if total_acc > best_acc: 
                sys.stdout.write(f"Update acc {best_acc:.4f} => {total_acc:.4f}\n")
                best_acc = total_acc
                model_path = os.path.join(log_dir, "best.pt")
                torch.save(train_model, model_path)
            else:
                sys.stdout.write(f"acc: {total_acc:.4f}\n")


if __name__ == "__main__":
    args = get_parser()

    # Get device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.stdout.write("Using device {} {}\n".format(device, os.environ['CUDA_VISIBLE_DEVICES']))

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
    sys.stdout.write("image shape and label shape: {} {}\n".format(r.shape, s.shape))

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create the model
    sys.stdout.write("Loading model to device\n")
    model = UNet(3, args.num_classes)
    train_model = model.train()
    train_model.cuda()

    # Train
    sys.stdout.write("Start Train\n")
    train(train_loader, val_loader, train_model, args)