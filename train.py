import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader

import os
from tqdm import tqdm
import numpy as np
import time

from model import UNET
from dataset import MyData
from loss import FocalLoss


def get_parser():                        
    import argparse

    parser = argparse.ArgumentParser(description='Train UNET')
    parser.add_argument("--epoch", type=int, default="10", help="train epochs")
    parser.add_argument("--lr", type=float, default="0.0005", help="initial learning rate")
    parser.add_argument("--batch", type=int, default="4", help="size to train each batch")
    parser.add_argument("--num_classes", type=int, default="5", help="number of classes")
    parser.add_argument("--criterion", type=str, default="Focal", help="loss function to use")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer to use")

    args = parser.parse_args()

    return args


def train(train_loader, train_model, args):
    init_epoch = args.epoch
    init_lr = args.lr
    init_batch = args.batch

    if args.criterion == "Focal":
        criterion = FocalLoss()
    elif args.criterion == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise RuntimeError("wrong type of criterion given:", args.criterion)
    criterion = criterion.cuda()

    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(train_model.parameters(), init_lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(train_model.parameters(), lr=init_lr)
    else:
        raise RuntimeError("wrong type of optimizer given:", args.optimizer)

    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
    log_dir = f"{time_stamp}_epoch-{args.epoch}_lr-{args.lr}_loss-{args.criterion}_optim-{args.optimizer}"
    os.makedirs(log_dir, exist_ok=True)

    best_acc = 0

    for epoch in range(init_epoch):
        batches = len(train_loader)
        pbar = tqdm(total=batches, desc=f"Epoch {epoch+1}/{init_epoch}: ", maxinterval=0.3, ascii=True)
        for iteration, (data, label) in enumerate(train_loader):
            data, label = data.cuda(), label.cuda()
            # label: N, H, W, C     pred_label: N, C, H, W
            pred_label = train_model(data)
            loss = criterion(pred_label, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1}/{init_epoch}: loss: {loss:.4f} acc: {acc:.4f}")
            pbar.update(1)
        pbar.close()

        print("Start Test")
        with torch.no_grad():
            for iteration, (data, label) in enumerate(val_loader):
                data, label = data.cuda(), label.cuda()
                pred_label = train_model(data)
            
            # copy the tensor to host memory first
            t_pred_label = pred_label.cpu().detach().numpy()
            t_label = label.cpu().detach().numpy()
            # get max arg as output label
            t_pred_label = np.argmax(t_pred_label, axis=1)
            t_label = np.transpose(t_label, [0, 3, 1, 2]).argmax(axis=1)
            print(t_pred_label.shape, t_label.shape)
            # update accuracy
            acc = np.sum(t_label == t_pred_label) / np.prod(t_label.shape)
            if acc > best_acc: 
                print(f"Update acc {best_acc:.4f} => {acc:.4f}")
                best_acc = acc
                model_name = os.path.join(log_dir, f"best_acc-{acc:.4f}.pt")
                torch.save(train_model.state_dict(), model_name)
        

if __name__ == "__main__":            
    args = get_parser()

    # Get device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # Load the data from the folders
    dataset = MyData("/home/data/1945", num_classes=args.num_classes, image_width=640, image_height=640)
    train_size = int(0.9 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset, val_dataset = data.random_split(dataset, (train_size, val_size))

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)

    # Create the model
    print("Loading model to device")
    model = UNET(in_channels=3, out_channels=args.num_classes)
    train_model = model.train()
    train_model.cuda()

    # Train
    print("Start Train")
    train(train_loader, train_model, args)