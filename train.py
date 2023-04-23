from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

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
    parser.add_argument("--criterion", type=str)

    args = parser.parse_args()

    return args


def train(train_loader, train_model, args):
    init_epoch = args.epoch
    init_lr = args.lr
    init_batch = args.batch

    criterion = FocalLoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = optim.AdamW(train_model.parameters(), init_lr)
    # optimizer = optim.SGD(train_model.parameters(), lr=init_lr)

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

            t_pred_label = np.argmax(pred_label, axis=1)
            t_label = np.transpose(label, [0, 3, 1, 2]).argmax(axis=1)
            acc = np.sum(t_label == t_pred_label) / np.prod(labels.shape)
            if acc > best_acc:
                
                model_name = 
                torch.save()

            pbar.set_description(f"Epoch {epoch+1}/{init_epoch}: loss: {loss} acc: {acc}")
            pbar.update(1)
        pbar.close()
            

if __name__ == "__main__":            
    args = get_parser()

    # Get device
    os.environ['CUDA_VISIBLE_DEVICES'] = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # Load the data from the folders
    train_dataset = MyData("/home/data/1945", num_classes=args.num_classes, image_width=640, image_height=640)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    # Create the model
    print("Loading model to device")
    model = UNET(in_channels=3, out_channels=args.num_classes)
    train_model = model.train()
    train_model.cuda()

    # Train
    print("Start Train")
    train(train_loader, train_model, args)