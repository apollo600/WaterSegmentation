from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image

from model import UNET
from dataset import MyData
import time


def get_parser():                        
    import argparse

    parser = argparse.ArgumentParser(description='Train UNET')
    parser.add_argument("--epoch", type=int, default="10", help="train epochs")
    parser.add_argument("--lr", type=float, default="0.01", help="initial learning rate")
    parser.add_argument("--batch", type=int, default="4", help="size to train each batch")

    args = parser.parse_args()

    return args


def train(train_loader, train_model, args):
    init_epoch = args.epoch
    init_lr = args.lr
    init_batch = args.batch

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = optim.AdamW(train_model.parameters(), init_lr)

    for epoch in range(init_epoch):
        batches = len(train_loader)
        pbar = tqdm(total=batches, desc=f"Epoch {epoch}/{init_epoch}: ", maxinterval=0.3, ascii=True)
        for iteration, (data, label) in enumerate(train_loader):
            pbar.set_description(f"Epoch {epoch}/{init_epoch}: processing")
            data, label = data.cuda(), label.cuda()
            pred_label = train_model(data)
            print(pred_label.size(), label.size())
            loss = criterion(pred_label, label.view(label.shape[0], -1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch}/{init_epoch}: loss: {loss}")
            pbar.update(1)
        pbar.close()
            

if __name__ == "__main__":            
    args = get_parser()

    # Load the data from the folders
    train_dataset = MyData("/home/data/1945", image_width=640, image_height=640)

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    # Create the model
    print("Loading model")
    model = UNET(in_channels=3, out_channels=1)
    train_model = model.train()
    train_model.cuda()

    # Train
    print("Start Train")
    train(train_loader, train_model, args)