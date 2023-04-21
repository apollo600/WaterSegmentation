import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from PIL import Image

from model import UNET
from dataset import MyData


def get_parser():                        
    import argparse

    parser = argparse.ArgumentParser(description='Train UNET')
    parser.add_argument("--epoch", type=int, help="train epochs")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--batch", type=int, help="size to train each batch")

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
        pbar = tqdm(total=batches, desc=f"Epoch {epoch}/{init_epoch}:", maxinterval=0.3)
        for iteration, (data, label) in enumerate(train_loader):
            pred_label = train_model(data)
            loss = criterion(pred_label, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update()
        pbar.close()
            

if __name__ == "__main__":            
    args = get_parser()
    
    # Define the transforms to be applied to each image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the data from the folders
    train_dataset = MyData("/home/data/1945")
    print(len(train_dataset))

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=8)

    # Create the model
    model = UNET(3, 1)
    train_model = model.train()

    # Train
    train(train_loader, train_model, args)