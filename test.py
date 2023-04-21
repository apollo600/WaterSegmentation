import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os
from PIL import Image

from model import UNET
from dataset import MyData

def load_label(filepath):
    label_path = os.path.splittext(os.path.basename(filepath))[0] + '.png'
    label = Image.open(label_path)
    return label
    

def get_parser():                        
    import argparse

    parser = argparse.ArgumentParser(description='Train UNET')
    parser.add_argument("--epoch", type=int, help="train epochs")
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--batch", type=int, help="size to train each batch")

    args = parser.parse_args()

    return args


def main():
    args = get_parser()

    init_epoch = args.epoch
    init_lr = args.lr
    init_batch = args.batch

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    for epoch in range(init_epoch):
        batches = train_data.length // init_batch

        pbar = tqdm(total=batches, desc="Batch:", maxinterval=0.3)
        for iteration, batch in enumerate(train_loader):
            pass
            

if __name__ == "__main__":            
    # Define the transforms to be applied to each image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the data from the folders
    train_dataset = MyData("/home/data/1945")
    print(len(train_dataset))

    # Create the loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    # Train
    for epoch in range(5):        
        