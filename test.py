import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import UNET

# Define the transforms to be applied to each image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data from the folders
train_data = ImageFolder('/home/data/1945', transform, is_valid_file=lambda x: x.endswith('.jpg'))

# Create the loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

def get_parser():                        
    import argparse

    parser = argparse.ArgumentParser(description='Train UNET')
    parser.add_argument("--epoch", type=int, help="train epochs")
    parser.add_argument("--lr", type=float, help="initial learning rate")

    args = parser.parse_args()

    return args


def main():
    args = get_parser()

    init_epoch = args.epoch
    init_lr = args.lr

    for epoch in tqdm(range(init_epoch), desc="Traning, Epoch:"):
        
                                                                                
        