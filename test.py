import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

from model import UNET

# Define the transforms to be applied to each image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the data from the folders
train_data = ImageFolder('/home/data/1945', transform, is_valid_file=lambda x: x.endswith('.jpg'))

# Create the loaders
train_loader = DataLoader(train_data)

def get_parser():                        
    
    