from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image

# tmp not use
class MyData(Dataset):        
    def __init__(self, path, image_width, image_height, augmentation=False):                
        super().__init__()
        self.image_paths = []
        self.label_paths = []
        self.image_width = image_width
        self.image_height = image_height

        file_list = os.listdir(self.path)
        file_list = [ x[:-4] for x in file_list if x.endswith('png') ]
        self.image_paths = [ x + ".jpg" for x in file_list ]
        self.label_paths = [ x + ".png" for x in file_list ]
        print(f"Found {len(file_list)} images")

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = Image.open(os.path.join(self.path, self.image_paths[index]))
        image = np.array(image)
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = np.transpose(image, [2, 0, 1])
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        label = Image.open(os.path.join(self.path, self.label_paths[index]))
        label = np.array(label)
        label = cv2.resize(label, (self.image_width, self.image_height))

        label = torch.from_numpy(label)
        
        return image, label

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    train_dataset = MyData("/home/data/1945", image_width=720, image_height=540)
    print(train_dataset.file_list)
    