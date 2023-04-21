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
        self.path = path
        self.data = []
        self.label = []
        self.image_width = image_width
        self.image_height = image_height

        file_list = os.listdir(self.path)
        file_list.sort()
        self.file_list = file_list
        assert(len(file_list) % 2 == 0)
        print(f"Found {len(file_list) // 2} images")

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # image = cv2.imread(os.path.join(self.path, self.file_list[2 * index + 1]))
        # image = cv2.resize(image, (self.image_width, self.image_height))
        # image = np.transpose(image, [2, 0, 1])
        # image = image.astype(np.float32)
        # image = torch.from_numpy(image)
        image = Image.open(os.path.join(self.path, self.file_list[2 * index]))
        image = transform(image)

        # label = cv2.imread(os.path.join(self.path, self.file_list[2 * index]))
        # label = cv2.resize(label, (self.image_width, self.image_height))
        # label = np.transpose(label, [2, 0, 1])
        # label = label.astype(np.float32)
        # label = torch.from_numpy(label)
        label = Image.open(os.path.join(self.path, self.file_list[2 * index + 1]))
        label = transform(label)
        
        return image, label

    def __len__(self):
        return len(self.file_list) // 2


if __name__ == "__main__":
    train_dataset = MyData("/home/data/1945", image_width=720, image_height=540)
    image, label = train_dataset[0]
    print(image.shape, label.shape)
    