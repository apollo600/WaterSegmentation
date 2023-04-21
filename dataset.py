from torch.utils.data.dataset import Dataset
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch

# tmp not use
class MyData(Dataset):        
    def __init__(self, path, image_width, image_height, augmentation=False):                
        super().__init__()
        self.path = path
        self.data = []
        self.label = []

        file_list = os.listdir(self.path)
        file_list.sort()
        assert(len(file_list) % 2 == 0)
        print(f"Found {len(file_list) // 2} images")
        for i in tqdm(range(len(file_list) // 2), desc="Read: "):
            image = cv2.imread(os.path.join(self.path, file_list[2 * i + 1]))
            image = cv2.resize(image, (image_width, image_height))
            image = np.transpose(image, [2, 0, 1])
            label = cv2.imread(os.path.join(self.path, file_list[2 * i]))
            label = cv2.resize(label, (image_width, image_height))
            label = np.transpose(label, [2, 0, 1])
            self.data.append(image)
            self.label.append(label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)
