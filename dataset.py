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
        self.image_width = image_width
        self.image_height = image_height

        file_list = os.listdir(self.path)
        file_list.sort()
        self.file_list = file_list
        assert(len(file_list) % 2 == 0)
        print(f"Found {len(file_list) // 2} images")
        # for i in tqdm(range(len(file_list) // 2), desc="Read: "):
        #     image = cv2.imread(os.path.join(self.path, file_list[2 * i + 1]))
        #     image = cv2.resize(image, (image_width, image_height))
        #     image = np.transpose(image, [2, 0, 1])
        #     image = image.astype(np.float32)
        #     label = cv2.imread(os.path.join(self.path, file_list[2 * i]))
        #     label = cv2.resize(label, (image_width, image_height))
        #     label = np.transpose(label, [2, 0, 1])
        #     label = label.astype(np.float32)
        #     self.data.append(image)
        #     self.label.append(label)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.path, self.file_list[2 * index + 1]))
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = np.transpose(image, [2, 0, 1])
        image = image.astype(np.float32)
        label = cv2.imread(os.path.join(self.path, self.file_list[2 * index]))
        label = cv2.resize(label, (self.image_width, self.image_height))
        label = np.transpose(label, [2, 0, 1])
        label = label.astype(np.float32)
        # data = self.data[index]
        # label = self.label[index]
        return image, label

    def __len__(self):
        # return len(self.data)
        return len(self.file_list) // 2
