from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image


class KittiData(Dataset):        
    def __init__(self, path, num_classes, image_width, image_height, augmentation=False, isTrain=True):                
        super().__init__()
        self.path = path
        self.class_num = num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.augmentation = augmentation
        self.isTrain = isTrain

        if isTrain:
            self.image_paths = os.listdir(os.path.join(path, "training", "image_2")).sort()
            self.label_paths = os.listdir(os.path.join(path, "training", "semantic")).sort()
        else:
            self.image_paths = os.listdir(os.path.join(path, "testing", "image_2")).sort()
            self.label_paths = os.listdir(os.path.join(path, "testing", "semantic")).sort()    
        print(f"Found {len(self.image_paths)} images")

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = Image.open(os.path.join(self.path, self.image_paths[index]))
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        image = np.array(image)
        image = np.transpose(image, [2, 0, 1])

        label = Image.open(os.path.join(self.path, self.label_paths[index]))
        label = label.resize((self.image_width, self.image_height), Image.BILINEAR)
        label = np.array(label)

        label_one_hot = np.zeros((label.shape[0], label.shape[1], self.class_num))
        for i in range(self.class_num):
            label_one_hot[:,:,i] = (label == i)

        image = torch.from_numpy(image).float()
        label_one_hot = torch.from_numpy(label_one_hot).long()
        
        return image, label_one_hot

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    train_dataset = KittiData("/project/train/src_repo/data_semantics", 8, 640, 640)
    image, data = 
            