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
    def __init__(self, path, class_num, image_width, image_height, augmentation=False):                
        super().__init__()
        self.path = path
        self.class_num = class_num
        self.image_width = image_width
        self.image_height = image_height

        file_list = os.listdir(path)
        file_list = [ x[:-4] for x in file_list if x.endswith('png') ]
        self.image_paths = [ x + ".jpg" for x in file_list ]
        self.label_paths = [ x + ".png" for x in file_list ]
        print(f"Found {len(file_list)} images")

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = Image.open(os.path.join(self.path, self.image_paths[index]))
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        image = np.array(image)

        label = Image.open(os.path.join(self.path, self.label_paths[index]))
        label = label.resize((self.image_width, self.image_height), Image.BILINEAR)
        label = np.array(label)

        label_one_hot = np.zeros((label.shape[0], label.shape[1], self.class_num))
        for i in range(self.class_num):
            label_one_hot[:,:,i] = (label == i).astype(np.float32)

        image = torch.from_numpy(image).float()
        label_one_hot = torch.from_numpy(label_one_hot).float()
        
        return image, label_one_hot

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    train_dataset = MyData("/home/data/1945", class_num=5, image_width=720, image_height=540)
    image, label = train_dataset[0]
    print(image.shape, label.shape)
    print(label[420, 110:115, :])
            