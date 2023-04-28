import os
import numpy as np
from PIL import Image
from typing import Tuple, Union

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class PascalData(Dataset):
    # root="/project/train/src_repo/VOCdevkit/VOC2012"
    def __init__(self, root, file_list, num_classes, image_width, image_height, augmentation=False, to_torch=True, one_hot=False):
                                                                                                                                                                                                                                                 
        super().__init__()
        self.path = path
        self.class_num = num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.to_torch = to_torch
        self.one_hot = one_hot

        self.image_paths = [ os.path.join(path, "JPEGImages", x + ".jpg") for x in file_list ]
        self.label_paths = [ os.path.join(path, "SegmentationClass", x + ".png") for x in file_list ]
        print(f"Found {len(file_list)} images")

    def __getitem__(self, index) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Input: image [H, W, C (RGB)]
        Train or Test: image [C (RGB), H, W], label [H, W]
        Note: label will be [H, W, C (Classes)] if one_hot is True
        """

        image = Image.open(os.path.join(self.path, self.image_paths[index]))
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        image = np.array(image)
        image = np.transpose(image, [2, 0, 1])

        label = Image.open(os.path.join(self.path, self.label_paths[index]))
        label = label.resize((self.image_width, self.image_height), Image.BILINEAR)
        label = np.array(label)

        if self.one_hot:
            label_one_hot = np.zeros((label.shape[0], label.shape[1], self.class_num))
            for i in range(self.class_num):
                label_one_hot[:,:,i] = (label == i)
            if self.to_torch:
                image = torch.from_numpy(image).float()
                label_one_hot = torch.from_numpy(label_one_hot).long()
            return image, label_one_hot
        else:
            if self.to_torch:
                image = torch.from_numpy(image).float()
                label = torch.from_numpy(label).long()
            return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_filelist(self):
        return self.file_list
    

if __name__ == "__main__":
    train_dataset = MyData("/home/data/1945", num_classes=5, image_width=720, image_height=540)
    image, label = train_dataset[0]
    print(image.shape, label.shape)
            