import os
import sys
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Union

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class KittiData(Dataset):
    def __init__(self, path, num_classes, image_width, image_height, augmentation=False, to_torch=True, one_hot=True):
        super().__init__()
        self.path = path
        self.class_num = num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.augmentation = augmentation
        self.to_torch = to_torch
        self.one_hot = one_hot

        if to_torch:
            image_paths = os.listdir(os.path.join(path, "training", "image_2"))
            image_paths.sort()
            self.image_paths = [os.path.join(path, "training", "image_2", x) for x in image_paths]
            label_paths = os.listdir(os.path.join(path, "training", "semantic"))
            label_paths.sort()
            self.label_paths = [os.path.join(path, "training", "semantic", x) for x in label_paths]
        else:
            image_paths = os.listdir(os.path.join(path, "testing", "image_2"))
            image_paths.sort()
            self.image_paths = [os.path.join(path, "testing", "image_2", x) for x in image_paths]
            label_paths = os.listdir(os.path.join(path, "testing", "semantic"))
            label_paths.sort()
            self.label_paths = [os.path.join(path, "testing", "semantic", x) for x in label_paths]
        sys.stdout.write(f"Found {len(self.image_paths)} images\n")

    def __getitem__(self, index) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Input: image [H, W, C (RGB)]
        Train or Test: image [C (RGB), H, W], label [H, W]
        Note: label will be [H, W, C (Classes)] if one_hot is True
        """

        assert(
            self.image_paths[index].replace('\\', '/').split('/')[-1]
            == self.label_paths[index].replace('\\', '/').split('/')[-1]
        )

        image = Image.open(self.image_paths[index])
        image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
        image = np.array(image)
        image = np.transpose(image, [2, 0, 1])

        label = Image.open(self.label_paths[index])
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


if __name__ == "__main__":
    train_dataset = KittiData("/project/train/src_repo/data_semantics", 34, 640, 640, one_hot=False)
    # Test case 1
    image, label = train_dataset[0]
    print(image.shape, label.shape)
    # Test case 2
    max_label = 0
    pbar = tqdm(total=len(train_dataset), ascii=True)
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        this_min_label = torch.min(label)
        this_max_label = torch.max(label)
        pbar.set_description(f"Llabel = {this_min_label} -- {this_max_label}")
        if max_label < this_max_label:
            max_label = this_max_label
        pbar.update(1)
    pbar.close()
    print("max label:", max_label)
                    