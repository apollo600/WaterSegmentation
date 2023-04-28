import os
import numpy as np
from PIL import Image
from typing import Tuple
from torch.utils.data.dataset import Dataset


class RawData(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path

        file_list = os.listdir(path)
        file_list = [ x[:-4] for x in file_list if x.endswith('png') ]
        self.image_paths = [ x + ".jpg" for x in file_list ]
        self.label_paths = [ x + ".png" for x in file_list ]
        print(f"Found {len(file_list)} images")

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """
        image [H, W, C (RGB)], label [H, W]
        """

        image = Image.open(os.path.join(self.path, self.image_paths[index]))
        image = np.array(image)

        label = Image.open(os.path.join(self.path, self.label_paths[index]))
        label = np.array(label)

        return image, label

    def __len__(self):
        return len(self.image_paths)
