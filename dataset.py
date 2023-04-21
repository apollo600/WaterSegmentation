from torch.utils.data.dataset import Dataset
import os
import cv2
from tqdm import tqdm

# tmp not use
class MyData(Dataset):        
    def __init__(self, path, augmentation=False):                
        super().__init__()
        self.path = path

        file_list = os.listdir(self.path)
        file_list.sort()
        assert(len(file_list) % 2 == 0)
        self.file_list = file_list
        print(f"Found {len(file_list) // 2} images")

    def __getitem__(self, index):
        data = cv2.imread(os.path.join(self.path, self.file_list[2 * i + 1]))
        label = cv2.imread(os.path.join(self.path, self.file_list[2 * i]))
        return data, label

    def __len__(self):
        return len(self.file_list) // 2
        
        