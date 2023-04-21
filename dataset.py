from torch.utils.data.dataset import Dataset
import os
import cv2
from tqdm import tqdm

# tmp not use
class MyData(Dataset):        
    def __init__(self, path, augmentation=False):                
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
            label = cv2.imread(os.path.join(self.path, file_list[2 * i]))
            self.data.append(image)
            self.label.append(label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)
        

def myCollate(x):            
    data, label = x
    return data, label