from torch.utils.data.dataset import Dataset
import os
import cv2
from tqdm import tqdm
import numpy as np
import torch

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
        

def myCollate(data):            
    # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    # data.sort(key=lambda x: len(x[0][0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    label_list = []
    for batch in range(0, len(data)): #
        data_list.append(data[batch][0])
        label_list.append(data[batch][1])
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor)
    return data_copy