import os
import numpy as np
from PIL import Image
from typing import Tuple, Union

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from model2.utils.utils import cvtColor, preprocess_input
import cv2


class MyData(Dataset):
    def __init__(self, dataset_path, annotation_lines, image_width, image_height, num_classes, train):
                                                                                                                
        super(MyData, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = (image_width, image_height)
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path), name + ".jpg"))
        png         = Image.open(os.path.join(os.path.join(self.dataset_path), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', [w, h], (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data      = np.array(image, np.uint8)

        #------------------------------------------#
        #   高斯模糊
        #------------------------------------------#
        blur = self.rand() < 0.25
        if blur: 
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        #------------------------------------------#
        #   旋转
        #------------------------------------------#
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data  = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128,128,128))
            label       = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label


# class PascalData(Dataset):
#     # root="/project/train/src_repo/VOCdevkit/VOC2012"
#     def __init__(self, root, file_list, num_classes, image_width, image_height, augmentation=False, to_torch=True):
                                                                                                                                                                                                                                                     
#         super().__init__()
#         self.root = root
#         self.file_list = file_list
#         self.class_num = num_classes
#         self.image_width = image_width
#         self.image_height = image_height
#         self.to_torch = to_torch
#         self.one_hot = one_hot

#         self.image_paths = [ os.path.join(root, "JPEGImages", x.strip() + ".jpg") for x in file_list ]
#         self.label_paths = [ os.path.join(root, "SegmentationClass", x.strip() + ".png") for x in file_list ]
#         print(f"Found {len(file_list)} images")

#     def __getitem__(self, index) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
#         """
#         Input: image [H, W, C (RGB)]
#         Train or Test: image [C (RGB), H, W], label [H, W]
#         Note: label will be [H, W, C (Classes)] if one_hot is True
#         """

#         image = Image.open(os.path.join(self.root, self.image_paths[index]))
#         image = image.resize((self.image_width, self.image_height), Image.BILINEAR)
#         image = np.array(image)
#         image = np.transpose(image, [2, 0, 1])

#         label = Image.open(os.path.join(self.root, self.label_paths[index]))
#         label = label.resize((self.image_width, self.image_height), Image.BILINEAR)
#         label = np.array(label)

#         if self.one_hot:
#             label_one_hot = np.zeros((label.shape[0], label.shape[1], self.class_num))
#             for i in range(self.class_num):
#                 label_one_hot[:,:,i] = (label == i)
#             if self.to_torch:
#                 image = torch.from_numpy(image).float()
#                 label_one_hot = torch.from_numpy(label_one_hot).long()
#             return image, label_one_hot
#         else:
#             if self.to_torch:
#                 image = torch.from_numpy(image).float()
#                 label = torch.from_numpy(label).long()
#             return image, label

#     def __len__(self):
#         return len(self.image_paths)

#     def get_filelist(self):
#         return self.file_list
    

if __name__ == "__main__":
    train_dataset = MyData("/home/data/1945", num_classes=5, image_width=720, image_height=540)
    image, label = train_dataset[0]
    print(image.shape, label.shape)
            