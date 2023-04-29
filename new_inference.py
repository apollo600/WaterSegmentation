import os
import sys
sys.path.append("/project/ev_sdk/src")
import ji
import numpy as np
import torch
import cv2
from PIL import Image
from model2.utils.utils import cvtColor, preprocess_input, resize_image
from model2.utils.utils_metrics import compute_mIoU


def calc_miou(dataset_path, miou_out_path, image_ids, num_classes):
                                                                                                                
    net    = ji.init()
    gt_dir      = os.path.join(dataset_path, "SegmentationClass/")
    pred_dir    = os.path.join(miou_out_path, 'detection-results')
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    print("Get miou.")
    for image_id in tqdm(image_ids, desc="Calculate miou read images", mininterval=1, ncols=64):
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        image_path  = os.path.join(dataset_path, "JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        #------------------------------#
        #   获得预测txt
        #------------------------------#
        ji.process_image(net, image, '{"mask_output_path": "' + str(mask_png_path).replace('\\', '/') + '"}')
        image = Image.open(mask_png_path)
        image.save(os.path.join(pred_dir, image_id + ".png"))
                
    print("Calculate miou.")
    _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, None)  # 执行计算mIoU的函数
    temp_miou = np.nanmean(IoUs) * 100

    print(temp_miou)

    print("Get miou done.")
    # shutil.rmtree(self.miou_out_path)


if __name__ == "__main__":  
    image_ids = 

    calc_miou("/home/data/1945", ".", image_ids, num_classes)                                           