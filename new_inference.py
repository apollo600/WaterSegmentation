import os
import sys
sys.path.append("/project/ev_sdk/src")
import ji
# import _deprecated_ji as ji
import numpy as np
import torch
import cv2
from PIL import Image
from model2.utils.utils import cvtColor, preprocess_input, resize_image
from model2.utils.utils_metrics import compute_mIoU
from tqdm import tqdm
from utils.visual import visualize
import time


def calc_miou(dataset_path, model_path, miou_out_path, image_ids, num_classes, mask_png_path, enable_visualize):
                                                                                                                                                        
    net    = ji.init(model_path)
    gt_dir      = os.path.join(dataset_path, "SegmentationClass/")
    pred_dir    = os.path.join(miou_out_path, 'detection-results')
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    print("Get miou.")

    time_costs = []

    for image_id in tqdm(image_ids, desc="Reading images and Predict", mininterval=1):
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        image_path  = os.path.join(dataset_path, "JPEGImages", image_id+".jpg")
        label_path  = os.path.join(dataset_path, "SegmentationClass", image_id+".png")
        image       = Image.open(image_path)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if enable_visualize:
            tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tmp_image = Image.fromarray(tmp_image)
            tmp_image.save(os.path.join(miou_out_path, image_id+"_src.jpg"))
        label       = Image.open(label_path)
        label       = np.array(label)
        #------------------------------#
        #   获得预测txt
        #------------------------------#
        time_start = time.perf_counter()
        ji.process_image(net, image, '{"mask_output_path": "' + str(mask_png_path).replace('\\', '/') + '"}')
        time_end = time.perf_counter()
        time_costs.append((time_end - time_start) * 1000)
        image = Image.open(mask_png_path)
        image.save(os.path.join(pred_dir, image_id + ".png"))
        image = np.array(image)
        # 可视化
        if enable_visualize:
            visualize(label, os.path.join(miou_out_path, image_id+"_gt.jpg"))
            visualize(image, os.path.join(miou_out_path, image_id+"_pred.jpg"))
                
    print("Calculate miou.")
    _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, None)  # 执行计算mIoU的函数
    temp_miou = np.nanmean(IoUs) * 100

    print(f"===> Average time: {np.mean(time_costs)} ms")

    print("Get miou done.")
    # shutil.rmtree(self.miou_out_path)


if __name__ == "__main__":  
    # image_ids = open("/project/train/src_repo/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r").readlines()
    # image_ids = [ x.strip() for x in image_ids ]
    # calc_miou("/project/train/src_repo/VOCdevkit/VOC2012", "/project/train/models/pascal/best_epoch_weights/pth", "/project/train/log/infer/pascal", image_ids, num_classes=21, mask_png_path="/project/ev_sdk/mask.png", enable_visualize=False)                    

    image_ids = open("/home/data/1945/ImageSets/Segmentation/val.txt", "r").readlines()
    image_ids = [ x.strip() for x in image_ids ]
    calc_miou("/home/data/1945", "/project/train/models/best_epoch_weights.pth", "/project/train/log/infer/my", image_ids, num_classes=6, mask_png_path="/project/ev_sdk/mask.png", enable_visualize=True)                             