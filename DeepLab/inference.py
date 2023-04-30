import os
import sys
import cv2
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append("/project/ev_sdk/src")

import ji

from model.utils_metrics import compute_mIoU
from utils.visual import visualize


def calc_miou(dataset_path, model_path, miou_out_path, image_ids, num_classes, mask_png_path, enable_visualize):
                                                                                                                                                        
    net    = ji.init(model_path)
    gt_dir      = os.path.join(dataset_path, "SegmentationClass/")
    pred_dir    = os.path.join(miou_out_path, 'detection-results')
    os.makedirs(miou_out_path, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    sys.stdout.write("Get miou.\n")

    time_costs = []

    for image_id in tqdm(image_ids, desc="Reading images and Predict", mininterval=1):
        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        image_path = os.path.join(dataset_path, "JPEGImages", image_id+".jpg")
        label_path = os.path.join(dataset_path, "SegmentationClass", image_id+".png")
        image = Image.open(image_path)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if enable_visualize:
            tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tmp_image = Image.fromarray(tmp_image)
            tmp_image.save(os.path.join(miou_out_path, image_id+"_src.jpg"))
        label = Image.open(label_path)
        label = np.array(label)
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
                
    sys.stdout.write("Calculate miou.\n")
    _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, None)  # 执行计算mIoU的函数
    temp_miou = np.nanmean(IoUs) * 100

    sys.stdout.write(f"===> Average time: {np.mean(time_costs)} ms\n")

    sys.stdout.write("Get miou done.\n")
    # shutil.rmtree(self.miou_out_path)


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument("--model_path", type=str, default="/project/train/models/None", help="path of model static dict to load")
    parser.add_argument("--data_root", type=str, default="./", help="data directory root path (where training/ testing/ or *.png is in)")
    parser.add_argument("--data_dir", type=str, default="dataset/", help="directory where data are saved")
    parser.add_argument("--log_root", type=str, default="./", help="log directory root path (where logs/ is in)")
    parser.add_argument("--log_dir", type=str, default="log/infer/", help="directory where logs are saved")
    parser.add_argument("--mask_root", type=str, default="./")
    parser.add_argument("--mask_dir", type=str, default="./")

    args = parser.parse_args()                  

    image_ids = open(os.path.join(args.data_root, args.data_dir, "ImageSets/Segmentation/val.txt").replace('\\', '/'), "r").readlines()
    image_ids = [ x.strip() for x in image_ids ]
    calc_miou(
        os.path.join(args.data_root, args.data_dir),
        args.model_path,
        os.path.join(args.log_root, args.log_dir),
        image_ids, num_classes=6,
        mask_png_path=os.path.join(args.mask_root, args.mask_dir, "mask.png"),
        enable_visualize=True
    )
