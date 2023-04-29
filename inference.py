import os
import sys

sys.path.append("/project/ev_sdk/src")

import ji
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import visual
from utils.raw_dataset import RawData


parser = argparse.ArgumentParser(description='Test')

parser.add_argument("--model_path", type=str, default="/project/train/models/None", help="path of model static dict to load")
parser.add_argument("--data_root", type=str, default="./", help="data directory root path (where training/ testing/ or *.png is in)")
parser.add_argument("--data_dir", type=str, default="dataset/", help="directory where data are saved")
parser.add_argument("--log_root", type=str, default="./", help="log directory root path (where logs/ is in)")
parser.add_argument("--log_dir", type=str, default="log/infer/", help="directory where logs are saved")
parser.add_argument("--mask_root", type=str, default="./")
parser.add_argument("--mask_dir", type=str, default="./")

args = parser.parse_args()

model = ji.init(args.model_path)

dataset = RawData(os.path.join(args.data_root, args.data_dir))
total_acc = 0

log_path = os.path.join(args.log_root, args.log_dir)
os.makedirs(log_path, exist_ok=True)

mask_path = os.path.join(args.mask_root, args.mask_dir)
mask_png_path = os.path.join(mask_path, "mask.png")
os.makedirs(mask_path, exist_ok=True)

for i in tqdm(range(len(dataset)), desc="Inferencing", ascii=True):

    # image: C, H, W    label: H, W
    image, label = dataset[i]
    h, w, _ = image.shape
    
    output_json = ji.process_image(
        model, image, '{"mask_output_path": "' + str(mask_png_path).replace('\\', '/') + '"}'
    )

    pred_label = Image.open(mask_png_path)
    pred_label = np.array(pred_label)

    # save
    visual.visualize(pred_label, os.path.join(log_path, f"{i:05d}_pred.png"))
    visual.visualize(label, os.path.join(log_path, f"{i:05d}_label.png"))
    Image.fromarray(np.uint8(image)).save(os.path.join(log_path, f"{i:05d}_src.png"))

    total_acc += (pred_label == label).sum() / np.prod(pred_label.shape)

print("Test acc:", total_acc / len(dataset))
