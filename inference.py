import os
import ji
import argparse
import numpy as np
from PIL import Image
from utils import visual

parser = argparse.ArgumentParser(description='Test UNET')
parser.add_argument("--model_path", type=str, default="/project/train/models/2023-04-26-15:26:55_epoch-100_lr-00005_loss-CrossEntropy_optim-AdamW_best_acc-0.7752.pt", help="path of model static dict to load")
parser.add_argument("--image_width", type=int, default=640)
parser.add_argument("--image_height", type=int, default=640)

args = parser.parse_args()

model = ji.init(args.model_path)
from utils.dataset import MyData
from tqdm import tqdm
dataset = MyData("/home/data/1945", 5, args.image_width, args.image_height, is_train=False, one_hot=False)
total_acc = 0
save_dir = "logs/infer"
os.makedirs(save_dir, exist_ok=True)
for i in tqdm(range(len(dataset)), desc="Inferencing", ascii=True):
    image, label = dataset[i]
    output_json = ji.process_image(model, image, 
        '{"mask_output_path": "/project/ev_sdk/mask.png"}')
    pred_label = Image.open("/project/ev_sdk/mask.png")
    pred_label = np.array(pred_label)
    # save pred label
    visual.visualize(pred_label, os.path.join(save_dir, f"{i:05d}_pred.png"))
    # save src and label
    Image.fromarray(image).save(os.path.join(save_dir, f"{i:05d}_src.png"))
    visual.visualize(label, os.path.join(save_dir, f"{i:05d}_gt.png"))
    total_acc += (pred_label == label).sum() / np.prod(pred_label.shape)

print("Test acc:", total_acc / len(dataset))
