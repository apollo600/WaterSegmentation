import os
import sys
import json
import argparse
 
from PIL import Image
import numpy as np
import cv2
import torch

sys.path.append("/project/train/src_repo/")

from model.model import UNET
 
def get_parser():
    parser = argparse.ArgumentParser(description='Test UNET')
    parser.add_argument("--model_path", type=str, default="/project/train/models/2023-04-26-15:26:55_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.7752.pt", help="path of model static dict to load")
    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--image_height", type=int, default=384)

    args = parser.parse_args()

    return args


def checkGPU():
    gpu_count = torch.cuda.device_count()

    print(">>> GPU List <<<")
    for i in range(gpu_count):
        print(f"{i+1}. {torch.cuda.get_device_name(i)}")
        

def init():
    
    """Initialize model
 
    Returns: model
 
    """

    args = get_parser()

    checkGPU()
    
    print("Load Model")
    model = UNET(3, 5)
    if args.model_path is not None and args.model_path is not "":
        model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    return model
 
def process_image(handle=None, input_image=None, args=None, **kwargs):
    
    """Do inference to analysis input_image and get output
 
    Attributes:
        handle: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: string in JSON format, format: {
            "mask_output_path": "/path/to/output/mask.png"
        }
 
    Returns: process result
 
    """

    args = json.loads(args)
    mask_output_path = args['mask_output_path']
    model = handle
    
    # Process image here
    image = input_image
    # H, W, C -> C, H, W
    image = np.transpose(image, [2, 0, 1])
    # C, H, W -> 1, Channels, H, W
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    image = image.cuda()
    # 1, Classes, H, W
    pred_label = model(image)
    # Generate dummy mask data
    t_pred_label = pred_label.cpu().detach().numpy()
    # print("label shape:", t_pred_label.shape)
    # print(t_pred_label.transpose([0, 2, 3, 1]))
    # 1, C, H, W -> 1, H, W
    t_pred_label = np.argmax(t_pred_label, axis=1)
    # print("after argmax shape:", t_pred_label.shape)
    # print(t_pred_label)
    # 1, H, W -> H, W
    t_pred_label = np.squeeze(t_pred_label, axis=0)
    # h, w, _ = input_image.shape
    # dummy_data = np.random.randint(low=0, high=2, size=(w, h), dtype=np.uint8)
    dummy_data = t_pred_label.astype(np.uint8)
    pred_mask_per_frame = Image.fromarray(dummy_data)
    pred_mask_per_frame.save(mask_output_path)
    return json.dumps({'mask': mask_output_path}, indent=4)


def visualize(label_img, output_path):
    color_map = {
        # tuple means R, G. B
        0: (0, 0, 0), # background
        1: (105, 119, 35), # algae
        2: (112, 6, 20), # dead_twigs_leaves
        3: (147, 112, 219), # rubbish
        4: (230, 153, 102), # water
    }

    h, w = label_img.shape[:2]
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in color_map.items():
        color_img[label_img == i, :] = c
    Image.fromarray(color_img).save(output_path)



if __name__ == "__main__":
    args = get_parser()
    
    # Best model now: /project/train/models/2023-04-26-15:26:55_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.7752.pt
    # Best local model: /project/train/models/2023-04-26-14\:34\:28_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.6238.pt
    model = init()

    from utils.dataset import MyData
    from tqdm import tqdm
    dataset = MyData("/home/data/1945", 5, args.image_width, args.image_height, is_train=False, one_hot=False)
    total_acc = 0

    save_dir = "logs/infer"
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc="Inferencing", ascii=True):
        # image: C, H, W    label: H, W
        image, label = dataset[i]
        image = image.numpy()
        label = label.numpy()

        image = np.transpose(image, [1, 2, 0])

        output_json = process_image(model, image, 
            '{"mask_output_path": "/project/ev_sdk/mask.png"}')

        pred_label = Image.open("/project/ev_sdk/mask.png")
        pred_label = np.array(pred_label)

        # save pred label
        visualize(pred_label, os.path.join(save_dir, f"{i:05d}_pred.png"))

        # save src and label
        image.astype(np.uint8)
        Image.fromarray(image).save(os.path.join(save_dir, f"{i:05d}_src.png"))
        visualize(label, os.path.join(save_dir, f"{i:05d}_gt.png"))

        total_acc += (pred_label == label).sum() / np.prod(pred_label.shape)
    print("Test acc:", total_acc / len(dataset))