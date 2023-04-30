import sys
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


def checkGPU():
    gpu_count = torch.cuda.device_count()
    sys.stdout.write(">>> GPU List <<<\n")
    for i in range(gpu_count):
        sys.stdout.write(f"{i+1}. {torch.cuda.get_device_name(i)}\n")


def init(model_path: str = None) -> nn.Module:

    """Initialize model
    Returns: model
    """

    checkGPU()

    sys.stdout.write("Load Model\n")
    if model_path is None:
        # /project/train/models/xxx.pt
        model_path = "真实环境所用pt"
    model = torch.load(model_path)
    model = model.eval()
    model = model.cuda()

    return model


def process_image(handle: nn.Module = None, input_image: np.ndarray = None, args: str = None, **kwargs):

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

    # Process image here
    h, w, _= input_image.shape
    input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float().cuda()
    resized_tensor= F.interpolate(input_tensor, size=512, mode='bilinear', align_corners=True)

    # 1, Classes, H, W
    pred_label = handle(resized_tensor)
    t_pred_label = pred_label.argmax(dim=1)

    # Convert tensor to PIL image
    pred_mask_per_frame = Image.fromarray(t_pred_label[0].byte().cpu().numpy())
    pred_mask_per_frame = pred_mask_per_frame.resize((w, h), Image.BILINEAR)
    pred_mask_per_frame.save(mask_output_path)

    return json.dumps({'mask': mask_output_path}, indent=4)
