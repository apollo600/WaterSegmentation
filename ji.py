import sys
import json
import torch
import numpy as np
import torch.nn as nn

sys.path.append("/project/train/src_repo/")
from model.model import UNET


def checkGPU():
    gpu_count = torch.cuda.device_count()
    print(">>> GPU List <<<")
    for i in range(gpu_count):
        print(f"{i+1}. {torch.cuda.get_device_name(i)}")


def init(model_path: str = None) -> nn.Module:

    """Initialize model
    Returns: model
    """

    checkGPU()

    print("Load Model")
    model = UNET(3, 5)
    if model_path is None:
        # Best model now: /project/train/models/2023-04-26-15:26:55_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.7752.pt
        # Best local model: /project/train/models/2023-04-26-14\:34\:28_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.6238.pt
        model_path = "真实环境所用pt"
    model.load_state_dict(torch.load(model_path))
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
    t_pred_label: np.ndarray = pred_label.cpu().detach().numpy()
    # print("label shape:", t_pred_label.shape)
    # print(t_pred_label.transpose([0, 2, 3, 1]))
    # 1, C, H, W -> 1, H, W
    t_pred_label = np.argmax(t_pred_label, axis=1)
    # print("after argmax shape:", t_pred_label.shape)
    # print(t_pred_label)
    # 1, H, W -> H, W
    t_pred_label = np.squeeze(t_pred_label, axis=0)
    # h, w, _ = input_image.shape
    # dummy_
    return json.dumps({'mask': mask_output_path}, indent=4)