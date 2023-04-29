import sys
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn


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
    if model_path is None:
        # Best model now: /project/train/models/2023-04-26-15:26:55_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.7752.pt
        # Best local model: /project/train/models/2023-04-26-14\:34\:28_epoch-100_lr-0.0005_loss-CrossEntropy_optim-AdamW_best_acc-0.6238.pt
        model_path = "真实环境所用pt"
    model = torch.load(model_path)
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
    ih, iw, _= input_image.shape
    h = w = 512

    # BGR -> RGB
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # 添加灰边
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv2.resize(input_image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    new_image[int((h-nh)/2):int((h-nh)/2)+nh, int((w-nw)/2):int((w-nw)/2)+nw] = image

    image = Image.fromarray(input_image)
    image = image.resize((512, 512), Image.BILINEAR)
    image = np.array(image)
    # H, W, C -> C, H, W
    image = np.transpose(image, [2, 0, 1])
    # C, H, W -> 1, C, H, W
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    image = image.cuda()

    # 1, Classes, H, W
    pred_label = handle(image)

    # Generate mask data
    t_pred_label: np.ndarray = pred_label.cpu().detach().numpy()
    # 1, C, H, W -> 1, H, W
    t_pred_label = np.argmax(t_pred_label, axis=1)
    # 1, H, W -> H, W
    t_pred_label = np.squeeze(t_pred_label, axis=0)
    pred_mask_per_frame = Image.fromarray(np.uint8(t_pred_label))
    pred_mask_per_frame = pred_mask_per_frame.resize((w, h), Image.BILINEAR)
    pred_mask_per_frame.save(mask_output_path)

    return json.dumps({'mask': mask_output_path}, indent=4)
