import sys
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import cv2
import torch.nn.functional as F
sys.path.append("/project/train/src_repo")


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
        model_path = "/project/train/models/best_epoch_weights.pth"
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

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.full((h, w, 3), 128, dtype=np.uint8)
    new_image[int((h-nh)/2):int((h-nh)/2)+nh, int((w-nw)/2):int((w-nw)/2)+nw] = image

    # 添加上 batch_size 维度
    new_image = np.expand_dims(np.transpose(np.array(new_image, np.float32) / 255.0, (2, 0, 1)), 0)

    with torch.no_grad():
        input_tensor = torch.from_numpy(new_image).cuda()
            
        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        pred_label = handle(input_tensor)[0]

        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pred_label = F.softmax(pred_label.permute(1,2,0),dim = -1).cpu().numpy()

        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        pred_label = pred_label[int((h - nh) // 2) : int((h - nh) // 2 + nh), \
                int((w - nw) // 2) : int((w - nw) // 2 + nw)]
        
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pred_label = cv2.resize(pred_label, (iw, ih), interpolation = cv2.INTER_LINEAR)

        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pred_label = pred_label.argmax(axis=-1)

        # np.array -> PIL Image
        pred_mask_per_frame = Image.fromarray(np.uint8(pred_label))

        # save
        pred_mask_per_frame.save(mask_output_path)

    return json.dumps({'mask': mask_output_path}, indent=4)
