import os
import sys
sys.path.append("/project/ev_sdk/src")
import ji
import numpy as np
import torch
import cv2
from PIL import Image
from model2.utils.utils import cvtColor, preprocess_input, resize_image


def get_miou_png(image, input_shape):
                    #---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #---------------------------------------------------------#
    image       = cvtColor(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    #---------------------------------------------------------#
    #   给图像增加灰条，实现不失真的resize
    #   也可以直接resize进行识别
    #---------------------------------------------------------#
    image_data, nw, nh  = resize_image(image, (input_shape[1],input_shape[0]))
    #---------------------------------------------------------#
    #   添加上batch_size维度
    #---------------------------------------------------------#
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()
            
        #---------------------------------------------------#
        #   图片传入网络进行预测
        #---------------------------------------------------#
        pr = self.net(images)[0]
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        #--------------------------------------#
        #   将灰条部分截取掉
        #--------------------------------------#
        pr = pr[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh), \
                int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        #   进行图片的resize
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        #   取出每一个像素点的种类
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)

    image = Image.fromarray(np.uint8(pr))
    return image


def calc_miou(dataset_path, miou_out_path, image_ids):
                                                                                            
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
        image       = get_miou_png(image)
        image.save(os.path.join(pred_dir, image_id + ".png"))
                
    print("Calculate miou.")
    _, IoUs, _, _ = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes, None)  # 执行计算mIoU的函数
    temp_miou = np.nanmean(IoUs) * 100

    self.mious.append(temp_miou)
    self.epoches.append(epoch)

    with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
        f.write(str(temp_miou))
        f.write("\n")
    
    plt.figure()
    plt.plot(self.epoches, self.mious, 'red', linewidth = 2, label='train miou')

    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Miou')
    plt.title('A Miou Curve')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
    plt.cla()
    plt.close("all")

    print("Get miou done.")
    shutil.rmtree(self.miou_out_path)