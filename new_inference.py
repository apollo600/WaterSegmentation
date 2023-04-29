import os
import sys
sys.path.append("/project/ev_sdk/src")
import ji


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
        image       = self.get_miou_png(image)
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