from tqdm import tqdm
import numpy as np
import torch
import os
from utils import visual
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


def Unet_trainer_one_epoch(train_loader: DataLoader, val_loader: DataLoader, train_model: nn.Module, args, criterion, optimizer):        
    for epoch in range(init_epoch):
        batches = len(train_loader)
        pbar = tqdm(total=batches, desc=f"Epoch {epoch+1}/{init_epoch}: ", maxinterval=0.3, ascii=True)

        for iteration, (data, label) in enumerate(train_loader):

            data, label = data.cuda(), label.cuda()
            # label: N, H, W; pred_label: N, C, H, W
            pred_label = train_model(data)

            if args.loss == "CrossEntropy":
                # N, C, H, W => C, N*H*W
                pred_label = pred_label.contiguous().permute(0, 2, 3, 1)
                pred_label = pred_label.reshape(-1, pred_label.size(3))
                # N, C, H, W => C*N*H*W
                label = label.view(-1)
            else:
                pass
            loss = criterion(pred_label, label)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # copy the tensor to host memory first
            t_pred_label = pred_label.cpu().detach().numpy()
            t_label = label.cpu().detach().numpy()
            # get max arg as output label
            t_pred_label = np.argmax(t_pred_label, axis=1)
            if args.loss == "CrossEntropy":
                pass
            elif args.loss == "Focal":
                t_label = np.transpose(t_label, [0, 3, 1, 2]).argmax(axis=1)
            # update accuracy
            acc = np.sum(t_label == t_pred_label) / np.prod(t_label.shape)

            pbar.set_description(f"Epoch {epoch+1}/{init_epoch} loss: {loss:.4f} train_acc: {acc:.4f}")
            pbar.update(1)
        pbar.close()

        print("Start Test")
        with torch.no_grad():
            batches = len(val_loader)
            total_acc = 0
            pbar = tqdm(total=batches, maxinterval=0.3, ascii=True)

            for iteration, (data, label) in enumerate(val_loader):

                pred_label = train_model(data.cuda())

                # copy the tensor to host memory first
                t_pred_label = pred_label.cpu().detach().numpy()
                t_label = label.detach().numpy()
                # get max arg as output label
                t_pred_label = np.argmax(t_pred_label, axis=1)
                if args.loss == "CrossEntropy":
                    pass
                elif args.loss == "Focal":
                    t_label = np.transpose(t_label, [0, 3, 1, 2]).argmax(axis=1)
                # update accuracy
                acc = np.sum(t_label == t_pred_label) / np.prod(t_label.shape[1:])
                total_acc += acc

                # visual pictures
                max_picture_shown_each_epoch = 3
                if args.log_visual and iteration < max_picture_shown_each_epoch:
                    log_path = os.path.join(args.log_root, args.log_dir)
                    os.makedirs(log_path, exist_ok=True)
                    visual.visualize(np.squeeze(t_pred_label, axis=0), os.path.join(log_path, f"e{epoch}_i{iteration}_pred.png"))
                    visual.visualize(np.squeeze(t_label, axis=0), os.path.join(log_path, f"e{epoch}_i{iteration}_label.png"))
                    Image.fromarray(np.squeeze(np.uint8(data.detach().numpy()), axis=0).transpose([1, 2, 0])).save(os.path.join(log_path, f"e{epoch}_i{iteration}_src.png"))

                pbar.update(1)
            pbar.close()

            total_acc /= batches
            if total_acc > best_acc: 
                print(f"Update acc {best_acc:.4f} => {total_acc:.4f}")
                best_acc = total_acc
                if args.save_dir == "":
                    model_path = os.path.join(log_dir, f"best_acc-{total_acc:.4f}.pt")
                else:
                    model_path = os.path.join(log_dir, f"{time_stamp}_epoch-{args.epoch}_lr-{args.lr}_loss-{args.loss}_optim-{args.optimizer}_best_acc-{total_acc:.4f}.pt")
                torch.save(train_model, model_path)
            else:
                print(f"acc: {total_acc:.4f}")
