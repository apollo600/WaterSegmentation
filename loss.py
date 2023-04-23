import torch
import torch.nn as nn

class FocalLoss(nn.Module):                                                                                                      
    def __init__(self, gamma=0, alpha=None, size_average=True):                                                
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N, C, H, W => N, C, H*W
            input = input.view(input.size(0), input.size(1), -1)
            # N, C, H*W => N, C, H, W     
            input = input.transpose(1, 2)     
            # N, C, H*W => N*H*W, C
            input = input.contiguous().view(-1, input.size(2))                                                                                                                                                                                                                                                                   