import torch.nn as nn

class FocalLoss(nn.Module):                                                                                                      
    def __init__(self, gamma=0, alpha=None, size_average=True):                                                
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor                                                                                                                                                                                