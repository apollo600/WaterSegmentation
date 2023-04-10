from torch.utils.data.dataset import Dataset

class Dataset(Dataset):        
    def __init__(self, path, augmentation=False):                
        super().__init__()
        self.path = path

        self.data = 