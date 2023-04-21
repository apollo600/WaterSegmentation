from torch.utils.data.dataset import Dataset

# tmp not use
class Dataset(Dataset):        
    def __init__(self, path, augmentation=False):                
        super().__init__()
        self.path = path
        self.data = []
        self.label = []

        file_list = os.listdir(self.path)
        file_prefix = [file.split]

        