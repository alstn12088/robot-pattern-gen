import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PatternDataset(Dataset):
    def __init__(self,path,train=True,transform=None,normalize = 20):
        self.path = path
        if train:
            self.image = np.load(path+'train/train_image.npy')
            self.score = np.load(path+'train/train_score.npy')
        else:
            self.image = np.load(path+'test/test_image.npy')
            self.score = np.load(path+'test/test_score.npy')
        self.transform = transform
        self.normalize = normalize        
    def __len__(self):
        return len(self.image)
    def __getitem__(self,idx):

        if self.transform is not None:
            img = self.transform(self.image[idx]).float()
        score = torch.from_numpy(self.score[idx]).float()/self.normalize
        return img, score

