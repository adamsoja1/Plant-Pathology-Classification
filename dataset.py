import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np

class PlantDataset(Dataset):
    def __init__(self, path, images_path, transform=None):
        self.df = pd.read_csv(path)
        self.transform = transform
        self.images_path = images_path
   
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = plt.imread(f"{self.images_path}/{self.df['image_id'].loc[idx]}.jpg")
        image = self.transform(image=image)['image'] if self.transform is not None else image
        target = np.array(self.df['class'].loc[idx])

        image = image/255.0
        target = np.squeeze(np.eye(4)[target.reshape(-1)])
        image = F.to_tensor(image)        
        image = F.resize(image, (100, 100))

        return image.float(), torch.tensor(target)