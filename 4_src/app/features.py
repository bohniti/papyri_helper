import torch
import pandas as pd
import numpy as np
import PIL
import os
import torch.nn.functional as F
import pandas as pd

class PypyrusDataset(torch.utils.data.Dataset):
    def __init__(self, data, csv, transform=None, unpatched=False):
        self.csv = csv
        self.data = data
        self.transform = transform
        self.df = pd.read_csv(csv, index_col=0)
        self.unpatched = unpatched
        if self.unpatched:
            self.targets = np.array(self.df["papyID"])
        else:
            self.targets = np.array(self.df["papyri"])
        
        
        
    def __len__(self):
        return len(self.df)       

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data,self.df.iloc[idx, 0])
        if self.unpatched:
            image = PIL.Image.open(img_name + '.png')
            label = self.df.iloc[idx,2]            
        else:
            image = PIL.Image.open(img_name)
            label = self.df.iloc[idx,1]            
        
        if self.transform:
            image = self.transform(image)

        
        return image, label
