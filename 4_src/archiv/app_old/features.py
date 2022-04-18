import torch
import pandas as pd
import numpy as np
import PIL
import os

class PypyrusDataset(torch.utils.data.Dataset):

    def __init__(self, data, csv, mode, transform=None, debug=False, batch_size=None):
        self.csv = csv
        self.data = data
        self.transform = transform
        df = pd.read_csv(csv)
        if mode == 'train':
            self.df = df[df.train==True]
        elif mode == 'val':
            self.df = df[df.val==True]
        elif mode == 'test':
            self.df = df[df.test==True]
        else:
            raise ValueError('Mode has to be train, val, or test.')
        if debug:
            if batch_size is not None:
                self.df = self.df.sample(n = batch_size)
            else:
                raise ValueError('To get a small dataset choose a batch_size')

        self.targets = np.array(self.df["label"])

        
    def __len__(self):
        return len(self.df)       

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img_name = os.path.join(self.data,self.df.iloc[idx, 1])
        image = PIL.Image.open(img_name)

        if self.transform:
            image = self.transform(image)         

        label = self.df.iloc[idx,2]
        return image, label