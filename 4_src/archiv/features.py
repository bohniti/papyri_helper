import torch
import pandas as pd
import numpy as np
import PIL
import os
from skimage import io
import skimage
import torch.nn.functional as F
from skimage import measure
import pandas as pd

class PypyrusDataset(torch.utils.data.Dataset):

    def __init__(self, data, csv, mode, transform=None, debug=False, batch_size=None, data_2=None):
        self.csv = csv
        self.data = data
        self.data_2 = data_2
        self.transform = transform
        df = pd.read_csv(csv)
        if mode == 'train':
            self.df = df[df.train==True]
        elif mode == 'val':
            self.df = df[df.val==True]
        elif mode == 'inference':
            self.df = df[(df.val==True) | ((df.train==True)) ]
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
        
        try:
            img_name = os.path.join(self.data,self.df.iloc[idx, 1])
            image = PIL.Image.open(img_name)
        except:
            img_name = os.path.join(self.data_2,self.df.iloc[idx, 1])
            image = PIL.Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)         

        label = self.df.iloc[idx,2]
        return image, label

def get_patches(uploaded_file, m=128, n=20, metric='mean', decending=False):
    img = io.imread(uploaded_file)

    # use torch -> cast2 tensor
    x = torch.tensor(np.array([img.T]))

    # define patch size properties for padding and unfolding
    kc, kh, kw = 3, m, m  # kernel size
    dc, dh, dw = 3, m, m  # stride

    # Pad to multiples of m. If image is too small -> go to next image
    try:
        x = F.pad(x, (x.size(2) % kw // 2, x.size(2) % kw // 2,
                      x.size(1) % kh // 2, x.size(1) % kh // 2,
                      x.size(0) % kc // 2, x.size(0) % kc // 2), mode='constant', value=255)

        # create_patches
        patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.contiguous().view(-1, kc, kh, kw)
    except:
        raise ValueError(f'The imag is too small on at least one dimension. Image has to be at leas {m}')

    org_img_labels = labels = np.ones_like(img, dtype=int)
    mean_img = measure.regionprops(org_img_labels, img)[0].intensity_mean

    # evaluate patch -> cast2 numpy
    patches = patches.numpy()
    selected_metric_values = []
    selected_patches = []
    selected_patch_ids = []

    if metric == 'variance':
        for patch_id, patch in enumerate(patches):
            # check if patch fits to metric

            if skimage.exposure.is_low_contrast(patch, fraction_threshold=0.025, lower_percentile=1,
                                                upper_percentile=99, method='linear') == False:
                selected_patches.append(patch)
                labels = np.ones_like(patch.T, dtype=int)
                variance = measure.regionprops(labels, patch.T, extra_properties=[image_stdev])[0].image_stdev
                selected_metric_values.append(variance)
                selected_patch_ids.append(patch_id)
    elif metric == 'mean':
        for patch_id, patch in enumerate(patches):
            if skimage.exposure.is_low_contrast(patch, fraction_threshold=0.025, lower_percentile=1,
                                                upper_percentile=99, method='linear') == False:
                selected_patches.append(patch)
                labels = np.ones_like(patch.T, dtype=int)
                intensity_mean = measure.regionprops(labels, patch.T)[0].intensity_mean
                intensity_mean = abs(intensity_mean - mean_img)
                selected_metric_values.append(intensity_mean)
                selected_patch_ids.append(patch_id)

    # sort
    selected_metric_values = np.array(selected_metric_values)
    idx = np.argsort(selected_metric_values)
    selected_patches = np.array(selected_patches)[idx]
    selected_patch_ids = np.array(selected_patch_ids)[idx]

    # get the first n patches which have the HIGHEST metric (e.g. variance)
    if decending:
        results = selected_patches[selected_patches.shape[0] - n:]
    else:
        results = selected_patches[:n]
    return results