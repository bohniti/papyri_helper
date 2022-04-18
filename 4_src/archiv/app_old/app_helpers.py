import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from skimage import io
import skimage
import torch.nn.functional as F
from skimage import measure

def print_decision(is_match):
    if is_match:
        print("Same class")
    else:
        print("Different class")


def imshow(image, figsize=(8, 4)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )

    image = inv_normalize(image)
    npimg = image.numpy()
    return np.transpose(npimg, (1, 2, 0))


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

            if not skimage.exposure.is_low_contrast(patch, fraction_threshold=0.025, lower_percentile=1,
                                                    upper_percentile=99, method='linear'):
                selected_patches.append(patch)
                labels = np.ones_like(patch.T, dtype=int)
                variance = measure.regionprops(labels, patch.T, extra_properties=[image_stdev])[0].image_stdev
                selected_metric_values.append(variance)
                selected_patch_ids.append(patch_id)
    elif metric == 'mean':
        for patch_id, patch in enumerate(patches):
            if not skimage.exposure.is_low_contrast(patch, fraction_threshold=0.025, lower_percentile=1,
                                                    upper_percentile=99, method='linear'):
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