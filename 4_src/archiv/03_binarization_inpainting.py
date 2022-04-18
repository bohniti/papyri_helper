import matplotlib
import matplotlib.pyplot as plt
from src.m1.archiv.io import get_images
import cv2
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

raw_data_path = '/Users/beantown/PycharmProjects/master-thesis/data/raw/michigan_data/1_analysis/'
scale = 0.25
matplot = False

files = ['Bodleian-Library-MS-Gr-class-a-1-P-1-10_00006_frame-6']

raw_data_path = '/Users/beantown/PycharmProjects/master-thesis/data/processed/16_patches/'

images = get_images(
    raw_data_path=raw_data_path,
    file_names=files,
    rand=False,
    scaler=None,
    print_info=False,
    print_overview=False,
    label=14)

image = images[0]

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

matplotlib.rcParams['font.size'] = 9
binary_global = image > threshold_otsu(image)

window_size = 25
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(image, window_size=window_size)

binary_niblack = image > thresh_niblack
binary_sauvola = image > thresh_sauvola

plt.figure(figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Global Threshold')
plt.imshow(binary_global, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_niblack, cmap=plt.cm.gray)
plt.title('Niblack Threshold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.title('Sauvola Threshold')
plt.axis('off')
#plt.show()

print(binary_sauvola.shape)

import numpy as np

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

mask = np.where((binary_sauvola == 0) | (binary_sauvola == 1), binary_sauvola ^ 1, binary_sauvola)

mask = mask[:550, :800]
mask = pad_along_axis(mask, image.shape[0], 0)
mask = pad_along_axis(mask, image.shape[1], 1)
mask = mask.reshape((1,mask.shape[0],mask.shape[1]))
mask = np.array(mask, dtype=np.float32)
#plt.imshow(mask)
#plt.show()

dst = cv2.inpaint(src=image, inpaintMask=mask, inpaintRadius=1, flags = cv2.INPAINT_NS)
plt.imshow(dst)
plt.show()
