import matplotlib
import matplotlib.pyplot as plt
from src.m1.preprocessing.build_features import create_patches_from_imgs
from src.m1.archiv.io import get_images
import cv2
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

raw_data_path = '/Users/beantown/PycharmProjects/master-thesis/data/raw/michigan_data/1_analysis/'
scale = 0.25
matplot = False

files = ['1203_C1_11V', '1203_C1_11R','1203_C1_11R']

create_patches_from_imgs(input_path=raw_data_path,
                         names=files,
                         output_path='/Users/beantown/PycharmProjects/master-thesis/data/processed/michigan/',
                         n=4,
                         gradients=False,
                         laplacian=True)

files = ['1203_C1_11R']

raw_data_paths = ['/Users/beantown/PycharmProjects/master-thesis/1_analysis/processed/michigan/8_laplace_patches/',
                  '/Users/beantown/PycharmProjects/master-thesis/1_analysis/raw/michigan_data/1_analysis/']

for i, raw_data_path in enumerate(raw_data_paths):
    laplace_images = get_images(
         raw_data_path=raw_data_path,
        file_names=files,
        rand=False,
        scaler=None,
        print_info=False,
        print_overview=False,
        label=0)

    image = laplace_images[0]
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
    plt.savefig(f'Example_Features_2_{i}.pdf',format='pdf')
    plt.show()