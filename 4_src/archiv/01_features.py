import matplotlib.pyplot as plt
from src.m1.preprocessing.build_features import create_patches_from_imgs
from src.m1.evaluation.visualize import show_images
from src.m1.archiv.io import get_images
import cv2

raw_data_path = '/1_analysis/raw/'

matplot = False

scale = 0.25

create_patches_from_imgs(input_path=raw_data_path,
                         output_path='/1_analysis/processed/',
                         n=4,
                         gradients=True)

create_patches_from_imgs(input_path=raw_data_path,
                         output_path='/1_analysis/processed/',
                         n=4,
                         gradients=False)

files = ['Bodleian-Library-MS-Gr-class-a-1-P-1-10_00001_frame-1_edges_x',
         'Bodleian-Library-MS-Gr-class-a-1-P-1-10_00001_frame-1_edges_y']
info = show_images(raw_data_path='/Users/beantown/PycharmProjects/master-thesis/data/processed/8_gradient_patches/',
                   all=False,
                   file_names=files,
                   rand=False,
                   scaler=scale,
                   print_info=True,
                   print_overview=False,
                   matplot=matplot,
                   label=1)

files = ['Bodleian-Library-MS-Gr-class-a-1-P-1-10_00001_frame-1_edges_x',
         'Bodleian-Library-MS-Gr-class-a-1-P-1-10_00001_frame-1_edges_y']
gradient_images = get_images(raw_data_path='/Users/beantown/PycharmProjects/master-thesis/data/processed/8_gradient_patches/',
                    file_names=files,
                    rand=False,
                    scaler=None,
                    print_info=False,
                    print_overview=False,
                    label=1)

print(len(gradient_images))

files = ['Bodleian-Library-MS-Gr-class-a-1-P-1-10_00001_frame-1']
patched_images = get_images(raw_data_path='/Users/beantown/PycharmProjects/master-thesis/data/processed/8_patches/',
                            file_names=files,
                            rand=False,
                            scaler=None,
                            print_info=False,
                            print_overview=False,
                            label=1)
print(len(patched_images))

test_img_vertical = patched_images - gradient_images[0]
test_img_horizontal = patched_images - gradient_images[1]
test_img = (patched_images - gradient_images[0]) - gradient_images[1]

if matplot:
    img = cv2.cvtColor(patched_images[0], cv2.COLOR_BGR2RGB)
    test_img_vertical = cv2.cvtColor(test_img_vertical[0], cv2.COLOR_BGR2RGB)
    test_img_horizontal = cv2.cvtColor(test_img_horizontal[0], cv2.COLOR_BGR2RGB)
    test_img = cv2.cvtColor(test_img[0], cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title('Img original.')
    plt.imshow(img)
    plt.figure()
    plt.title('Img without vertical edges.')
    plt.imshow(test_img_vertical)
    plt.show()
    plt.figure()
    plt.title('Img without horizontal edges.')
    plt.imshow(test_img_horizontal)
    plt.show()
    plt.figure()
    plt.title('Img without both edges.')
    plt.imshow(test_img)
    plt.show()
else:
    print(len(patched_images))
    cv2.imshow('Original - patched', patched_images)
    cv2.imshow('Img without vertical edges.', test_img_vertical[0])
    cv2.imshow('Img without horizontal edges.', test_img_horizontal[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
