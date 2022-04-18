from src.m1.archiv import get_info_df, show_info
from src.m1.archiv.utils import create_info_subset
import cv2


def show_images(raw_data_path, all=False, file_names=None, n=None, rand=False, scaler=None, print_info=True,
                print_overview=True, matplot=True, label=None):
    """
    Functions print images with cv2.imshow.
    Note, matplotlib tends to not show full resolution, this is why it's usage is omitted here.
    Args:
        matplot: ust matplotlib to print (lower resultion but works within jupyter notebook)
        raw_data_path: absolut path to the raw 1_analysis
        all: show all images
        file_names: how subset determined by file names
        n: show first n images if rand is false; else shows random n images
        rand: see n
        scaler: determines the value by which you want to up- or downsclae the img before it is shown
        print_info: use project specific info file/ function to print info for each file
        print_overview: print general information about the whole dataset
    """

    # Load info file or create one
    info = get_info_df(raw_data_path, overwrite_info=False)

    if print_info and print_overview:
        show_info(data_path=raw_data_path, file_names=file_names, n=n, rand=rand, overview=True, label=label)

    elif print_info and not print_overview:
        show_info(data_path=raw_data_path, file_names=file_names, n=n, rand=rand, overview=False, label=label)

    # Print all images
    subset_info = create_info_subset(info, all=all, file_names=file_names, rand=rand, label=label)

    for index, row in subset_info.iterrows():
        name = row['names'] + '.' + str(row['labels']) + '.jpg'
        show_image(raw_data_path=raw_data_path, name=name, label=row['labels'], scaler=scaler, matplot=matplot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image(raw_data_path, name, label=0, scaler=None, matplot=True):
    """
    Builtin-function used to print on image with corresponding headline.
    Args:
        raw_data_path: see show_images
        name: file name
        label: label in the format raw_dat/name.label.file_type specifies if the img is a fragment of a papyri.
        scaler:see show_images
    """
    img = cv2.imread(raw_data_path + name, 1)

    if scaler is None:
        if matplot:
            plt.title(f'{name}\nOriginal Size\n{label = }')
            plt.imshow(img)
            plt.show()
        else:
            cv2.imshow(f'{name}\nOriginal Size\n{label = }', img)

    else:
        img_scaled = cv2.resize(img, (0, 0), fx=scaler, fy=scaler)
        if scaler > 1:
            title_string = f'{name}\nUpscale by {(scaler - 1) * 100}%\n{label = }'
        else:
            title_string = f'{name}\nDownscaled by {(1 - scaler) * 100}%\n{label = }'
        if matplot:
            plt.title(title_string)
            plt.imshow(img_scaled)
            plt.show()
        else:
            cv2.imshow(title_string, img_scaled)
