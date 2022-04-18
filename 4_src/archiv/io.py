from pathlib import Path
from PIL import Image
import numpy as np
from tabulate import tabulate
import pandas as pd
import cv2
from src.m1.archiv.utils import create_info_subset


def get_info_df(data_path, n=None, overwrite_info=False):
    """
    Function reads or creates a .csv file in given dir.
    The file is for creating a pandas.DataFrame out of it.
    The dataframe contains the name, width, height and the label of each image in a given dir.

    Args
        raw_data_path: absolut path to the raw 1_analysis
        n: use n if you want to create df just for first n! Not recommended!!!


    Returns
        pandas.DataFrame: info df can be used to read and write 1_analysis and to calculate naive statistics.
    """

    info_file = Path(data_path + 'info_file.csv')

    if info_file.is_file() and not overwrite_info:
        print(f'\nRead info from {str(info_file)} ...')
        info = pd.read_csv(data_path + 'info_file.csv')
        return info

    else:
        print(f'Start collecting info from {data_path} ...')
        p = Path(data_path).glob('**/*.jpg')
        files = np.array([x for x in p if x.is_file()])
        heights = []
        widths = []
        names = []
        labels = []

        for file in files:
            im = Image.open(file)
            width, height = im.size
            heights.append(height)
            widths.append(width)
            file_splits = str(file).split('/')[-1]
            file_splits = str(file_splits).split('.')
            names.append(file_splits[0])
            labels.append(int(file_splits[1]))

        info = pd.DataFrame()
        info['names'] = names
        info['widths'] = widths
        info['heights'] = heights
        info['labels'] = labels

        print(f'Write info to info_file.csv')
        info.to_csv(data_path + 'info_file.csv')
        return info


def show_info(data_path, overview=True, file_names=None, n=None, rand=False, overwrite_info=False, label=None,
              print_all=False):
    """
    Function uses the an project specific pandas df given by a info.csv file to print img or dataset information.
    Note,Just use filenames or n and rand parameters. File_names creats a subset out of the specified files.

    Args
        raw_data_path: path to 1_analysis (.csv file)
        overview: if functions calculates and print naive statistics about the dataset
        file_names: if function prints sample-specific information
        n: prints n file-information strings
        rand: if true n doesn't print the first but instead random sample-information
    """
    info = get_info_df(data_path=data_path, overwrite_info=overwrite_info)

    if overview:
        nr_of_samples = info.shape[0]
        nr_of_labels = str(info['labels'].value_counts())
        max_width = info['widths'].max()
        max_height = info['heights'].max()
        median_width = info['widths'].median()
        median_height = info['heights'].median()
        mean_width = info['widths'].mean()
        mean_height = info['heights'].mean()
        print(
            f'\n\033[1mDataset Overview\033[0m\n'
            f'\n{  nr_of_samples = }'
            f'\n{  nr_of_labels = }'
            f'\n{  max_width = }'
            f'\n{  max_height = }'
            f'\n{  median_width = }'
            f'\n{  median_height = }'
            f'\n{  mean_width = }'
            f'\n{  mean_height = }\n')

    subset_info = create_info_subset(info, all=print_all, file_names=file_names, rand=rand, label=label, n=n)

    if file_names is not None:
        print(
            f'\n\033[1mFile Info\033[0m\n\n' + tabulate(subset_info, headers='keys', tablefmt='github',
                                                        showindex=False))
    elif n is not None and not rand:
        print('\n\033[1mFile Info\033[0m\n\n' + tabulate(subset_info, headers='keys', tablefmt='github',
                                                         showindex=False))
    elif n is not None and rand:
        print('\n\033[1mFile Info\033[0m\n\n' + tabulate(subset_info, headers='keys', tablefmt='github',
                                                         showindex=False))

    return info


def get_images(raw_data_path, print_all=False, file_names=None, n=None, rand=False, scaler=None, print_info=True,
               print_overview=True, label=0):
    """
    Functions print images with cv2.imshow.
    Note, matplotlib tends to not show full resolution, this is why it's usage is omitted here.

    Args:
        label: Just get img with defined label
        matplot: ust matplotlib to print (lower resultion but works within jupyter notebook)
        raw_data_path: absolut path to the raw 1_analysis
        print_all: show all images
        file_names: how subset determined by file names
        n: show first n images if rand is false; else shows random n images
        rand: see n
        scaler: determines the value by which you want to up- or downsclae the img before it is shown
        print_info: use project specific info file/ function to print info for each file
        print_overview: print general information about the whole dataset
    """

    def get_image(raw_data_path, name, scaler=None):

        img = cv2.imread(raw_data_path + name, 1)

        if scaler is None:
            return img

        else:
            img_scaled = cv2.resize(img, (0, 0), fx=scaler, fy=scaler)
            return img_scaled

    # Load info file or create one
    info = get_info_df(raw_data_path)

    if print_info and print_overview:
        show_info(data_path=raw_data_path, file_names=file_names, n=n, rand=rand, overview=True)

    elif print_info and not print_overview:
        show_info(data_path=raw_data_path, file_names=file_names, n=n, rand=rand, overview=False)

    subset_info = create_info_subset(info, all=print_all, file_names=file_names, rand=rand, label=label)

    imgs = []

    for index, row in subset_info.iterrows():
        name = row['names'] + '.' + str(row['labels']) + '.jpg'
        img = get_image(raw_data_path=raw_data_path, name=name, scaler=scaler)
        imgs.append(img)

    return imgs
