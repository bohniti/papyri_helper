import os
import matplotlib.pyplot as plt
from cycler import cycler
import logging
import numpy as np

def create_output_dir(name, experiment_name, x=1):
    while True:
        dir_name = (name + (str(x) + '_' if x is not 0 else '') + '_' + experiment_name).strip()
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

            return dir_name
        else:
            x = x + 1


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, epoch, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=12)

    plt.title(f'{split_name} umap {epoch}')

    if split_name == 'val':        
        plt.savefig(f"./umap_val/epoch_{epoch}.png")
    else:
        plt.savefig(f"./umap_train/epoch_{epoch}.png")