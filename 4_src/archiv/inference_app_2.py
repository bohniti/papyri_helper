import numpy as np
import torch
from torchvision import transforms
import torchvision
from pytorch_metric_learning.utils import common_functions as c_f
import features
import models
import os
from skimage import io
import pandas as pd
import streamlit as st
import plotly.express as px

os.environ['KMP_DUPLICATE_LIB_OK']='True'
trained = False

def print_decision(is_match):
    if is_match:
        print("Same class")
    else:
        print("Different class")


mean = [0.6143, 0.6884, 0.7665]
std = [0.2909, 0.2548, 0.2122]

inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
)
# arguments must be in the specified order, matching regionprops
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=1)

def imshow(img, figsize=(8, 4)):
    img = inv_normalize(img)
    npimg = img.numpy()
    #plt.figure(figsize=figsize)
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()
    st.image(np.transpose(npimg, (1, 2, 0)), caption='Input Patches')

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)

dataset = features.PypyrusDataset(data='/Users/beantown/PycharmProjects/master-thesis/1_analysis/processed/06_dataset_3/val',
                                  csv='/Users/beantown/PycharmProjects/master-thesis/1_analysis/processed/06_dataset_3/info.csv',
                                  mode='val',
                                  transform=transform,
                                  debug=False,
                                  batch_size=64)

labels_to_indices = c_f.get_labels_to_indices(dataset.targets)

trunk = torchvision.models.densenet121(pretrained=False)
trunk_output_size = trunk.classifier.in_features
trunk.classifier = c_f.Identity()
trunk.load_state_dict(torch.load("/Users/beantown/PycharmProjects/master-thesis/results/11_Dataset_3_Scheduler_Densenet121_MultiSimilarityLoss_MultiSimilarityMiner/saved_models/trunk_best13.pth", map_location=torch.device('cpu')))
embedder = models.MLP([trunk_output_size, 64]).to(torch.device('cpu'))
embedder.load_state_dict(torch.load("/Users/beantown/PycharmProjects/master-thesis/results/11_Dataset_3_Scheduler_Densenet121_MultiSimilarityLoss_MultiSimilarityMiner/saved_models/embedder_best13.pth", map_location=torch.device('cpu')))
classifier = models.MLP([64, 50]).to(torch.device('cpu'))
classifier.load_state_dict(torch.load("/Users/beantown/PycharmProjects/master-thesis/results/11_Dataset_3_Scheduler_Densenet121_MultiSimilarityLoss_MultiSimilarityMiner/saved_models/classifier_best13.pth", map_location=torch.device('cpu')))
trunk.eval()
embedder.eval()
classifier.eval()

trunk.eval()
embedder.eval()
classifier.eval()


def get_acc(trunk, embedder, classifier, labels, img, k):
    x = trunk(img)
    x = embedder(x)
    result = classifier(x)
    result = torch.topk(result, k)
    result_labels = []
    result_probabilities = []
    for i in result.indices[0].numpy():
        result_labels.append(labels[i])

    result_probabilities = result.values[0].detach().numpy()[:k]
    return result_probabilities, result_labels


metric = 'mean'
m = 128
n=24
decending = False
cleansed = '/Users/beantown/PycharmProjects/master-thesis/1_analysis/processed/03_post_cleansing/'

# get img from dataset
#img = io.imread(cleansed + '0_15569_3592A4V.png')

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img = io.imread(uploaded_file)
    st.image(img, caption='Input Image:')

    # use torch -> cast2 tensor
    x = torch.tensor(np.array([img.T]))

    labels = list(np.unique(dataset.targets))
    distances, nearest_labels = get_acc(trunk, embedder,classifier, labels, img, k=10)

    nearest_labels
    patches_distances.append(distances)
    #patches_distances.append((distances[0].numpy()))
    #print("nearest images")
    #plt.title('Results')

    imshow(torchvision.utils.make_grid(input_patches))


    label_df = pd.DataFrame(patches_labels)
    distance_df = pd.DataFrame(patches_distances)

    label_df.to_csv('labels.csv')
    distance_df.to_csv('distances.csv')

    labels_list = label_df.stack().to_list()
    dist_list = distance_df.stack().to_list()
    df = pd.DataFrame(labels_list, columns=['Label'])
    df['Distance'] = dist_list
    df['Label'] = df['Label'].astype('string')

    fig = px.histogram(df, x="Label", title='Predicted Labels')
    st.plotly_chart(figure_or_data=fig)
    fig = px.box(df, x="Label", y="Distance",title='Median Distances')
    st.plotly_chart(figure_or_data=fig)

