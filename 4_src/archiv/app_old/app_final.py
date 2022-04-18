import numpy as np
import PIL
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.inference import InferenceModel
import os
from skimage import io
st.set_page_config(layout="wide")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = 'cpu'
mean = [0.6143, 0.6884, 0.7665]
std = [0.2909, 0.2548, 0.2122]

# load models
trunk = torchvision.models.densenet121(pretrained=False)
trunk_output_size = trunk.classifier.in_features
trunk.classifier = common_functions.Identity()
trunk = trunk.to(device)


class MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=True):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

embedder = MLP([trunk_output_size, 512]).to(device)
embedder.load_state_dict(torch.load('/Users/beantown/PycharmProjects/master-thesis/results/7_baseline_big_big/saved_models/embedder_best21.pth',map_location=torch.device('cpu') ))
trunk.load_state_dict(torch.load('/Users/beantown/PycharmProjects/master-thesis/results/7_baseline_big_big/saved_models/trunk_best21.pth',map_location=torch.device('cpu') ))

class SquarePad:
    def __call__(self, image):
        w = image.shape[1]
        h = image.shape[2]
        max_size = 3000
        if w > max_size:
            # image = transforms.PILToTensor()(image)
            image = transforms.CenterCrop((h, max_size))(image)
        if h > max_size:
            # image = transforms.PILToTensor()(image)
            image = transforms.CenterCrop((max_size, w))(image)
        return image

val_transform = transforms.Compose([
    transforms.PILToTensor(),
    SquarePad(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)])

def get_img(img):
    mean = [0.6143, 0.6884, 0.7665]
    std = [0.2909, 0.2548, 0.2122]

    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])

    img = inv_normalize(img)
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


class PypyrusDataset(torch.utils.data.Dataset):
    def __init__(self, data, csv, transform=None, unpatched=False):
        self.csv = csv
        self.data = data
        self.transform = transform
        self.df = pd.read_csv(csv, index_col=0)
        self.unpatched = unpatched
        if self.unpatched:
            self.targets = np.array(self.df["papyID"])
        else:
            self.targets = np.array(self.df["papyri"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data, self.df.iloc[idx, 0])
        if self.unpatched:
            image = PIL.Image.open(img_name + '.png')
            label = self.df.iloc[idx, 2]
        else:
            image = PIL.Image.open(img_name)
            label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

val_dataset = PypyrusDataset(data='/Users/beantown/PycharmProjects/master-thesis/data/preprocessed/val_cleansed',
                               csv='/Users/beantown/PycharmProjects/master-thesis/data/preprocessed/val.csv',
                               transform=val_transform,
                               unpatched=True)

inference_model = InferenceModel(trunk, embedder=embedder, normalize_embeddings=True)
inference_model.load_knn_func("filename.index")

st.title("Papyri Finder")
col1, col2 = st.columns(2)
uploaded_file = col2.file_uploader("")


transform = transforms.Compose([
    transforms.ToTensor(),
    SquarePad(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=mean, std=std)])



if (uploaded_file is not None):
    st.image(io.imread(uploaded_file))
    top_1 = 0
    top_labels = []
    dist_per_patch = []
    near_patches = []
    near_labels = []

    with st.spinner('Wait for it...'):
        img = io.imread(uploaded_file)
        img = transform(img)
        img = img[None, :]
        distances, indices = inference_model.get_nearest_neighbors(img, k=100)
        nearest_imgs = [val_dataset[i][0] for i in indices.cpu()[0]]
        near_labels = [val_dataset[i][1] for i in indices.cpu()[0]]
        col1.header('Your suggestions')
        for i, nearest in enumerate(nearest_imgs):
            if near_labels[i]
            st.text(near_labels[i])
            st.image(get_img(nearest), f"Nearest  #{i}")
