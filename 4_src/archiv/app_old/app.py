import skimage.util
import torch
from torchvision import transforms
import torchvision
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
import models
import app_helpers
import os
import streamlit as st
from skimage import io
import PIL
import toml


st.set_page_config(layout="wide")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = toml.load('./app_config.toml')
setting = config.get('settings')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)




labels_to_indices = c_f.get_labels_to_indices(dataset.targets)


import torch.nn as nn
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
        return self.net(sx)

trunk = torchvision.models.densenet121(pretrained=False)
trunk_output_size = trunk.classifier.in_features
trunk.classifier = c_f.Identity()
trunk = trunk.to('cpu')

embedder = models.MLP([trunk_output_size, 512]).to('cpu')
embedder.load_state_dict(torch.load(setting['embedder_model'],map_location=torch.device('cpu') ))
trunk.load_state_dict(torch.load(setting['trunk_model'],map_location=torch.device('cpu') ))



match_finder = MatchFinder(distance=CosineSimilarity(), threshold=setting['threshold'])




inference_model = InferenceModel(trunk,
                                 embedder=embedder,
                                 match_finder=match_finder,
                                 normalize_embeddings=True)



st.title("Papyri Finder")

col1, col2 = st.columns(2)
label = col1.selectbox("If you wan't to have an accuracy, selct label", test_labels)
show_patch_neighbours = col1.checkbox('I want to see the results of each query')
show_fragments = col1.checkbox('I want to see all relevant fragments')
uploaded_file = col2.file_uploader("")

#if (uploaded_file is not None) and (user_input is not None):
if (uploaded_file is not None):
    #inference_model.train_knn(dataset)
    #inference_model.save_knn_func("knn_func.index")
    inference_model.load_knn_func("knn_func.index")

    patches = app_helpers.get_patches(uploaded_file)
    col1, col2 = st.columns(2)

    col1.header('You uploaded images')
    col1.image(io.imread(uploaded_file))
    col2.header('Created patches')
    col2.image(skimage.util.montage(patches, channel_axis=1))

    top_1 = 0
    top_labels = []
    dist_per_patch = []
    near_patches = []
    near_labels = []

    with st.spinner('Wait for it...'):
        for j, patch in enumerate(patches):
            io.imsave('./2_temp/' + f'{int(j)}.jpg', patches[j].T)
            img = image = PIL.Image.open('./2_temp/' + f'{int(j)}.jpg')
            img = transform(img)
            img = img[None, :]
            query_img = app_helpers.imshow(torchvision.utils.make_grid(img))
            distances, indices = inference_model.get_nearest_neighbors(img, k=2)

            dist_per_patch.append(distances[0].numpy())
            near_pa = [dataset[i][0] for i in indices.cpu()[0]]
            near_pa = app_helpers.imshow(torchvision.utils.make_grid(near_pa))
            near_patches.append(near_pa)
            near_labels.append([dataset[i][1] for i in indices.cpu()[0]])


    import pandas as pd
    import plotly.express as px

    label_df = pd.DataFrame(near_labels)
    distance_df = pd.DataFrame(dist_per_patch)
    labels_list = label_df.stack().to_list()
    dist_list = distance_df.stack().to_list()
    df = pd.DataFrame(labels_list, columns=['Label'])
    df['Distance'] = dist_list
    df['Label'] = df['Label'].astype('string')
    result = str(df['Label'].mode()[0])
    prediction = f"The most likely papyri ID given the input image and the resulting patch-quers is: {result}"
    st.title(prediction)
    col1, col2 = st.columns(2)
    fig = px.histogram(df, x="Label", title='Predicted Labels')
    col1.plotly_chart(figure_or_data=fig)
    fig = px.box(df, x="Label", y="Distance", title='Median Distances')
    col2.plotly_chart(figure_or_data=fig)
    labels_list = label_df.stack().to_list()

    info_df = pd.read_csv(setting['csv'])
    result_df = info_df[info_df['label'] == int(df['Label'].mode()[0])]

    result_imgs = []
    org_imgs = result_df.original_image.unique()
    for img in org_imgs:
        result_imgs.append(io.imread('/Users/beantown/PycharmProjects/master-thesis/data/processed/03_post_cleansing/' + img))
    if show_fragments:
        for i, frag in enumerate(result_imgs):
            st.image(frag, f"Fragment #{i}")

    if show_patch_neighbours:
        for i, nearest_imgs in enumerate(near_patches):
            col1, col2 = st.columns(2)
            col1.header(f"Query {i}")
            col1.image(patches[i].T, "Input", use_column_width=True)
            col1.image(nearest_imgs, "Outout", use_column_width=True)
            col2.header(f"Labels")
            col2.write(near_labels[i])

    #class_top_1 = top_1 / (j + 1)
    #st.write(f"Top-1 Acc of Papy {user_input}: {top_1} / {j + 1} = {class_top_1}")

