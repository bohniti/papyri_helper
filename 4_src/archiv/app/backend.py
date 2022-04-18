from pathlib import Path
import torch
import torchvision
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.inference import InferenceModel
from torchvision import transforms
import models
import helpers
import features
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


output_dir = '/Users/beantown/PycharmProjects/master-thesis/results/7_baseline_big_big'
embedding_space = '512'
device = 'cpu'
means = [0.6143, 0.6884, 0.7665]
std = [0.2909, 0.2548, 0.2122]
class_label = 15110
knn_model = Path('knn_model_2.index')
n = 20


def load_dataset(samples, labels):
    # transform the dataset
    val_transform = transforms.Compose([transforms.PILToTensor(),
                                        helpers.SquarePad(),
                                        transforms.ConvertImageDtype(torch.float),
                                        transforms.Normalize((means[0], means[1], means[2]), (std[0], std[1], std[2]))])

    # create a pytorch dataset
    return features.PypyrusDataset(data=samples, csv=labels, transform=val_transform, unpatched=False)


def load_indices_and_labels(val_dataset):
    # retrieve labels from the dataset
    labels_to_indices = common_functions.get_labels_to_indices(val_dataset.targets)
    labels = list(labels_to_indices.keys())
    return labels_to_indices, labels


def get_samples_images(dataset, query_class, i):
    query_img = dataset[query_class[i]][0].unsqueeze(0)
    visualize_img = helpers.save_img(torchvision.utils.make_grid(query_img), 'query.png')
    return query_img, visualize_img


def load_train_model(dataset):
    p = Path(output_dir + '/saved_models/').glob('**/trunk_best*')
    trunk_model = [x for x in p if x.is_file()][0]
    trunk = torchvision.models.densenet121(pretrained=False)
    trunk_output_size = trunk.classifier.in_features
    trunk.classifier = common_functions.Identity()
    trunk.load_state_dict(torch.load(trunk_model, map_location=torch.device('cpu')))

    # define and load embedder model
    p = Path(output_dir + '/saved_models/').glob('**/embedder_best*')
    embedder_model = [x for x in p if x.is_file()][0]
    embedder = models.MLP([trunk_output_size, embedding_space]).to(device)
    embedder.load_state_dict(torch.load(embedder_model, map_location=torch.device('cpu')))
    inference_model = InferenceModel(trunk, embedder=embedder, normalize_embeddings=True)

    # if model has been trained load, else train
    if knn_model.is_file():
        #print('yes')
        inference_model.load_knn_func("knn_model.index")
    else:
        inference_model.train_knn(dataset, batch_size=1)
        inference_model.save_knn_func("knn_model_2.index")


def get_predictions(dataset, query_img, K):
    p = Path(output_dir + '/saved_models/').glob('**/trunk_best*')
    trunk_model = [x for x in p if x.is_file()][0]
    trunk = torchvision.models.densenet121(pretrained=False)
    trunk_output_size = trunk.classifier.in_features
    trunk.classifier = common_functions.Identity()
    trunk.load_state_dict(torch.load(trunk_model, map_location=torch.device('cpu')))

    # define and load embedder model
    p = Path(output_dir + '/saved_models/').glob('**/embedder_best*')
    embedder_model = [x for x in p if x.is_file()][0]
    embedder = models.MLP([trunk_output_size, embedding_space]).to(device)
    embedder.load_state_dict(torch.load(embedder_model, map_location=torch.device('cpu')))
    inference_model = InferenceModel(trunk, embedder=embedder, normalize_embeddings=True)

    # if model has been trained load, else train
    if knn_model.is_file():
        # print('yes')
        inference_model.load_knn_func("knn_model.index")
    else:
        print('train model')
        inference_model.train_knn(dataset, batch_size=1)
        inference_model.save_knn_func("knn_model_2.index")

    distances, indices = inference_model.get_nearest_neighbors(query_img, k=K)
    nearest_imgs = [dataset[i][0] for i in indices.cpu()[0]]
    results = []
    for i, image in enumerate(nearest_imgs):
        results.append(helpers.save_img(image, f'{i}_nearest_image.png'))
    return results

