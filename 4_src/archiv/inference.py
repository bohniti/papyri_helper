import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
import features
import models
import helpers
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
trained = False


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

path_trunk_model = "/results/11_Dataset_3_Scheduler_Densenet121_MultiSimilarityLoss_MultiSimilarityMiner/saved_models/trunk_best13.pth"
path_embedder_model = "/results/11_Dataset_3_Scheduler_Densenet121_MultiSimilarityLoss_MultiSimilarityMiner/saved_models/embedder_best13.pth"

inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
)


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


# Init Trunk Model
trunk = torchvision.models.densenet121(pretrained=False)
trunk_output_size = trunk.classifier.in_features
trunk.classifier = c_f.Identity()
trunk.load_state_dict(torch.load(path_trunk_model, map_location=torch.device('cpu')))

# Init Embedder Model
embedder = models.MLP([trunk_output_size, 64]).to(torch.device('cpu'))
embedder.load_state_dict(torch.load(path_embedder_model, map_location=torch.device('cpu')))
match_finder = MatchFinder(distance=CosineSimilarity(), threshold=1)

inference_model = InferenceModel(trunk,
                embedder=embedder,
                match_finder=match_finder,
                normalize_embeddings=True,
                knn_func=None,
                data_device=torch.device('cpu'),
                dtype=None)

# get_samples
classA, classB = labels_to_indices[14726], labels_to_indices[11457]

inference_model.train_knn(dataset)


# get 10 nearest neighbors for a car image
for img_type in [classA, classB]:
    img = dataset[img_type[2]][0].unsqueeze(0)
    print(type(img))
    print(img.shape)
    print("query image")
    helpers.imshow(torchvision.utils.make_grid(img))
    distances, indices = inference_model.get_nearest_neighbors(img, k=16)
    nearest_imgs = [dataset[i][0] for i in indices.cpu()[0]]
    nearest_labels = [dataset[i][1] for i in indices.cpu()[0]]
    print("nearest images")
    plt.title('Results')
    #helpers.imshow(torchvision.utils.make_grid(nearest_imgs))
    print(nearest_labels)

for i in range(len(classA)):
    for j in range(len(classB)):
        (x, _), (y, _) = dataset[classA[i]], dataset[classB[j]]
        #helpers.imshow(torchvision.utils.make_grid(torch.stack([x, y], dim=0)))
        decision = inference_model.is_match(x.unsqueeze(0), y.unsqueeze(0))
        helpers.print_decision(decision)

# compare two images of a different class
for i in range(len(classA)):
    (x, _), (y, _) = dataset[classA[0]], dataset[classA[i]]
    #helpers.imshow(torchvision.utils.make_grid(torch.stack([x, y], dim=0)))
    decision = inference_model.is_match(x.unsqueeze(0), y.unsqueeze(0))
    helpers.print_decision(decision)