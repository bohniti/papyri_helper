import streamlit as st
from PIL import Image
from efficientnet_pytorch import EfficientNet
from fpdf import FPDF
import torch.nn as nn
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import pandas as pd
import PIL
import os
import toml
from google.colab import drive
import matplotlib
import torch
import torchvision
from torchvision import datasets, transforms
from pytorch_metric_learning.testers import GlobalEmbeddingSpaceTester
import numpy as np
import cv2
from matplotlib import pyplot as plt

st.write('Papyri Inference APP')
val_path = '/Users/beantown/PycharmProjects/master-thesis/data/processed/05_val/'
df = pd.read_csv(val_path + 'processed_info.csv')
papyIDs = sorted(df.papyID.unique())

papyID = st.selectbox('Select PapyID', papyIDs)
fragmentIDs = sorted(df.loc[df['papyID'] == papyID].fragmentID.unique())
fragmentID = st.selectbox('Select PapyID', fragmentIDs)

# show Input image
input_fragments = df.loc[df['papyID'] == papyID]
input_img_path = val_path + input_fragments.loc[input_fragments['fragmentID']==fragmentID].fnames.values[0] + '.png'
image = Image.open(input_img_path)


st.image(image, caption=f'{fragmentID}th fragment of papyri with ID {papyID}')



