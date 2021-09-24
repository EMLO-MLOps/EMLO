'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import math
import glob
import pandas as pd
from utils import *


# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'model.pth'
train_data_dir = os.path.join('data', 'train')
validation_data_dir = os.path.join('data', 'validation')
nb_validation_samples = 800
epochs = 10
batch_size = 10
num_workers = 2
epochs = 3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'--- Device --- {device}')

train_transform = get_train_transform()
val_transform = get_val_transform()

train_data = datasets.ImageFolder(train_data_dir, transform=train_transform)
val_data = datasets.ImageFolder(validation_data_dir, transform=val_transform)

model = get_model(2)
model = model.to(device)
print(f'--- Model Loaded --- ')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

print(f'--- Data Loaded --- ')
train_loader = get_data_loader(train_data, batch_size, num_workers)
val_loader = get_data_loader(val_data, batch_size, num_workers, train=False)

print(f'--- Start Training --- ')
returned = train_model(model, epochs, train_loader, device, criterion, optimizer, scheduler, top_model_weights_path)

print(returned)

eval_model(model, val_loader, device)