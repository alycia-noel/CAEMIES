# -*- coding: utf-8 -*-
"""
Main script to pre-train the autoencoders
Created on Mon Nov 16 23:05:38 2020

@author: Alycia
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from autoenc import ConvAutoencoder

from timeit import default_timer as timer

''' Parameters '''
input_w = 256
input_h = 256
input_nc = 3
train_split = 0.7
val_split = 0.3
batch_size = 32
learning_rate = 0.002
horse_dataset_dir = './data/horse2zebra/train'
ciphertext_dataset_dir = './data/ciphertext/train'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Train-val-test split function
def train_val_test_split(dataset, splits):
    datasets = {}

    # Two-way split
    if len(splits) == 2:
        first_idx, second_idx = train_test_split(list(range(dataset.__len__())), test_size=splits[1])
        datasets['first'] = Subset(dataset, first_idx)
        datasets['second'] = Subset(dataset, second_idx)
        
    return datasets

# Set up the dataloaders
trans = transforms.Compose([
    transforms.Resize((input_h, input_w)),
    transforms.ToTensor()
])

horse_files = datasets.ImageFolder(root=horse_dataset_dir, transform=trans)
horse_split = train_val_test_split(horse_files, splits=[train_split, val_split])

ciphertext_files = datasets.ImageFolder(root=ciphertext_dataset_dir, transform=trans)
ciphertext_split = train_val_test_split(ciphertext_files, splits=[train_split, val_split])

horse_train_set = horse_split['first']
horse_val_set = horse_split['second']

ciphertext_train_set = ciphertext_split['first']
ciphertext_val_set = ciphertext_split['second']

#train_loader = torch.utils.data.DataLoader(horse_train_set, batch_size, shuffle=True)
#val_loader = torch.utils.data.DataLoader(horse_val_set, batch_size, shuffle=True)

train_loader = torch.utils.data.DataLoader(ciphertext_train_set, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(ciphertext_val_set, batch_size, shuffle=True)

autoencoder = ConvAutoencoder(
    in_channels = 3, 
    out_channels = 3, 
    kernel_size = 3,
    device = device,
    downsampling = 1
    )

autoencoder.train_and_validate(train_loader, val_loader, batch_size=batch_size, lr=learning_rate, device=device, epochs=50,  which_model = 'Ciphertext')