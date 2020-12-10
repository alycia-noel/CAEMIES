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
import matplotlib.pyplot as py
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
chest_train_dataset_dir = './data/chest_xray/train'
chest_val_dataset_dir = './data/chest_xray/val'
chest_test_dataset_dir = './data/chest_xray/test'
ciphertext_train_dataset_dir = './data/ciphertext/train'
ciphertext_val_dataset_dir = './data/ciphertext/val'
ciphertext_test_dataset_dir = './data/ciphertext/test'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Set up the dataloaders
trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_h, input_w)),
    transforms.ToTensor()
])

chest_train_set = datasets.ImageFolder(root=chest_train_dataset_dir, transform=trans)
chest_val_set = datasets.ImageFolder(root=chest_val_dataset_dir, transform=trans)
chest_test_set = datasets.ImageFolder(root=chest_test_dataset_dir, transform=trans)

ciphertext_train_set = datasets.ImageFolder(root=ciphertext_train_dataset_dir, transform=trans)
ciphertext_val_set = datasets.ImageFolder(root=ciphertext_val_dataset_dir, transform=trans)
ciphertext_test_set = datasets.ImageFolder(root=ciphertext_test_dataset_dir, transform=trans)

chest_train_loader = torch.utils.data.DataLoader(chest_train_set, batch_size, shuffle=True)
chest_val_loader = torch.utils.data.DataLoader(chest_val_set, batch_size, shuffle=True)

cipher_train_loader = torch.utils.data.DataLoader(ciphertext_train_set, batch_size, shuffle=True)
cipher_val_loader = torch.utils.data.DataLoader(ciphertext_val_set, batch_size, shuffle=True)

#pre-training

autoencoder = ConvAutoencoder(
    in_channels = 1, 
    out_channels = 1, 
    kernel_size = 3,
    device = device,
    downsampling = 1
    )

autoencoder.train_and_validate(chest_train_loader, chest_val_loader, batch_size=batch_size, lr=learning_rate, device=device, epochs=5,  which_model = 'Chest')
chest_auto_path = './pre_trained_networks/chest_autoencoder.pt'
autoencoder.save_model(chest_auto_path)

autoencoder.train_and_validate(cipher_train_loader, cipher_val_loader, batch_size=batch_size, lr=learning_rate, device=device, epochs=5,  which_model = 'Ciphertext')
cipher_auto_path = './pre_trained_networks/cipher_autoencoder.pt'
autoencoder.save_model(cipher_auto_path)



