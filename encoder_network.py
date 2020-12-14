# -*- coding: utf-8 -*-
"""
Encryption script: takes in a medical image and produces a ciphertext
Created on Wed Dec  9 13:49:49 2020

@author: ancarey
"""

from __future__ import division
import os
import time
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from functools import partial
from collections import OrderedDict
import itertools
from torch.backends import cudnn
import torch.cuda
import torchvision.utils as v_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from autoenc import ConvAutoencoder
import torchvision.datasets as datasets
from sklearn.metrics import f1_score
from torchvision.utils import save_image, make_grid
from PIL import Image

input_w = 256
input_h = 256
input_nc = 3
train_split = 0.7
val_split = 0.3
batch_size = 32
learning_rate = 0.001
chest_train_dataset_dir = './data/chest_xray/train'
chest_val_dataset_dir = './data/chest_xray/val'
chest_test_dataset_dir = './data/chest_xray/test'
ciphertext_train_dataset_dir = './data/ciphertext/train'
ciphertext_val_dataset_dir = './data/ciphertext/val'
ciphertext_test_dataset_dir = './data/ciphertext/test'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
chest_auto_path = './pre_trained_networks/chest_autoencoder.pt'
cipher_auto_path = './pre_trained_networks/cipher_autoencoder.pt'

Tensor = torch.cuda.FloatTensor 

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
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((input_h, input_w)),
    transforms.ToTensor()
])

chest_train_set = datasets.ImageFolder(root=chest_train_dataset_dir, transform=trans)
chest_val_set = datasets.ImageFolder(root=chest_val_dataset_dir, transform=trans)
chest_test_set = datasets.ImageFolder(root=chest_test_dataset_dir, transform=trans)

ciphertext_files = datasets.ImageFolder(root=ciphertext_train_dataset_dir, transform=trans)
ciphertext_val_files = datasets.ImageFolder(root=ciphertext_val_dataset_dir, transform=trans)

#ciphertext_train_set = ciphertext_split['first']
#ciphertext_val_set = ciphertext_split['second']
ciphertext_test_set = datasets.ImageFolder(root=ciphertext_test_dataset_dir, transform=trans)

#encryption_train_set = chest_train_set + ciphertext_train_set
#encryption_val_set = chest_val_set + ciphertext_val_set
#encryption_test_set = chest_test_set + ciphertext_test_set

train_loader = torch.utils.data.DataLoader(chest_train_set, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(chest_val_set, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(chest_test_set, batch_size, shuffle=True)

cipher_loader = torch.utils.data.DataLoader(ciphertext_files, batch_size, shuffle=True)
cipher_val_loader = torch.utils.data.DataLoader(ciphertext_val_files, batch_size, shuffle=True)

def load_pretrained_models(path):
    network = ConvAutoencoder(in_channels = 1, 
                             out_channels = 1, 
                             kernel_size = 3,
                             device = device,
                             downsampling = 1
                            )
    network.load_state_dict(torch.load(chest_auto_path))
    return(network)

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        x1 = self.model(x)
        return x1
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #load pre-trained models 
        chest_auto = load_pretrained_models(chest_auto_path)
        cipher_auto = load_pretrained_models(cipher_auto_path)
        
        self.encode = chest_auto.encoder
        self.decode = cipher_auto.decoder
        self.discriminator = discriminator
        #feature mapping layer 
        self.feature_map = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          )
        
    def forward(self, x):
        x1 = self.encode(x) # [32, 256, 64, 64]
        x2 = self.feature_map(x1)
        x3 = x2 + x1
        x4 = self.decode(x3)
        return(x4)
     
class cyclegan(nn.Module):
    def __init__(self, batch_size, fine_size, input_nc, output_nc, L1_lambda, dataset_dir):
        super(cyclegan, self).__init__()
            self.batch_size = batch_size
            self.image_size = fine_size
            self.input_c_dim = input_nc
            self.output_c_dim = output_nc
            self.L1_lambda = L1_lambda
            self.dataset_dir = dataset_dir
            
            self.dicriminator = Discriminator
            self.generator = Generator 
            
            self.criterionGAN = nn.L1Loss()
        
            self._build_model()

        def _build_model(self):
            Tensor = torch.cuda.FloatTensor
            real_chest = Tensor(batchSize, input_nc, size, size)
            real_cipher = Tensor(batchSize, output_nc, size, size)
            target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
            target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)
                 
            self.fake_cipher = self.generator(self.real_chest)
            self.fake_chest_ = self.generator(self.fake_cipher)
            self.fake_chest = self.generator(self.real_cipher)
            self.fake_cipher_ = self.generator(self.fake_chest)
     
            self.DB_fake = self.discriminator(self.fake_cipher)
            self.DA_fake = self.discriminator(self.fake_chest)
            
            self.g_loss_chest2cipher = self.criterionGAN(self.DB_fake, target_real)     
            self.g_loss_cipher2chest = self.criterionGAN(self.DA_fake, target_real)
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
    # def train_and_val(self, trainLoader, validLoader, cipherLoader, cipherValidLoader, batch_size, lr, device, epochs=10):
    #     print("====== Train Encoder Network ======")
    #     optimizer = optim.Adam(self.parameters(), lr = 0.002, betas=(.5, .999))
    #     loss_function = nn.MSELoss()
    #     self.to(device)
        
    #     train_losses = []
    #     val_losses = []
        
    #     for epoch in range(epochs):
    #         print("=== Starting epoch %i ===" % (epoch + 1))
    #         torch.set_grad_enabled
    #         running_train_losses = []
    #         running_val_losses = []
            
    #         iteration = 0
    #         print("== Training ==")
    #         for (img_batch, _), (tester, _) in list(zip(trainLoader, cipherLoader)):
    #             img_batch = img_batch.to(device)
    #             tester = tester.to(device)
    #             optimizer.zero_grad()
    #             output = self(img_batch)
    #             #print(output.shape)
    #             #print('test', tester.shape)
    #             recon_loss = loss_function(output, tester)
    #             recon_loss.backward()
    #             optimizer.step()
                
    #             if iteration % 10 == 0:
    #                 print("At img_batch %i. Loss %4f." % (iteration + 1, recon_loss.item()))
    #                 test_img = output[1].permute(1, 2, 0).cpu().detach().numpy()
    #                 tester_image = tester[1].permute(1, 2, 0).cpu().detach().numpy()
    #                 plt.figure()
    #                 plt.imshow(test_img, cmap='Greys')
    #                 plt.show()
    #                 plt.figure()
    #                 plt.imshow(tester_image, cmap='Greys')
    #                 plt.show()
                    
    #             running_train_losses.append(recon_loss.item())
    #             iteration = iteration + 1
    #         print("")
            
    #         iteration = 0
    #         # Validate
    #         print("== Validating ==" )
    #         with torch.no_grad():
    #             for (image, _), (tester, _) in list(zip(validLoader, cipherValidLoader)):
    
    #                 #image = Variable(image.view(-1, 256*256)).to(device)
    #                 tester = tester.to(device)
    #                 image = image.to(device)
    #                 output = self(image)
    #                 recon_loss = loss_function(output, tester)
                    
    #                 if iteration % 10 == 0:
    #                     print("At img_batch %i. Loss %4f." % (iteration + 1, recon_loss.item()))
                        
    #                 running_val_losses.append(recon_loss.item())
    #                 iteration = iteration + 1
    #         print("")
    #         print("Epoch %i finished! " % (epoch + 1))
    #         print("Training loss of %.4f; Validation loss of %.4f" % (np.average(running_train_losses), np.average(running_val_losses)))
    #         print("")
    #         train_losses.append(np.average(running_train_losses))
    #         val_losses.append(np.average(running_val_losses))
        
    #     #Plot the graph
    #     plt.figure(1)
    #     plt.title('Loss of Training and Validating the Encryption Network')
    #     plt.plot([i + 1 for i in range(len(train_losses))], train_losses, 'b-', label='Training')
    #     plt.plot([i + 1 for i in range(len(val_losses))], val_losses, 'r-', label='Validation')
    #     plt.legend(loc='upper right')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss (%)')
    #     plt.axis([1, epochs, 0, .5])
    #     plt.show()
    
    # def sample_images(self, batches_done):
    #     """Saves a generated sample from the test set"""
    #     for batch_num, (imgs, _) in enumerate(test_loader):
    #         print(imgs.shape)
    #         imgs.to(device)
    #         self.eval()
            
    #         real_A = imgs[1]
    #         fake_B = self(real_A)
    
    #         # Arange images along x-axis
    #         real_A = make_grid(real_A, nrow=5, normalize=True)
    #         fake_B = make_grid(fake_B, nrow=5, normalize=True)
            
    #         # Arange images along y-axis
    #         image_grid = torch.cat((real_A, fake_B), 1)
    #         save_image(image_grid, "results/encryption/%s.png" % (batches_done), normalize=False)

                
    # def test(self, test_loader):
    #     print("======TEST Stacked Network======")
    #     correct = 0
    #     total = 0

    #     y_true = []
    #     y_pred = []
        
    #     for batch_num, (image, label) in enumerate(test_loader):
    #         image = image.to(device)
    #         label = label.to(device)

    #         output = self(image)
            

    #         y_true.append(label)
    #         y_pred.append(predicted)

        # # Report test accuracy and F! score
        # print('Accuracy of the network on the %d testing images: %.2f %%' % (total, 100.0 * correct / total))
        # print('F1 Score of the network on the %d testing images: %.2f %%' % (total, 100.0 * f1_score(y_true, y_pred)))
   
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
encryption_network = encoder_network()
encryption_network.train_and_val(train_loader, val_loader, cipher_loader, cipher_val_loader, batch_size=batch_size, lr=learning_rate, device=device, epochs=20) 
encryption_network.save_model('./pre_trained_networks/encryption_network.pt')
#encryption_network.test(test_loader)