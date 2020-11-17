# -*- coding: utf-8 -*-
"""
Script to pre-train the convolutional autoencoders and save them
Created on Mon Nov 16 18:17:05 2020

@author: Alycia
"""

from __future__ import print_function
from timeit import default_timer as timer
#from progress.bar import Bar
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from functools import partial
from collections import OrderedDict
import numpy as np

#Creating the class and functions for the residual layer based on ResNet
# The residual block takes an input with in_channels, applies some block
# of convolutional layers to reduce it to out_channels and sum it up to 
# the original input
# Identity = convolutional layer followed by an activation layer
# Most code in this section taken from https://github.com/FrancescoSaverioZuppichini/ResNet
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_short(self):
        return self.in_channels != self.out_channnels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion = expansion
        self.downsampling = downsampling
        self.conv = conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                   stride=self.downsampling, bias=False),
                'bn' : nn.BatchNorm2d(self.expanded_channels)
                })) if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict(
        {
            'conv' : conv(in_channels, out_channels, *args, **kwargs),
            'bm' : nn.BatchNorm2d(out_channels)
            }))

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.block = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv = self.conv, bias = False, stride = self.downsampling ),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias = False)
            )
        
#Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, downsampling=1):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 7, stride = 1, padding=(3,3), padding_mode='reflect'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size= 3, stride = 2, padding=(1,1), padding_mode='reflect'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size= 3, stride = 2, padding=(1,1), padding_mode='reflect'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            )
       
        # self.feature_mapping = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size = 1, stride = 1),
        #     nn.BatchNorm2d(),
        #     nn.ReLU()
        #     )
        
        #Decoder
        self.decoder = nn.Sequential(
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            ResNetBasicBlock(256, 256, downsampling=downsampling),
            nn.ConvTranspose2d(256, 128, kernel_size= 3, stride = 2, padding=(1,1), padding_mode='zeros', output_padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size= 3, stride = 2, padding=(1,1), padding_mode='zeros', output_padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size= 7, stride = 1, padding=(3,3), padding_mode='zeros'),
            nn.Tanh()
            )


    def forward(self, x):
        x1 = self.encoder(x)
        out = self.decoder(x1)
        return out

    def train_and_validate(self, trainLoader, validLoader, batch_size, lr, device, epochs=10,  which_model = 'Horse'):
        print("====== Pre-Training %s Autoencoder ======" %(which_model))
        optimizer = optim.Adam(self.parameters(), lr = 0.002, betas=(.5, .999))
        loss_function = nn.MSELoss()
        self.to(device)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            print("=== Starting epoch %i ===" % (epoch + 1))
            start = timer()
            running_train_losses = []
            running_val_losses = []
            
            
            print("== Training ==" )
            for batch_num, (img_batch, _) in enumerate(trainLoader):
                #mini_batch = Variable(img_batch.view(-1, 256*256)).to(device)
                img_batch = img_batch.to(device)
                optimizer.zero_grad()
                output = self(img_batch)
                recon_loss = loss_function(output, img_batch)
                recon_loss.backward()
                optimizer.step()
                
                if batch_num % 3 == 0:
                    print("At img_batch %i. Loss %4f." % (batch_num + 1, recon_loss.item()))
            
                running_train_losses.append(recon_loss.item())
            print("")
                
                # Validate
            print("== Validating ==" )
            with torch.no_grad():
                for batch_num, (image, _) in enumerate(validLoader):
    
                    #image = Variable(image.view(-1, 256*256)).to(device)
                    image = image.to(device)
                    hidden_out = self.encoder(image)
                    output = self.decoder(hidden_out)
                    recon_loss = loss_function(output, image)
                    
                    if batch_num % 3 == 0:
                        print("At img_batch %i. Loss %4f." % (batch_num + 1, recon_loss.item()))
                    
                    running_val_losses.append(recon_loss.item())
    
            end = timer()
            print("")
            print("Epoch %i finished! It took: %.4f seconds" % (epoch + 1, end - start))
            print("Training loss of %.4f; Validation loss of %.4f" % (np.average(running_train_losses), np.average(running_val_losses)))
            print("")
            train_losses.append(np.average(running_train_losses))
            val_losses.append(np.average(running_val_losses))
        
        #Plot the graph
        plt.figure(1)
        plt.title('Loss of Training and Validating the %s Autoencoder' % (which_model))
        plt.plot([i + 1 for i in range(len(train_losses))], train_losses, 'b-', label='Training')
        plt.plot([i + 1 for i in range(len(val_losses))], val_losses, 'r-', label='Validation')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (%)')
        plt.axis([1, epochs, 0, .3])
        plt.show()
# #Instantiate the model
# model = ConvAutoencoder()
# print(model)

# #Loss function
# criterion = nn.BCELoss()

# #Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# #Epochs
# n_epochs = 100

# for epoch in range(1, n_epochs+1):
#     # monitor training loss
#     train_loss = 0.0

#     #Training
#     for data in train_loader:
#         images, _ = data
#         images = images
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, images)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*images.size(0)
          
#     train_loss = train_loss/len(train_loader)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
# #Batch of test images
# dataiter = iter(test_loader)
# images, labels = dataiter.next()

# #Sample outputs
# output = model(images)
# images = images.numpy()

# output = output.view(32, 3, 32, 32)
# output = output.detach().numpy()

# #Original Images
# print("Original Images")
# fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
# for idx in np.arange(5):
#     ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
#     imshow(images[idx])
#     ax.set_title(classes[labels[idx]])
# plt.show()

# #Reconstructed Images
# print('Reconstructed Images')
# fig, axes = plt.subplots(nrows=1, ncols=5, sharex=True, sharey=True, figsize=(12,4))
# for idx in np.arange(5):
#     ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
#     imshow(output[idx])
#     ax.set_title(classes[labels[idx]])
# plt.show() 
    
    
