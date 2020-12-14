# -*- coding: utf-8 -*-
"""
Image-to-image translation: chest -> cipher

Code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and
https://github.com/aitorzip/PyTorch-CycleGAN/blob/master

Created on Mon Nov  9 01:18:40 2020

@author: ancarey
"""
import os
import glob
import random
import datetime
import sys
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from visdom import Visdom
from autoenc import ConvAutoencoder

input_w = 256
input_h = 256
input_nc = 3
train_split = 0.7
val_split = 0.3
batch_size = 16
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
root = 'datasets/chest2cipher/'
Tensor = torch.cuda.FloatTensor 

'''Helper functions'''
def load_pretrained_models(path):
    network = ConvAutoencoder(in_channels = 1, 
                             out_channels = 1, 
                             kernel_size = 3,
                             device = device,
                             downsampling = 1
                            )
    network.load_state_dict(torch.load(chest_auto_path))
    return(network)

def imshow(img):
    img = img / 2 + 0.5 #un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)
    
'''Classes'''
# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()

#         channels, height, width = input_shape

#         # Calculate output shape of image discriminator (PatchGAN)
#         self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

#         def discriminator_block(in_filters, out_filters, normalize=True):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalize:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *discriminator_block(channels, 64, normalize=False),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(256, 512),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(512, 1, 4, padding=1)
#         )
    
#     def forward(self, x):
#         x1 = self.model(x)
#         return x1

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        
        model = [ nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        
        model += [ nn.Conv2d(64, 128, 4, stride=2, padding =1),
                   nn.InstanceNorm2d(128),
                   nn.LeakyReLU(0.2, inplace=True)]
        
        model += [ nn.Conv2d(128, 256, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(256),
                   nn.LeakyReLU(0.2, inplace=True)]
        
        model += [ nn.Conv2d(256, 512, 4, padding = 1),
                   nn.InstanceNorm2d(512),
                   nn.LeakyReLU(0.2, inplace=True)]
        
        model += [ nn.Conv2d(512, 1, 4, padding=1)]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #load pre-trained models 
        chest_auto = load_pretrained_models(chest_auto_path)
        cipher_auto = load_pretrained_models(cipher_auto_path)
        
        self.encode = chest_auto.encoder
        self.decode = cipher_auto.decoder
    
        #feature mapping layer 
        self.feature_map = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          )
        
    def forward(self, x):
        #print(x.shape)
        x1 = self.encode(x) # [32, 256, 64, 64]
        x2 = self.feature_map(x1)
        x3 = x2 + x1
        x4 = self.decode(x3)
        return(x4)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
class ReplayBuffer():
    def __init__(self, max_size = 50):
        assert (max_size > 0), "Empty buffer or trying to create a black-hole. Be careful."
        self.max_size = max_size
        self.data = []
        
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else: 
                    to_return.append(element)
                    
        return Variable(torch.cat(to_return))

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                             transforms.Resize((input_h, input_w)),
                                             transforms.ToTensor()
                                            ])
        self.unaligned = unaligned
        
        self.files_chest = sorted(glob.glob(os.path.join(root, '%s/chest' % mode) + '/*.*'))
        self.files_cipher = sorted(glob.glob(os.path.join(root, '%s/cipher' % mode) + '/*.*'))
        
    def __getitem__(self, index):
        item_chest = self.transform(Image.open(self.files_chest[index % len(self.files_chest)]))
        
        if self.unaligned:
            item_cipher = self.transform(Image.open(self.files_cipher[random.randint(0, len(self.files_cipher) - 1)]))
        else:
            item_cipher = self.transfrom(Image.open(self.files_cipher[index % len(self.files_cipher)]))
            
        return {'CHEST': item_chest, 'CIPHER': item_cipher}
    
    def __len__(self):
        return max(len(self.files_chest), len(self.files_cipher))
    
class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs= n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        
    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()
        
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))
        
        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item() 
            else:
                self.losses[loss_name] += losses[loss_name].item()
                
            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
        
''' Constants '''
input_nc = 1
output_nc = 1
LR = 0.0002
epoch = 0
n_epochs = 200
batchSize = 16
dataroot = 'datasets/chest2cipher/'
decay_epoch = 100
size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''Setup Network'''
generator_chest2cipher = Generator()#(input_nc, output_nc)
generator_cipher2chest = Generator()#(output_nc, input_nc)
discriminator_chest = Discriminator(input_nc)
discriminator_cipher = Discriminator(output_nc)

generator_chest2cipher.to(device)
generator_cipher2chest.to(device)
discriminator_chest.to(device)
discriminator_cipher.to(device)

generator_chest2cipher.apply(weights_init_normal)
generator_cipher2chest.apply(weights_init_normal)
discriminator_chest.apply(weights_init_normal)
discriminator_cipher.apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_gen = torch.optim.Adam(itertools.chain(generator_chest2cipher.parameters(), generator_cipher2chest.parameters()), lr = LR, betas=(0.5, 0.999))
optimizer_disc_chest = torch.optim.Adam(discriminator_chest.parameters(), lr = LR, betas=(0.5, 0.999))
optimizer_disc_cipher = torch.optim.Adam(discriminator_cipher.parameters(), lr = LR, betas=(0.5, 0.999))

lr_scheduler_gen = torch.optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_disc_chest = torch.optim.lr_scheduler.LambdaLR(optimizer_disc_chest, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_disc_cipher = torch.optim.lr_scheduler.LambdaLR(optimizer_disc_cipher, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor #if opt.cuda else torch.Tensor
input_chest = Tensor(batchSize, 1, size, size)
input_cipher = Tensor(batchSize, 1, size, size)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

#Load datasets
'''Load Datasets and save example training images'''
transforms_ = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                             transforms.Resize((input_h, input_w)),
                                             transforms.ToTensor()
                                            ])
dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True, num_workers=0)

#Loss plot
logger = Logger(n_epochs, len(dataloader))

'''Training'''
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(dataloader):
        real_chest = Variable(input_chest.copy_(batch['CHEST']))
        real_cipher = Variable(input_cipher.copy_(batch['CIPHER']))
        
        optimizer_gen.zero_grad()
        same_cipher = generator_chest2cipher(real_cipher)
        loss_identity_cipher = criterion_identity(same_cipher, real_cipher)*5.0
        
        same_chest = generator_cipher2chest(real_chest)
        loss_identity_chest = criterion_identity(same_chest, real_chest)*5.0

        # GAN loss
        fake_cipher = generator_chest2cipher(real_chest)
        pred_fake = discriminator_cipher(fake_cipher)
        loss_GAN_chest2cipher = criterion_GAN(pred_fake, target_real)

        fake_chest = generator_cipher2chest(real_cipher)
        pred_fake = discriminator_chest(fake_chest)
        loss_GAN_cipher2chest = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_chest = generator_cipher2chest(fake_cipher)
        loss_cycle_chestcipherchest = criterion_cycle(recovered_chest, real_chest)*10.0

        recovered_cipher = generator_chest2cipher(fake_chest)
        loss_cycle_cipherchestcipher = criterion_cycle(recovered_cipher, real_cipher)*10.0

        # Total loss
        loss_G = loss_identity_chest + loss_identity_cipher + loss_GAN_chest2cipher + loss_GAN_cipher2chest + loss_cycle_chestcipherchest +  loss_cycle_cipherchestcipher
        loss_G.backward()
        
        optimizer_gen.step()
        ###################################

        ###### Discriminator A ######
        optimizer_disc_chest.zero_grad()

        # Real loss
        pred_real = discriminator_chest(real_chest)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_chest = fake_A_buffer.push_and_pop(fake_chest)
        pred_fake = discriminator_chest(fake_chest.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_disc_chest.step()
        ###################################

        ###### Discriminator B ######
        optimizer_disc_cipher.zero_grad()

        # Real loss
        pred_real = discriminator_cipher(real_cipher)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_cipher = fake_B_buffer.push_and_pop(fake_cipher)
        pred_fake = discriminator_cipher(fake_cipher.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_disc_cipher.step()
        ###################################

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_chest + loss_identity_cipher), 'loss_G_GAN': (loss_GAN_chest2cipher + loss_GAN_cipher2chest),
                    'loss_G_cycle': (loss_cycle_chestcipherchest + loss_cycle_cipherchestcipher), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_chest, 'real_B': real_cipher, 'fake_A': fake_chest, 'fake_B': fake_cipher})

    # Update learning rates
    lr_scheduler_gen.step()
    lr_scheduler_disc_chest.step()
    lr_scheduler_disc_cipher.step()

    # Save models checkpoints
    torch.save(generator_chest2cipher.state_dict(), 'output/generator_chest2cipher.pth')
    torch.save(generator_cipher2chest.state_dict(), 'output/generator_cipher2chest.pth')
    torch.save(discriminator_chest.state_dict(), 'output/discriminator_chest.pth')
    torch.save(discriminator_cipher.state_dict(), 'output/discriminator_cipher.pth')

