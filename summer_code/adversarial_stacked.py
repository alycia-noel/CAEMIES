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

from models.stacked_model import StackedNetwork
from dataset import ExecutableDataset

from timeit import default_timer as timer
from progress.bar import Bar

#====================PARAMETERS==========================
input_w = 128
input_h = 128
learning_rate = 0.0001
momentum = 0.9
n_epochs = 10
N_epochs = 10
batch_size = 32

# Splits
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Encoder Layer Parameters
hidden_enc_dim1 = 2000
hidden_enc_dim2 = 1000

# Perceptron Hidden Layer Parameters
hidden_mp_dim1 = 20
hidden_mp_dim2 = 20
hidden_mp_dim3 = 150
#========================================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Train-val-test split function
def train_val_test_split(dataset, splits):
    datasets = {}

    # Two-way split
    if len(splits) == 2:
        first_idx, second_idx = train_test_split(list(range(dataset.__len__())), test_size=splits[1])
        datasets['first'] = Subset(dataset, first_idx)
        datasets['second'] = Subset(dataset, second_idx)

    # Three-way split
    elif len(splits) == 3:
        first_idx, second_third_idx = train_test_split(list(range(dataset.__len__())), test_size=1-splits[0])
        second_idx, third_idx = train_test_split(second_third_idx, test_size=splits[2]/(splits[1]+splits[2]))

        datasets['first'] = Subset(dataset, first_idx)
        datasets['second'] = Subset(dataset, second_idx)
        datasets['third'] = Subset(dataset, third_idx)

    return datasets

#============================================================================================================

# DATA TRANSFORMATION

transform_exec = transforms.Compose([
    transforms.Resize((input_h, input_w)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

#============================================================================================================

# PATHS

#windows_mal_train_path = '/Users/huymai/Datasets/executables/windows_mal_train'
#windows_mal_test_path = '/Users/huymai/Datasets/executables/windows_mal_test'
#windows_benign_path = '/Users/huymai/Datasets/executables/windows_benign'
#elf_mal_path = '/Users/huymai/Datasets/executables/elf_mal'
#elf_benign_path = '/Users/huymai/Datasets/executables/elf_ben'

windows_mal_path = r'D:\executables\windows_mal'
windows_benign_path = r'D:\executables\windows_benign'
elf_mal_path = r'D:\executables\elf_mal_images'
elf_benign_path = r'D:\executables\elf_benign'
ae_pe_elf_path = r'D:\executables\PE_ELF_Full_AE'

#============================================================================================================

# SPLIT DATA
"""
Encoded labels:
0 - Benign
1 - Malware
"""

# Split Windows Benign files
windows_benign_files = ExecutableDataset(transform_exec, windows_benign_path, all='Benign')
windows_benign_split = train_val_test_split(windows_benign_files, splits=[train_split, val_split, test_split])

windows_benign_train_set = windows_benign_split['first']
windows_benign_val_set = windows_benign_split['second']
windows_benign_test_set = windows_benign_split['third']

# Split Windows Malware Files
windows_mal_files = ExecutableDataset(transform_exec, windows_mal_path, all='Malware')
windows_mal_split = train_val_test_split(windows_mal_files, splits=[train_split, val_split, test_split])

windows_mal_train_set = windows_mal_split['first']
windows_mal_val_set = windows_mal_split['second']
windows_mal_test_set = windows_mal_split['third']

# Split Elf Malware files
elf_mal_files = ExecutableDataset(transform_exec, elf_mal_path, all='Malware')
elf_mal_split = train_val_test_split(elf_mal_files, splits=[train_split, val_split, test_split])

elf_mal_train_set = elf_mal_split['first']
elf_mal_val_set = elf_mal_split['second']
elf_mal_test_set = elf_mal_split['third']

# Split Elf Benign files
elf_benign_files = ExecutableDataset(transform_exec, elf_benign_path, all='Benign')
elf_benign_split = train_val_test_split(elf_benign_files, splits=[train_split, val_split, test_split])

elf_benign_train_set = elf_benign_split['first']
elf_benign_val_set = elf_benign_split['second']
elf_benign_test_set = elf_benign_split['third']

# Split Elf Benign files
ae_pe_elf_files = ExecutableDataset(transform_exec, ae_pe_elf_path, all='Malware')
ae_pe_elf_split = train_val_test_split(ae_pe_elf_files, splits=[train_split, val_split, test_split])

ae_pe_elf_train_set = ae_pe_elf_split['first']
ae_pe_elf_val_set = ae_pe_elf_split['second']
ae_pe_elf_test_set = ae_pe_elf_split['third']

#============================================================================================================

# PUT SETS INTO DATALOADERS
windows_elf_train_dataset = windows_benign_train_set + windows_mal_train_set + elf_benign_train_set + elf_mal_train_set + ae_pe_elf_train_set
windows_elf_val_dataset = windows_benign_val_set + windows_mal_val_set + elf_benign_val_set + elf_mal_val_set + ae_pe_elf_val_set
windows_elf_test_dataset = windows_mal_test_set + windows_benign_test_set + elf_mal_test_set + elf_benign_test_set + ae_pe_elf_test_set

train_loader = torch.utils.data.DataLoader(windows_elf_train_dataset, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(windows_elf_val_dataset, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(windows_elf_test_dataset, shuffle=False)

#============================================================================================================

# MODEL

# Instantiation
stacked_model = StackedNetwork(
    input_size=input_w*input_h,
    encode_layers=[hidden_enc_dim1, hidden_enc_dim2],
    mp_layers=[hidden_mp_dim1, hidden_mp_dim2, hidden_mp_dim3, 1],
    device=device
)

print('=========ADVERSARIAL TRAINING ON PE AND ELF========')

# Layer-wise pretraining
train_mp_percentages, val_mp_percentages = stacked_model.pretrain(train_loader, val_loader, epochs=n_epochs, batch_size=batch_size, lr=learning_rate, momentum=momentum)

# Fine-tune model
train_st_percentages, val_st_percentages, train_losses, val_losses = stacked_model.train_and_val(train_loader, val_loader, epochs=N_epochs, batch_size=batch_size, lr=learning_rate, momentum=momentum)

# Test model (Don't do until parameters have been optimized!)
stacked_model.test(test_loader)

# Save results
stacked_model_path = r'D:\saved_models\pretrained_adv_stacked.pt'
stacked_model.save_model(stacked_model_path)

#============================================================================================================

# RESULTS
accuracies_path = r'D:\results\stackedaccuraciesadv.png'
losses_path =r'D:\results\stackedlossesadv.png'

# Stacked Accuracies over N epochs
plt.figure(1)
plt.title('Stacked Network Accuracies (PE and ELF Adversarial)')
plt.plot([i + 1 for i in range(len(train_st_percentages))], train_st_percentages, 'b-', label='Training')
plt.plot([i + 1 for i in range(len(val_st_percentages))], val_st_percentages, 'r-', label='Validation')
plt.legend(loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.axis([1, N_epochs, 0, 100])
plt.savefig(accuracies_path)

# Stacked Losses over N epochs
plt.figure(2)
plt.title('Stacked Network Losses (PE and ELF Adversarial)')
plt.plot([i + 1 for i in range(len(train_losses))], train_losses, 'b-', label='Training')
plt.plot([i + 1 for i in range(len(val_losses))], val_losses, 'r-', label='Validation')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(losses_path)
