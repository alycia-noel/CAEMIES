import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from models.ae_model import Autoencoder
from models.mp_model import BinaryPerceptron
from dataset import ExecutableDataset

from sklearn.metrics import f1_score

from timeit import default_timer as timer
from progress.bar import Bar

def build_parts(layers, binary=False):
    net = []
    for i in range(1, len(layers) - 1):
        net.extend([nn.Linear(layers[i-1], layers[i]), nn.ReLU()])

    if binary:
        net.extend([nn.Linear(layers[-2], layers[-1]), nn.Sigmoid()])
    else:
        net.extend([nn.Linear(layers[-2], layers[-1]), nn.ReLU()])
    return nn.Sequential(*net)

class StackedNetwork(nn.Module):
    def __init__(self, input_size, encode_layers, mp_layers, device):
        super(StackedNetwork, self).__init__()

        self.in_features = input_size

        self.enc_ = [input_size] + encode_layers
        self.mp_ = [encode_layers[-1]] + mp_layers
        self.stacked_encoder = build_parts(self.enc_)
        self.perceptron = build_parts(self.mp_, binary=True)

        self.device = device

    def forward(self, x):
        out1 = self.stacked_encoder(x)
        out2 = self.perceptron(out1)
        return out2


    def pretrain(self, trainLoader, validLoader, epochs, batch_size, lr, momentum):
        trLoader = trainLoader
        valLoader = validLoader

        # Pretrain Each Autoencoder
        weights = []
        biases = []
        print("======PRETRAINING======")
        for i in range(1, len(self.enc_)):
            in_features = self.enc_[i-1]
            out_features = self.enc_[i]
            ae = Autoencoder(in_features, out_features)
            ae.train_and_val(trLoader, valLoader, epochs, batch_size, lr, momentum, self.device)

            # Get encoded representations
            train_enc, train_labels = ae.encode_batch(trLoader, self.device)
            val_enc, val_labels = ae.encode_batch(valLoader, self.device)

            train_set = ExecutableDataset(imgs=train_enc, labels=train_labels)
            val_set = ExecutableDataset(imgs=val_enc, labels=val_labels)

            # Use output feature as input to next autoencoder
            trLoader = DataLoader(train_set, batch_size, shuffle=True)
            valLoader = DataLoader(val_set, batch_size, shuffle=True)

            # Copy encoder parameters
            k = 0
            for param in ae.encoder.parameters():
                if k % 2 == 0:
                    weights.append(param.data)
                else:
                    biases.append(param.data)
                k += 1

        # Pretrain Multi-layer Perceptron
        in_features = self.mp_[0]
        hidden_layers = self.mp_[1:-1]
        mp = BinaryPerceptron(in_features, hidden_layers)
        train_mp_percentages, val_mp_percentages, mp_path = mp.train_and_val(trLoader, valLoader, epochs, batch_size, lr, momentum, self.device)
        mp.load_state_dict(torch.load(mp_path))
        mpLayer = [mp]

        # Copy mp weights
        k = 0
        for param in mp.parameters():
            if k % 2 == 0:
                weights.append(param.data)
            else:
                biases.append(param.data)
            k += 1

        # Copy all weights and biases onto stacked network
        self.copy_parameters(weights, biases)

        return train_mp_percentages, val_mp_percentages


    def copy_parameters(self, weights, biases):
        print('======COPYING PARAMETERS======')
        k = 0
        #print("=====================")
        for param in self.parameters():
            # Copy weights
            if k % 2 == 0:
                param.data = weights[int(k / 2)].detach().clone()

            # Copy biases
            else:
                param.data = biases[int(k / 2)].detach().clone()
            k += 1


    def train_and_val(self, trainLoader, validLoader, epochs, batch_size, lr, momentum, freeze=0):
        print("======Stacked Network======")
        self.to('cpu')
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        loss_function = nn.BCELoss()
        best_acc = 0.0

        train_percentages = []
        val_percentages = []
        train_losses = []
        val_losses = []

        # Freeze encoder (only for transfer learning)
        count = 0
        for child in self.children():
            print('=========')
            count += 1
            if count == freeze:
                for param in child.parameters():
                    param.requires_grad = False

        for epoch in range(epochs):
            start = timer()
            running_train_losses = []
            running_val_losses = []

            # Train
            correct_t = 0
            total_t = 0
            for batch_num, batch in enumerate(trainLoader):

                img_batch = Variable(batch[0].view(-1, self.in_features)).to('cpu')
                lbl_batch = Variable(batch[1]).to('cpu')
                lbl_batch = lbl_batch.float()
                optimizer.zero_grad()

                output = self(img_batch)
                loss = loss_function(output, lbl_batch)
                loss.backward()
                optimizer.step()

                predicted_t = torch.round(output.data)
                total_t += lbl_batch.size(0)
                correct_t += (predicted_t == lbl_batch.reshape((lbl_batch.size(0),1))).sum()

                # Print statistics
                if batch_num % 50 == 0:
                    print("At minibatch %i. Loss: %.4f." % (batch_num + 1, loss.item()))

                running_train_losses.append(loss.item())

            # Validate
            correct = 0
            total = 0
            print("=======Validating=======")
            with torch.no_grad():
                for batch_num, (image, label) in enumerate(validLoader):
                    # if batch_num == 100:
                    #     break

                    image = Variable(image.view(-1, self.in_features)).to('cpu')
                    label = Variable(label).to('cpu')
                    label = label.float()

                    output = self(image)
                    predicted = torch.round(output.data).long()
                    total += label.size(0)
                    correct += (predicted == label.reshape((label.size(0), 1))).sum()

                    running_val_losses.append(loss.item())

            train_acc = 100.0 * correct_t / total_t
            val_acc = 100.0 * correct / total

            print('Accuracy of the network on the %d training images: %.2f %%' % (total_t, train_acc))
            print('Accuracy of the network on the %d validation images: %.2f %%' % (total, val_acc))

            end = timer()

            print("Epoch %i finished! It took: %.4f seconds" % (epoch + 1, end - start))
            print("Training loss of %.4f" % (np.average(running_train_losses)))
            print("Validating loss of %.4f" % (np.average(running_val_losses)))

            train_percentages.append(train_acc)
            val_percentages.append(val_acc)
            train_losses.append(np.average(running_train_losses))
            val_losses.append(np.average(running_val_losses))

            print("====================")

        return train_percentages, val_percentages, train_losses, val_losses


    def test(self, testLoader):
        print("======TEST Stacked Network======")
        correct = 0
        total = 0

        y_true = []
        y_pred = []

        for batch_num, (image, label) in enumerate(testLoader):

            image = Variable(image.view(-1, self.in_features)).to('cpu')
            label = Variable(label).to('cpu')

            output = self(image)
            predicted = torch.round(output.data).long()
            total += label.size(0)
            correct += (predicted == label).sum()

            y_true.append(label)
            y_pred.append(predicted)

        # Report test accuracy and F! score
        print('Accuracy of the network on the %d testing images: %.2f %%' % (total, 100.0 * correct / total))
        print('F1 Score of the network on the %d testing images: %.2f %%' % (total, 100.0 * f1_score(y_true, y_pred)))

    def save_model(self, path):
        torch.save(self.state_dict(), path)
