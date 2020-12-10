import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from weight_initializer import init_model_weights
from timeit import default_timer as timer
from progress.bar import Bar

class Autoencoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Autoencoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(out_features, in_features),
            nn.ReLU()
        )

        # Initialize weights
        self.apply(init_model_weights)

    def forward(self, x):
        out1 = self.encoder(x)
        out2 = self.decoder(out1)
        return out2

    def train_and_val(self, trainLoader, validLoader, epochs, batch_size, lr, momentum, device):
        print("======Autoencoding Layer======")
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        loss_function = nn.MSELoss()
        self.to(device)

        for epoch in range(epochs):
            start = timer()
            running_train_losses = []
            running_val_losses = []

            # Train
            for batch_num, (img_batch, _) in enumerate(trainLoader):
                # if batch_num == 300:
                #     break

                mini_batch = Variable(img_batch.view(-1, self.in_features)).to(device)
                optimizer.zero_grad()

                output = self(mini_batch)
                recon_loss = loss_function(output, mini_batch)
                recon_loss.backward()
                optimizer.step()

                # Print statistics
                # (Testing stuff)
                if batch_num % 50 == 0:
                    print("At minibatch %i. Loss: %.4f." % (batch_num + 1, recon_loss.item()))

                running_train_losses.append(recon_loss.item())
            print("")

            # Validate
            print("======Validating======")
            with torch.no_grad():
                for batch_num, (image, _) in enumerate(validLoader):
                    # if batch_num == 100:
                    #     break

                    image = Variable(image.view(-1, self.in_features)).to(device)

                    hidden_out = self.encoder(image)
                    output = self.decoder(hidden_out)
                    recon_loss = loss_function(output, image)

                    running_val_losses.append(recon_loss.item())

            end = timer()

            print("Epoch %i finished! It took: %.4f seconds" % (epoch + 1, end - start))
            print("Training loss of %.4f; Validation loss of %.4f" % (np.average(running_train_losses), np.average(running_val_losses)))
            print("====================")


    def encode_batch(self, dataLoader, device):
        encoded = []
        labels = []
        print("======Getting encoded======")
        for batch_num, mini_batch in enumerate(dataLoader):
            # if batch_num == 300:
            #     break

            batch_images = Variable(mini_batch[0].view(-1, self.in_features)).to(device)
            batch_labels = Variable(mini_batch[1]).to(device)

            hidden_out = self.encoder(batch_images)
            encoded.append(hidden_out.data)
            labels.append(batch_labels)

        encoded = torch.cat(encoded, dim=0)
        labels = torch.cat(labels, dim=0)

        return encoded, labels
