import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from weight_initializer import init_model_weights
from timeit import default_timer as timer
from progress.bar import Bar

class BinaryPerceptron(nn.Module):
    def __init__(self, in_features, hidden):
        super(BinaryPerceptron, self).__init__()

        self.in_features = in_features

        self.in_layer = nn.Sequential(
            nn.Linear(self.in_features, hidden[0]),
            nn.LeakyReLU()
        )

        hidden_list = []
        if len(hidden) > 1:
            for i in range(1, len(hidden)):
                hidden_list.extend([nn.Linear(hidden[i-1], hidden[i]), nn.LeakyReLU()])

        self.hidden_layer = nn.Sequential(*hidden_list)

        #self.output_layer = nn.Sequential(
            #nn.Linear(hidden[-1], 1),
            #nn.Sigmoid()
        #)

        self.in_layer.apply(init_model_weights)
        self.hidden_layer.apply(init_model_weights)

        self.output_layer = nn.Linear(hidden[-1], 1)
        nn.init.xavier_normal_(self.output_layer.weight.data, gain=4.0)
        self.output_layer.bias.data.fill_(0.01)

    def forward(self, x):
        out1 = self.in_layer(x)
        out2 = self.hidden_layer(out1)
        out3 = F.sigmoid(self.output_layer(out2))
        return out3

    def train_and_val(self, trainLoader, validLoader, epochs, batch_size, lr, momentum, device):
        print("======Multi-Layer Perceptron======")
        self.to('cpu')
        loss_function = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        best_acc = 0.0
        prev_acc = 0.0

        train_percentages = []
        val_percentages = []

        for epoch in range(epochs):
            start = timer()
            running_train_losses = []

            # Train
            correct_t = 0
            total_t = 0
            for batch_num, batch in enumerate(trainLoader):
                # if batch_num == 300:
                #     break

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

                '''
                if batch_num % 10 == 0:
                    print(output)
                    print(lbl_batch)
                    print('---------')
                '''

                # Print statistics
                if batch_num % 50 == 0:
                    print("At minibatch %i. Loss: %.4f." % (batch_num + 1, loss.item()))

                running_train_losses.append(loss.item())

            # Validate
            correct = 0
            total = 0
            print("======VALIDATING======")
            with torch.no_grad():
                for batch_num, (image, label) in enumerate(validLoader):
                    # if batch_num == 100:
                    #     break

                    image = Variable(image.view(-1, self.in_features)).to('cpu')
                    label = Variable(label).to('cpu')
                    label = label.float()

                    output = self(image)
                    predicted = torch.round(output.data)
                    total += label.size(0)
                    correct += (predicted == label.reshape(label.size(0), 1)).sum()

                    # if batch_num % 1000 == 0:
                    #     print("TOTAL: " + str(total))
                    #     print("CORRECT: " + str(correct.data))

            train_acc = 100.0 * correct_t / total_t
            val_acc = 100.0 * correct / total

            print('Accuracy of the network on the %d training images: %.2f %%' % (total_t, train_acc))
            print('Accuracy of the network on the %d validation images: %.2f %%' % (total, val_acc))

            end = timer()

            print("Epoch %i finished! It took: %.4f seconds" % (epoch + 1, end - start))
            print("Training loss of %.4f" % (np.average(running_train_losses)))

            # Early stopping
            if prev_acc != 0.0 and torch.abs(train_acc - val_acc) < 5.0 and (prev_acc - val_acc) > 5.0:
                print("========EARLY STOPPING MLP========")
                break

            # Save previous epoch
            mp_path = r'D:\saved_models\pretrained_mp.pt'
            torch.save(self.state_dict(), mp_path)
            prev_acc = val_acc

            train_percentages.append(train_acc)
            val_percentages.append(val_acc)

            print("====================")

        return train_percentages, val_percentages, mp_path
