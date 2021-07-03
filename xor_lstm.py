import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences


# hyperparameters
batch_size = 8
hidden_size = 2
data_size = 100000


# torch.manual_seed(42)
# data generation and pre-procesing
# inputs = torch.randint(0, 2, (data_size, 50, 1)).float()
inputs = []
for i in range(data_size):
    # Choose random length
    length = np.random.randint(1, 51)
    inputs.append(np.random.randint(2, size=(length)).astype('float32'))
inputs = np.asarray(inputs)
# Pad binary strings with 0's to make sequence length same for all
inputs = pad_sequences(inputs, maxlen=50, dtype='float32', padding='pre')
inputs = torch.from_numpy(inputs).view(data_size, 50, 1)
labels_onehot = [1 if torch.sum(inputs[i:i+1])%2 == 1 else 0 for i in range(data_size)]
labels_onehot = torch.FloatTensor(labels_onehot).view(data_size, 1)
inputs = torch.cat(torch.chunk(inputs, dim=0, chunks=data_size), 1).view(50, data_size, -1)

train_inputs = inputs[:, :int(0.8*data_size), :]
test_inputs = inputs[:, int(0.8*data_size):, :]
train_labels_onehot = labels_onehot[:int(0.8*data_size)]
test_labels_onehot = labels_onehot[int(0.8*data_size):]
print("----Data Loaded-----")

# LSTM
class LSTM(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber: 'Long-Short Term Memory' cell
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.w_ih = nn.Parameter(torch.randn(4*self.hidden_size, self.input_size, requires_grad=True))
        self.w_hh = nn.Parameter(torch.randn(4*self.hidden_size, self.hidden_size, requires_grad=True))
        self.b_ih = nn.Parameter(torch.randn(4*hidden_size, requires_grad=True))
        self.b_hh = nn.Parameter(torch.randn(4*hidden_size, requires_grad=True))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        h, c = hidden
        for i in x:
            gates = F.linear(i, self.w_ih, self.b_ih) + F.linear(h, self.w_hh, self.b_hh)

            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * torch.tanh(c)

        return (h, c)


# model
class lstm_xor(nn.Module):
    def __init__(self, hidden_size, batch_size):
        super(lstm_xor, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # LSTM
        self.lstm = nn.LSTM(1, hidden_size)

        # The linear layer that maps from hidden state space to score
        self.linear = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers*numdirections, minibatch_size, hidden_size)
        return (torch.rand(1, self.batch_size, self.hidden_size).normal_(),
                torch.rand(1, self.batch_size, self.hidden_size).normal_())

    def forward(self, x):
        _, self.hidden = self.lstm(x, self.hidden)
        # print(self.hidden)
        # print self.hidden[0].size(), x.size()
        out = self.linear(self.hidden[0].view(batch_size, -1))
        # print(out)
        score = torch.sigmoid(out)
        return score

model = lstm_xor(hidden_size, batch_size)
print([i.size() for i in list(model.parameters())])
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

with torch.no_grad():
    score = model(inputs[:, 0:batch_size, :])
    print(score)

# training
for epoch in range(100):
    for i in range(int(int(0.8*data_size)/batch_size)):
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Forward pass. inputs[i*batch_size:i*batch_size + batch_size]
        score = model(train_inputs[:, i*batch_size:i*batch_size + batch_size, :])
        # print score.size(), labels_onehot[i:i+1].size()

        # Compute the loss, gradients, and update the parameters by calling optimizer.step()
        loss = loss_fn(score, train_labels_onehot[i*batch_size:i*batch_size + batch_size])
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print("{}th sample crossed, loss: {}".format(i+1, loss), score, train_labels_onehot[i*batch_size:i*batch_size + batch_size])

    #testing
    with torch.no_grad():

        correct = 0
        test_outputs = torch.Tensor(())
        for i in range(int(int(0.2*data_size)/batch_size)):
            score = model(test_inputs[:, i*batch_size:i*batch_size + batch_size, :])
            test_outputs = torch.cat((test_outputs, torch.round(score)), dim=0)

        for a, b in zip(test_outputs, test_labels_onehot):
            if torch.equal(a, b):
                correct += 1

        acc = 100 * correct / (0.2*data_size)

    print("epoch: {}, loss: {}, acc: {}".format(epoch, loss, acc))

# testing
# with torch.no_grad():
#
#     test_outputs = torch.Tensor(())
#     for i in range(int(int(0.2*data_size)/batch_size)):
#         score = model(test_inputs[:, i*batch_size:i*batch_size + batch_size, :])
#         test_outputs = torch.cat((test_outputs, torch.round(score)), dim=0)
#
#     correct = 0
#     for a, b in zip(test_outputs, test_labels_onehot):
#         if torch.equal(a, b):
#             correct += 1
#     acc = float(100 * correct / (int(0.2*data_size)))
#     print acc
