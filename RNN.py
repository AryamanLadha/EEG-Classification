import numpy as np
import torch
from utils.preprocess import getData
import matplotlib.pyplot as plt

X_train, y_train, X_valid, y_valid, X_test, y_test, person_train, person_valid, person_test, original = getData()


X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[2], X_train.shape[1]))
X_valid= torch.reshape(X_valid, (X_valid.shape[0], X_valid.shape[2], X_valid.shape[1]))
X_test= torch.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))




torch.set_default_tensor_type(torch.DoubleTensor)

from networks.rnn import LSTMModel
from utils.preprocess import getData
from utils.validate import validate
from utils.test_accuracy import test


def _LSTM_(input_dim, hidden_dim, layer_dim, output_dim, criterion, num_epochs):

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    valid_acc = []
    test_acc = []
    train_acc = []
    train_loss = []
    valid_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        outputs = model.forward(X_train) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train)

        loss.backward() #calculates the loss of the loss function

        optimizer.step() #improve from loss, i.e backprop
        
        train_a, train_l = validate(model, trainloader, criterion)
        valid_a, valid_l = validate(model, valloader, criterion)
        test_a, test_l = validate(model, testloader, criterion)
        
        
        train_acc.append(train_a)
        valid_acc.append(valid_a)
        test_acc.append(test_a)
        train_loss.append(loss.item())
        valid_loss.append(valid_l)
        test_loss.append(test_l)
        
        if epoch % 2 == 0:
            
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))#,'valid acc: %1.5f', valid_acc, 'test_acc: %1.5f', test_acc)

    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss



X_train, y_train, X_valid, y_valid, X_test, y_test, person_train, person_valid, person_test, original = getData()
#we need (N, 250, 22) #(N,L,H) input of batch_dim x seq_dim x feature_dim
X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[2], X_train.shape[1]))
X_valid= torch.reshape(X_valid, (X_valid.shape[0], X_valid.shape[2], X_valid.shape[1]))
X_test= torch.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))

batch_size=64
trainset = torch.utils.data.TensorDataset(X_train,y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# Shuffle is set to false for validation and test sets since no training is done on them, all we do is evaluate.
valset =  torch.utils.data.TensorDataset(X_valid, y_valid)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

testset = torch.utils.data.TensorDataset(X_test, y_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=0)






num_epochs = 20 #1000 epochs
learning_rate = 0.1 #0.001 lr

feature_dim = X_train.shape[2]#(X_train.shape[1],X_train.shape[2]) #number of features
seq_dim = X_train.shape[1]


input_dim = 22
hidden_dim = 2 #number of features in hidden state
layer_dim = 1 #number of stacked lstm layers
output_dim = 4 #number of output classes
seq_dim = 250  # Number of steps to unroll

criterion = torch.nn.L1Loss()

train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss = _LSTM_(input_dim, hidden_dim, layer_dim, output_dim, criterion, num_epochs)


acc_name = 'figures/lstm_acc_hid_'+str(hidden_dim)+'_lay_'+str(layer_dim)+'_loss_'+'L1'+'.png'

plt.figure()
plt.plot(train_acc)
plt.plot(valid_acc)
plt.plot(test_acc)
plt.legend(['train','validation','test'])
plt.title('Accuracy')
plt.savefig(acc_name)

loss_name = 'figures/lstm_loss_hid_'+str(hidden_dim)+'_lay_'+str(layer_dim)+'_loss_'+'L1'+'.png'
plt.figure()
plt.plot(train_loss)
plt.plot(valid_loss)
plt.plot(test_loss)
plt.legend(['train','validation','test'])
plt.title('loss')
plt.savefig(loss_name)


