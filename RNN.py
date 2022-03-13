import numpy as np
import torch
from utils.preprocess import getData
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

from networks.rnn import *
from utils.validate import validate
from utils.test_accuracy import test

def get_idxs(in_array, to_find):
    """Utility function for finding the positions of observations of one array in another an array.
    Args:
        in_array (array): Array in which to locate elements of to_find
        to_find (array): Array of elements to locate in in_array
    Returns:
        TYPE: Indices of all elements of to_find in in_array
    """
    targets = ([np.where(in_array == x) for x in to_find])
    return np.squeeze(np.concatenate(targets, axis=1))

def _LSTM_(input_dim, hidden_dim, layer_dim, output_dim, criterion, num_epochs):
    """
    Here you can switch between LSTMModel and LSTM_CNN.
    Just note that LSTMModel has different input size so don't forget to check dimensions
    """

    #model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    
    model = LSTM_CNN(input_dim, hidden_dim, layer_dim, output_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print(model)
    
    stats = {
    'train_accuracies': [],
    'train_losses': [],
    'val_accuracies': [],
    'val_losses': []
    }
    for epoch in range(num_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #inputs = inputs.float()
            #labels = labels.float()
      
            outputs = model.forward(inputs) #forward pass
            optimizer.zero_grad() #caluclate the gradient, manually setting to 0
            
            
            
            # obtain the loss function
            
            loss = criterion(outputs.double(), labels.double())
    
            loss.backward() #calculates the loss of the loss function
    
            optimizer.step() #improve from loss, i.e backprop
            
            # accumulate loss
            running_loss += loss.item()
            
            # Make prediction for batch
            _, predicted = outputs.max(1)
            
            # Store accuracy for batch
            # WE convert back from one-hot to integer for checking accuracy
            
            #print(predicted.shape)
            #print(torch.argmax(labels, dim=1).shape)
            total += labels.size(0)
            correct += predicted.eq(torch.argmax(labels, dim=1)).sum().item()
            
        # Store accuracy,loss for epoch
        train_loss=running_loss/len(trainloader)
        train_accuracy=100.*correct/total
        
        # At the end of each epoch, calculate validation accuracy
        
        # Set the network in eval mode since we're not training here
        model.eval()
        
         # Turn gradient computation off
        with torch.no_grad():
            val_accuracy, val_loss = validate(model, valloader, criterion)
        
        # Set the network back in training mode
        model.train()
        
        stats['train_accuracies'].append(train_accuracy)
        stats['train_losses'].append(train_loss)
        stats['val_accuracies'].append(val_accuracy)
        stats['val_losses'].append(val_loss)
        
        
        # Display results
        print(f'Epoch: {epoch}')
        print(f'\t -- Train Loss: {train_loss} | Train Accuracy: {train_accuracy}')
        print(f'\t -- Val Loss: {val_loss} | Val Accuracy: {val_accuracy}')
        
    
    return stats
######################################  HYPERPARAMETERS #######################
batch_size=10
num_epochs = 20 
learning_rate = 0.01 #0.001 lr
hidden_dim = 4 #number of features in hidden state
layer_dim = 1 #number of stacked lstm layers
###############################################################################

X_train, y_train, X_valid, y_valid, X_test, y_test, person_train, person_valid, person_test, original = getData()
############################## keep specific subjects ##########################
idx_train = get_idxs(person_train, np.array([1]))[0]
idx_test = get_idxs(person_test, np.array([1]))[0]
X_train = X_train[idx_train,...]
X_test = X_test[idx_test,...]
y_train = y_train[idx_train,...]
y_test = y_test[idx_test,...]

###################################### INPUT RESHAPING ########################
#LSTM: we need (N, 250, 22) #(N,L,H) input of batch_dim x seq_dim x feature_dim
"""
X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[2], X_train.shape[1]))
X_valid= torch.reshape(X_valid, (X_valid.shape[0], X_valid.shape[2], X_valid.shape[1]))
X_test= torch.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))
"""
#CNN_LSTM: 
X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_valid= torch.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], X_valid.shape[2]))
X_test= torch.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))   
################################################################################


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


feature_dim = X_train.shape[2]#(X_train.shape[1],X_train.shape[2]) #number of features
seq_dim = X_train.shape[1]


input_dim = 22
output_dim = 4 #number of output classes
seq_dim = 250  # Number of steps to unroll

criterion = torch.nn.MSELoss()#torch.nn.L1Loss()

stats = _LSTM_(input_dim, hidden_dim, layer_dim, output_dim, criterion, num_epochs)


acc_name = 'figures/1_lstm_cnn_acc_hid_'+str(hidden_dim)+'_lay_'+str(layer_dim)+'_loss_'+'MSE'+'_lr_'+str(learning_rate)+'_batch_'+str(batch_size)+'.png'

plt.figure()
plt.plot(stats['train_accuracies'])
plt.plot(stats['val_accuracies'])

plt.legend(['train','validation'])
plt.title('subject1 Accuracy')
plt.savefig(acc_name)

loss_name = 'figures/1_lstm_cnn_loss_hid_'+str(hidden_dim)+'_lay_'+str(layer_dim)+'_loss_'+'MSE'+'_lr_'+str(learning_rate)+'_batch_'+str(batch_size)+'.png'
plt.figure()
plt.plot(stats['train_losses'])
plt.plot(stats['val_losses'])

plt.legend(['train','validation'])
plt.title('subject1 loss')
plt.savefig(loss_name)

#don't forget to edit figure names, here it's subject specific
