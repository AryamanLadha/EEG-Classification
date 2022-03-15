#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:04:20 2022

@author: golaraahmadiazar
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import Conv2D,LSTM,BatchNormalization,MaxPooling2D,Reshape
#from keras.utils import to_categorical
from utils.preprocess import getData
import matplotlib.pyplot as plt
from utils.validate import validate
from utils.test_accuracy import test

def keras_hybrid(batch_size,num_epochs):
    
    # Building the CNN model using sequential class
    hybrid_cnn_lstm_model = Sequential()
    
    # Conv. block 1
    hybrid_cnn_lstm_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.5))
    
    # Conv. block 2
    hybrid_cnn_lstm_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.5))
    
    # Conv. block 3
    hybrid_cnn_lstm_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.5))
    
    # Conv. block 4
    hybrid_cnn_lstm_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
    hybrid_cnn_lstm_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
    hybrid_cnn_lstm_model.add(BatchNormalization())
    hybrid_cnn_lstm_model.add(Dropout(0.5))
    
    # FC+LSTM layers
    hybrid_cnn_lstm_model.add(Flatten()) # Adding a flattening operation to the output of CNN block
    hybrid_cnn_lstm_model.add(Dense((100))) # FC layer with 100 units
    hybrid_cnn_lstm_model.add(Reshape((100,1))) # Reshape my output of FC layer so that it's compatible
    
    hybrid_cnn_lstm_model.add(LSTM(50, dropout=0.6, recurrent_dropout=0.1, input_shape=(100,1), return_sequences=False))
    hybrid_cnn_lstm_model.add(LSTM(10, dropout=0.6, recurrent_dropout=0.1, input_shape=(50,), return_sequences=False))
    
    # Output layer with Softmax activation 
    hybrid_cnn_lstm_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
    
    
    # Printing the model summary
    hybrid_cnn_lstm_model.summary()
    
    #hybrid_cnn_lstm_optimizer = keras.optimizers.Adam(lr=learning_rate)
    
    # Compiling the model
    hybrid_cnn_lstm_model.compile(loss='categorical_crossentropy',
                     #optimizer=hybrid_cnn_lstm_optimizer,
                     metrics=['accuracy'])
    
    # Training and validating the model
    hybrid_cnn_lstm_model_results = hybrid_cnn_lstm_model.fit(X_train,
                 y_train,
                 batch_size=batch_size,
                 epochs=num_epochs,
                 validation_data=(X_valid, y_valid), verbose=True)
    
    return hybrid_cnn_lstm_model_results#.history()

######################################  HYPERPARAMETERS #######################
batch_size=5
num_epochs = 20
learning_rate = 0.001 #0.001 lr
hidden_dim = 2 #number of features in hidden state
layer_dim = 2 #number of stacked lstm layers
###############################################################################

X_train, y_train, X_valid, y_valid, X_test, y_test, original = getData(lib='keras')#, subject=[1]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

  

################################################################################

input_dim = 22
output_dim = 4 #number of output classes
seq_dim = 250  # Number of steps to unroll
stats = keras_hybrid(batch_size, num_epochs)


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

print('best train loss:',min(stats['train_losses']))
print('best validation loss:',min(stats['val_losses']))
print('best train acc:',max(stats['train_accuracies']))
print('best validation acc:',max(stats['val_accuracies']))


