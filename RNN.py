import numpy as np
import tensorflow.compat.v1 as tf
import os
from scipy import signal
import pandas as pd
from tensorflow import keras as K
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Dropout, LSTM, Conv1D, BatchNormalization, PReLU
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding
from tensorflow.keras import regularizers
import torch


from utils.preprocess import getData
from networks.rnn import RNN

X_train, y_train, X_valid, y_valid, X_test, y_test = getData()


X_train = torch.reshape(X_train, (X_train.shape[0], X_train.shape[2], X_train.shape[1]))
X_valid= torch.reshape(X_valid, (X_valid.shape[0], X_valid.shape[2], X_valid.shape[1]))
X_test= torch.reshape(X_test, (X_test.shape[0], X_test.shape[2], X_test.shape[1]))
# X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[2], X_train.shape[1]))
# X_valid = np.reshape(X_valid,(X_valid.shape[0], X_valid.shape[2], X_valid.shape[1]))
# X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[2], X_test.shape[1]))

# X_train = tf.convert_to_tensor(X_train, tf.int64)
# X_test = tf.convert_to_tensor(X_test, tf.int64)
# X_valid = tf.convert_to_tensor(X_valid, tf.int64)




shape = X_train.shape


RNN_network = RNN(shape[1:])
RNN_network.LSTM()
history = RNN_network.fit_model(X_train, y_train, X_valid, y_valid)

plt.figure()
plt.plot(history['val_categorical_accuracy'])
plt.legend(['validation accuracy'])
plt.title('LSTM')

RNN_network.evaulate(X_test,y_test)

RNN_network = RNN(shape[1:])
RNN_network.GRU()
history = RNN_network.fit_model(X_train.numpy(), y_train, X_valid.numpy(), y_valid)

plt.figure()
plt.plot(history['val_categorical_accuracy'])
plt.legend(['validation accuracy'])
plt.title('GRU')

RNN_network.evaulate(X_test.numpy(),y_test)


