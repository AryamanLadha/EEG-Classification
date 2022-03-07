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


from utils.preprocess import getData
from networks.rnn import RNN

X_train, y_train, X_valid, y_valid, X_test, y_test = getData()

RNN_network = RNN(X_train)
RNN_network.LSTM()
history = RNN_network.fit_model(X_train, y_train, X_valid, y_valid)

plt.figure()
plt.plot(history['val_categorical_accuracy'])
plt.legend(['validation accuracy'])
plt.title('LSTM')

RNN_network.evaulate(X_test,y_test)

RNN_network = RNN(X_train)
RNN_network.GRU()
history = RNN_network.fit_model(X_train, y_train, X_valid, y_valid)

plt.figure()
plt.plot(history['val_categorical_accuracy'])
plt.legend(['validation accuracy'])
plt.title('GRU')

RNN_network.evaulate(X_test,y_test)


