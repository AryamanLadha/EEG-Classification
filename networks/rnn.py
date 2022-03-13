"""
import tensorflow.compat.v1 as tf
import numpy as np
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
from tensorflow.keras import layers


class RNN:
  
  def __init__(self, in_shape):
      super().__init__()
      self.in_shape = in_shape
      
      
      
  def LSTM(self):
        

      #base model
      inputs = Input(shape=self.in_shape, name='eeg_in')
      
      LSTM_1, state_h1, state_c1 = LSTM(64, dropout=0.5, return_sequences=True, return_state= True)(inputs)
      LSTM_2, state_h2, state_c2 = LSTM(64, dropout=0.5, return_sequences=True, return_state= True)(LSTM_1)
      LSTM_3, state_h3, state_c3 = LSTM(64, dropout=0.5, return_sequences=True, return_state= True)(LSTM_2)
      LSTM_4, state_h4, state_c4 = LSTM(64, dropout=0.5, return_sequences=True, return_state= True)(LSTM_3)


      lstm_out = tf.reshape(state_h4, shape=[-1, 64, 1], name='lstm_out')
      lstm_flat = Flatten()(lstm_out)
      predictions = Dense(4, activation='softmax', name = 'dense2')(lstm_flat)

      self.model = Model(inputs=inputs,outputs=predictions) 
      #self.model.summary()
      
            
      
  def GRU(self):
    
      self.model = K.Sequential()
      self.model.add(layers.Embedding(input_dim=self.in_shape, output_dim=64))
      # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
      self.model.add(layers.GRU(256, return_sequences=True))
      # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
      self.model.add(layers.SimpleRNN(128))
      self.model.add(layers.Dense(4))
      #self.model.summary()
      
      
    
  def fit_model(self, X, y, X0, y0):
    
      opt_adam = K.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
      self.model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
      es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=30)
      #mc = ModelCheckpoint(os.path.join(SAVEPATH, BESTMODEL), monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
      history = self.model.fit(x=X, y=y, epochs=15, shuffle=True, 
                    verbose=1, validation_data = (X0, y0), callbacks=[es])
      
      return history
    
  def evaluate(self, X, y):
    
      self.model.evaluate(X,y)

"""
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 

#input of batch_dim x seq_dim x feature_dim   
#torch.set_default_tensor_type(torch.DoubleTensor)     
class LSTMModel(nn.Module):
    """
    input_dim = 22
    output_dim = 4 #number of output classes
    seq_dim = 250
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.double()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out
    
class LSTM_CNN(nn.Module):

    """
    
    Our input (C,H,W) is (22,250,1)
    
       
    input_dim = 22
    output_dim = 4 #number of output classes
    seq_dim = 250
    
    
    """
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=22, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(in_channels=64,out_channels=128, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        
        self.lstm1 = torch.nn.LSTM(
            input_size=238 ,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
        )
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        return (x)

    
