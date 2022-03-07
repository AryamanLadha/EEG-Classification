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
  
  def __init__(self, X):
      super().__init__()
      self.in_shape = X.shape[1:]
      
      
      
  def LSTM(self):
        

      #base model
      inputs = Input(shape=self.in_shape, name='eeg_in')
      LSTM_1, state_h1, state_c1 = LSTM(128, dropout=0.5, return_sequences=True, return_state= True)(inputs)
      LSTM_2, state_h2, state_c2 = LSTM(128, dropout=0.5, return_sequences=True, return_state= True)(LSTM_1)
      LSTM_3, state_h3, state_c3 = LSTM(128, dropout=0.5, return_sequences=True, return_state= True)(LSTM_2)
      LSTM_4, state_h4, state_c4 = LSTM(128, dropout=0.5, return_sequences=True, return_state= True)(LSTM_3)


      lstm_out = tf.reshape(state_h4, shape=[-1, 128, 1], name='lstm_out')
      lstm_flat = Flatten()(lstm_out)
      predictions = Dense(4, activation='softmax', name = 'dense2')(lstm_flat)

      self.model = Model(inputs=inputs,outputs=predictions) 
      self.model.summary()
      
            
      
  def GRU(self):
    
      self.model = K.Sequential()
      self.model.add(layers.Embedding(input_dim=self.in_shape, output_dim=64))
      # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
      self.model.add(layers.GRU(256, return_sequences=True))
      # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
      self.model.add(layers.SimpleRNN(128))
      self.model.add(layers.Dense(4))
      self.model.summary()
      
      
    
  def fit_model(self, X_train, y_train, X_test, y_test):
    
      opt_adam = K.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
      self.model.compile(loss='categorical_crossentropy' , optimizer=opt_adam, metrics=['categorical_accuracy'])
      es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=30)
      #mc = ModelCheckpoint(os.path.join(SAVEPATH, BESTMODEL), monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
      history = self.model.fit(x=X_train, y=y_train, epochs=15, shuffle=True, 
                    verbose=1, validation_data = (X_test, y_test), callbacks=[es])
      
      return history
    
  def evaluate(self, X, y):
    
      self.model.evaluate(X,y)

        
    
