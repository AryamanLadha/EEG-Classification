import numpy as np
import torch
import torch.nn.functional as F
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def data_prep(X,y,sub_sample,average,noise):
    """
    Preprocess data using Tonmoy's discussion 9A code.
    """
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:500]
    # print('Shape of X after trimming:',X.shape)
    
    # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    
    
    total_X = X_max
    total_y = y
    # print('Shape of X after maxpooling:',total_X.shape)
    
    # Averaging + noise 
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
    
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    # print('Shape of X after averaging+noise and concatenating:',total_X.shape)
    
    # Subsampling
    
    for i in range(sub_sample):
        
        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
        
    
    # print('Shape of X after subsampling and concatenating:',total_X.shape)
    return total_X,total_y


def getData(lib='tensorflow'):
    """
    Load the data from the .npy files, preprocess it, and return it.
    We return data in the form (N,C,H,W), since this is the format used by Pytorch's convolutional layers
    Inputs
    - lib: The library you're using. Options are 'tensorflow', 'keras', 'torch'. Default is 'tensorflow'
    Outputs
    - x_train : [6960, 22, 250, 1] if on torch else [6960, 250, 1, 22]
    - y_train: [6960, 4]
    - person_train: [6960, 1]
    - x_valid: [1500, 22, 250, 1] if on torch else [1500, 250, 1, 22]
    - y_valid: [1500, 4]
    - person_valid: [1500, 1]
    - x_test: [1772, 22, 250, 1] if on torch else [1772, 250, 1, 22]
    - y_test:  [1772, 4]
    - person_test: [1772,1]
    """
    
    # Load the data
    X_test = np.load("./data/X_test.npy")
    y_test = np.load("./data/y_test.npy")
    person_train_valid = np.load("./data/person_train_valid.npy")
    X_train_valid = np.load("./data/X_train_valid.npy")
    y_train_valid = np.load("./data/y_train_valid.npy")
    person_test = np.load("./data/person_test.npy")
    
    ## Adjusting the labels so that 

    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3

    y_train_valid -= 769
    y_test -= 769
    
    original = {'y_test': y_test}
    
    
    # Preprocess
    X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)
    person_train_valid_prep = np.concatenate([person_train_valid]*4)
    person_test_prep = np.concatenate([person_test]*4)
    
    
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(8460, 1500, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))
   

    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
    (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
    (person_train, person_valid) = person_train_valid_prep[ind_train], person_train_valid_prep[ind_valid]
    
    
    
 
    if (lib == 'torch'):
        
        # Converting the labels to categorical variables for multiclass classification
        y_train = F.one_hot(torch.tensor(y_train).type(torch.int64), num_classes=4)
        y_valid = F.one_hot(torch.tensor(y_valid).type(torch.int64), num_classes=4)
        y_test = F.one_hot(torch.tensor(y_test_prep).type(torch.int64), num_classes=4)

        # Adding width of the segment to be 1 -> We are using Cov2d layers so we need a width.
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
        x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)

        # Convert all x's to tensors
        x_train = torch.from_numpy(x_train)
        x_valid = torch.from_numpy(x_valid)
        x_test = torch.from_numpy(x_test)

        person_train = torch.from_numpy(person_train)
        person_valid = torch.from_numpy(person_valid)
        person_test = torch.from_numpy(person_test)
        
    
    elif(lib == 'tensorflow' or lib == 'keras'):
        
        # Converting the labels to categorical variables for multiclass classification 
        
        y_train = to_categorical(y_train, 4)
        y_valid = to_categorical(y_valid, 4)
        y_test = to_categorical(y_test, 4)
    

        # Adding width of the segment to be 1 -> We are using Cov2d layers so we need a width.
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)
        x_test = X_test_prep.reshape(X_test_prep.shape[0], X_test_prep.shape[1], X_test_prep.shape[2], 1)
        
        # Reshaping the training and validation dataset for Keras API
        x_train = np.swapaxes(x_train, 1,3)
        x_train = np.swapaxes(x_train, 1,2)
        x_valid = np.swapaxes(x_valid, 1,3)
        x_valid = np.swapaxes(x_valid, 1,2)
        x_test = np.swapaxes(x_test, 1,3)
        x_test = np.swapaxes(x_test, 1,2)
    
        # Convert all to tensorflow tensors
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        
        x_valid = tf.convert_to_tensor(x_valid)
        y_valid = tf.convert_to_tensor(y_valid)
        
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.convert_to_tensor(y_test)

        person_train = tf.convert_to_tensor(person_train)
        person_valid = tf.convert_to_tensor(person_valid)
        person_test = tf.convert_to_tensor(person_test)
        
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test, person_train, person_valid, person_test, original
    
    
    
    
    
    