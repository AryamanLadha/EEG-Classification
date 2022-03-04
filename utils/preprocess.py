import numpy as np
import torch
import torch.nn.functional as F

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


def getData():
    """
    Load the data from the .npy files, preprocess it, and return it.
    We return data in the form (N,C,H,W), since this is the format used by Pytorch's convolutional layers
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
    
    
    # Preprocess
    X_train_valid_prep,y_train_valid_prep = data_prep(X_train_valid,y_train_valid,2,2,True)
    X_test_prep,y_test_prep = data_prep(X_test,y_test,2,2,True)
    
    
    # First generating the training and validation indices using random splitting
    ind_valid = np.random.choice(8460, 1500, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid] 
    (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]
    
    # Converting the labels to categorical variables for multiclass classification -> Find pytorch package to do this
    # We need to convert all y's to tensors for this.
    # x.type(torch.LongTensor))
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
    
    
    # Right now, our x's are of the form (N,C,H,W), as desired
    # We don't return person_data right now.
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test
    
    
    
    
    
    