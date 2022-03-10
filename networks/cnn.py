import torch.nn as nn # torch neural network modules
import torch.nn.functional as F # torch functions, usually w/o learnable parameter
import torch

class BasicCNN(nn.Module):
    """
    A convolutional neural network. All filters are 10x1, but different layers have different number of filters.
    We pad to keep the height and width of the output the same as the input.
    We use a relu activation function after each convolutional layer.
    
    Our input (C,H,W) is (22,250,1)
    
    Code inspired from Tonmoy'w Week 9 Discussion Notebook.
    
    At some point, we should convert this to EEGNet.
    Other ideas: 
        -- Batchnorm vs dropout?
        -- Using [conv-relu-batchnorm-conv-relu-batchnorm-pool]Xn i.e multiple convolutions before a pool
        --- Using [conv-batchnorm-relu-conv-batchnorm-relu-pool]Xn i.e multiple convolutions before a pool, batchnorm before relu
    
    """
    
    def __init__(self):
        super().__init__()
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=25, kernel_size=(10,1), padding='same') 
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=25)
        self.dropout1 = nn.Dropout(p=0.4)
        
        
        # Conv layer 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10,1), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=50)
        self.dropout2 = nn.Dropout(p=0.4)
        
        # Conv layer 3
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10,1), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=100)
        self.dropout3 = nn.Dropout(p=0.4)
        
        # Conv layer 4
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10,1), padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=200)
        self.dropout4 = nn.Dropout(p=0.4)
        
        # Affine layer
        self.affine = nn.Linear(200*250*1,4)
        
    def forward(self,x):
        # Each layer does conv -> relu -> pool -> batchnorm
        x = self.dropout1(self.batchnorm1(self.pool1(F.relu(self.conv1(x)))))
        x = self.dropout2(self.batchnorm2(self.pool2(F.relu(self.conv2(x)))))
        x = self.dropout3(self.batchnorm3(self.pool3(F.relu(self.conv3(x)))))
        x = self.dropout4(self.batchnorm4(self.pool4(F.relu(self.conv4(x)))))
      
        # Flatten and pass through affine layer to get a vector (4,1) vector to pass into the softmax function per example.
        x = torch.flatten(x, 1)
        x = self.affine(x)
        return x
    
class OptimizedCNN(nn.Module):
    """
    A convolutional neural network. All filters are the same size, but different layers have different number of filters.
    We pad to keep the height and width of the output the same as the input.
    We use a relu activation function after each convolutional layer.
    
    Our input (C,H,W) is (22,250,1)
    
    The model architecture is:

    [conv-batchnorm-relu-dropout-conv-batchnorm-relu-dropout-pool]X2 - affine -> output into softmax
    
    i.e multiple convolutions before a pool, batchnorm before relu
    
    Things to try 
    -> increase dropout
    -> change filter size for convolutional layer
    -> change filter size, stride for pooling layer to reduce dimensions as we pass through network.
    -> varying the number of filters for each layer
    
    
    """
    
    def __init__(self, filter_size=(10,1), dropout=0.4):
        super().__init__()
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=25, kernel_size=filter_size, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(num_features=25)
        self.dropout1 = nn.Dropout(p=dropout)
        
        
        # Conv layer 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=filter_size, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(num_features=50)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.pool1 =  nn.MaxPool2d(kernel_size=(10, 1))
        
        # Conv layer 3
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=filter_size, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(num_features=100)
        self.dropout3 = nn.Dropout(p=dropout)
        
        # Conv layer 4
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=filter_size, padding='same')
        self.batchnorm4 = nn.BatchNorm2d(num_features=200)
        self.dropout4 = nn.Dropout(p=dropout)
        
        self.pool2 = nn.MaxPool2d(kernel_size=(10, 1))
        
        # Affine layer
        self.affine = nn.Linear(200*250*1,4)
        
    def forward(self,x):
        
        # [conv-batchnorm-relu-dropout-conv-batchnorm-relu-dropout]
        
        # (22,250,1)
        x = self.dropout1(self.batchnorm1(F.relu(self.conv1(x))))
        x = self.dropout2(self.batchnorm2(F.relu(self.conv2(x))))
        
        # pool
        
        # (22,250,1)
        x = self.pool1(x)
        
        #
       
        # [conv-batchnorm-relu-dropout-conv-batchnorm-relu-dropout]
        x = self.dropout3(self.batchnorm3(F.relu(self.conv3(x))))
        x = self.dropout4(self.batchnorm4(F.relu(self.conv4(x))))
        
        
        # pool
        x = self.pool2(x)
        
        # Flatten and pass through affine layer to get a vector (4,1) vector to pass into the softmax function per example.
        x = torch.flatten(x, 1)
        x = self.affine(x)
        
        return x
    
class OptimizedCNNV2(nn.Module):
    """
    A convolutional neural network. All filters are the same size, but different layers have different number of filters.
    We pad to keep the height and width of the output the same as the input.
    We use a relu activation function after each convolutional layer.
    
    Our input (C,H,W) is (22,250,1)
    
    The model architecture is:

    [conv-relu-pool-batchnorm-droput]X4 - affine -> output into softmax
    
    Things to try 
    -> increase dropout
    -> change filter size for convolutional layer
    -> change filter size, stride for pooling layer to reduce dimensions as we pass through network.
    -> varying the number of filters for each layer
    
    
    """
    
    def __init__(self, filter_size=(10,1), dropout=0.8):
        super().__init__()
        
        # (22,250,1)
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=25, kernel_size=filter_size, padding='same') 
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=25)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # (22,250,1)
        
        
        # Conv layer 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=filter_size, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=50)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # (22,250,1)
        
        # Conv layer 3
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=filter_size, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=100)
        self.dropout3 = nn.Dropout(p=dropout)
        
         # (22,250,1)
        
        # Conv layer 4
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=filter_size, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=200)
        self.dropout4 = nn.Dropout(p=dropout)
        
        # (22,250,1)
        
        # Affine layer
        self.affine = nn.Linear(200*250*1,4)
        
    def forward(self,x):
        
        # Each layer does conv -> relu -> pool -> batchnorm
        x = self.dropout1(self.batchnorm1(self.pool1(F.relu(self.conv1(x)))))
        x = self.dropout2(self.batchnorm2(self.pool2(F.relu(self.conv2(x)))))
        x = self.dropout3(self.batchnorm3(self.pool3(F.relu(self.conv3(x)))))
        x = self.dropout4(self.batchnorm4(self.pool4(F.relu(self.conv4(x)))))
      
        # Flatten and pass through affine layer to get a vector (4,1) vector to pass into the softmax function per example.
        x = torch.flatten(x, 1)
        x = self.affine(x)
        return x
    
class DeepCNN(nn.Module):
    """
    A convolutional neural network. All filters are the same size, but different layers have different number of filters.
    We pad to keep the height and width of the output the same as the input.
    We use a relu activation function after each convolutional layer.
    
    Our input (C,H,W) is (22,250,1)
    
    The model architecture is:

    [conv-relu-pool-batchnorm-droput]X6 - affine -> output into softmax
    
    Things to try 
    -> increase dropout
    -> change filter size for convolutional layer
    -> change filter size, stride for pooling layer to reduce dimensions as we pass through network.
    -> varying the number of filters for each layer
    
    
    """
    
    def __init__(self, filter_size=(10,1), dropout=0.4):
        super().__init__()
        
        # (22,250,1)
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=25, kernel_size=filter_size, padding='same') 
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=25)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # (25,250,1)
        
        
        # Conv layer 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=filter_size, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=50)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # (50,250,1)
        
        # Conv layer 3
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=filter_size, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=100)
        self.dropout3 = nn.Dropout(p=dropout)
        
         # (100,250,1)
        
        # Conv layer 4
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=filter_size, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=200)
        self.dropout4 = nn.Dropout(p=dropout)
        
        # (200,250,1)
        
        # Conv layer 5
        self.conv5 = nn.Conv2d(in_channels=200, out_channels=150, kernel_size=filter_size, padding='same')
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm5 = nn.BatchNorm2d(num_features=150)
        self.dropout5 = nn.Dropout(p=dropout)
        
        # (150,250,1)
        
        # Conv layer 6
        self.conv6 = nn.Conv2d(in_channels=150, out_channels=50, kernel_size=filter_size, padding='same')
        self.pool6 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm6 = nn.BatchNorm2d(num_features=50)
        self.dropout6 = nn.Dropout(p=dropout)
        
        # (50,250,1)
        
        # Affine layer
        self.affine = nn.Linear(50*250*1,4)
        
    def forward(self,x):
        
        # Each layer does conv -> relu -> pool -> batchnorm
        x = self.dropout1(self.batchnorm1(self.pool1(F.relu(self.conv1(x)))))
        x = self.dropout2(self.batchnorm2(self.pool2(F.relu(self.conv2(x)))))
        x = self.dropout3(self.batchnorm3(self.pool3(F.relu(self.conv3(x)))))
        x = self.dropout4(self.batchnorm4(self.pool4(F.relu(self.conv4(x)))))
        
        x = self.dropout5(self.batchnorm5(self.pool5(F.relu(self.conv5(x)))))
        x = self.dropout6(self.batchnorm6(self.pool6(F.relu(self.conv6(x)))))
      
        # Flatten and pass through affine layer to get a vector (4,1) vector to pass into the softmax function per example.
        x = torch.flatten(x, 1)
        x = self.affine(x)
        
        return x
    