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
        
        
        # Conv layer 2
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(10,1), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=50)
        
        # Conv layer 3
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(10,1), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm3 = nn.BatchNorm2d(num_features=100)
        
        # Conv layer 4
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(10,1), padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 1), padding=(1,0), stride=1)
        self.batchnorm4 = nn.BatchNorm2d(num_features=200)
        
        # Affine layer
        self.affine = nn.Linear(200*250*1,4)
        
    def forward(self,x):
        # Each layer does conv -> relu -> pool -> batchnorm
        x = self.batchnorm1(self.pool1(F.relu(self.conv1(x))))
        x = self.batchnorm2(self.pool2(F.relu(self.conv2(x))))
        x = self.batchnorm3(self.pool3(F.relu(self.conv3(x))))
        x = self.batchnorm4(self.pool4(F.relu(self.conv4(x))))
      
        # Flatten and pass through affine layer to get a vector (4,1) vector to pass into the softmax function per example.
        x = torch.flatten(x, 1)
        x = self.affine(x)
        return x