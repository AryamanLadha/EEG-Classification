import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 

DROPOUT = 0.4

class LSTM(nn.Module):
    
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(22, 64, 3, batch_first=True, dropout=DROPOUT)

        self.fc = nn.Sequential(
            nn.Linear(64, 54),
            nn.BatchNorm1d(num_features=54, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(54, 44),
            nn.BatchNorm1d(num_features=44, eps=1e-05, momentum=0.2, affine=True),
            nn.ReLU(inplace = True),
            nn.Linear(44, 4)
        )
        self.affine = nn.Linear(200*250*1,4)
    
    def forward(self, x, h=None):

        # LSTM
        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 2, 1)
        out, _ = self.lstm(x)

        # Fully Connected Layer
        #out = self.fc(out[:, -1, :])
        out = torch.flatten(out,1)
        out = self.affine(out)

        return out

class LSTM_CNN(nn.Module):

    """
    
    Our input (C,H,W) is (22,250,1)
    
       
    input_dim = 22
    output_dim = 4 #number of output classes
    seq_dim = 250
    
    
    """
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
    """
    
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(

            # Input: N x 1 x 22 x 1000

            ### Conv-Pool Block 1
            # Convolution (temporal)
            #nn.Conv2d(1, 25, kernel_size=(1, 10), stride=1, padding=0),
            nn.Conv2d(22, 25, kernel_size=(10,1), stride=1, padding=0),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(3, 3), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            nn.Conv2d(25, 25, kernel_size=(18, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            
            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 2
            # Convolution
            #nn.Conv2d(1, 50, kernel_size=(25, 10), stride=1, padding=0),
            nn.Conv2d(25, 50, kernel_size=(10,1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 3
            # Convolution
            #nn.Conv2d(1, 100, kernel_size=(50, 10), stride=1, padding=0),
            nn.Conv2d(50, 100, kernel_size=(10,1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.2, affine=True),

            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            # Dropout
            nn.Dropout(p=DROPOUT),


            ### Conv-Pool Block 4
            # Convolution
            #nn.Conv2d(1, 200, kernel_size=(100, 10), stride=1, padding=0),
            nn.Conv2d(100, 200, kernel_size=(10, 1), stride=1, padding=0),
            nn.ELU(alpha=0.9, inplace=True),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.2, affine=True),
            
            # Max Pooling
            Permute(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

        )
        
        self.lstm = nn.LSTM(7, 64, 3, batch_first=True, dropout=DROPOUT)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 4),
        )
        self.affine = nn.Linear(200*250*1,4)
    
    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # LSTM
        N, C, H, W = x.size()
        x = x.view(N, H, W).permute(0, 1, 2)
        out, _ = self.lstm(x)

        # Fully Connected Layer
        #out = self.fc(out[:, -1, :])
        out = torch.flatten(out,1)
        out = self.affine(out)

        return out
    #"""
