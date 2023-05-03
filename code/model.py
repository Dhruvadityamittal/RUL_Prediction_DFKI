import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CNN_Model(nn.Module):
    
    def __init__(self,input_size):
        super(CNN_Model, self).__init__()
        filter_size_1 = 21
        filter_size=21
        self.conv1 = nn.Conv1d(1,16,kernel_size = filter_size_1, stride=1,padding=filter_size_1//2)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.max_pool1 = nn.MaxPool1d(2)
        

        self.conv2 = nn.Conv1d(16,32, kernel_size = filter_size_1, stride = 1,padding=filter_size_1 //2)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32,64, kernel_size = filter_size, stride = 1,padding=filter_size //2)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(64,128, kernel_size = filter_size, stride = 1,padding=filter_size //2)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.max_pool4 = nn.MaxPool1d(2)

        self.flatten_size = 128*math.floor(input_size/(2*2*2*2))
        self.flatten = nn.Flatten(start_dim=1)
        
        self.Linear1 = nn.Linear(self.flatten_size, input_size)
        self.batch_norm_linear = nn.BatchNorm1d(input_size)
        # self.a = nn.Linear()
        self.Linear2 = nn.Linear(input_size,1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)
        # print(self.flatten_size)
        
        
        
    def forward(self,x):
        x= x.view(x.shape[0],1,x.shape[1])
        
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.max_pool1(out)

        out = self.conv2(out)
        out = self.gelu(out)
        out = self.batch_norm2(out)
        out = self.dropout(out)
        out = self.max_pool2(out)   

        out = self.conv3(out)
        out = self.gelu(out)
        out = self.batch_norm3(out)
        out = self.dropout(out)
        out = self.max_pool3(out) 


        out = self.conv4(out)
        out = self.gelu(out)
        out = self.batch_norm4(out)
        out = self.dropout(out)
        out = self.max_pool4(out) 

        out = self.flatten(out)
        
        out = self.Linear1(out)
        out = self.gelu(out)    
        out = self.Linear2(out)
        
        return out