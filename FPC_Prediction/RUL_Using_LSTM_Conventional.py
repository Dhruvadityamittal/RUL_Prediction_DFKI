import numpy as np
import torch
import torch.nn as nn
import pandas

from matplotlib.pylab import plt
import warnings

from load_data_conventional import *
from dataloader_conventional import *
from train_model_conventional import *
from model_conventional import *


warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


discharge_capacities = np.load(r"./Datasets/discharge_capacity.npy", allow_pickle=True)
discharge_capacities = discharge_capacities.tolist()
percentage = 0.40

np.array(discharge_capacities[0]).shape

max_length_train,max_length_out = get_lengths(discharge_capacities,percentage)
max_length_out = max_length_out+1


channels = [0,1,2,3,4,5,6]


model = LSTM_Model_Conventional(max_length_train,len(channels))
# model = CNN_Model_Conventional(max_length_train,len(channels))
ch = ''.join(map(str,channels))
dataset = "MIT"
version = 1
fld = 3
print(model.name)

model_dir = "./Weights/Conventional/"
model_path = f'{model_dir}/{dataset}_{model.name}_Conventional_Channels={ch}_Version={version}_Fold{fld}.pth'
epochs = 500
load_pretrained = False
pretrained = False
lr = 0.001
n_folds = 5
parameters = {"epochs" : epochs,
                "learning_rate" : lr ,
                "percentage" : percentage,
                "max_length_train" :max_length_train,
                "max_length_out" :max_length_out,
                "channels" : channels
}

optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas= (0.9, 0.99))
criterion = nn.MSELoss()
early_stopping_patiance = 50


if(pretrained):
    model.load_state_dict(torch.load(model_path,map_location= device))
else:
    if(load_pretrained):
       model.load_state_dict(torch.load(model_path,map_location= device))

       model= perform_n_folds_conventional(model, n_folds,discharge_capacities, criterion, 
                    optimizer, early_stopping_patiance, model_path,
                    parameters,version, dataset)
    else:
        model = perform_n_folds_conventional(model, n_folds,discharge_capacities, criterion, 
                    optimizer, early_stopping_patiance, model_path,
                    parameters,version, dataset)





