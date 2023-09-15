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
from load_data import get_discharge_capacities_HUST,get_discharge_capacities_MIT


warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = "HUST"
# dataset = "Combined"

m_name = "CNN"
# m_name = "LSTM"

if(dataset == "MIT"):
    discharge_capacities = np.load(r"./Datasets/discharge_capacity.npy", allow_pickle=True)
    discharge_capacities = discharge_capacities.tolist()
    channels = [0]

elif(dataset == "HUST"):    
    discharge_capacities = get_discharge_capacities_HUST(fea_num=1)
    channels = [0]
else:
    discharge_capacities_MIT = get_discharge_capacities_MIT()
    discharge_capacities_HUST = get_discharge_capacities_HUST(fea_num=1)
    discharge_capacities = discharge_capacities_MIT[0:100] + discharge_capacities_HUST[0:70] + discharge_capacities_MIT[100:] + discharge_capacities_HUST[70:]
    channels = [0]


percentage = 0.40

max_length_train,max_length_out = get_lengths(discharge_capacities,percentage)
max_length_out = max_length_out+1

if(m_name == "LSTM"):
    model = LSTM_Model_Conventional(max_length_train,len(channels),max_length_out)
else:
    model = CNN_Model_Conventional(max_length_train,len(channels),max_length_out)


ch = ''.join(map(str,channels))
version = 1
fld = 3
print(device,model.name, dataset)

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





