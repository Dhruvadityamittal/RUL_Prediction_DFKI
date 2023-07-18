import pickle
import numpy as np
import matplotlib.pylab as plt
import os
from scipy import interpolate
import torch

from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new, Autoencoder, LSTM_Model
from util_FPC import EarlyStopping, get_fpc,plot_RUL, get_data, get_change_indices
from train_model import train_model,perform_n_folds
from import_file import *
from dataloader import get_RUL_dataloader
from torch.utils.data import Dataset, DataLoader
from dataloader import battery_dataloader


fea_num =1
discharge_capacities_SNL = np.load(f"./Datasets/snl_data_{fea_num}.npy",allow_pickle=True)

name_start_train = 0
name_start_test = 70

d = []
for battery_temp in discharge_capacities_SNL:
    a = np.squeeze(battery_temp[0], axis = 1)     # Voltage/Current Features
    b = np.expand_dims(battery_temp[1], axis =1)  # Discharge Capccity
    c = np.concatenate((b,a), axis =1).T
    d.append(c)

discharge_capacities_SNL = d

percentage  = 0.10  # 10 percent data
window_size = 50    # window size
stride = 1          # stride
channels  =[0] # channels


train_data_SNL, FPC_data_SNL, FPC_data_dict_SNL= get_data(discharge_capacities_SNL[0:70], 
                                                              percentage, window_size , stride, channels, "train" , name_start = name_start_train)
test_data_SNL,test_data_dict_SNL  = get_data(discharge_capacities_SNL[70:],None,window_size,stride,channels,type= "test",name_start = name_start_test)


obj_train_SNL  = battery_dataloader(train_data_SNL)
obj_FPC_SNL  = battery_dataloader(FPC_data_SNL)
obj_test_SNL  = battery_dataloader(test_data_SNL)

train_dataloader_SNL = DataLoader(obj_train_SNL, batch_size=8,shuffle=True)
FPC_dataloader_SNL   = DataLoader(obj_FPC_SNL,batch_size=1,shuffle=False)
test_dataloader_SNL = DataLoader(obj_test_SNL, batch_size=1,shuffle=False)


print("Shape of a batch    :",next(iter(train_dataloader_SNL))[0].shape)


device = "cpu"
epochs = 2
learning_rate = 0.001

pretrained = True
load_pretrained = False
version = 1

ch = ''.join(map(str,channels))

# model = CNN_Model(window_size,len(channels))
model = LSTM_Model(window_size,len(channels))

model_dir = "./Weights/FPC/"
model_path = f'{model_dir}/model_SNL_f{ch}_f{window_size}_f{model.name}_f{version}.pth'

if(load_pretrained):
    model.load_state_dict(torch.load(model_path, map_location=device ))

model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas= (0.9, 0.99))
criterion = nn.BCELoss()

early_stopping = EarlyStopping(patience=20)


if(pretrained):
    
    model.load_state_dict(torch.load(model_path, map_location=device ))
    model.to(device)
else:
    model = train_model(model, optimizer, criterion, early_stopping,train_dataloader_SNL,epochs,learning_rate,load_pretrained,model_path,version)

version = 2
# Get Change Indices


change_indices_train,change_indices_test, _, _ = get_change_indices(model,discharge_capacities_SNL,
                                                                    channels,get_saved_indices = True, version = 1, name_start_train=name_start_train, name_start_test= name_start_test, dataset = "SNL")
change_indices_all = np.concatenate((change_indices_train,change_indices_test))


channels_RUL = [0]
window_size_RUL = 50
stride_RUL =1
c_RUL = ''.join(map(str,channels_RUL))

n_folds = 5
scenario = 1
# learning_rate = 0.01
learning_rate = 0.0001
epochs = 10

parameters = {
    "window_size" : window_size,
    "stride": stride,
    "channels": channels_RUL,
    "epochs": epochs,
    "learning_rate": learning_rate
}



model_RUL = LSTM_Model_RUL(window_size,len(channels))  # LSTM Model
# model_RUL = Net(len(channels))    # Transformer Model
#model_RUL = CNN_Model_RUL(window_size,channels)    # CNN Model

optimizer = torch.optim.Adam(model_RUL.parameters(), lr = learning_rate, betas= (0.9, 0.99))
# criterion = nn.L1Loss()
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=50)

version = 1
pretrained_RUL_scenario1 = False
load_pretrained_scenario1  = False

model_dir_scenario1 = "./Weights/Scenario1/"
model_path_scenario1 = f'{model_dir_scenario1}/model_SNL_f{model_RUL.name}_f{c_RUL}_f{window_size_RUL}_f{version}.pth'

if(pretrained_RUL_scenario1):
    print("Loading a Pre-trained Model")
    model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
    test_batteries = np.load("./Test_data/test_batteries_SNL.npy",allow_pickle=True)
    train_batteries = np.load("./Test_data/train_batteries_SNL.npy",allow_pickle=True)
    
    
else:
    if(load_pretrained_scenario1):
        print("Training further on already trained model")
        model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
        model_RUL, test_dataloader_RUL, test_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities_SNL,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version)
    else:
        print("Training a new Model")
        model_RUL, test_dataloader_RUL, test_batteries, train_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities_SNL,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version)
        np.save(f"./Test_data/test_batteries_SNL.npy", test_batteries, allow_pickle=True)
        np.save(f"./Test_data/train_batteries_SNL.npy", train_batteries, allow_pickle=True)


_, _, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities_SNL, train_batteries, test_batteries, 
                                              change_indices_all, parameters["window_size"],
                                              parameters["stride"],parameters["channels"] ,scenario)

plot_RUL(model_RUL,discharge_capacities_SNL,test_batteries,test_dataloader_RUL,
         change_indices_all,"Outputs/scenario1_RUL_prediction_test_SNL")