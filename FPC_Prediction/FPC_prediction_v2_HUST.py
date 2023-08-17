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
from load_data import get_discharge_capacities_HUST


fea_num =1
discharge_capacities_HUST = get_discharge_capacities_HUST(fea_num)

name_start_train = 0
name_start_test = 70


percentage  = 0.10  # 10 percent data
window_size = 50    # window size
stride = 1          # stride
channels  =[0] # channels


train_data_HUST, FPC_data_HUST, FPC_data_dict_HUST= get_data(discharge_capacities_HUST[0:70], 
                                                              percentage, window_size , stride, channels, "train" , name_start = name_start_train)
test_data_HUST,test_data_dict_HUST  = get_data(discharge_capacities_HUST[70:],None,window_size,stride,channels,type= "test",name_start = name_start_test)


obj_train_HUST  = battery_dataloader(train_data_HUST)
obj_FPC_HUST  = battery_dataloader(FPC_data_HUST)
obj_test_HUST  = battery_dataloader(test_data_HUST)

train_dataloader_HUST = DataLoader(obj_train_HUST, batch_size=8,shuffle=True)
FPC_dataloader_HUST   = DataLoader(obj_FPC_HUST,batch_size=1,shuffle=False)
test_dataloader_HUST = DataLoader(obj_test_HUST, batch_size=1,shuffle=False)


print("Shape of a batch    :",next(iter(train_dataloader_HUST))[0].shape)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on ", device)
epochs = 10
learning_rate = 0.0001

pretrained = True
load_pretrained = False
version = 1

ch = ''.join(map(str,channels))

# model = CNN_Model(window_size,len(channels))
dataset = "HUST"
print("Dataset :",dataset)
model = LSTM_Model(window_size,len(channels))
fld = 1
model_dir = "./Weights/FPC/"
model_path = f'{model_dir}/{dataset}_{model.name}_FPC_Channels={ch}_WindowSize={window_size}_Version={version}.pth'

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
    model = train_model(model, optimizer, criterion, early_stopping,train_dataloader_HUST,epochs,learning_rate,load_pretrained,model_path,version)

version = 2
# Get Change Indices


change_indices_train,change_indices_test, _, _ = get_change_indices(model,discharge_capacities_HUST,
                                                                    channels,get_saved_indices = False, version = 1, name_start_train=name_start_train, name_start_test= name_start_test, dataset = "HUST")
change_indices_all = np.concatenate((change_indices_train,change_indices_test))

exit()
# RUL Code Start from here

channels_RUL = [0]
window_size_RUL = 50
stride_RUL =1
c_RUL = ''.join(map(str,channels_RUL))

n_folds = 5
scenario = 1
epochs = 600



parameters = {
    "window_size" : window_size,
    "stride": stride,
    "channels": channels_RUL,
    "epochs": epochs,
    "learning_rate": learning_rate
}



# model_RUL = LSTM_Model_RUL(window_size,len(channels))  # LSTM Model
model_RUL = Net(len(channels))    # Transformer Model
#model_RUL = CNN_Model_RUL(window_size,channels)    # CNN Model
print("Training RUL on :", model_RUL.name)

if(model_RUL.name == "LSTM"):
    learning_rate = 0.0001
else:
    learning_rate = 0.0001
print("Learning Rate :", learning_rate)
optimizer = torch.optim.Adam(model_RUL.parameters(), lr = learning_rate, betas= (0.9, 0.99))
# criterion = nn.L1Loss()
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=50)

version = 1
pretrained_RUL_scenario1 = False
load_pretrained_scenario1  = False

model_dir_scenario1 = "./Weights/Scenario1/"
model_path_scenario1 = f'{model_dir_scenario1}/{dataset}_f{model_RUL.name}_RUL_Channels={c_RUL}_WindowSize={window_size_RUL}_Version={version}_Fold={fld}.pth'



if(pretrained_RUL_scenario1):
    print("Loading a Pre-trained Model")
    model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
    test_batteries = np.load("./Test_data/test_batteries_HUST.npy",allow_pickle=True)
    train_batteries = np.load("./Test_data/train_batteries_HUST.npy",allow_pickle=True)
    
    
else:
    if(load_pretrained_scenario1):
        print("Training further on already trained model")
        model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
        model_RUL, test_dataloader_RUL, test_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities_HUST,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version,dataset)
    else:
        print("Training a new Model")
        model_RUL, test_dataloader_RUL, test_batteries, train_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities_HUST,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version,dataset)
        np.save(f"./Test_data/test_batteries_HUST.npy", test_batteries, allow_pickle=True)
        np.save(f"./Test_data/train_batteries_HUST.npy", train_batteries, allow_pickle=True)


_, _, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities_HUST, train_batteries, test_batteries, 
                                              change_indices_all, parameters["window_size"],
                                              parameters["stride"],parameters["channels"] ,scenario)

plot_RUL(model_RUL,discharge_capacities_HUST,test_batteries,test_dataloader_RUL,
         change_indices_all,"Outputs/scenario1_RUL_prediction_test_HUST")