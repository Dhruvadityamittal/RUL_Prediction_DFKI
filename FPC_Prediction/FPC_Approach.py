import numpy as np
import torch
import torch.nn as nn
import pandas
from torch.utils.data import DataLoader
from matplotlib.pylab import plt
import warnings
from torchmetrics.classification import BinaryAccuracy
import os
import shutil
from util import EarlyStopping, get_fpc, plot_RUL
from load_data import get_data,get_data_RUL
from dataloader import battery_dataloader,battery_dataloader_RUL
from model import CNN_Model,LSTM_Model_RUL
from train_model import train_model,train_model_RUL
import argparse



warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--epoch", type=int, default=100)
    # parser.add_argument("--lr", type=float, default=1e-4)
    
    # args = parser.parse_args()

    # Loading the data
    discharge_capacities = np.load(r"./Datasets/discharge_capacity.npy", allow_pickle=True)
    discharge_capacities = discharge_capacities.tolist()

    # Parameters
    percentage  = 0.10  # 5 percent data
    window_size = 50    # window size
    stride = 1          # stride
    channels  = 7       # channels
    epochs = 100


    train_data,FPC_data = get_data(discharge_capacities[:100],percentage,window_size,stride,channels,type = "train")
    test_data = get_data(discharge_capacities[100:],None,window_size,stride,channels,type= "test")

    obj_train  = battery_dataloader(train_data)
    obj_FPC  = battery_dataloader(FPC_data)
    obj_test  = battery_dataloader(test_data)


    train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
    FPC_dataloader   = DataLoader(obj_FPC,batch_size=1,shuffle=False)
    test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)


    print("Number of Channels  :", channels)
    print("Shape of a batch    :",next(iter(train_dataloader))[0].shape)


    train_model(window_size,channels,train_dataloader,epochs, pretrained = True)
    model = CNN_Model(window_size,channels)
    model.load_state_dict(torch.load("./Weights/model_f7_f50.pth", map_location=device ))
    model.to(device)


    no_of_channels = [1,2,3,4,5,6,7]
    no_of_channels = [7]
    changes_train = []
    changes_test = []
    epochs = 50
    # os.mkdir("/kaggle/working/change_indices")
    get_saved_indices = True


    batteries_train_show =[i for i in range (14)]
    batteries_test_to_show= [i+100 for i in range(0,14)]
    
    change_percentage_train, change_indices_train =  get_fpc(model,batteries_train_show,discharge_capacities,FPC_dataloader,True, True,"Train")
    change_percentage_test, change_indices_test =  get_fpc(model,batteries_test_to_show,discharge_capacities,test_dataloader,True, False,"Test")

    print("Average FPC on training :", np.mean(change_percentage_train))
    print("Average FPC on testing :", np.mean(change_percentage_test))
    
    if(get_saved_indices == False):

            batteries_train =[i for i in range (100)]
            batteries_test= [i+100 for i in range(0,24)]

            change_percentage_train, change_indices_train =  get_fpc(batteries_train,discharge_capacities,FPC_dataloader,False, False)
            change_percentage_test, change_indices_test =  get_fpc(batteries_test,discharge_capacities,test_dataloader,False, False)
                        
            if(os.path.exists("./change_indices") == False):
                os.mkdir("./change_indices")

            np.save("./change_indices/change_indices_train.npy",change_indices_train, allow_pickle=True)
            np.save("./change_indices/change_indices_test.npy",change_indices_test, allow_pickle=True)
    else:
        print("Loading Pre")
        change_indices_train = np.load("./change_indices/change_indices_train.npy" , allow_pickle=True)
        change_indices_test = np.load("./change_indices/change_indices_test.npy",allow_pickle=True)

    # RUL Prediction
    
    train_data_RUL= get_data_RUL(discharge_capacities[:100],change_indices_train,window_size,stride,channels,"Train")
    obj_train_RUL  = battery_dataloader_RUL(train_data_RUL)

    test_data_RUL= get_data_RUL(discharge_capacities[100:],change_indices_test,window_size,stride,channels,"Test")
    obj_test_RUL  = battery_dataloader_RUL(test_data_RUL)


    train_dataloader_RUL = DataLoader(obj_train_RUL, batch_size=128,shuffle=True)
    train_dataloader_RUL_temp = DataLoader(obj_train_RUL, batch_size=1,shuffle=False)

    test_dataloader_RUL = DataLoader(obj_test_RUL, batch_size=1,shuffle=False)


    train_model_RUL(window_size, channels,train_dataloader_RUL,200,pretrained = True) 

    model_RUL = LSTM_Model_RUL(window_size,channels)
    model_RUL.load_state_dict(torch.load("./Weights_RUL/model_RUL_f7_f50.pth",map_location= device))
    model_RUL.to(device)


    batteries =[0]
    plot_RUL(model_RUL,discharge_capacities,batteries,train_dataloader_RUL_temp,change_indices_train,False)





