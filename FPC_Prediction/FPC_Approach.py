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
from load_data import get_data,get_data_RUL_scenario1,get_data_RUL_scenario2
from dataloader import battery_dataloader,battery_dataloader_RUL
from model import CNN_Model,LSTM_Model_RUL
from train_model import train_model,train_model_RUL
import argparse
torch.manual_seed(0)



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
    percentage  = 0.10  # 10 percent data
    window_size = 50    # window size
    stride = 1          # stride
    channels  = 7       # channels


    train_data,FPC_data = get_data(discharge_capacities[:100],percentage,window_size,stride,channels,type = "train")
    test_data  = get_data(discharge_capacities[100:],None,window_size,stride,channels,type= "test")

    obj_train  = battery_dataloader(train_data)
    obj_FPC  = battery_dataloader(FPC_data)
    obj_test  = battery_dataloader(test_data)


    train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
    FPC_dataloader   = DataLoader(obj_FPC,batch_size=1,shuffle=False)
    test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)


    print("Number of Channels  :", channels)
    print("Shape of a batch    :",next(iter(train_dataloader))[0].shape)


    epochs = 100
    window_size = 50
    channels = 7
    learning_rate = 0.001
    stride =1

    pretrained = False
    load_pretrained = True
    version = 1

    if(pretrained):
        model = CNN_Model(window_size,channels)
        model.load_state_dict(torch.load("./Weights/FPC/model_f7_f50.pth", map_location=device ))
        model.to(device)
    else:
        model = train_model(window_size,channels,train_dataloader,epochs,learning_rate,load_pretrained,"./Weights/FPC/model_f7_f50.pth" ,"./Weights/FPC/",version)

    changes_train = []
    changes_test = []

    batteries_train =[i for i in range (100)]
    batteries_test= [i+100 for i in range(0,24)]

    change_percentage_train, change_indices_train =  get_fpc(model,batteries_train,discharge_capacities,FPC_dataloader,False, False,True)
    change_percentage_test, change_indices_test =  get_fpc(model,batteries_test,discharge_capacities,test_dataloader,False, False,False)


    changes_train.append(np.mean(change_percentage_train))
    changes_test.append(np.mean(change_percentage_test))
    
    
    if(os.path.exists("./change_indices") == False):
        os.mkdir("./change_indices")

    np.save("./change_indices/change_indices_train_f{channels}.npy",change_indices_train, allow_pickle=True)
    np.save("./change_indices/change_indices_test_f{channels}.npy",change_indices_test, allow_pickle=True)

    np.save("./change_indices/change_percentage_train_f{channels}.npy",change_percentage_train, allow_pickle=True)
    np.save("./change_indices/change_percentage_test_f{channels}.npy",change_percentage_test, allow_pickle=True)

    

    train_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities[:100],change_indices_train,window_size,stride,channels,"Train")
    obj_train_RUL_scenario1  = battery_dataloader_RUL(train_data_RUL_scenario1)

    test_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities[100:],change_indices_test,window_size,stride,channels,"Test")
    obj_test_RUL_scenario1  = battery_dataloader_RUL(test_data_RUL_scenario1)


    train_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities[:100],change_indices_train,window_size,stride,channels,"Train")
    obj_train_RUL_scenario2  = battery_dataloader_RUL(train_data_RUL_scenario2)

    test_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities[100:],change_indices_test,window_size,stride,channels,"Test")
    obj_test_RUL_scenario2  = battery_dataloader_RUL(test_data_RUL_scenario2)




    train_dataloader_RUL_scenario1 = DataLoader(obj_train_RUL_scenario1, batch_size=128,shuffle=True)
    train_dataloader_RUL_temp_scenario1 = DataLoader(obj_train_RUL_scenario1, batch_size=1,shuffle=False)
    test_dataloader_RUL_scenario1 = DataLoader(obj_test_RUL_scenario1, batch_size=1,shuffle=False)

    train_dataloader_RUL_scenario2 = DataLoader(obj_train_RUL_scenario2, batch_size=128,shuffle=True)
    train_dataloader_RUL_temp_scenario2 = DataLoader(obj_train_RUL_scenario2, batch_size=1,shuffle=False)
    test_dataloader_RUL_scenario2 = DataLoader(obj_test_RUL_scenario2, batch_size=1,shuffle=False)

    learning_rate = 0.00001
    epochs = 2
    pretrained_RUL_scenario1 = False
    pretrained_RUL_scenario2 = False

    version = 1


    if(pretrained_RUL_scenario1):
        model_RUL_scenario1 = LSTM_Model_RUL(window_size,channels)
        model_RUL_scenario1.load_state_dict(torch.load("./Weights/Scenario1/model_RUL_Scenario1_f7_f50.pth",map_location= device))
        model_RUL_scenario1.to(device)  

    else:
        load_pretrained_scenario1 = True
        model_RUL_scenario1 = train_model_RUL(window_size, channels,train_dataloader_RUL_scenario1,epochs,learning_rate,load_pretrained_scenario1,"./Weights/Scenario1/model_RUL_Scenario1_f7_f50.pth","./Weights_RUL/Scenario1/",version) 

    if(pretrained_RUL_scenario2):
        model_RUL_scenario2 = LSTM_Model_RUL(window_size,channels)
        model_RUL_scenario2.load_state_dict(torch.load("./Weights/Scenario2/model_RUL_Scenario2_f7_f50.pth",map_location= device))
        model_RUL_scenario2.to(device)  
    else:
        load_pretrained_scenario2 = True
        model_RUL_scenario2 = train_model_RUL(window_size, channels,train_dataloader_RUL_scenario2,epochs,learning_rate,load_pretrained_scenario2,"./Weights/Scenario2/model_RUL_Scenario2_f7_f50.pth","./Weights_RUL/Scenario2/",version) 


    

    batteries =[0,1]
    plot_RUL(model_RUL_scenario2,discharge_capacities,batteries,train_dataloader_RUL_temp_scenario2,change_indices_train,"scenario2_RUL_prediction_train")  

    test_batteries  = [i+100 for i in [1,2,3]]
    plot_RUL(model_RUL_scenario2,discharge_capacities,test_batteries,test_dataloader_RUL_scenario2,change_indices_test,"scenario2_RUL_prediction_test")
    
    test_batteries  = [i+100 for i in range(24)]
    plot_RUL(model_RUL_scenario1,discharge_capacities,test_batteries,test_dataloader_RUL_scenario1,change_indices_test,"scenario1_RUL_prediction_test")

    batteries =[0,1,2,3]
    plot_RUL(model_RUL_scenario1,discharge_capacities,batteries,train_dataloader_RUL_temp_scenario1,change_indices_train,"scenario1_RUL_prediction_train")

