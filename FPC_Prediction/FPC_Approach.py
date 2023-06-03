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


    # Loading the data
    discharge_capacities = np.load(r"./Datasets/discharge_capacity.npy", allow_pickle=True)
    discharge_capacities = discharge_capacities.tolist()

    # Parameters
    percentage  = 0.10  # 10 percent data
    window_size = 50    # window size
    stride = 1          # stride
    channels_to_use  =[0,2,3]     # channels

    channels_to_use_s = ''.join(map(str,channels_to_use))

    epochs = 50
    window_size = 50
    learning_rate = 0.001

    pretrained = False
    load_pretrained = False
    version = 1

    save_dirs = ["Weights","Weights/FPC","Weights/Scenario1","Weights/Scenario2","Outputs"]
    for dir in save_dirs:
        if(os.path.isdir(dir)==False):
            print("Creating Directory :", dir)
            os.mkdir(dir)

    # Training the FPC model

    model_dir = "./Weights/FPC/"
    model_path = f'{model_dir}/model_FPC_f{channels_to_use_s}_f{window_size}_f{version}.pth'

    changes_train = []
    changes_test = []
   
    get_saved_indices = False

    if(not get_saved_indices):

        for channels in [channels_to_use]: 
            print("Channels used : ", channels)     

            train_data,FPC_data,FPC_data_dict = get_data(discharge_capacities[:100],percentage,window_size,stride,channels,type = "train")
            test_data,test_data_dict  = get_data(discharge_capacities[100:],None,window_size,stride,channels,type= "test")

            obj_train  = battery_dataloader(train_data)
            obj_FPC  = battery_dataloader(FPC_data)
            obj_test  = battery_dataloader(test_data)

            train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
            FPC_dataloader   = DataLoader(obj_FPC,batch_size=1,shuffle=False)
            test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)

            print("Shape of a batch    :",next(iter(train_dataloader))[0].shape)
    

            

            # Getting the FPC Model
            print("Getting FPC Model")
            if(pretrained):
                print("Getting Pretrained Model for FPC prediction")
                model = CNN_Model(window_size,len(channels))
                model.load_state_dict(torch.load(model_path, map_location=device ))
                model.to(device)
            else:
                print("Training the FPC model again")
                model = train_model(window_size,len(channels),train_dataloader,epochs,learning_rate,load_pretrained,model_path,version)

            # Getting Change Indices/ FPC points

            # Just checking the output on sample batteries
            val_batteries = [i for i in range(0,4)]
            _,_ = get_fpc(model,val_batteries,discharge_capacities,FPC_data_dict,True, True,True,"Outputs/FPC_Training")        
            val_batteries = [i+100 for i in range(0,4)]
            _,_ = get_fpc(model,val_batteries,discharge_capacities,test_data_dict,True, False,False,"Outputs/FPC_Testing")

            # Getting FPC for all battteries

            batteries_train =[i for i in range (100)]
            batteries_test= [i+100 for i in range(0,24)]

            change_percentage_train, change_indices_train =  get_fpc(model,batteries_train,discharge_capacities,FPC_data_dict,False, False,True,"Outputs/FPC_train")
            change_percentage_test, change_indices_test =  get_fpc(model,batteries_test,discharge_capacities,test_data_dict,False, False,False,"Outputs/FPC_test")

            print("Mean FPC for Training = {} and Testing = {}".format(np.mean(change_percentage_train), np.mean(change_percentage_train)))
            changes_train.append(np.mean(change_percentage_train))
            changes_test.append(np.mean(change_percentage_train))
            
            
            if(os.path.exists("./change_indices") == False):
                os.mkdir("./change_indices")

            np.save(f"./change_indices/change_indices_train_{channels_to_use_s}.npy",change_indices_train, allow_pickle=True)
            np.save(f"./change_indices/change_indices_test_{channels_to_use_s}.npy",change_indices_test, allow_pickle=True)

            np.save(f"./change_indices/change_percentage_train_{channels_to_use_s}.npy",change_percentage_train, allow_pickle=True)
            np.save(f"./change_indices/change_percentage_test_{channels_to_use_s}.npy",change_percentage_test, allow_pickle=True)

    else:
        print("Loading Old Indices and FPC Model")
        change_indices_train = np.load(f"./change_indices/change_indices_train_{channels_to_use_s}.npy" , allow_pickle=True)
        change_indices_test = np.load(f"./change_indices/change_indices_test_{channels_to_use_s}.npy",allow_pickle=True)




    # RUL Prediction

    train_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities[:100],change_indices_train,window_size,stride,channels_to_use,"Train")
    obj_train_RUL_scenario1  = battery_dataloader_RUL(train_data_RUL_scenario1)

    test_data_RUL_scenario1= get_data_RUL_scenario1(discharge_capacities[100:],change_indices_test,window_size,stride,channels_to_use,"Test")
    obj_test_RUL_scenario1  = battery_dataloader_RUL(test_data_RUL_scenario1)


    train_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities[:100],change_indices_train,window_size,stride,channels_to_use,"Train")
    obj_train_RUL_scenario2  = battery_dataloader_RUL(train_data_RUL_scenario2)

    test_data_RUL_scenario2= get_data_RUL_scenario2(discharge_capacities[100:],change_indices_test,window_size,stride,channels_to_use,"Test")
    obj_test_RUL_scenario2  = battery_dataloader_RUL(test_data_RUL_scenario2)




    train_dataloader_RUL_scenario1 = DataLoader(obj_train_RUL_scenario1, batch_size=128,shuffle=True)
    train_dataloader_RUL_temp_scenario1 = DataLoader(obj_train_RUL_scenario1, batch_size=1,shuffle=False)
    test_dataloader_RUL_scenario1 = DataLoader(obj_test_RUL_scenario1, batch_size=1,shuffle=False)

    train_dataloader_RUL_scenario2 = DataLoader(obj_train_RUL_scenario2, batch_size=128,shuffle=True)
    train_dataloader_RUL_temp_scenario2 = DataLoader(obj_train_RUL_scenario2, batch_size=1,shuffle=False)
    test_dataloader_RUL_scenario2 = DataLoader(obj_test_RUL_scenario2, batch_size=1,shuffle=False)


    learning_rate_RUL = 0.00001
    epochs_RUL = 200
    pretrained_RUL_scenario1 = False
    load_pretrained_scenario1 = False
    pretrained_RUL_scenario2 = True

    model_dir_scenario1 = "./Weights/Scenario1/"
    model_path_scenario1 = f'{model_dir_scenario1}/model_f{channels_to_use_s}_f{window_size}_f{version}.pth'

    model_dir_scenario2 = "./Weights/Scenario2/"
    model_path_scenario2 = f'{model_dir_scenario2}/model_f{channels_to_use_s}_f{window_size}_f{version}.pth'


    version = 1

    print("Getting RUL Prediction Scenario1 ")
    if(pretrained_RUL_scenario1):
        print("Getting Pretrained Model for RUL prediction Scenario1")
        model_RUL_scenario1 = LSTM_Model_RUL(window_size,len(channels_to_use))
        model_RUL_scenario1.load_state_dict(torch.load(model_path_scenario1,map_location= device))
        model_RUL_scenario1.to(device)  

    else:
        print("Training the RUL model Scenario1 again")
     
        model_RUL_scenario1 = train_model_RUL(window_size, len(channels_to_use),train_dataloader_RUL_scenario1,epochs_RUL,learning_rate_RUL,load_pretrained_scenario1,model_path_scenario1,version) 


    # if(pretrained_RUL_scenario2):
#     model_RUL_scenario2 = LSTM_Model_RUL(window_size,len(channels))
#     model_RUL_scenario2.load_state_dict(torch.load(model_path_scenario2,map_location= device))
#     model_RUL_scenario2.to(device)  
# else:
#     load_pretrained_scenario2 = False
#     model_RUL_scenario2 = train_model_RUL(window_size, len(channels),train_dataloader_RUL_scenario2,epochs,learning_rate,load_pretrained_scenario2,model_path_scenario2,version) ``


    # batteries =[0,1,2,3]
    # plot_RUL(model_RUL_scenario1,discharge_capacities,batteries,train_dataloader_RUL_temp_scenario1,change_indices_train,"scenario1_RUL_prediction_train")

    test_batteries  = [i+100 for i in range(24)]
    plot_RUL(model_RUL_scenario1,discharge_capacities,test_batteries,test_dataloader_RUL_scenario1,change_indices_test,"Outputs/scenario1_RUL_prediction_test")

    # batteries =[0,1,2]
    # plot_RUL(model_RUL_scenario2,discharge_capacities,batteries,train_dataloader_RUL_temp_scenario2,change_indices_train,"scenario2_RUL_prediction_train")

    # test_batteries  = [i+100 for i in [16,17,18]]
    # plot_RUL(model_RUL_scenario2,discharge_capacities,test_batteries,test_dataloader_RUL_scenario2,change_indices_test,"scenario2_RUL_prediction_test")


        
