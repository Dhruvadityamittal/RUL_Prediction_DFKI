import os
os.chdir(".")

from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new, Autoencoder, LSTM_Model
from load_data import get_data, get_data_RUL_scenario1, get_discharge_capacities_MIT,get_discharge_capacities_HUST, get_dirs, get_data_RUL_scenario2
from dataloader import battery_dataloader, battery_dataloader_RUL, get_RUL_dataloader
from import_file import *
from train_model import train_model, train_model_RUL, test_model_RUL, perform_n_folds
from util_FPC import get_fpc_window, get_data, get_fpc, get_change_indices, EarlyStopping, plot_RUL, weight_reset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Training on ", device)

# dataset = "HUST"
dataset = "MIT"
# RUL_model_name = "LSTM"
RUL_model_name = "Net"

print("Using Dataset :", dataset)

percentage  = 0.10  # 10 percent data
window_size_FPC = 50    # window size
stride_FPC = 1          # stride
window_size_RUL = 64
stride_RUL =1

get_dirs()

epochs_FPC = 5   # 5 for HUST , 15 for MIT
epochs_RUL = 600

if(dataset == "MIT"):
    channels  =[0,1,2,3,4,5,6] # channels
    channels_RUL = [0,1,2,3,4,5,6]
    name_start_train = 0
    name_start_test = 100
    
    discharge_capacities = get_discharge_capacities_MIT()
    learning_rate_FPC = 0.0001
    
    model_FPC = LSTM_Model(window_size_FPC,len(channels))

    if(RUL_model_name == "LSTM"):
        model_RUL = LSTM_Model_RUL(window_size_RUL,len(channels))  # LSTM Model
        learning_rate_RUL = 0.0001
    elif(RUL_model_name == "Net"):
        model_RUL = Net(len(channels), feature_size=window_size_RUL)    # Transformer Model
        learning_rate_RUL = 0.0001      
    else:
        model_RUL = CNN_Model_RUL(window_size_RUL,len(channels_RUL))  # CNN Model
        learning_rate_RUL = 0.01      
    
else:
    channels  =[0] # channels
    channels_RUL = [0]
    name_start_train = 0
    name_start_test = 70
    
    discharge_capacities = get_discharge_capacities_HUST(fea_num =1)
    learning_rate_FPC = 0.0001
    # model_FPC = CNN_Model(window_size,len(channels))
    model_FPC = LSTM_Model(window_size_FPC,len(channels))

    if(RUL_model_name == "LSTM"):
        model_RUL = LSTM_Model_RUL(window_size_RUL,len(channels))  # LSTM Model
        learning_rate_RUL = 0.0001
    elif(RUL_model_name == "Net"):
        model_RUL = Net(len(channels))    # Transformer Model
        learning_rate_RUL = 0.001      # CNN Model
    else:
        model_RUL = CNN_Model_RUL(window_size_RUL,len(channels_RUL))
        learning_rate_RUL = 0.01      # CNN Model



# Training for Stage 1
train_data,FPC_data,FPC_data_dict = get_data(discharge_capacities[:name_start_test],percentage,window_size_FPC,stride_FPC,channels,type = "train", name_start=name_start_train)
test_data,test_data_dict  = get_data(discharge_capacities[name_start_test:],None,window_size_FPC,stride_FPC,channels,type= "test", name_start=name_start_test)


obj_train  = battery_dataloader(train_data)
obj_FPC  = battery_dataloader(FPC_data)
obj_test  = battery_dataloader(test_data)

train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
FPC_dataloader   = DataLoader(obj_FPC,batch_size=1,shuffle=False)
test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)

print("Number of Channels  :", channels)
print("Shape of a batch    :",next(iter(train_dataloader))[0].shape)


pretrained = True
load_pretrained = False
version = 1

ch = ''.join(map(str,channels))

model_dir = "./Weights/FPC/"
model_path = f'{model_dir}/{dataset}_{model_FPC.name}_FPC_Channels={ch}_WindowSize={window_size_FPC}_Version={version}.pth'
print(model_path)

if(load_pretrained):
    model_FPC.load_state_dict(torch.load(model_path, map_location=device ))

model_FPC.to(device)


optimizer = torch.optim.Adam(model_FPC.parameters(), lr = learning_rate_FPC, betas= (0.9, 0.99))
criterion = nn.BCELoss()

early_stopping = EarlyStopping(patience=20)

if(pretrained):    
    model_FPC.load_state_dict(torch.load(model_path, map_location=device ))
    model_FPC.to(device)
else:
    model_FPC = train_model(model_FPC, optimizer, criterion, early_stopping,train_dataloader,epochs_FPC,learning_rate_FPC,load_pretrained,model_path,version)

version = 1
print("Version", version)
# Get Change Indices
change_indices_train,change_indices_test, _, _ = get_change_indices(model_FPC,discharge_capacities,channels,get_saved_indices = True, version = 1, name_start_train = name_start_train,name_start_test= name_start_test , dataset= "MIT") 
change_indices_all = np.concatenate((change_indices_train,change_indices_test))



# Inference on Test and Train
# batteries = [i for i in range(0,24)]
# _,_ = get_fpc(model_FPC,batteries,discharge_capacities,FPC_data_dict,True, True,True,"./Outputs/FPC_Training_" + dataset + "_train_new_latest")

# batteries = [i+name_start_test for i in range(0,len(discharge_capacities)-name_start_test)]
# _,_= get_fpc(model_FPC,batteries,discharge_capacities,test_data_dict,True, False,False,"Outputs/FPC_Testing_" + dataset + "_test_new_latest")

# exit()
# **************************************************************************
# RUL prediction code Starts here


c_RUL = ''.join(map(str,channels_RUL))
n_folds = 5
scenario = 1

parameters = {
    "window_size" : window_size_RUL,
    "stride": stride_RUL,
    "channels": channels_RUL,
    "epochs": epochs_RUL,
    "learning_rate": learning_rate_RUL
}


print("Learning Rate :", learning_rate_RUL)
print("Training RUL on :", model_RUL.name)

optimizer = torch.optim.Adam(model_RUL.parameters(), lr = learning_rate_RUL, betas= (0.9, 0.99))
# criterion = nn.L1Loss()
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=50)

version = 1
pretrained_RUL_scenario1 = False
load_pretrained_scenario1  = False
fld = 1
model_dir_scenario1 = "./Weights/Scenario1/"
model_path_scenario1 = f'{model_dir_scenario1}/{dataset}_f{model_RUL.name}_RUL_Channels={c_RUL}_WindowSize={window_size_RUL}_Version={version}_Fold={fld}.pth'

if(pretrained_RUL_scenario1):
    print("Loading a Pre-trained Model")
    model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
    test_batteries = np.load(f"./Test_data/test_batteries_{dataset}_{model_RUL.name}_fold{fld}.npy",allow_pickle=True)
    train_batteries = np.load(f"./Test_data/train_batteries_{dataset}_{model_RUL.name}_fold{fld}.npy",allow_pickle=True)
    val_batteries = np.load(f"./Test_data/val_batteries_{dataset}_{model_RUL.name}_fold{fld}.npy",allow_pickle=True)
else:
    if(load_pretrained_scenario1):
        print("Training further on already trained model")
        model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
        model_RUL, test_dataloader, test_batteries, train_batteries, val_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, dataset)
    else:
        model_RUL, test_dataloader, test_batteries, train_batteries, val_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, dataset)
        # np.save(f"./Test_data/test_batteries_{dataset}.npy", test_batteries, allow_pickle=True)
        # np.save(f"./Test_data/train_batteries_{dataset}.npy", train_batteries, allow_pickle=True)
        # np.save(f"./Test_data/val_batteries_{dataset}.npy", val_batteries, allow_pickle=True)


# test_batteries  = [i+100 for i in range(24)]



_, _,  _, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities, train_batteries,val_batteries, test_batteries,
                                              change_indices_all, parameters["window_size"],
                                              parameters["stride"],parameters["channels"] ,scenario)

plot_RUL(model_RUL,discharge_capacities,test_batteries,test_dataloader,change_indices_all,"Outputs/scenario1_RUL_prediction_"+dataset+"_test")