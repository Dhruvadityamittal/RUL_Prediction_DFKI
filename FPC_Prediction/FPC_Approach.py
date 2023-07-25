import os
os.chdir(".")

from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new, Autoencoder, LSTM_Model
from load_data import get_data, get_data_RUL_scenario1, get_discharge_capacities, get_dirs, NormalizeData, get_data_RUL_scenario2
from dataloader import battery_dataloader, battery_dataloader_RUL, get_RUL_dataloader
from import_file import *
from train_model import train_model, train_model_RUL, test_model_RUL, perform_n_folds
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from util_FPC import get_fpc_window, get_data, get_fpc, get_change_indices, EarlyStopping, plot_RUL, weight_reset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

get_dirs()
discharge_capacities = get_discharge_capacities()


percentage  = 0.10  # 10 percent data
window_size = 50    # window size
stride = 1          # stride
channels  =[0,1,2,3,4,5,6] # channels
name_start_train = 0
name_start_test = 100


train_data,FPC_data,FPC_data_dict = get_data(discharge_capacities[:name_start_test],percentage,window_size,stride,channels,type = "train", name_start=name_start_train)
test_data,test_data_dict  = get_data(discharge_capacities[name_start_test:],None,window_size,stride,channels,type= "test", name_start=name_start_test)

obj_train  = battery_dataloader(train_data)
obj_FPC  = battery_dataloader(FPC_data)
obj_test  = battery_dataloader(test_data)


train_dataloader = DataLoader(obj_train, batch_size=8,shuffle=True)
FPC_dataloader   = DataLoader(obj_FPC,batch_size=1,shuffle=False)
test_dataloader = DataLoader(obj_test, batch_size=1,shuffle=False)


print("Number of Channels  :", channels)
print("Shape of a batch    :",next(iter(train_dataloader))[0].shape)



epochs = 10
window_size = 50
learning_rate = 0.001

pretrained = True
load_pretrained = False
version = 1

ch = ''.join(map(str,channels))
dataset = "MIT"

# model = CNN_Model(window_size,len(channels))
model = LSTM_Model(window_size,len(channels))

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
    model = train_model(model, optimizer, criterion, early_stopping,train_dataloader,epochs,learning_rate,load_pretrained,model_path,version)

version = 1
# Get Change Indices
change_indices_train,change_indices_test, _, _ = get_change_indices(model,discharge_capacities,channels,get_saved_indices = False, version = 2, name_start_train = name_start_train,name_start_test= name_start_test , dataset= "MIT") 
change_indices_all = np.concatenate((change_indices_train,change_indices_test))

batteries = [i for i in range(0,24)]
_,_ = get_fpc(model,batteries,discharge_capacities,FPC_data_dict,True, True,True,"./Outputs/FPC_Training_MIT")

batteries = [i+100 for i in range(0,24)]
_,_= get_fpc(model,batteries,discharge_capacities,test_data_dict,True, False,False,"Outputs/FPC_Testing_MIT_Test")



channels_RUL = [0,1,2,3,4,5,6]
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



model_RUL = LSTM_Model_RUL(window_size,len(channels))  # LSTM Model
# model_RUL = Net(len(channels))    # Transformer Model
#model_RUL = CNN_Model_RUL(window_size,channels)    # CNN Model
if(model_RUL.name == "LSTM"):
    learning_rate = 0.0001
else:
    learning_rate = 0.01

print("Learning Rate :", learning_rate)
print("Training RUL on :", model_RUL.name)
optimizer = torch.optim.Adam(model_RUL.parameters(), lr = learning_rate, betas= (0.9, 0.99))
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
    test_batteries = np.load("./Test_data/test_batteries.npy",allow_pickle=True)
    train_batteries = np.load("./Test_data/train_batteries.npy",allow_pickle=True)
else:
    if(load_pretrained_scenario1):
        print("Training further on already trained model")
        model_RUL.load_state_dict(torch.load(model_path_scenario1,map_location= device))
        perform_n_folds(model_RUL,n_folds,discharge_capacities,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, dataset)
    else:
        model_RUL, test_dataloader, test_batteries, train_batteries = perform_n_folds(model_RUL,n_folds,discharge_capacities,change_indices_all,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, dataset)
        np.save(f"./Test_data/test_batteries.npy", test_batteries, allow_pickle=True)
        np.save(f"./Test_data/train_batteries.npy", train_batteries, allow_pickle=True)


# test_batteries  = [i+100 for i in range(24)]
_, _,  test_dataloader_RUL = get_RUL_dataloader(discharge_capacities, train_batteries, test_batteries, 
                                              change_indices_all, parameters["window_size"],
                                              parameters["stride"],parameters["channels"] ,scenario)

plot_RUL(model_RUL,discharge_capacities,test_batteries,test_dataloader,change_indices_all,"Outputs/scenario1_RUL_prediction_test")