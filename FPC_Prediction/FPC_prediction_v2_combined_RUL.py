import os
os.chdir(".")

from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new, Autoencoder, LSTM_Model,TransformerLSTM
from load_data import get_data, get_data_RUL_scenario1, get_discharge_capacities_MIT,get_discharge_capacities_HUST, get_dirs, get_data_RUL_scenario2
from dataloader import  battery_dataloader, battery_dataloader_RUL, get_RUL_dataloader
from import_file import *
from train_model import train_model, train_model_RUL, test_model_RUL, perform_n_folds
from util_FPC import get_fpc_window, get_data, get_fpc, get_change_indices, EarlyStopping, plot_RUL, weight_reset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
percentage  = 0.10  # 10 percent data
window_size = 50    # window size

stride = 1          # stride
channels_MIT  =[0,1,2,3,4,5,6] # channels MIT
channels_HUST  =[0]           # channels HUST

name_start_train_MIT = 0
name_start_test_MIT = 100

name_start_train_HUST = 0
name_start_test_HUST = 70

version = 1

discharge_capacities_MIT = get_discharge_capacities_MIT()
discharge_capacities_HUST = get_discharge_capacities_HUST(fea_num=1)

ch_MIT = ''.join(map(str,channels_MIT))
ch_HUST  = ''.join(map(str,channels_HUST))

# model = CNN_Model(window_size,len(channels))
model_MIT = LSTM_Model(window_size,len(channels_MIT))
model_HUST = LSTM_Model(window_size,len(channels_HUST))

model_dir = "./Weights/FPC/"
model_path_MIT = f'{model_dir}/{"MIT"}_{model_MIT.name}_FPC_Channels={ch_MIT}_WindowSize={window_size}_Version={version}.pth'
model_path_HUST = f'{model_dir}/{"HUST"}_{model_HUST.name}_FPC_Channels={ch_HUST}_WindowSize={window_size}_Version={version}.pth'
# model_path_MIT = r"C:\Users\Dhruv\OneDrive\Desktop\MIT_LSTM_Classifier_FPC_Channels=0123456_WindowSize=50_Version=1.pth"


model_MIT.load_state_dict(torch.load(model_path_MIT, map_location=device ))
model_HUST.load_state_dict(torch.load(model_path_HUST, map_location=device ))

change_indices_train_MIT,change_indices_test_MIT, _, _ = get_change_indices(model_MIT,discharge_capacities_MIT,channels_MIT,get_saved_indices = True, version = 1, name_start_train = name_start_train_MIT,name_start_test= name_start_test_MIT , dataset= "MIT") 
change_indices_all_MIT = np.concatenate((change_indices_train_MIT,change_indices_test_MIT))

change_indices_train_HUST,change_indices_test_HUST, _, _ = get_change_indices(model_HUST,discharge_capacities_HUST,channels_HUST,get_saved_indices = True, version = 1, name_start_train = name_start_train_HUST,name_start_test= name_start_test_HUST , dataset= "HUST") 
change_indices_all_HUST = np.concatenate((change_indices_train_HUST,change_indices_test_HUST))



disharge_capacity_combined = discharge_capacities_MIT[0:100] + discharge_capacities_HUST[0:70] + discharge_capacities_MIT[100:] + discharge_capacities_HUST[70:]
change_indices_combined = list(change_indices_all_MIT[0:100]) + list(change_indices_all_HUST[0:70]) + list(change_indices_all_MIT[100:]) + list(change_indices_all_HUST[70:])


n_folds = 5
scenario = 1
# learning_rate = 0.01
learning_rate = 0.0001
epochs = 500
channels_RUL = [0]
window_size_RUL = 64
c_RUL  = ''.join(map(str,channels_RUL))

parameters = {
    "window_size" : window_size,
    "stride": stride,
    "channels": channels_RUL,
    "epochs": epochs,
    "learning_rate": learning_rate
}


dataset = "Combined"
learning_rate_RUL = 0.0001

# model_RUL = LSTM_Model_RUL(window_size,len(c_RUL))  # LSTM Model
# model_RUL = Net(len(channels_RUL))    # Transformer Model
# model_RUL = CNN_Model_RUL(window_size,len(channels_RUL))    # CNN Model


model_RUL = TransformerLSTM(len(channels_RUL), window_size)

print(model_RUL.name,dataset,learning_rate_RUL)
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
        model_RUL, test_dataloader, test_batteries, train_batteries, val_batteries = perform_n_folds(model_RUL,n_folds,disharge_capacity_combined,change_indices_combined,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, dataset)
    else:
        model_RUL, test_dataloader, test_batteries, train_batteries, val_batteries = perform_n_folds(model_RUL,n_folds,disharge_capacity_combined,change_indices_combined,criterion, optimizer, early_stopping,
                    pretrained_RUL_scenario1, model_path_scenario1,scenario,parameters, version, dataset)
        # np.save(f"./Test_data/test_batteries_{dataset}.npy", test_batteries, allow_pickle=True)
        # np.save(f"./Test_data/train_batteries_{dataset}.npy", train_batteries, allow_pickle=True)
        # np.save(f"./Test_data/val_batteries_{dataset}.npy", val_batteries, allow_pickle=True)


# test_batteries  = [i+100 for i in range(24)]



_, _,  _, test_dataloader_RUL = get_RUL_dataloader(disharge_capacity_combined, train_batteries,val_batteries, test_batteries,
                                              change_indices_combined, parameters["window_size"],
                                              parameters["stride"],parameters["channels"] ,scenario)

plot_RUL(model_RUL,disharge_capacity_combined,test_batteries,test_dataloader,change_indices_combined,"Outputs/scenario1_RUL_prediction_"+model_RUL.name+"_"+dataset+"_test")