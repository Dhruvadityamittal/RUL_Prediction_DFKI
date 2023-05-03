# Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import os

from battery_dataloader import battery_dataloader
from model import CNN_Model
from util import EarlyStopping

import torch 
from torch.utils.data import DataLoader,Dataset, random_split, Subset
import torchmetrics

# This function return the cycle when battery reaches it's threshold
def get_batteries(discharge_capacities,threshold,input_size):
   
   data_x = []
   data_y = []
   max_cycle_length =[]
   threshold_values = []
   data_x_index = []
   cycles = [cell.shape[0] for cell in discharge_capacities]
   for bat in range(len(discharge_capacities)): 
      c= 0
     
      for i in range(0,len(discharge_capacities[bat])):
      # Just checking if there is a anamoly where the dischare suddenly drops in the initial cycles 
         if(discharge_capacities[bat][i] <= threshold*max(discharge_capacities[bat]) and i >=0.1*cycles[bat]):
            threshold_values.append(discharge_capacities[bat][i])     # saving threshold value of the battery
            break
         c=c+1

      # if c==the total cycles it means it does not have that threshold in its charge
      if(c!=int(cycles[bat]) and cycles[bat]>input_size):     
         data_x.append(list(discharge_capacities[bat][0:input_size]))
         data_y.append(c)
         data_x_index.append(bat)
         max_cycle_length.append(len(discharge_capacities[bat]))

   return data_x,data_y,threshold_values,data_x_index,max_cycle_length

def get_batteries1(discharge_capacities,threshold_cycles,input_size):
    data_x =[]
    data_y = []
    cycles = [cell.shape[0] for cell in discharge_capacities]

    for i in range(len(discharge_capacities)):
        if(cycles[i] >= threshold_cycles  and cycles[i] > input_size):
            data_x.append(discharge_capacities[i][:input_size])
            c = discharge_capacities[i][threshold_cycles]
            data_y.append(c)
    return data_x,data_y


from torchmetrics import MeanAbsolutePercentageError

def loss_function(predicted, actual,max_cycle_length):
    # return torch.mean(torch.square(predicted - actual) / max_cycle_length)
    return torch.mean(torch.abs(predicted - actual) / max_cycle_length)

def train_model(threshold, input_size, fold, train_dataloader, valid_dataloader, learning_rate):
    model = CNN_Model(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas= (0.9, 0.99))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    early_stopping = EarlyStopping(patience=50)
    try:
        os.mkdir("weights")
    except:
        pass

    mse_loss  = torch.nn.MSELoss()
    mae_loss  = torch.nn.L1Loss()
    
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0
        total_mae = 0
        total_mse = 0
        total_mape = 0
        total = 0
        model.train()
        model.requires_grad_(True)
        for x,y,max_cycle_length in train_dataloader:
            noise = torch.randn(x.size()) * 0.1
            x = x + noise
            x = x.to(device=device)
            y = y.to(device=device)
            max_cycle_length = max_cycle_length.to(device)
            
            out = model(x)
            # print("Batch ",out,y,max_cycle_length)
            # exit()
            loss = loss_function(out,y,max_cycle_length)
            mse = mse_loss(out,y)
            mae = mae_loss(out,y)
            mape = torch.mean(torch.abs((out - y) / y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size()[0]
            total_mae += mae.item() * x.size()[0]
            total_mse += mse.item() * x.size()[0]
            total_mape += mape.item() * x.size()[0]
            total += x.size()[0]

            
        model.eval()
        model.requires_grad_(False)
        total_valid_loss = 0
        total_valid_mse = 0
        total_valid_mae = 0
        total_valid_mape = 0
        valid_step = 0
        
        for x,y,max_cycle_length in valid_dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
            max_cycle_length = max_cycle_length.to(device)

            out = model(x)
            valid_loss = loss_function(out,y,max_cycle_length)
            valid_mse = mse_loss(out,y)
            valid_mae = mae_loss(out,y)
            valid_mape = torch.mean(torch.abs((out - y) / y))
            # valid_mape = mean_abs_percentage_error(out,y)
            total_valid_loss += valid_loss.item() * x.size()[0]
            total_valid_mse+= valid_mse.item() * x.size()[0]
            total_valid_mae+= valid_mae.item() * x.size()[0]
            total_valid_mape+= valid_mape.item() * x.size()[0]
            valid_step += x.size()[0]
        
        
        print('Epoch {}/{} train_loss={:.4f} train_mae={:.2f} train_mse={:.2f} train_mape={:.4f} valid_loss={:.4f} valid_mae={:.2f} valid_mse={:.2f} valid_mape={:.4f}'.format(
                                epoch +1, epochs,
                                total_loss/total,
                                total_mae/total,
                                total_mse/total,
                                total_mape/total,
                                total_valid_loss/valid_step,
                                total_valid_mae/valid_step,
                                total_valid_mse/valid_step,
                                total_valid_mape/valid_step
                                ))

        valid_evaluation = total_valid_loss/valid_step + 0.01 * total_valid_mae/valid_step + total_valid_mape/valid_step
        # valid_evaluation = total_valid_loss/valid_step + total_valid_mape/valid_step
        early_stopping(valid_evaluation, \
                        model, f'weights/model_f{threshold}_f{input_size}_f{fold}.pth')
        if early_stopping.early_stop:
            print('Early stopping')
            break

    

def test_model(threshold, input_size, fold, test_dataloader):
    model = CNN_Model(input_size)
    model.load_state_dict(torch.load(f'weights/model_f{threshold}_f{input_size}_f{fold}.pth'))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    mse_loss  = torch.nn.MSELoss()
    mae_loss  = torch.nn.L1Loss()
    mean_abs_percentage_error = MeanAbsolutePercentageError()

    total_test_loss =0 
    test_step = 0
    total_test_mse =0 
    total_test_mae =0 
    total_test_mape =0 
    
    model.eval()
    model.requires_grad_(False)
    for x,y,max_cycle_length in test_dataloader:
        x = x.to(device=device)
        y = y.to(device=device)
        max_cycle_length = max_cycle_length.to(device)
        out = model(x)
        
        test_loss = loss_function(out,y,max_cycle_length)
        test_mse = mse_loss(out,y)
        test_mae = mae_loss(out,y)

        test_mape = torch.mean(torch.abs((out - y) / y))
        # test_mape = mean_abs_percentage_error(out,y)
        total_test_loss += test_loss.item() * x.size()[0]
        total_test_mse+= test_mse.item() * x.size()[0]
        total_test_mae+= test_mae.item() * x.size()[0]
        total_test_mape+= test_mape.item() * x.size()[0]
        test_step += x.size()[0]

        print("predicted:",out)
        print("actual:",y)
        
    print("Testing Loss custom= {}  MAE ={} MSE ={} MAPE ={}  ".format(total_test_loss/test_step,total_test_mae/test_step, total_test_mse/test_step, total_test_mape/test_step))
    return total_test_loss/test_step,total_test_mae/test_step, total_test_mse/test_step, total_test_mape/test_step
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    thresholds = [0.8,0.85,0.90,0.95]
    # thresholds = [0.90]
    initial_cycles = [50, 60,70,80,90,100,150,200,250,300,350,400,500]
    initial_cycles = [500]
    
    n_folds = 5
    batch_size = args.batch_size

    all_maes = []
    all_mses = []
    all_mape = []
    all_custom = []

    discharge_capacities = np.load(r'./Datasets/discharge_capacity.npy', allow_pickle=True)

    for threshold in thresholds:
        for initial_cycle in initial_cycles:
            mses = []
            maes = []
            mapes= []
            customs = []
            for fold in range(n_folds):
                print("Threshold {} , initial cycles = {}, folds = {}".format(threshold,initial_cycle, fold))
                data_x,data_y,threshold_values,data_x_index,max_cycle_length = get_batteries(discharge_capacities,threshold,initial_cycle)
                
                battery_data = battery_dataloader(data_x,data_y,max_cycle_length)
                testIndex = [i for i in range(len(battery_data)) if i % n_folds == fold]
                trainIndex = [i for i in range(len(battery_data)) if i not in testIndex]

                print("train and test:", len(trainIndex), len(testIndex))

                trainIndex, validIndex = random_split(trainIndex, [0.8, 0.2])

                train_subset = Subset(battery_data, trainIndex)
                valid_subset = Subset(battery_data, validIndex)
                test_subset = Subset(battery_data, testIndex)

                # print(train_subset.dataset.data_y)
                # exit()

                train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=True)
                test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
            
                train_model(threshold, initial_cycle, fold, train_dataloader, valid_dataloader, args.lr)
                custom, mae, mse, mape = test_model(threshold, initial_cycle, fold, test_dataloader)

                customs.append(custom)
                mses.append(mse)
                maes.append(mae)
                mapes.append(mape)
                exit()
            all_custom.append(customs)
            all_mses.append(mses)
            all_maes.append(maes)
            all_mape.append(mapes)
        
    import pandas as pd
    d = pd.DataFrame([all_maes,all_mses, all_mape ,all_custom])
    d.to_excel("Test.xlsx",index= False) 
