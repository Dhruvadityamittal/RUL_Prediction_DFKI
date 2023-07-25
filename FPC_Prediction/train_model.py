import shutil
import os
import torch
from torchmetrics.classification import BinaryAccuracy
from util_FPC import EarlyStopping, weight_reset
import torch.nn as nn
from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new
import time
import numpy as np
from dataloader import get_RUL_dataloader
from torchmetrics import MeanAbsolutePercentageError



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train_model(model, optimizer, criterion, early_stopping,train_dataloader,epochs,lr, load_pretrained, path,version):
    
    

    metric = BinaryAccuracy().to(device)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        model.requires_grad_(True)
        acc = 0
        total_loss = 0
        total = 0
        total_batches = 0
        for x, y ,_ in train_dataloader:

            x = x.to(device=device)
            y = y.to(device=device)
            out = model(x)
            acc += metric(out, y.unsqueeze(1))

            loss = criterion(out,y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size()[0]
            total += x.size()[0]
            total_batches +=1


        print("Loss = {} Accuarcy ={}".format(total_loss/total,acc/total_batches))

        evaluation = total_loss/total
        early_stopping(evaluation, model, path)
        
        if early_stopping.early_stop:
            print('Early stopping')
            break
    model.load_state_dict(torch.load(path, map_location=device ))    

    return model




def train_model_RUL(model_RUL,criterion, optimizer,train_dataloader,epochs,lr,load_pretrained,path,early_stopping,version):
    times = []
    model_RUL.to(device) 
    model_RUL.train()

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        
        model_RUL.requires_grad_(True)
        total_loss = 0
        total = 0
        total_batches = 0
        for x, y ,_ in train_dataloader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            if(model_RUL.name == "Transformer"):
                out,d = model_RUL(x)
                loss = criterion(out,y.unsqueeze(1))  + 0*criterion(d,x)
            else:
                out =  model_RUL(x)
                loss = criterion(out,y.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size()[0]
            total += x.size()[0]
            total_batches +=1

        end = time.time()
        times.append(end-start)
        print("Epoch = {}, Loss = {} ".format(epoch, total_loss/total))
        
        evaluation = total_loss/total
        early_stopping(evaluation, model_RUL,path)

        if early_stopping.early_stop:
            print('Early stopping')
            break
    print("\n Average Time per Epoch :" ,np.mean(times))       
    model_RUL.load_state_dict(torch.load(path, map_location=device ))  
    return model_RUL


def test_model_RUL(model_RUL, criterion, test_dataloader):
    model_RUL.eval()
    
    total_loss_mse = 0
    total_mse = 0
    total_loss_mae = 0
    total_mae = 0
    total_loss_mape = 0
    total_mape = 0
    
    total_batches = 0

    

    l1 = nn.L1Loss().to(device)
    l2 = MeanAbsolutePercentageError().to(device)
    for x, y ,_ in test_dataloader:
        x = x.to(device=device)
        y = y.to(device=device)
        
        if(model_RUL.name == "Transformer"):
            out,d = model_RUL(x)
            loss_mse = criterion(out,y.unsqueeze(1))  + 0*criterion(d,x)
            loss_mae = l1(out,y.unsqueeze(1))  + 0*l1(d,x)
            loss_mape = l2(out,y.unsqueeze(1))  + 0*l2(d,x)

        else:
            out =  model_RUL(x)
            loss_mse = criterion(out,y.unsqueeze(1))
            loss_mae = l1(out,y.unsqueeze(1))  
            loss_mape = l2(out,y.unsqueeze(1))


        total_loss_mse += loss_mse.item() * x.size()[0]
        total_mse += x.size()[0]

        total_loss_mae += loss_mae.item() * x.size()[0]
        total_mae += x.size()[0]

        total_loss_mape += loss_mape.item() * x.size()[0]
        total_mape += x.size()[0]

        total_batches +=1
    print("\n\nTest loss : MSE = {}, MAE = {}, MAPE = {} \n\n".format(total_loss_mse/total_mse, 
                                                                      total_loss_mae/total_mae,
                                                                        total_loss_mape/total_mape))
            

def perform_n_folds(model, n_folds,discharge_capacities,change_indices,criterion, 
                    optimizer, early_stopping, load_pretrained, path,
                     scenario, parameters,version,dataset):
    for fold in range(n_folds):
        print("*********************  Fold = {}  ********************* \n\n".format(fold))
        test_batteries = [i for i in range(len(discharge_capacities)) if i % n_folds == fold]
        train_batteries = [i for i in range(len(discharge_capacities)) if i not in test_batteries]
        
        fold_path = path[:-10]+"Fold="+str(fold)+".pth"
    

        train_dataloader_RUL, train_dataloader_RUL_temp, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities, 
                                                                                              train_batteries, test_batteries, 
                                                                                              change_indices, parameters["window_size"],
                                                                                                parameters["stride"],parameters["channels"] ,scenario)
        
        early_stopping = EarlyStopping(patience=50)
        model = train_model_RUL(model, criterion, optimizer,train_dataloader_RUL,parameters["epochs"],
                                parameters["learning_rate"],load_pretrained,fold_path,early_stopping,version)
        
        test_model_RUL(model,criterion,test_dataloader_RUL)
        
        np.save(f"./Test_data/test_batteries_{dataset}_{model.name}_fold{fold}.npy", test_batteries, allow_pickle=True)
        np.save(f"./Test_data/train_batteries_{dataset}_{model.name}_fold{fold}.npy", train_batteries, allow_pickle=True)

        if(fold !=n_folds-1):
            model.apply(weight_reset)
        else:
            
            return model, test_dataloader_RUL, test_batteries, train_batteries
    
        

            
            

