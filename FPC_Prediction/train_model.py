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



device = 'cuda' if torch.cuda.is_available() else 'cpu'



def train_model(window_size,channels,train_dataloader,epochs,lr, load_pretrained, path,version):
    
    

    
    model = CNN_Model(window_size,channels)
    if(load_pretrained):
        model.load_state_dict(torch.load(path, map_location=device ))

    
    model.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas= (0.9, 0.99))
    criterion = nn.BCELoss()
    metric = BinaryAccuracy().to(device)
    early_stopping = EarlyStopping(patience=20)


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
    
    total_loss = 0
    total = 0
    total_batches = 0
    for x, y ,_ in test_dataloader:
        x = x.to(device=device)
        y = y.to(device=device)
        
        if(model_RUL.name == "Transformer"):
            out,d = model_RUL(x)
            loss = criterion(out,y.unsqueeze(1))  + 0*criterion(d,x)
        else:
            out =  model_RUL(x)
            loss = criterion(out,y.unsqueeze(1))

        total_loss += loss.item() * x.size()[0]
        total += x.size()[0]
        total_batches +=1
    print("\n\nTest loss = {} \n\n".format(total_loss/total))
            

def perform_n_folds(model, n_folds,discharge_capacities,change_indices,criterion, optimizer, early_stopping, load_pretrained, path,
                     scenario, parameters,version):
    for fold in range(n_folds):
        print("*********************  Fold = {}  ********************* \n\n".format(fold))
        test_batteries = [i for i in range(len(discharge_capacities)) if i % n_folds == fold]
        train_batteries = [i for i in range(len(discharge_capacities)) if i not in test_batteries]
        

        train_dataloader_RUL, train_dataloader_RUL_temp, test_dataloader_RUL = get_RUL_dataloader(discharge_capacities, 
                                                                                              train_batteries, test_batteries, 
                                                                                              change_indices, parameters["window_size"],
                                                                                                parameters["stride"],parameters["channels"] ,scenario)
        
        model = train_model_RUL(model, criterion, optimizer,train_dataloader_RUL,parameters["epochs"],
                                parameters["learning_rate"],load_pretrained,path,early_stopping,version)
        
        test_model_RUL(model,criterion,test_dataloader_RUL)
        if(fold !=n_folds-1):
            model.apply(weight_reset)
        else:
            return model, test_dataloader_RUL, test_batteries
    
        

            
            

