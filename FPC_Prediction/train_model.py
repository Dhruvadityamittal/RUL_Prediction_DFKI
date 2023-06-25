import shutil
import os
import torch
from torchmetrics.classification import BinaryAccuracy
from util import EarlyStopping
import torch.nn as nn
from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL, Net, Net_new
import time
import numpy as np



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




def train_model_RUL(window_size,channels,train_dataloader,epochs,lr,load_pretrained,path,version):
    times = []
#     model_RUL = LSTM_Model_RUL(window_size,channels)
    
    model_RUL = Net()    # Transformer Model
#     model_RUL = CNN_Model_RUL(window_size,channels)    # CNN Model
    print("Model :", model_RUL.name)
    if(load_pretrained):
        print("Loading a Pre-trained Model")
        model_RUL.load_state_dict(torch.load(path,map_location= device))
    else:
        print("Training a new model")
    model_RUL.to(device) 
        
    optimizer = torch.optim.Adam(model_RUL.parameters(), lr = lr, betas= (0.9, 0.99))
#     criterion = nn.L1Loss()
    criterion = nn.MSELoss()
    model_RUL.train()
    
    early_stopping = EarlyStopping(patience=50)
    
    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        
        model_RUL.requires_grad_(True)
        acc = 0
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
    print("Average Epoch Times:" ,np.mean(times))       
    model_RUL.load_state_dict(torch.load(path, map_location=device ))  
    return model_RUL
            
    
        

            
            

