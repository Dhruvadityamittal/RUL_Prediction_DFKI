import shutil
import os
import torch
from torchmetrics.classification import BinaryAccuracy
from util import EarlyStopping
import torch.nn as nn
from model import CNN_Model, LSTM_Model_RUL, CNN_Model_RUL


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(window_size,channels,train_dataloader,epochs, pretrained = True):
    
    if(pretrained != True):
        try:
            shutil.rmtree("./Weights")
            os.mkdir("./Weights")
        except:
            os.mkdir("./Weights")
        
        
    
        model = CNN_Model(window_size,channels)
        model.to(device)

        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas= (0.9, 0.99))
        criterion = nn.BCELoss()
        metric = BinaryAccuracy().to(device)
        early_stopping = EarlyStopping(patience=50)



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
            early_stopping(evaluation, \
                            model, f'./Weights/model_f{channels}_f{window_size}.pth')
            if early_stopping.early_stop:
                print('Early stopping')
                break


from torchmetrics.classification import BinaryAccuracy
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model_RUL(window_size,channels,train_dataloader,epochs,pretrained = True):
    
    if(pretrained != True):
        try:
            shutil.rmtree("./Weights_RUL")
            os.mkdir("./Weights_RUL")
        except:
            os.mkdir("./Weights_RUL")

        model_RUL = LSTM_Model_RUL(window_size,channels)
        learning_rate = 0.00001
        optimizer = torch.optim.Adam(model_RUL.parameters(), lr = learning_rate, betas= (0.9, 0.99))
        criterion = nn.L1Loss()
               
        model_RUL.to(device)
        model_RUL.train()
        
        early_stopping = EarlyStopping(patience=30)
        for epoch in range(epochs):
            total_loss = 0
            
            model_RUL.requires_grad_(True)
            acc = 0
            total_loss = 0
            total = 0
            total_batches = 0
            for x, y ,_ in train_dataloader:

                x = x.to(device=device)
                y = y.to(device=device)
                out = model_RUL(x)

                loss = criterion(out,y.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size()[0]
                total += x.size()[0]
                total_batches +=1


            print("Epoch = {}, Loss = {} ".format(epoch, total_loss/total))
            
            evaluation = total_loss/total
            early_stopping(evaluation, model_RUL, f'./Weights_RUL/model_RUL_f{channels}_f{window_size}.pth')
            if early_stopping.early_stop:
                print('Early stopping')
                break

            
    
        

            
            

