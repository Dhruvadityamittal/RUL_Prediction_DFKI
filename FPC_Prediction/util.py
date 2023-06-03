import torch
import numpy as np
import matplotlib.pylab as plt
from load_data import NormalizeData
from torchmetrics import MeanAbsolutePercentageError
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

percentage  = 0.10  # 5 percent data
window_size = 50    # window size
stride = 1          # stride
channels  = 7       # channels


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def get_fpc_window(pred,patiance):
    
    count = 0
    for window,pred_value in enumerate(pred):
        if(pred_value.item() ==0):
            count =  count +1
        if(pred_value.item() ==1):
            count =0
        if(window == len(pred)-1):
            change_index = window-count
            return change_index,[1.0 if i<change_index else 0.0 for i in range(len(pred))]
        if(count == patiance):
            change_index = window - patiance
            return change_index,[1.0 if i<change_index else 0.0 for i in range(len(pred))]
        

def get_fpc(model,batteries,discharge_capacities,data_loader,plot,show_FPC_curve,add_initial,save_path):
    
    plt.figure()
    if(plot):
        rows = 4
        col  = 1
        fig, ax = plt.subplots(rows,col,figsize=(10,6),sharex=True, sharey=True)
        ax = ax.flatten()
        plt.suptitle("FPC Prediction", fontsize = 20)
        # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    
    change_percentage = []
    change_indices    = []
    model.eval()
    pred = []
    
    
    for ind,battery in enumerate(batteries):
#         pred = []
#         count = 0
#         for x, y ,_ in data_loader:
#             x = x.to(device=device)
#             y = y.to(device=device)
            
#             initial_count = count                # This is used to avoid iterating over all batteries in the dataset 
#             if(_[0][7:] == str(battery)):
#                 out = torch.where(model(x) > 0.5, 1, 0)
#                 pred.append(out.cpu().detach().numpy()[0][0].astype(float))
#                 count = count +1
#             if(initial_count==count and count >1):
#                 break
        pred = []
        count = 0
        battery_name = "battery"+ str(battery)
        for x in data_loader[battery_name]:
            x = x.view(1,x.shape[0],x.shape[1])
            x = x.to(device=device)
            out = torch.where(model(x) > 0.5, 1, 0) 
            pred.append(out.cpu().detach().numpy()[0][0].astype(float))

        index,smoothed_output = get_fpc_window(pred,patiance=10)   # Index where the the transition occurs
        index = index*stride

        if(add_initial):
            change_indices.append(index+int(percentage*len(discharge_capacities[battery][0])))
        else:
            change_indices.append(index)
        
        change_percentage.append(100*discharge_capacities[battery][0][index]/max(discharge_capacities[battery][0]))
        
        if(show_FPC_curve):
            FPC_curve = np.copy(discharge_capacities[battery][0])
            FPC_curve[1:int(percentage*len(discharge_capacities[battery][0]))] = None
            FPC_curve[int((1-percentage)*len(discharge_capacities[battery][0])):-1] = None

            Non_FPC_curve = np.copy(discharge_capacities[battery][0])
            Non_FPC_curve[int(percentage*len(discharge_capacities[battery][0])):int((1-percentage)*len(discharge_capacities[battery][0]))] = None
    
            pred_padded = np.pad(pred, (int(percentage*len(discharge_capacities[battery][0])), 0), constant_values=(np.nan,))
            smoothed_output_padded = np.pad(smoothed_output, (int(percentage*len(discharge_capacities[battery][0])), 0), constant_values=(np.nan,))
            
            if(plot == True):
                ax[ind].plot(FPC_curve, color = 'orange')
                ax[ind].plot(Non_FPC_curve, color ='red')
                ax[ind].plot(pred_padded,color ='blue')
                ax[ind].plot(smoothed_output_padded,color ='black')
        
                ax[ind].legend(["FPC", "NON-FPC","Prediction","Smoothed Output"])
                ax[ind].set_title("Battery =" +str(battery+1))
                
        else:
            if(plot):
                
                ax[ind].plot(discharge_capacities[battery][0], color = 'orange')
                ax[ind].plot(pred, color ='red')
                ax[ind].plot(smoothed_output, color ='black')
                ax[ind].legend(["Actual", "Prediction", "Smoothed Prediction"])
                ax[ind].set_title("Battery =" +str(battery+1))
                
    plt.savefig(save_path+".png")
   
#     fig.supxlabel('Cycles')
#     fig.supylabel('Discharge Capacity')
    
    return change_percentage, change_indices



def plot_RUL(model,discharge_capacities,batteries,data_loader,change_indices,save_path):

    rows = 1
    col  = len(batteries)
    fig, ax = plt.subplots(col,rows,figsize=(12,2*len(batteries)))
    ax = ax.flatten()
    plt.suptitle("Results", fontsize = 20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    mse_loss = 0
    mae_loss =0 
    mape_loss =0 
    for ind,battery in enumerate(batteries):
        pred = []
        count = 0
        actual = []
        for x, y ,_ in data_loader:
            x = x.to(device)
            y = y.to(device)

            initial_count = count

            if(_[0][7:] == str(battery)):
                out = model(x)
                pred.append(out.cpu().detach().numpy()[0][0].astype(float))
                actual.append(y.cpu().detach().numpy()[0].astype(float))
                count = count +1
            if(initial_count==count and count >1):
                break
        
        if(battery>=100):
            change_indices_battery = battery - 100
        else:
            change_indices_battery = battery

        l = nn.MSELoss()
        l1 = nn.L1Loss()
        l2 = MeanAbsolutePercentageError()
        if(len(pred)!=0):
            mse_loss += l(torch.Tensor(pred),torch.Tensor(actual))
            mae_loss += l1(torch.Tensor(pred),torch.Tensor(actual))
            mape_loss += l2(torch.Tensor(pred),torch.Tensor(actual))
        
        x = [change_indices[change_indices_battery]+i for i in range(len(pred))]
        # print(len(discharge_capacities[battery][0]))
        ax[ind].plot(x,pred)
        ax[ind].plot(x,actual)
        
        ax[ind].legend(['Predicted', 'Actual'])
        ax[ind].set_title("Battery"+str(battery))

    print("MSE= {}, MAE ={} , MAPE = {}".format(mse_loss/len(batteries),mae_loss/len(batteries),mape_loss/len(batteries)))
    
    plt.savefig(save_path+".png")


