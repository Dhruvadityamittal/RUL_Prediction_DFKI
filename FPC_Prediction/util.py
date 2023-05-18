import torch
import numpy as np
import matplotlib.pylab as plt
from load_data import NormalizeData

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
        

def get_fpc(model,batteries,discharge_capacities,data_loader,plot,show_FPC_curve,title):
    

    if(plot):
        rows = 2
        col  = 7
       
        fig, ax = plt.subplots(col,rows,figsize=(16,16))
        ax = ax.flatten()
        plt.suptitle("Results for :" +title, fontsize = 20)
        fig.tight_layout(rect=[0, 1, 1, 0.95])

    
    change_percentage = []
    change_indices    = []
    model.eval()
    pred = []
    
    
    for ind,battery in enumerate(batteries):
        pred = []
        count = 0
        for x, y ,_ in data_loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            initial_count = count                # This is used to avoid iterating over all batteries in the dataset 
            if(_[0][7:] == str(battery)):
                out = torch.where(model(x) > 0.5, 1, 0)
                pred.append(out.cpu().detach().numpy()[0][0].astype(float))
                count = count +1
            if(initial_count==count and count >1):
                break

        index,smoothed_output = get_fpc_window(pred,patiance=10)   # Index where the the transition occurs
        index = index*stride

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
        
                ax[ind].legend(["FPC", "NON-FPC","Prediction","Smoothed Output"], fontsize = 'x-small')
                ax[ind].set_title("Battery =" +str(battery))
        else:
            if(plot):
                
                ax[ind].plot(discharge_capacities[battery][0], color = 'orange')
                ax[ind].plot(pred, color ='red')
                ax[ind].plot(smoothed_output, color ='black')
                ax[ind].legend(["Actual", "Prediction", "Smoothed Prediction"])
                ax[ind].set_title("Battery =" +str(battery))

    if(plot):
        plt.show()

    return change_percentage, change_indices
   

def plot_RUL(model,discharge_capacities,batteries,data_loader,change_indices,plot_nomalized):
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
        # print(len(pred)-len(discharge_capacities[battery][0][change_indices[battery-100]:]))
        normalized,(ma,mi) = NormalizeData(discharge_capacities[battery][0][change_indices[battery-100]:])

        
        if(plot_nomalized):
            plt.plot(normalized)
            plt.plot(actual)
            plt.plot(pred)
            plt.legend([ "Complete Actual",'Actual', "Prediction"])
        else:
            denonmalized = (np.array(pred)*(ma-mi))+mi
            # plt.plot(denonmalized)
            # plt.plot(discharge_capacities[battery][0][change_indices[battery-100]:])
            a = list(discharge_capacities[battery][0][0:change_indices[battery-100]]) + list(denonmalized)
        
            plt.plot(a)
            plt.plot(discharge_capacities[battery][0][:-window_size])
            
    else:
       
        # print(len(pred)-len(discharge_capacities[battery][0][change_indices[battery]:]))
        normalized,(ma,mi) = NormalizeData(discharge_capacities[battery][0][change_indices[battery]:])
        
        

        if(plot_nomalized):
            plt.plot(normalized)
            plt.plot(actual)
            plt.plot(np.array(pred))
            plt.legend([ "Complete Actual",'Actual', "Prediction"])
        else:
            denonmalized = (np.array(pred)*(ma-mi))+mi
            # plt.plot(denonmalized)
            # plt.plot(discharge_capacities[battery][0][change_indices[battery]:])
            a = list(discharge_capacities[battery][0][0:change_indices[battery-100]]) + list(denonmalized)
        
            plt.plot(a)
            plt.plot(discharge_capacities[battery][0][:-window_size])
            plt.legend(['Actual', "Prediction"])

    plt.show()

