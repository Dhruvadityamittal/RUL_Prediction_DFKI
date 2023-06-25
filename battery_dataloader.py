import torch 
from torch.utils.data import DataLoader,Dataset, random_split, Subset

class battery_dataloader(Dataset):
    
    def __init__(self,data_x,data_y,max_cycle_length):
        self.data_x = data_x
        self.data_y = data_y
        self.max_cycle_length = max_cycle_length


    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self,idx):
        
        output = torch.tensor(self.data_y[idx]).float()
        inp =  torch.tensor(self.data_x[idx]).float()
        max_len = torch.tensor(self.max_cycle_length[idx])

        return inp, output, max_len
class battery_dataloader1(Dataset):
    
    def __init__(self,data_x,data_y):
        self.data_x = data_x
        self.data_y = data_y
        


    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self,idx):
        
        output = torch.tensor(self.data_y[idx]).float()
        inp =  torch.tensor(self.data_x[idx]).float()
        

        return inp, output, 0 
   
