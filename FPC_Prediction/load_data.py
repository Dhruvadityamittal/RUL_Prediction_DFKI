import numpy as np

def get_data(discharge_capacities,percentage,window_size,stride,channels,type):

    train_data =[]
    FPC_data  =[]
    name = 0
    test_data = []
    
    if(type == "train"):
        
        for battery in discharge_capacities:
            a = len(FPC_data)
            battery = np.asarray(battery)
            
            battery_name = 'battery' + str(name)
            name = name+1
            
            # Taking inital x% as input and giving the output as 1
            i= 0
            target = 1
            while(i+stride+window_size <= int(percentage*len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                train_data.append((battery[:channels,i:i+window_size], target,battery_name ))
                i = i+stride

            # Taking inputs in the middle for FPC
            i = int(percentage*len(battery[0]))
            target = -1
            while(i+stride+window_size <= int((1-percentage)*len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                FPC_data.append((battery[:channels,i:i+window_size], target,battery_name))
                i = i+stride

            # Taking last x% as input and giving the output as 0
            i = int((1-percentage)*len(battery[0]))
            target = 0
            while(i+stride <= len(battery[0]) and len(battery[0][i:i+window_size]) == window_size):
                train_data.append((battery[:channels,i:i+window_size], target ,battery_name))
                i = i+stride
            # print(len(FPC_data)-a, len(battery[0]), len(FPC_data)-a- .90*len(battery[0]))

        return train_data,FPC_data

    else:
        name = 100
        for battery in discharge_capacities:
            battery = np.asarray(battery)
            i= 0
            battery_name = 'battery' + str(name)
            name = name+1
            while(i+stride <= len(battery[0]) and len(battery[0][i:i+window_size]) == window_size):
                test_data.append((battery[:channels,i:i+window_size], 1,battery_name))
                i = i+stride

        return test_data


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data)), (max(data), min(data))


        
def get_data_RUL_scenario1(discharge_capacities,change_indices, window_size,stride,channels, type):
        
        if(type == "Train"):
            
            train_data =[]
            for index,battery in enumerate(discharge_capacities):
                    battery = np.array(battery)
                    battery_name = "battery" + str(index)
                    i = change_indices[index]   # FPC cycle
                    
                    percentage_index = 0
                    
                    EOL = len(battery[0])

                    while(i+stride+window_size+1 <= int(len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                            train_data.append((battery[:channels,i:i+window_size], 1-((i-change_indices[index])/(EOL - change_indices[index])),battery_name ))
                            i = i+stride
                            percentage_index = percentage_index+1

            return train_data
        else:
            print(type)
            test_data =[]
            for index,battery in enumerate(discharge_capacities):
                    battery = np.array(battery)
                    battery_name = "battery" + str(index+100)
                    i = change_indices[index]   # FPC cycle
                    percentage_index = 0
                    
                    EOL = len(battery[0])
                    

                    while(i+stride+window_size+1 <= int(len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                            test_data.append((battery[:channels,i:i+window_size], 1-(i-change_indices[index])/(EOL - change_indices[index]),battery_name ))
                            i = i+stride
                            percentage_index = percentage_index+1
                        
            return test_data

def get_data_RUL_scenario2(discharge_capacities,change_indices, window_size,stride,channels, type):
        
        if(type == "Train"):
            
            train_data =[]
            for index,battery in enumerate(discharge_capacities):
                    battery = np.array(battery)
                    battery_name = "battery" + str(index)
                    i = change_indices[index]
                    
                    percentage_index = 0
                    normalized_capacity,_ = NormalizeData(battery[0][i:])

                    while(i+stride+window_size+1 <= int(len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                            train_data.append((battery[:channels,i:i+window_size], normalized_capacity[percentage_index],battery_name ))
                            i = i+stride
                            percentage_index = percentage_index+1

            return train_data
        else:
            print(type)
            test_data =[]
            for index,battery in enumerate(discharge_capacities):
                    battery = np.array(battery)
                    battery_name = "battery" + str(index+100)
                    i = change_indices[index]
                    percentage_index = 0
                    normalized_capacity,_ = NormalizeData(battery[0][i:])

                    while(i+stride+window_size+1 <= int(len(battery[0])) and len(battery[0][i:i+window_size]) == window_size):
                            test_data.append((battery[:channels,i:i+window_size], normalized_capacity[percentage_index],battery_name ))
                            i = i+stride
                            percentage_index = percentage_index+1
            return test_data