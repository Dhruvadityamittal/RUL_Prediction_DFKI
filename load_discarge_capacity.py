# Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
import os

import torchmetrics

# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

batch1 = pickle.load(open(r'./Datasets/batch1.pkl', 'rb'))
#remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']
numBat1 = len(batch1.keys())
batch2 = pickle.load(open(r'./Datasets/batch2.pkl','rb'))

# There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
# and put it with the correct cell from batch1
batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482];

for i, bk in enumerate(batch1_keys):
    batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
    for j in batch1[bk]['summary'].keys():
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
    last_cycle = len(batch1[bk]['cycles'].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]


del batch2['b2c7']
del batch2['b2c8']
del batch2['b2c9']
del batch2['b2c15']
del batch2['b2c16']

numBat2 = len(batch2.keys())


batch3 = pickle.load(open(r'./Datasets/batch3.pkl','rb'))
# remove noisy channels from batch3
del batch3['b3c37']
del batch3['b3c2']
del batch3['b3c23']
del batch3['b3c32']
del batch3['b3c42']
del batch3['b3c43']

numBat3 = len(batch3.keys())

numBat = numBat1 + numBat2 + numBat3

bat_dict = {**batch1, **batch2, **batch3}

bat_dict[list(bat_dict.keys())[0]]['summary'].keys()
features = [  'Discharge Capacity', 'Charge Time','Internal Resistance', 'Charge Capacity','Temperature avg', 'Temperature min', 'Temperature max']

cycles = []
charge_time = []
discharge_capacities = []
IR = []
QC = []
QD = []
Tavg = []
Tmin = []
Tmax = []


for i in bat_dict.keys():
    plt.plot(bat_dict[i]['summary']['cycle'], bat_dict[i]['summary']['QD'])
    cycles.append(bat_dict[i]['summary']['cycle'][-1])
    discharge_capacities.append(bat_dict[i]['summary']['QD'])
#     charge_time.append(bat_dict[i]['summary']['chargetime'])
#     IR.append(bat_dict[i]['summary']['IR'])
#     QC.append(bat_dict[i]['summary']['QC'])
#     Tavg.append(bat_dict[i]['summary']['Tavg'])
\
#     Tmin.append(bat_dict[i]['summary']['Tmin'])
#     Tmax.append(bat_dict[i]['summary']['Tmax'])
    
    
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')
plt.title("Combined Data for 124 cells")

# Smoothing the input

maxes = []
mins = []
for  i in discharge_capacities:
    maxes.append(max(i))
    mins.append(min(i))

# Only removing the maximum outliers
for i in range(len(discharge_capacities)):
    for j in range(len(discharge_capacities[i])):
        if(discharge_capacities[i][j]>np.mean(maxes)):
            
            discharge_capacities[i][j]= np.mean(maxes)

    plt.plot(discharge_capacities[i])
plt.show()

        # if(discharge_capacities[i][j]<np.mean(mins)):
            
        #     discharge_capacities[i][j] =np.mean(mins)

discharge_capacities_np = np.array(discharge_capacities)
np.save(r'./Datasets/discharge_capacity.npy', discharge_capacities_np, allow_pickle=True)
