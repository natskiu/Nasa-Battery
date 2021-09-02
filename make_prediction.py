import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import joblib
import random
import warnings
from scipy import io
from modules.utils import *

DATAPATH = 'Data/'
BATTERY_IDS = ['B0005', 'B0006', 'B0007', 'B0018', 'B0049', 'B0051', 'B0053', 'B0054', 'B0055','B0056']

def get_random_battery(battery_ids=BATTERY_IDS):
    return random.choice(battery_ids)

def mat_to_discharge_indices(battery_id, datapath=DATAPATH):
    '''
    datapath: string, path to where the mat files are stored
    battery_id: string, battery id for the battery of interest
    '''
    file = io.loadmat(datapath + battery_id +'.mat')
    raw_cycles = np.vstack(file[battery_id][0,0])
    discharge_indices = get_indices(raw_cycles, is_charge=False)
    if battery_id in ['B0049', 'B0051']:
        discharge_indices.remove(8)
        discharge_indices.pop()
    elif battery_id in ['B0053', 'B0054']:
        discharge_indices.pop()
    vectorized_cycles = capacity_vectorizer(discharge_indices, raw_cycles)
    return discharge_indices, vectorized_cycles

def import_csv(battery_id):
    filepath = 'processed_csv/'
    name = '_processed.csv'
    dataframe_capacity = pd.read_csv((filepath+battery_id+name), index_col = 0)
    dataframe = dataframe_capacity.drop(['capacity', 'time_for_max_temp_C', 'remaining_cycles', 'max_temp_C'], axis=1)
    scaler = joblib.load('scaler.pkl')
    dataframe[dataframe.columns] = scaler.transform(dataframe)
    if battery_id in ['B0005', 'B0006', 'B0007', 'B0018']:
        dataframe["ambient_temp_4"] = 0
        dataframe["ambient_temp_24"] = 1
    else:
        dataframe["ambient_temp_4"] = 1
        dataframe["ambient_temp_24"] = 0
    return dataframe, dataframe_capacity

def prepare_row_for_model(dataframe):
    dataframe.dropna(inplace=True)
    chosen_index = random.choice(dataframe.index.values.tolist()) #list with one entry
    row = dataframe.loc[[chosen_index]]
    
    '''max_temp_time, max_temp = extract_feature_1_2_6_7(chosen_index, vectorized_cycles, l_threshold = 250, r_threshold = 750, peak_width = 3)
    slope_temp = extract_feature_3(chosen_index, vectorized_cycles, l_threshold = 250, r_threshold = 750, peak_width = 3)
    time_3V = extract_feature_4(vectorized_cycles, chosen_index, threshold=500, voltage_cutoff=3)
    slope_V = extract_feature_5(vectorized_cycles, chosen_index, start_time=100, end_time=500)
    features_dict = {'time_for_max_temp_D':max_temp_time,'max_temp_D':max_temp,
                 'slope_temp_D': slope_temp, 'time_voltage_measured_below3_D':time_3V,
                 'slope_voltage_measured_D':slope_V, 'ambient_temp_4':[0], 'ambient_temp_24':[0]}
    df = pd.DataFrame(data = features_dict)'''
    return row, chosen_index

def get_true_capacity(dataframe_capacity, chosen_index):
    return dataframe_capacity.capacity.loc[chosen_index]

def final_plot(vectorized_cycles, discharge_indices, chosen_index, prediction, true_capacity, battery_id):
    features = ['voltage/ V', 'temperature/ C']
    discharge_index = discharge_indices[chosen_index]
    voltage = np.vstack(vectorized_cycles[0,discharge_index][3][0,0])[0]
    temp = np.vstack(vectorized_cycles[0,discharge_index][3][0,0])[2]
    time = np.vstack(vectorized_cycles[0,discharge_index][3][0,0])[5]
    plt.subplot(1, 2, 1)
    plt.plot(time, voltage,marker='o' ,markersize = 3, linestyle='')
    plt.xlabel('time/ s')
    plt.ylabel('voltage_measured during discharge/ V')
    plt.title(battery_id + ' voltage measured over time for cycle ' + str(chosen_index))
    plt.subplot(1, 2, 2)
    plt.plot(time, temp, marker='o' ,markersize = 3, linestyle='')
    plt.xlabel('time/ s')
    plt.ylabel('temperature during discharge/ C')
    plt.title(battery_id + ' temperature over time for cycle ' + str(chosen_index))
    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()
    plt.text(xmax-(xmax-xmin)/10, ymin+(ymax-ymin)/10, 'Capacity prediction: ' + str(round(prediction, 4)) + 'Ah', horizontalalignment='right', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(xmax-(xmax-xmin)/10, ymin+1.6*(ymax-ymin)/10, 'True capacity: ' + str(round(true_capacity, 4)) + 'Ah', horizontalalignment='right', verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

def main():
    font = {'family': 'normal', 'size': 16}
    matplotlib.rc('font', **font)
    warnings.filterwarnings('ignore', category=UserWarning)
    battery_id = get_random_battery()
    discharge_indices, vectorized_cycles = mat_to_discharge_indices(battery_id)
    dataframe, dataframe_capacity = import_csv(battery_id) #can have NaN
    row, chosen_index = prepare_row_for_model(dataframe)
    true_capacity = get_true_capacity(dataframe_capacity, chosen_index)
    model = joblib.load('best_voting.pkl')
    try:
        prediction = float(model.predict(row))
    except ValueError:
        print('Failed to make prediction.')
    final_plot(vectorized_cycles, discharge_indices, chosen_index, prediction, true_capacity, battery_id)

if __name__ == "__main__":
    main()