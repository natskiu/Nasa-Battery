import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import joblib
import random
from scipy import io
from modules.utils import *

DATAPATH = 'Data/'
BATTERY_IDS = ['B0005', 'B0006', 'B0007', 'B0018', 'B0049', 'B0051', 'B0053', 'B0054', 'B0055','B0056']

def get_random_battery(battery_ids=BATTERY_IDS):
    return random.choice(battery_ids)

def mat_to_vectorized_cycle(battery_id, datapath=DATAPATH):
    '''
    datapath: string, path to where the mat files are stored
    battery_id: string, battery id for the battery of interest
    '''
    file = io.loadmat(datapath + battery_id +'.mat')
    raw_cycles = np.vstack(file[battery_id][0,0])
    discharge_indices = get_indices(raw_cycles, is_charge=False)
    vectorized_cycles = capacity_vectorizer(discharge_indices, raw_cycles)
    return vectorized_cycles, discharge_indices

def prepare_row_for_model(vectorized_cycles, discharge_indices):
    chosen_index = [random.choice(discharge_indices)] #list with one entry
    max_temp_time, max_temp = extract_feature_1_2_6_7(chosen_index, vectorized_cycles, l_threshold = 250, r_threshold = 750, peak_width = 3)
    slope_temp = extract_feature_3(chosen_index, vectorized_cycles, l_threshold = 250, r_threshold = 750, peak_width = 3)
    time_3V = extract_feature_4(vectorized_cycles, chosen_index, threshold=500, voltage_cutoff=3)
    slope_V = extract_feature_5(vectorized_cycles, chosen_index, start_time=100, end_time=500)
    features_dict = {'time_for_max_temp_D':max_temp_time,'max_temp_D':max_temp,
                 'slope_temp_D': slope_temp, 'time_voltage_measured_below3_D':time_3V,
                 'slope_voltage_measured_D':slope_V, 'ambient_temp_4':[0], 'ambient_temp_24':[0]}
    df = pd.DataFrame(data = features_dict)
    return df

def main():
    battery_id = get_random_battery()
    vectorized_cycles, discharge_indices = mat_to_vectorized_cycle(battery_id)
    df = prepare_row_for_model(vectorized_cycles, discharge_indices) #selects a random row
    if battery_id in ['B0005', 'B0006', 'B0007', 'B0018']:
        df["ambient_temp_24"].replace({0: 1}, inplace=True)
    else:
        df["ambient_temp_4"].replace({0: 1}, inplace=True)
    model = joblib.load('best_voting.pkl')
    prediction = model.predict(df)
    print(battery_id, df, prediction)

if __name__ == "__main__":
    main()