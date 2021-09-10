import numpy as np 
import matplotlib.pyplot as plt 
from scipy import interpolate
import modules.utils as tools
from scipy import io
import os
import os.path
from scipy.interpolate import interp1d
import argparse
import pickle

DATAPATH = 'Data/'
battery_ids = ['B0005', 'B0006', 'B0007']

def get_uninterpolated_data(battery_ids, datapath = DATAPATH):
    uninterpolated_data = []
    for battery_id in battery_ids:
        v = io.loadmat(datapath + battery_id + '.mat')
        raw_cycles = np.vstack(v[battery_id][0,0])
        discharge_indices = tools.get_indices(raw_cycles, is_charge = False)
        new_cycles = tools.capacity_vectorizer(discharge_indices, raw_cycles)
        voltage_timeseries_all = []
        for discharge_index in discharge_indices:
            voltage_measured = new_cycles[0,discharge_index][3][0,0][0].flatten().tolist()
            time = new_cycles[0,discharge_index][3][0,0][5].flatten().tolist()
            single_cycle = np.array([time, voltage_measured]).transpose()
            voltage_timeseries_all.append(single_cycle)
        uninterpolated_data.append(np.array(voltage_timeseries_all))
    return uninterpolated_data

def get_interpolated_data(uninterpolated_data, step_size=10):
    interpolated_data = []
    for battery_no in range(len(uninterpolated_data)):
        battery_interpolated = []
        for cycle_no in range(len(uninterpolated_data[battery_no])):
            times = uninterpolated_data[battery_no][cycle_no][:,0]
            voltages = uninterpolated_data[battery_no][cycle_no][:,1]
            f = interp1d(times, voltages) #linear interpolation
            new_times = np.arange(0, times[-1], step_size)
            new_voltages = f(new_times)
            single_cycle = np.array([new_times, new_voltages]).transpose()
            battery_interpolated.append(single_cycle)
        interpolated_data.append(battery_interpolated)
    return interpolated_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for specified batteries')
    parser.add_argument('batteries', nargs='+')
    args = parser.parse_args()
    uninterpolated_data = get_uninterpolated_data(args.batteries)
    interpolated_data = get_interpolated_data(uninterpolated_data)
    filepath = "Data/pickles"
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    with open(filepath+'/interpolated_data.pkl', 'wb') as f:
        pickle.dump(interpolated_data, f)