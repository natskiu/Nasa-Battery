import sys
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
sys.path.append('..')

# Functions to preprocess raw data
def capacity_vectorizer(discharge_indices, cycles):
    '''
    input
    --------
    discharge_indices: list, 
    cycles: array, the stacked raw cycle data
    '''
    for discharge_index in discharge_indices:
        N = (cycles[0,discharge_index][3][0,0][5]).shape[1] # gets number of entries
        capacity_scalar = cycles[0,discharge_index][3][0,0][6]
        cycles[0,discharge_index][3][0,0][6] = np.pad(capacity_scalar.flatten().tolist(), (0, N-1), 'constant')
        vectorized_cycles = cycles
    return vectorized_cycles


def get_indices(cycles, is_charge=True):
    index_list =[]
    if is_charge:
        label = 'charge'
    else:
        label = 'discharge'
    for i in range(cycles.shape[1]):
        if cycles[0,i][0] == np.array([label]):
             index_list.append(i)
    return index_list

# Functions to visualize data
def cycle_plotter(cycles, discharge_indices, cycle_indices):
    '''
    cycles: array, the vectorized array containing the raw data
    discharge_indices: list, containing all the discharging indices
    cycle_indices: list, containing the cycle numbers we want to plot 
    '''
    # the reason for adding an extra param entry_indices is so that we can keep track of the cycle number by discharge indices while having 
    # the freedom to plot only the cycles of interest
    features = ['voltage_measured/ V', 'currenet_measured/ A', 'temperature/ C', 'current_load/charge/ A', 'voltage_load/charge/ V']  
    for i, label in enumerate(features):
        for discharge_cycle_number, discharge_index in enumerate(discharge_indices):
            y = np.vstack(cycles[0,discharge_index][3][0,0])[i]
            x = np.vstack(cycles[0,discharge_index][3][0,0])[5]
            #f=plt.figure()
            if (discharge_cycle_number+1) in cycle_indices:
                plt.plot(x, y,marker='o' ,markersize = 3, linestyle='',label=('cycle'+str(discharge_cycle_number+1)))
        plt.ylabel(label)
        plt.xlabel('time/s')
        title = input('Please enter the titile of the graph for'+label)       
        plt.title(title) 
        plt.legend()   
        plt.show()

# Functions to extract features
def extract_feature_1_2_6_7(indices, cycles, l_threshold = 250, r_threshold = 750, peak_width = 3):
    '''
  This function returns two lists - first is the time to reach the maximum 
  temperature for each cycle, second is the maximum temperature for each cycle.
  This can be used for both charging and discharging cycles, just input the 
  correct list of indices. NB for charging cycles remove the extra two cycles
  first
  --------
  indices: list, a list that can be generated using function get_indices 
          containing indices of discharging/ charging cycles
  cycles: array, an array containing all the data of the battery. It has to be
          preprocessed to be stackable and indexable.
  threshold: int, set by default to 1000 to avoid taking in to account of 
             anomalous max temp data
  outputs
  --------
  max_temp_time_list: list, a list that contains the corresponding time of the 
                      maximum temperature in every discharging/ charging cycle
  max_temp_list: list, a list that contains the maximum temperature in every
                 discharging/ charging cycle 
                
  note: the lengths of the lists should be the same = the length of the input 
  list of indices.
    '''
    max_temp_time_list = []
    max_temp_list = []
    for index in indices:
        times = cycles[0,index][3][0,0][5].flatten().tolist()
        temps = cycles[0,index][3][0,0][2].flatten().tolist()
        l_threshold_index = next(time[0] for time in enumerate(times) if time[1] > l_threshold)
        try:
            r_threshold_index = next(time[0] for time in enumerate(times) if time[1] > r_threshold)
        except StopIteration:
            r_threshold_index = len(times) - 1
        temps = temps[l_threshold_index:r_threshold_index]
        times = times[l_threshold_index:r_threshold_index]

        decreasing_count = 0
        max_temp_index = 0
        for i, temp in enumerate(temps):
            if temp < temps[i-1]:
                decreasing_count += 1
                if decreasing_count == peak_width:
                    max_temp_index = i-(peak_width-1)
                    break
            else:
                decreasing_count = 0
        
        if max_temp_index == 0:
            max_temp_time_list.append(float('nan'))
            max_temp_list.append(float('nan'))
        else:
            max_temp = temps[max_temp_index] ## Need to change
            max_temp_time_list.append(times[max_temp_index])
            max_temp_list.append(max_temp)
    return max_temp_time_list, max_temp_list

def extract_feature_3(discharge_indices, cycles, l_threshold = 250, r_threshold = 750, peak_width = 3):
    '''
    Inputs
    --------
    indices: list, a list that can be generated using function get_indices 
          containing indices of discharging/ charging cycles
    cycles: array, an array containing all the data of the battery. It has to be
          preprocessed to be stackable and indexable.
    threshold: int, set by default to 1000 to avoid taking in to account of 
             anomalous max temp data

    '''
    max_temp_times, max_temps = extract_feature_1_2_6_7(discharge_indices, cycles,l_threshold, r_threshold, peak_width)
    initial_temps = []
    for discharge_index in discharge_indices:
      initial_temp = cycles[0,discharge_index][3][0,0][2].flatten().tolist()[0]
      initial_temps.append(initial_temp)
    slopes = (np.array(max_temps)-np.array(initial_temps))/np.array(max_temp_times)
    return slopes.tolist()

def extract_feature_4(dataset, indices, threshold=500, voltage_cutoff=3): #time for voltage_measured to drop below 3V during discharge
    '''
    Feature 4 is the time for voltage_measured to drop below 3V during discharge.
    **Input**
    data = full cycle data
    indices = list of indices from which we would like to extract features
    threshold = time from which to start checking (to avoid anomalous data at the start)

    **Output**
    feature_4_list = list of feature values for each cycle
    '''
    feature_4_list = []
    for index in indices:
        voltage_measured_list = (dataset[0,index][3][0,0][0]).flatten().tolist() #turn voltage_measured numpy array into list
        time_list = (dataset[0,index][3][0,0][5]).flatten().tolist() #turn time vector into list
        threshold_index = next(i for i, time in enumerate(time_list) if time > threshold) #getting index of threshold
        voltage_measured_list = voltage_measured_list[threshold_index:] #shortening voltage_measured_list and time_list
        time_list = time_list[threshold_index:]
        try:
            index_3V = next(i for i, voltage in enumerate(voltage_measured_list) if voltage < voltage_cutoff) #getting index of when 3V is reached
            feature_4_list.append(time_list[index_3V]) #getting the corresponding time
        except StopIteration:
            feature_4_list.append(float('NaN'))
    return feature_4_list

def extract_feature_5(dataset, indices, start_time=100, end_time=500): #slope of voltage_measured during discharge
    '''
  Feature 5 is the slope of voltage_measured using the first N data points.
  **Input**
  data = full cycle data
  indices = list of indices from which we would like to extract features
  start_time = start time to measure slope
  end_time = end time to measure slope

  **Output**
  feature_5_list = list of feature values for each cycle
    '''
    feature_5_list = []

    for index in indices:
        voltage_measured_list = (dataset[0,index][3][0,0][0]).flatten().tolist() #turn voltage_measured numpy array into list
        time_list = (dataset[0,index][3][0,0][5]).flatten().tolist() #turn time vector into list
        start_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i]-start_time)) #get index of start_time
        end_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i]-end_time)) #get index of end_time
        slope = (voltage_measured_list[end_time_index] - voltage_measured_list[start_time_index])/(time_list[end_time_index] - time_list[start_time_index])
        feature_5_list.append(slope)
  
    return feature_5_list

def extract_label(dataset, indices):
  labels = []
  for index in indices:
    label = (dataset[0,index][3][0,0][6]).flatten().tolist()
    labels.append(label[0])
  return labels

def remaining_cycles(dataset, indices,threshold = 0.7):
    '''
    input
    ---------
    threshold: int, the battery is considered to have reached its end-of-life if the capacity is lower than initial_capacity*threshold
    output
    --------
    remaining_cycles_list: list, a list that 
    '''
    #capacity_list = []
    initial_capacity = (dataset[0,1][3][0,0][6]).flatten().tolist()[0]
    cutoff = initial_capacity*threshold
    capacity = initial_capacity
    i = 0
    while capacity > cutoff:
        try:
            critical_cycle = i-1
            capacity = (dataset[0,indices[i]][3][0,0][6]).flatten().tolist()[0]
            i += 1
        except IndexError:
            capacity = 0
            critical_cycle = i-1

    remaining_cycles_list = [(critical_cycle - j) for j in range(len(indices))]
    return remaining_cycles_list