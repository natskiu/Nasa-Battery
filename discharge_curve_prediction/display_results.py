import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
import pickle
import argparse

###
# This program first loads our trained model for predicting the next 50 voltage discharge curves given the previous 10. 
# It then uses this model to make a prediction on user-specified data, and plots the predictions against the actual discharge curves.
# by Venus Lee and Jesse Wang
###

window_size = 10
forecasting_range = 50
dimensionality = 230
cutoff = 5

### Functions for data preprocessing ###

def clip_interpolated_data(interpolated_data, cutoff=5):
    for i in range(3):
        for j in range(168):
            interpolated_data[i,j] = interpolated_data[i, j][cutoff:,:]
    return interpolated_data

def min_max_scaling(all_data, minimum = 1.7555322021036255, maximum = 4.2331462538907605):
    return (all_data - minimum)/(maximum-minimum)

def inverse_min_max_scaling(scaled_data, minimum = 1.7555322021036255, maximum = 4.2331462538907605):
    return (scaled_data)*(maximum-minimum) + minimum

def preprocess_data(interpolated_data):
    interpolated_data = clip_interpolated_data(interpolated_data, cutoff=cutoff)
    interpolated_data = min_max_scaling(interpolated_data).tolist()

    num_examples = len(interpolated_data[0]) - window_size - forecasting_range
    X_5 = np.empty((num_examples, window_size, dimensionality))
    Y_5 = np.empty((num_examples, forecasting_range, dimensionality))
    for i in range(num_examples):
        for j in range(window_size):
            X_5[i, j, :] = interpolated_data[0][i+j][:,1][:dimensionality]
        for k in range(forecasting_range):
            Y_5[i, k, :] = interpolated_data[0][i+k+window_size][:,1][:dimensionality]
    num_examples = len(interpolated_data[1]) - window_size - forecasting_range
    X_6 = np.empty((num_examples, window_size, dimensionality))
    Y_6 = np.empty((num_examples, forecasting_range, dimensionality))
    for i in range(num_examples):
        for j in range(window_size):
            X_6[i, j, :] = interpolated_data[1][i+j][:,1][:dimensionality]
        for k in range(forecasting_range):
            Y_6[i, k, :] = interpolated_data[1][i+k+window_size][:,1][:dimensionality]
    num_examples = len(interpolated_data[2]) - window_size - forecasting_range
    X_7 = np.empty((num_examples, window_size, dimensionality))
    Y_7 = np.empty((num_examples, forecasting_range, dimensionality))
    for i in range(num_examples):
        for j in range(window_size):
            X_7[i, j, :] = interpolated_data[2][i+j][:,1][:dimensionality]
        for k in range(forecasting_range):
            Y_7[i, k, :] = interpolated_data[2][i+k+window_size][:,1][:dimensionality]

    return X_5, X_6, X_7, Y_5, Y_6, Y_7


### Functions to create model object ###

def get_weights(type="weights_1"):
    if type == "weights_1":
        row = np.linspace(start=1.5, stop=0.5, num=forecasting_range)
        weights = np.tile(row, (dimensionality, 1))
    return weights

def weighted_loss(weights, error_func='mse'): # Custom loss function
    def loss(y_true, y_pred): # y_true, y_pred have dimensions (dimensionality x forecasting_range)
        if error_func == 'mse':
            squared_difference = tf.math.square(y_true - y_pred)
            weighted_squared_difference = tf.math.multiply(squared_difference, weights)
            return tf.reduce_mean(weighted_squared_difference, axis=-1)
        if error_func == 'mae':
            abs_difference = tf.math.abs(y_true - y_pred)
            weighted_abs_difference = tf.math.multiply(abs_difference, weights)
            return tf.reduce_mean(weighted_abs_difference, axis=-1)
    return loss

def create_model(loss,forecasting_range=50,window_size=10, n_neurons=100, additional_layers=1, 
activation='tanh', optimizer='Adam', include_dropout=True, dropout_prob=0.2, lr=3e-4):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(n_neurons, input_shape=[None, window_size], activation=activation, return_sequences=True))
    if include_dropout:
        model.add(keras.layers.Dropout(dropout_prob))
    for _ in range(additional_layers):
        model.add(keras.layers.LSTM(n_neurons, activation=activation, return_sequences=True))
        if include_dropout:
            model.add(keras.layers.Dropout(dropout_prob))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(forecasting_range)))
    if optimizer == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss)
    return model

### Functions to visualise results ###

def prepare_data_for_plot(model, X_5, X_6, X_7):
    X_5_swapped = np.swapaxes(X_5, 1, 2)
    Y_5_preds = np.swapaxes(model.predict(X_5_swapped), 1, 2)
    X_6_swapped = np.swapaxes(X_6, 1, 2)
    Y_6_preds = np.swapaxes(model.predict(X_6_swapped), 1, 2)
    X_7_swapped = np.swapaxes(X_7, 1, 2)
    Y_7_preds = np.swapaxes(model.predict(X_7_swapped), 1, 2)
    return Y_5_preds, Y_6_preds, Y_7_preds

def plot_projections(X_5, X_6, X_7, Y_5, Y_6, Y_7, Y_5_preds, Y_6_preds, Y_7_preds,
    battery_no=0, starting_cycle=0, step=20, is_actual=False, dimensionality=200):
    if battery_no == 0:
        X_data, Y_data, Y_preds = X_5, Y_5, Y_5_preds
        battery_id = 'B0005'
    if battery_no == 1:
        X_data, Y_data, Y_preds = X_6, Y_6, Y_6_preds
        battery_id = 'B0006'
    if battery_no == 2:
        X_data, Y_data, Y_preds = X_7, Y_7, Y_7_preds
        battery_id = 'B0007'
    if is_actual:
        text = 'Actual'
    else:
        text = 'Projected'
        Y_data = Y_preds
    plt.figure
    plt.title(text + ' discharge curves for the next 50 cycles \n Starting cycle: {}, Battery {}'.format(starting_cycle, battery_id))
    plt.xlim(left=0, right=dimensionality)
    plt.ylim(bottom=3, top=4)
    plt.xlabel('Time step (1 unit = 10s)')
    plt.ylabel('Voltage')
    x = np.arange(start=0, stop=dimensionality, step=1)
    for i in range(window_size):
        voltages = inverse_min_max_scaling(X_data[starting_cycle, i, :])
        plt.plot(x, voltages, color = 'orange', label='Cycles {} to {}'.format(starting_cycle, starting_cycle+window_size-1))
    for j in range(0,forecasting_range,step):
        voltages = inverse_min_max_scaling(Y_data[starting_cycle, j, :])
        plt.plot(x, voltages, color = 'green', label=text + ' cycles {} to {}'.format(starting_cycle+window_size, starting_cycle+window_size+forecasting_range-1))
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

### Main function ###

def main():
    parser = argparse.ArgumentParser(description='Handle inputs')
    parser.add_argument('-b', '--battery_number', type=int, default=0, help='0 corresponds to B0005, 1 corresponds to B0006, 2 corresponds to B0007')
    parser.add_argument('-s', '--starting_cycle', type=int, default=100, help='must be between 0 and 107')
    args = parser.parse_args()
    
    with open('pickles/interpolated_data.pkl', 'rb') as f:
        interpolated_data = pickle.load(f)
    interpolated_data = np.array(interpolated_data)
    X_5, X_6, X_7, Y_5, Y_6, Y_7 = preprocess_data(interpolated_data)
    weights = get_weights()
    weights_model = create_model(loss=weighted_loss(weights, error_func='mse'), additional_layers=1, n_neurons=120, include_dropout=False)
    weights_model.load_weights('h5_files/weights_1_dim_230_neurons_120.h5')
    Y_5_preds, Y_6_preds, Y_7_preds = prepare_data_for_plot(weights_model, X_5, X_6, X_7)

    fig1 = plt.figure()
    fig1.set_figheight(4)
    fig1.set_figwidth(13)
    plt.subplot(1,2,1)
    plot_projections(X_5, X_6, X_7, Y_5, Y_6, Y_7, Y_5_preds, Y_6_preds, Y_7_preds,
        battery_no=args.battery_number, starting_cycle=args.starting_cycle, step=5, is_actual=False, dimensionality=dimensionality)
    plt.subplot(1,2,2)
    plot_projections(X_5, X_6, X_7, Y_5, Y_6, Y_7, Y_5_preds, Y_6_preds, Y_7_preds,
        battery_no=args.battery_number, starting_cycle=args.starting_cycle, step=5, is_actual=True, dimensionality=dimensionality)
    plt.show()

if __name__ == '__main__':
    main()
    