import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 
import modules.utils as tools
import pickle
import argparse
from sklearn.model_selection import train_test_split

def preprocess_data(interpolated_data, window_size = 50, forecasting_range = 10):
    X = []
    y = []
    for battery_no in range(len(interpolated_data)):
        for cycle_no in range(len(interpolated_data[battery_no])):
            for step in range(len(interpolated_data[battery_no][cycle_no])- window_size - forecasting_range):
                x_row = interpolated_data[battery_no][cycle_no][step:step+window_size,1].tolist()
                y_row = interpolated_data[battery_no][cycle_no][step+window_size:step+window_size+forecasting_range,1]
                X.append(x_row)
                y.append(y_row)

    X = np.array(X).reshape(-1, window_size)
    y = np.array(y).reshape(-1, forecasting_range)
    return X, y

def min_max_scaling(all_data, minimum, maximum):
    return (all_data - minimum)/(maximum-minimum)

#custom metric for evaluation
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:,-1], Y_pred[:, -1])

def build_model(forecasting_range,n_neurons=80, n_hidden=1, l2_activity=0.0001, l2_kernel=0, lr=0.01, is_LSTM=True, activation='tanh'):
    model = keras.models.Sequential()
    if is_LSTM:
        model.add(keras.layers.LSTM(units=n_neurons, return_sequences=True, activation=activation, input_shape=[None,1]))
        for layer in range(n_hidden):
            model.add(keras.layers.LSTM(units=n_neurons, return_sequences=True, activation=activation))
    else:
        model.add(keras.layers.SimpleRNN(units=n_neurons, return_sequences=True, activation=activation, input_shape=[None,1]))
        for layer in range(n_hidden):
            model.add(keras.layers.SimpleRNN(units=n_neurons, return_sequences=True, activation=activation))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=forecasting_range, kernel_regularizer=keras.regularizers.l2(l2=l2_kernel), activity_regularizer=keras.regularizers.l2(l2=l2_activity))))
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])
    return model


def main():
    parser = argparse.ArgumentParser(description='Handle inputs')
    parser.add_argument('-w', '--window_size', type=int, default=50)
    parser.add_argument('-f', '--forecasting_range', type=int, default=200)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-n', '--filename', type=str)
    args = parser.parse_args()
    window_size = args.window_size
    forecasting_range = args.forecasting_range
    epochs = args.epochs
    filename = args.filename

    with open('Data/pickles/interpolated_data.pkl', 'rb') as f:
        interpolated_data = pickle.load(f)
    X, y = preprocess_data(interpolated_data, window_size=window_size, forecasting_range=forecasting_range)
    all_data = np.concatenate((X, y), axis=1)
    minimum = all_data.min()
    maximum = all_data.max()
    all_data_scaled = min_max_scaling(all_data, maximum=maximum, minimum=minimum)
    X = all_data_scaled[:, :window_size]
    y = all_data_scaled[:,window_size:]

    X = X.reshape(-1, window_size, 1)
    y = y.reshape(-1, forecasting_range, 1)
    series = np.concatenate((X, y), axis=1)

    Y = np.empty((X.shape[0], window_size, forecasting_range))
    for step_ahead in range(1, forecasting_range+1):
        Y[:,:,step_ahead -1] = series[:,step_ahead:step_ahead+window_size,0]
    X_seq_train, X_seq_valtest, Y_seq_train, Y_seq_valtest = train_test_split(X, Y, random_state=42, train_size=0.6)
    X_seq_val, X_seq_test, Y_seq_val, Y_seq_test = train_test_split(X_seq_valtest, Y_seq_valtest, random_state=42, train_size=0.5)

    wrapped_model = build_model(forecasting_range)
    wrapped_model.fit(X_seq_train, Y_seq_train, epochs=epochs, validation_data=(X_seq_val, Y_seq_val))
    wrapped_model.save(filename)

if __name__ == '__main__':
    main()