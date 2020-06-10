import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow import keras

from neural_nets_fn import find_learning_rate, callbacks_fn, plot_lr_vs_loss
from neural_nets_fn import build_model, run_network
from util import describe_Xy_data, test_mse, test_models, DatasetType, load_dataset

from functools import partial

np.random.seed(42)
tf.random.set_seed(42)

#%% INFO
initializers = [name for name in dir(keras.initializers) if not name.startswith("_")]
activations = [m for m in dir(keras.activations) if not m.startswith("_")]

#%%
DS_TYPE = DatasetType.SCALED 

all_data= load_dataset(DS_TYPE, folder='dataset')
X_train, y_train, X_test, y_test, X_valid, y_valid = all_data
describe_Xy_data(X_train, y_train, X_test, y_test, X_valid, y_valid)

#%%
LOSS          = 'mean_squared_error'
N_NEURONS     = 100
N_LAYERS      = 7
PATIENCE      = 15
EPOCHS        = 500
MODELS_FOLDER = "TFmodels"
INPUT_SHAPE = X_train.shape[1:]

run_bike_network= partial(run_network, loss=LOSS, dataset_type=DS_TYPE, n_neurons=N_NEURONS, n_layers=N_LAYERS,
                          epochs=EPOCHS, input_shape=INPUT_SHAPE, save_folder=MODELS_FOLDER, 
                          with_cb=True, patience=PATIENCE)
        
#%% BATCH NORMALIZATION and ALPHA DROP did not help

#%% 
name = 'elu_sgdm'
optimizer = keras.optimizers.SGD(lr=(10**-6)*1, momentum=0.9)
run_bike_network(name, all_data, optimizer, activation='elu')

#%%
name='selu_sgdm'
optimizer = keras.optimizers.SGD(lr=(10**-6)*1, momentum=0.9)
run_bike_network(name, all_data, optimizer, activation='selu')

#%% ELU SGDM same as SELU SGDM

#%%
name='selu_rmsprop'
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
run_bike_network(name, all_data, optimizer, activation='selu')

#%%
name='elu_adam'
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
run_bike_network(name, all_data, optimizer, activation='elu')
    
#%% TESTS
skmodels = test_models(models_folder='SKmodels',data_folder='dataset')
tfmodels = test_models(models_folder='TFmodels',data_folder='dataset')

result_tests = pd.concat([skmodels,tfmodels])
result_tests.sort_values(by=['mse'],inplace=True)
print(result_tests)