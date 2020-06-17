import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow import keras

from neural_nets_fn import find_learning_rate, callbacks_fn, plot_lr_vs_loss
from neural_nets_fn import build_model, run_network, plot_learning_curves
from util import describe_Xy_data, test_rmsle, test_models, DatasetType, load_dataset

from functools import partial

np.random.seed(42)
tf.random.set_seed(42)

#%% INFO
initializers = [name for name in dir(keras.initializers) if not name.startswith("_")]
activations = [m for m in dir(keras.activations) if not m.startswith("_")]

#%%
DATA_FOLDER = ['dataset', 'prepared_data_and_models', 'train']
DS_TYPE = DatasetType.SCALED 
USE_YLOG = True

X_train, y_train, X_test, y_test, X_valid, y_valid = load_dataset(DS_TYPE, folder=DATA_FOLDER)
if USE_YLOG:
    y_train = np.log1p(y_train)
describe_Xy_data(X_train, y_train, X_test, y_test, X_valid, y_valid)#%% log data is better
all_data = (X_train, y_train, X_test, y_test, X_valid, y_valid)

#%%
LOSS          = 'mean_squared_logarithmic_error'
N_NEURONS     = 100
N_LAYERS      = 7
PATIENCE      = 15
EPOCHS        = 500
MODELS_FOLDER = ["TFmodels","ylog"]
INPUT_SHAPE = X_train.shape[1:]
maxLossPlot = 1.2
nEpochsPlot = 100
      
#%% PARTIAL FUNC
run_bike_network= partial(run_network, loss=LOSS, dataset_type=DS_TYPE, n_neurons=N_NEURONS, n_layers=N_LAYERS,
                          epochs=EPOCHS, input_shape=INPUT_SHAPE, save_folder=MODELS_FOLDER, 
                          with_cb=True, patience=PATIENCE, maxLossPlot=maxLossPlot,nEpochsPlot=nEpochsPlot,ylog_model=USE_YLOG)
#%% BATCH NORMALIZATION and ALPHA DROP did not help
# name = 'elu_sgdm_alpha'
# optimizer = keras.optimizers.SGD(lr=(10**-2)*1, momentum=0.9)
# run_bike_network(name, all_data, optimizer, activation='elu', alpha_drop=True)
#%% 
name = 'elu_sgdm'
optimizer = keras.optimizers.SGD(lr=(10**-3)*1, momentum=0.9)
run_bike_network(name, all_data, optimizer, activation='elu')

#%%
name='selu_sgdm'
optimizer = keras.optimizers.SGD(lr=(10**-3)*1, momentum=0.9)
run_bike_network(name, all_data, optimizer, activation='selu')

#%% ELU SGDM same as SELU SGDM

#%%
name='selu_rmsprop'
optimizer = keras.optimizers.RMSprop(lr=(10**-4)*1, rho=0.9)
run_bike_network(name, all_data, optimizer, activation='selu')

#%%
name='elu_adam'
optimizer = keras.optimizers.Adam(lr=(10**-4)*1, beta_1=0.9, beta_2=0.999)
run_bike_network(name, all_data, optimizer, activation='elu')
    
#%% TESTS
# Best Models
tfmodels_ymsle = test_models(models_folder=['TFmodels','y_msle'],data_folder=DATA_FOLDER)
tfmodels_ylog = test_models(models_folder=['TFmodels','ylog'],data_folder=DATA_FOLDER, ylog_model=True)
result_best = pd.concat([tfmodels_ymsle,tfmodels_ylog])
result_best.sort_values(by=['rmsle'],inplace=True)
print(result_best)

