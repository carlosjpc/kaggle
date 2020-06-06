import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow import keras

from neural_nets_fn import find_learning_rate, callbacks_fn, plot_lr_vs_loss
from neural_nets_fn import build_model, run_network
from util import describe_Xy_data, test_mse, test_models

from functools import partial

np.random.seed(42)
tf.random.set_seed(42)

#%% INFO
initializers = [name for name in dir(keras.initializers) if not name.startswith("_")]
activations = [m for m in dir(keras.activations) if not m.startswith("_")]

#%%
all_data = joblib.load("dataset/XY.pkl")
X_train, X_test, y_train, y_test, X_valid, y_valid  = all_data
describe_Xy_data(X_train,y_train,X_test,y_test,X_valid,y_valid)     
        
#%% BATCH NORMALIZATION and ALPHA DROP did not help

#%% TRY MORE LAYERS OR NEURONS
LOSS = 'mean_squared_error'
INPUT_SHAPE = X_train.shape[1:]
n_neurons = 100
n_layers = 7
EPOCHS = 500
MODELS_FOLDER = 'TFmodels'
run_bike_network= partial(run_network, loss=LOSS, n_neurons=n_neurons, n_layers=n_layers,
                          epochs=EPOCHS, input_shape=INPUT_SHAPE, save_folder=MODELS_FOLDER, with_cb=True)

#%%
name='elu_sgdm'
name += '_' +str(n_layers) + '_' + str(n_neurons)
cb = callbacks_fn(name)
lr=(10**-6)*1
keras.backend.clear_session()
model = build_model(n_layers,n_neurons,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid), callbacks=cb)
test_mse(model,X_test,y_test,name)
model.save('TFmodels/'+name+".h5")

#%%
name='selu_sgdm'
name += '_'+ str(n_layers) + '_' + str(n_neurons)
keras.backend.clear_session()
cb = callbacks_fn(name)
lr=(10**-6)*1
keras.backend.clear_session()
model = build_model(n_layers,n_neurons,input_shape=INPUT_SHAPE, activation='selu')
optimizer=keras.optimizers.SGD(lr=lr,  momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid), callbacks=cb)
test_mse(model,X_test,y_test,name)
model.save('TFmodels/'+name+".h5")

#%% ELU SGDM same as SELU SGDM

#%%
name='selu_rmsprop'
name += '_' + str(n_layers) + '_' + str(n_neurons)
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(n_layers,n_neurons,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid),callbacks=cb)
test_mse(model,X_test,y_test,name)
model.save('TFmodels/'+name+".h5")

#%%
name='elu_adam'
name += '_' + str(n_layers) + '_' + str(n_neurons)
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(n_layers,n_neurons,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid),callbacks=cb)
test_mse(model,X_test,y_test,name)
model.save('TFmodels/'+name+".h5")

model = run_bike_network('elu_adam', all_data, optimizer, activation='elu')

#%%

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
run_network('elu_adam', all_data, optimizer, activation='elu')

#%%
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model = run_bike_network('elu_adam', all_data, optimizer, activation='elu')

# run_network(name, all_data, optimizer, input_shape=[17], with_cb=False)
    
#%% TESTS
test_models('TFmodels',X_test,y_test)
test_models('SKmodels',X_test,y_test)