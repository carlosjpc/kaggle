import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow import keras
from functools import partial

from neural_nets_fn import find_learning_rate, callbacks_fn, plot_lr_vs_loss
from neural_nets_fn import build_model, plot_learning_curves
from util import describe_Xy_data, test_rmsle
from util import describe_Xy_data, DatasetType, load_dataset, test_rmsle

np.random.seed(42)
tf.random.set_seed(42)

#%% INFO
initializers = [name for name in dir(keras.initializers) if not name.startswith("_")]
activations = [m for m in dir(keras.activations) if not m.startswith("_")]

#Momentum optimization
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
#Nesterov Accelerated Gradient
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
#AdaGrad do not use for deep neural networks
optimizer = keras.optimizers.Adagrad(lr=0.001)
#RMSProp
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
#Adam Optimization
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#Adamax
optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)
#Nadam Optimization
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)

#%%
LOSS = 'mean_squared_logarithmic_error'
N_NEURONS = 30
N_LAYERS = 3
DS_TYPE = DatasetType.SCALED 
DATA_FOLDER = ['dataset', 'prepared_data_and_models', 'train']
plot_learning_curves_bikes = partial(plot_learning_curves,nEpochs=100,maxLoss=1.2)

#%%
X_train, y_train, X_test, y_test, X_valid, y_valid = load_dataset(DS_TYPE, folder=DATA_FOLDER)
describe_Xy_data(X_train, y_train, X_test, y_test, X_valid, y_valid)#%% log data is better
ylog_train = np.log1p(y_train)
ylog_valid = np.log1p(y_valid)
ylog_test = np.log1p(y_test)

INPUT_SHAPE = X_train.shape[1:]

#%% ELU
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=1)
plot_lr_vs_loss(rates, losses)

#%%
name='elu_sgdm'
keras.backend.clear_session()
cb = callbacks_fn(name)
lr=(10**-2)*1
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=cb)
test_rmsle(model,X_test,y_test,name)
plot_learning_curves_bikes(history,name)

#%% SELU
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=1)
plot_lr_vs_loss(rates, losses)

#%% SELU with momentum = nan

#%%
name='elu_rmsprop'
keras.backend.clear_session()
cb = callbacks_fn(name)
lr=(10**-2)*1
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_rmsle(model,X_test,y_test,name)
plot_learning_curves_bikes(history,name)

#%%
name='selu_rmsprop'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_rmsle(model,X_test,y_test,name)
plot_learning_curves_bikes(history,name)

#%%
name='elu_adam'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_rmsle(model,X_test,y_test,name)
plot_learning_curves_bikes(history,name)

#%%
name='selu_adam'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_rmsle(model,X_test,y_test,name)
plot_learning_curves_bikes(history,name)
