import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow import keras
from IPython.display import display
from functools import partial

from neural_nets_fn import find_learning_rate, callbacks_fn, plot_lr_vs_loss
from neural_nets_fn import build_model, plot_learning_curves
from util import describe_Xy_data, DatasetType, load_dataset, test_rmsle, rmsle, tf_rmsle

np.random.seed(42)
tf.random.set_seed(42)

#%% INFO
initializers = [name for name in dir(keras.initializers) if not name.startswith("_")]
activations = [m for m in dir(keras.activations) if not m.startswith("_")]

#%%
DATA_FOLDER = ['dataset', 'prepared_data_and_models', 'train']
DS_TYPE = DatasetType.SCALED 
USE_YLOG = False

X_train, y_train, X_test, y_test, X_valid, y_valid = load_dataset(DS_TYPE, folder=DATA_FOLDER)
if USE_YLOG:
    y_train = np.log1p(y_train)
    y_valid = np.log1p(y_valid)
    y_test = np.log1p(y_test)
describe_Xy_data(X_train, y_train, X_test, y_test, X_valid, y_valid)#%% log data is better

#%%
LOSS = "mean_squared_logarithmic_error" #"mean_squared_logarithmic_error" #'mean_squared_error'
N_NEURONS = 30
N_LAYERS = 3
MODELS_FOLDER = ['TFmodels','y_msle']
INPUT_SHAPE = X_train.shape[1:]
plot_learning_curves_bikes = partial(plot_learning_curves,nEpochs=100,maxLoss=1.2)

#%% BASELINE
name='baseline'
keras.backend.clear_session()
callbacks = callbacks_fn(name)
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE)
optimizer=keras.optimizers.SGD(lr=1e-1)
model.compile(loss=LOSS, optimizer=optimizer)
display(keras.utils.plot_model(model, name+'.png', show_shapes=True))

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid),
                    callbacks=callbacks)
test_rmsle(model,X_test,y_test,name,ylog_model=USE_YLOG)

#%%
rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-8, max_rate=10)
plot_lr_vs_loss(rates, losses)

#%% ELU
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
model.compile(loss=LOSS, optimizer='sgd')

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=10)
plot_lr_vs_loss(rates, losses)


#%%
name='elu_sgd'
keras.backend.clear_session()
cb = callbacks_fn(name)
lr=(10**-2)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer=keras.optimizers.SGD(lr=lr)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=cb)
test_rmsle(model,X_test,y_test,name,ylog_model=USE_YLOG)
plot_learning_curves_bikes(history,name)

#%% SELU
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
model.compile(loss=LOSS, optimizer='sgd')

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=10)
plot_lr_vs_loss(rates, losses)

#%%
name='selu_sgd'
keras.backend.clear_session()
cb = callbacks_fn(name)
lr=(10**-1)*1
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer=keras.optimizers.SGD(lr=lr)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=cb)
test_rmsle(model,X_test,y_test,name,ylog_model=USE_YLOG)
plot_learning_curves_bikes(history,name)

#%% LeakyReLU
keras.backend.clear_session()
model = keras.models.Sequential([
    keras.layers.Dense(N_NEURONS, input_shape=INPUT_SHAPE),
    keras.layers.Dense(N_NEURONS, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(N_NEURONS, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(1, activation="linear")
])
model.compile(loss=LOSS, optimizer='sgd')

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=10)
plot_lr_vs_loss(rates, losses)

#%%
name='leaky_relu'
lr=(10**-1)*1
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
keras.backend.clear_session()
model = keras.models.Sequential([
    keras.layers.Dense(N_NEURONS, input_shape=INPUT_SHAPE),
    keras.layers.Dense(N_NEURONS, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(N_NEURONS, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(N_NEURONS, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(1, activation="linear")
])
optimizer=keras.optimizers.SGD(lr=lr)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=cb)
test_rmsle(model,X_test,y_test,name,ylog_model=USE_YLOG)
plot_learning_curves_bikes(history,name)

