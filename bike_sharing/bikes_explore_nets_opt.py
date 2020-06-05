import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib

from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from functools import partial
from IPython.display import display
from keras.callbacks import TensorBoard

np.random.seed(42)
tf.random.set_seed(42)
K = keras.backend
root_logdir = os.path.join(os.curdir, "my_logs")

#%%
def reshape_array(var):
    if type(var) is not np.ndarray:
        var = var.to_numpy()    
    return var.reshape(-1,1)

def test_mse(model, X_test, y_test, name='Model'):
    y_pred = reshape_array(model.predict(X_test))
    y_test = reshape_array(y_test)
    sub = y_pred-y_test
    perc=abs(sub)/y_test*100
    print("\n{} MSE: {:.0f} ~= {:.1f}%".format(name,mean_squared_error(y_pred, y_test),perc.mean()))
    return mean_squared_error(y_pred, y_test)

def get_run_logdir(name=''):
    import time
    time_stamp = time.strftime("run_%m_%d-%H_%M_%S")
    run_id = name + '_' + time_stamp
    return os.path.join(root_logdir, run_id)

def build_model(n_hidden=1, n_neurons=30, input_shape=[10],
                activation='relu', initializer="he_normal", output_shape=1,
                output_activation='linear', batch_norm=False, alpha_drop=False):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_neurons, input_shape=input_shape))
    for layer in range(n_hidden):
        model = add_batch_drop_layer(model, batch_norm, alpha_drop)
        model.add(keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=initializer))
    model = add_batch_drop_layer(model, batch_norm, alpha_drop)
    model.add(keras.layers.Dense(output_shape, activation=output_activation))
    return model

def add_batch_drop_layer(model, batch_norm=False, alpha_drop=False):
    if batch_norm:
        model.add(keras.layers.BatchNormalization())
    if alpha_drop:
        model.add(keras.layers.AlphaDropout(rate=0.2))
    return model

def describe_Xy_data(X_train,y_train,X_test,y_test,X_valid=None,y_valid=None):
    print('DATA DESCRIPTION')
    print('X_train: ' + str(X_train.shape))
    print('y_train: ' + str(y_train.shape))
    print('y range: ' + str(min(y_train)) +' - ' + str(max(y_train)))
    print('X_test : ' + str(X_test.shape))
    print('y_test : ' + str(y_test.shape))
    if X_valid.all() is not None:
        print('X_valid: ' + str(X_valid.shape))
    if y_valid.all() is not None:
        print('y_valid: ' + str(y_valid.shape))

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
        
class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
        
def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = len(X) // batch_size * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")

def callbacks_fn(name,patience=15):
    model_file_name = 'my_model_'+name
    run_logdir          = get_run_logdir(name)
    checkpoint_save_cb  = keras.callbacks.ModelCheckpoint(model_file_name, save_best_only=True)
    early_stop_cb       = keras.callbacks.EarlyStopping(patience=patience)
    tensorboard_cb      = TensorBoard(run_logdir)
    callbacks=[checkpoint_save_cb,early_stop_cb,tensorboard_cb]
    return [tensorboard_cb, early_stop_cb]

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
X_train, X_test, y_train, y_test, X_valid, y_valid = joblib.load("dataset/XY.pkl")
describe_Xy_data(X_train,y_train,X_test,y_test,X_valid,y_valid)

#%%
LOSS = 'mean_squared_error'
INPUT_SHAPE = X_train.shape[1:]
N_NEURONS = 30
N_LAYERS = 3

#%% ELU
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=1e-3)
plot_lr_vs_loss(rates, losses)

#%%
name='elu_sgdm'
keras.backend.clear_session()
cb = callbacks_fn(name)
lr=(10**-6)*1
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=cb)
test_mse(model,X_test,y_test,name)

#%% SELU
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss=LOSS, optimizer=optimizer)

rates, losses = find_learning_rate(model, X_train, y_train, epochs=5, min_rate=10**-7, max_rate=1e-3)
plot_lr_vs_loss(rates, losses)

#%% SELU with momentum = nan

#%%
name='elu_rmsprop'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_mse(model,X_test,y_test,name)

#%%
name='selu_rmsprop'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_mse(model,X_test,y_test,name)

#%%
name='elu_adam'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='elu')
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_mse(model,X_test,y_test,name)

#%%
name='selu_adam'
keras.backend.clear_session()
cb = callbacks_fn(name)
keras.backend.clear_session()
model = build_model(N_LAYERS,N_NEURONS,input_shape=INPUT_SHAPE, activation='selu')
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss=LOSS, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),callbacks=cb)
test_mse(model,X_test,y_test,name)
