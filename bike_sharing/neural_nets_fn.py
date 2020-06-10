import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import TensorBoard

from util import test_mse

np.random.seed(42)
tf.random.set_seed(42)
K = keras.backend
root_logdir = os.path.join(os.curdir, "my_logs")

#%%
def get_run_logdir(name=''):
    time_stamp = time.strftime("%m_%d-%H_%M_%S")
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
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
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

def callbacks_fn(name,patience=15, save_folder=''):
    file_path =  save_folder+'/'+name+".h5"
    run_logdir          = get_run_logdir(name)
    checkpoint_save_cb  = keras.callbacks.ModelCheckpoint(file_path, save_best_only=True)
    early_stop_cb       = keras.callbacks.EarlyStopping(patience=patience)
    tensorboard_cb      = TensorBoard(run_logdir)
    return [early_stop_cb, tensorboard_cb, checkpoint_save_cb]

def run_network(name, data, optimizer, input_shape, dataset_type, activation='elu', loss='mean_squared_error', n_neurons=4,
                n_layers=1,epochs=10, save_folder='', with_cb=True, patience=15, batch_norm=False, alpha_drop=False):
    X_train, y_train, X_test, y_test, X_valid, y_valid = data
    name += '_' +str(n_layers) + '_' + str(n_neurons)
    cb = callbacks_fn(name, patience) if with_cb else []
    keras.backend.clear_session()
    model = build_model(n_layers,n_neurons,input_shape=input_shape, 
                        activation=activation, batch_norm=batch_norm, alpha_drop=alpha_drop)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=cb)
    test_mse(model,X_test,y_test,name)
    save_tfmodel_in_dir(model, name, dataset_type, save_folder)
    return model
    
def save_tfmodel_in_dir(model,name,dataset_type,folder):
    file_name = dataset_type.value + '_' + name + ".h5"
    if folder is not '':
        if os.path.isdir(folder):
            model.save(os.path.join(folder,file_name))
        else:
            print('folder not found')