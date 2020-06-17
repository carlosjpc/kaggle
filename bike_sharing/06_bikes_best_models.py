import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
from util import DatasetType, load_dataset, describe_Xy_data, test_rmsle, rmsle, test_models
from functools import partial

from sklearn_fn import feature_importances, random_search, train_estimators, save_skmodel_in_dir
from sklearn_fn import grid_search, cross_validation, create_results_train,estimator_name

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor

import tensorflow as tf
from tensorflow import keras

from neural_nets_fn import find_learning_rate, callbacks_fn, plot_lr_vs_loss
from neural_nets_fn import build_model, run_network, plot_learning_curves

np.random.seed(42)
tf.random.set_seed(42)

#%%
DATA_FOLDER = ['dataset', 'prepared_data_and_models', 'final']
DS_TYPE = DatasetType.SCALED 

all_data = load_dataset(DS_TYPE, folder=DATA_FOLDER)
X_train, y_train, X_test, y_test = all_data
describe_Xy_data(X_train, y_train, X_test, y_test)#%% log data is better
ylog_train = np.log1p(y_train)
ylog_test = np.log1p(y_test)

#%% REGRESSORS
MODELS_FOLDER = ['SKmodels','final']
save_skmodel = partial(save_skmodel_in_dir,dataset_type=DS_TYPE,folder=MODELS_FOLDER)

#%% RandomForestRegressor
random_forest_best = RandomForestRegressor(max_features=14, n_estimators=300)
random_forest_best.fit(X_train,ylog_train)
save_skmodel(random_forest_best, estimator_name(random_forest_best))

#%% ExtraTreesRegressor
extra_forest_best = ExtraTreesRegressor(max_features=17, max_depth=25, n_estimators=350)
extra_forest_best.fit(X_train,ylog_train)
save_skmodel(extra_forest_best, estimator_name(extra_forest_best))

#%% ENSEMBLE
extra_forest_best = ExtraTreesRegressor(max_features=17, max_depth=25, n_estimators=350)
random_forest_best = RandomForestRegressor(max_features=14, n_estimators=300)
Voter = VotingRegressor([('rfr', random_forest_best), 
                         ('etr',extra_forest_best)])
Voter.fit(X_train,ylog_train)
save_skmodel(Voter, estimator_name(Voter))

#%% NEURAL NETWORKS
LOSS          = 'mean_squared_logarithmic_error'
N_NEURONS     = 100
N_LAYERS      = 7
PATIENCE      = 15
EPOCHS        = 500
MODELS_FOLDER = ["TFmodels","final"]
INPUT_SHAPE = X_train.shape[1:]
maxLossPlot = 1.2
nEpochsPlot = 100
      
run_bike_network= partial(run_network, loss=LOSS, dataset_type=DS_TYPE, n_neurons=N_NEURONS, n_layers=N_LAYERS,
                          epochs=EPOCHS, input_shape=INPUT_SHAPE, save_folder=MODELS_FOLDER, 
                          with_cb=True, patience=PATIENCE, maxLossPlot=maxLossPlot,nEpochsPlot=nEpochsPlot)
#%% elu_sgdm
name = 'elu_sgdm'
optimizer = keras.optimizers.SGD(lr=(10**-2)*1, momentum=0.9)
run_bike_network(name, all_data, optimizer, activation='elu')

#%% selu_sgdm
name='selu_sgdm'
optimizer = keras.optimizers.SGD(lr=(10**-2)*1, momentum=0.9)
run_bike_network(name, all_data, optimizer, activation='selu')

#%%
name='elu_adam'
optimizer = keras.optimizers.Adam(lr=(10**-4)*1, beta_1=0.9, beta_2=0.999)
run_bike_network(name, all_data, optimizer, activation='elu')

#%% TESTS
# Best Models
skmodels_ylog = test_models(models_folder=['SKmodels','final'],data_folder=DATA_FOLDER, ylog_model=True)
tfmodels_ymsle = test_models(models_folder=['TFmodels','final'],data_folder=DATA_FOLDER)

result_best = pd.concat([skmodels_ylog,tfmodels_ymsle])
result_best.sort_values(by=['rmsle'],inplace=True)
print(result_best)

            