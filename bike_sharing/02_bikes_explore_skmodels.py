import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
from util import DatasetType, load_dataset, describe_Xy_data, test_mse
from functools import partial


from sklearn_fn import feature_importances, random_search, train_estimators, save_skmodel_in_dir
from sklearn_fn import grid_search, cross_validation, create_results_train,estimator_name

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(42)

#%%
MODELS_FOLDER = "SKmodels"
DS_TYPE = DatasetType.PCA 
save_skmodel = partial(save_skmodel_in_dir,dataset_type=DS_TYPE,folder=MODELS_FOLDER)

#%%
X_train, y_train, X_test, y_test, X_valid, y_valid = load_dataset(DS_TYPE, folder='dataset')
describe_Xy_data(X_train, y_train, X_test, y_test, X_valid, y_valid)

#%% PLAIN MODELS
verbose = 0
random_state = 42
n_estimators = 100

decision_tree = DecisionTreeRegressor(random_state=random_state)
random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
extra_trees = ExtraTreesRegressor(n_estimators=n_estimators, random_state=random_state, verbose=verbose)
lin_svr = LinearSVR(random_state=random_state, verbose=verbose)
svr = SVR(verbose=verbose)
sgd = SGDRegressor(random_state=random_state, verbose=verbose)
lin_reg = LinearRegression()
gbr = GradientBoostingRegressor()

#%%
estimators = [decision_tree, random_forest, extra_trees, svr, sgd, lin_reg, gbr]
estimators_trained, train_results = train_estimators(estimators,X_train, y_train)
print('\n MODELS SCORES \n' +str(train_results))

#%%
cross_results = cross_validation(estimators, X_train, y_train,cv=5)
print('\n CROSS VALIDATION SCORES \n' +str(cross_results))

#%% RANDOM SEARCH 
param_grid = {
    "n_estimators": np.arange(100,500),
    'max_features': ['auto', 10,12,14,16,17],
    "min_samples_split": np.arange(2,10),
    "max_depth": [None,15,20,25,30,35,40,50],
}
rfg_rs, rfg_rs_res = random_search(RandomForestRegressor(),param_grid,X_train,y_train, cv=2, n_iter=100, verbose=2)
print(rfg_rs_res)

#%%
param_grid = {
    "n_estimators": np.arange(100,500),
    'max_features': ['auto', 10,12,14,16,17],
    "min_samples_split": np.arange(2,10),
    "max_depth": [None,15,20,25,30,35,40,50],
}
etr_rs, etr_rs_res = random_search(ExtraTreesRegressor(),param_grid,X_train,y_train, cv=2, n_iter=100, verbose=2)
print(etr_rs_res)

#%%
param_grid = {
    'max_features': ['auto', 10,12,14,16,17],
    "min_samples_split": np.arange(2,10),
    "max_depth": [None,15,20,25,30,35,40,50],
}
dtr_rs, dtr_rs_res = random_search(DecisionTreeRegressor(),param_grid,X_train,y_train, cv=2, n_iter=100, verbose=2)
print(dtr_rs_res)

#%% GRID SEARCH
param_grid = [
    {'n_estimators': [300, 350, 400], 'max_features': [14, 15, 16, 17], "max_depth": [25,30,35]},
  ]
rfg_gs, rfg_gs_res = grid_search(RandomForestRegressor(),param_grid,X_train,y_train, verbose=1)
print(rfg_gs_res)

#%% 
param_grid = [
    {'max_features': ['auto',14,15,16,17], "max_depth": [None,15,25,30,35,40,50]},
  ]
drt_gs, drt_gs_res = grid_search(DecisionTreeRegressor(),param_grid,X_train,y_train, verbose=1)
print(drt_gs_res)

#%%
param_grid = [
    {'n_estimators': [300, 350, 400], 'max_features': ['auto', 17], "max_depth": [25,30,35, 40]},
  ]
etr_gs, etr_gs_res = grid_search(ExtraTreesRegressor(),param_grid,X_train,y_train, verbose=1)
print(etr_gs_res)

#%% BEST MODELS
#%% RandomForestRegressor
random_forest_best = RandomForestRegressor(max_features=15, n_estimators=300)
random_forest_best.fit(X_train,y_train)
save_skmodel(random_forest_best, estimator_name(random_forest_best))

#%% DecisionTreeRegressor
decision_tree_best = DecisionTreeRegressor(max_depth=15, max_features=16)
decision_tree_best.fit(X_train,y_train)
save_skmodel(decision_tree_best, estimator_name(decision_tree_best))

#%% ExtraTreesRegressor
extra_forest_best = ExtraTreesRegressor(max_features=17, max_depth=25, n_estimators=350)
extra_forest_best.fit(X_train,y_train)
save_skmodel(extra_forest_best, estimator_name(extra_forest_best))

#%% FEATURES
features = X_train.columns
feature_importances_rfr = feature_importances(random_forest_best, features)
feature_importances_dtr = feature_importances(decision_tree_best, features)
feature_importances_etr = feature_importances(extra_forest_best, features)

#%% TEST
test_mse(random_forest_best, X_test, y_test, estimator_name(random_forest_best))
test_mse(decision_tree_best, X_test, y_test, estimator_name(decision_tree_best))
test_mse(extra_forest_best, X_test, y_test, estimator_name(extra_forest_best))

#%% ENSEMBLE
extra_forest_best = ExtraTreesRegressor(max_features=17, max_depth=25, n_estimators=350)
random_forest_best = RandomForestRegressor(max_features=15, n_estimators=300)

Voter = VotingRegressor([('rfr', random_forest_best), 
                         ('etr',extra_forest_best)])

Voter.fit(X_train,y_train)
test_mse(Voter,X_test, y_test)

save_skmodel(Voter, estimator_name(Voter))





