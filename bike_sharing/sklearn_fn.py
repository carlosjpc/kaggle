import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
import os
from util import DatasetType, load_dataset

from scipy.stats import uniform, reciprocal, probplot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from util import describe_Xy_data, test_mse

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
def estimator_name(estimator):
    return estimator.__class__.__name__

def train_estimators(estimators,X_train, y_train):
    times = []
    scores = []
    print("TRAINING MODELS:")
    for estimator in estimators:
        t0 = time.time()
        estimator.fit(X_train, y_train)
        t1 = time.time()
        times.append(t1-t0)
        scores.append(estimator.score(X_train,y_train))
        # print("{}: {:.1f} seconds".format(estimator_name(estimator), t1 - t0),end='')
        print(estimator_name(estimator) + ', ',end='')
    print("")  
    result = create_results_train(estimators,times,scores)
    return estimators, result
   
def create_results_train(estimators,times,scores):
    result = pd.DataFrame()
    result['estimator'] = list(map(estimator_name,estimators))
    result['train_time'] = times
    result['scores'] = scores
    result.sort_values(by=['scores'], axis=0, ascending=False, inplace=True)
    return result

def cross_validation(estimators, X_train, y_train,cv=5):
    result = []
    for estimator in estimators:
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
        rmse_scores = (-scores).mean() #np.sqrt
        result.append(rmse_scores)
    df = pd.DataFrame(data=result, index=list(map(estimator_name,estimators)), columns=['mean_error'])
    return df.sort_values(by=['mean_error'], axis=0, ascending=True)

def grid_search(model, param_grid, X, y, cv=3, verbose=2):
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv,
                               scoring='neg_mean_squared_error',
                               return_train_score=True, verbose=verbose)
    grid_search.fit(X, y)
    gs_res = compile_results_search(grid_search)   
    return grid_search, gs_res

def random_search(model, param_distributions, X, y, n_iter=10, cv=3, verbose=2):
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=n_iter, cv=cv,
                               scoring='neg_mean_squared_error',
                               return_train_score=True, verbose=verbose)
    random_search.fit(X, y)
    random_res = compile_results_search(random_search)
    
    return random_search, random_res

def compile_results_search(search):
    cvres = search.cv_results_
    search_res = pd.DataFrame(zip(-cvres["mean_test_score"], cvres["params"]),
                 columns=['mean_test_score','params'])
    search_res.sort_values('mean_test_score', inplace=True, ascending=True)
    return search_res

def feature_importances(model,features):
    print(estimator_name(model))
    feat_imp = pd.DataFrame(data=model.feature_importances_,
                                   index=features, columns=['importance'])
    feat_imp.sort_values(by='importance', inplace=True,ascending=False)
    print(str(feat_imp)+'\n')
    return feat_imp
            
def save_skmodel_in_dir(model,name,dataset_type,folder='models'):
   if os.path.isdir(folder):
       file_name = dataset_type.value + '_' + name + ".pkl"
       joblib.dump(model, os.path.join(folder,file_name))
   else:
       print('folder not found')
    