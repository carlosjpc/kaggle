import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib
import os
from util import DatasetType, load_dataset, rmsle

from scipy.stats import uniform, reciprocal, probplot
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from util import describe_Xy_data, test_rmsle

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import make_scorer

np.random.seed(42)
rmsle_scorer = make_scorer(rmsle)

#%%
def estimator_name(estimator):
    return estimator.__class__.__name__

def train_estimators(estimators,X_train, y_train, X_valid, y_valid, ylog=False):
    times = []
    scores = []
    if ylog:
        y_train, y_valid = (np.log1p(y_train), np.log1p(y_valid))
    print("TRAINING MODELS:")
    for estimator in estimators:
        t0 = time.time()
        estimator.fit(X_train, y_train)
        t1 = time.time()
        times.append(t1-t0)
        y_pred = estimator.predict(X_valid)
        if ylog:
            scores.append(rmsle(np.exp(y_pred),np.exp(y_valid)))
        else:
            scores.append(rmsle(y_pred,y_valid))
        # print("{}: {:.1f} seconds".format(estimator_name(estimator), t1 - t0),end='')
        # print("{}: {:.3f} , {:.3f}".format(estimator_name(estimator), np.sqrt(mean_squared_log_error(y_pred,y_valid)), rmsle(y_pred,y_valid)))
        print(estimator_name(estimator) + ', ' , end='')
    print("")  
    result = create_results_train(estimators,times,scores)
    return estimators, result
   
def create_results_train(estimators,times,scores):
    result = pd.DataFrame()
    result['estimator'] = list(map(estimator_name,estimators))
    result['train_time'] = times
    result['scores'] = scores
    result.sort_values(by=['scores'], axis=0, ascending=True, inplace=True)
    return result

def cross_validation(estimators, X_train, y_train,cv=5):
    result = []
    for estimator in estimators:
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=rmsle)
        rmse_scores = (-scores).mean() #np.sqrt
        result.append(rmse_scores)
        print(estimator_name(estimator) + ', ' , end='')
    df = pd.DataFrame(data=result, index=list(map(estimator_name,estimators)), columns=['rmsle'])
    return df.sort_values(by=['rmsle'], axis=0, ascending=True)

def grid_search(model, param_grid, X, y, cv=3, verbose=2):
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv,
                               scoring=rmsle_scorer,
                               return_train_score=True, verbose=verbose)
    grid_search.fit(X, y)
    gs_res = compile_results_search(grid_search)   
    return grid_search, gs_res

def random_search(model, param_distributions, X, y, n_iter=10, cv=3, verbose=2):
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=n_iter, cv=cv,
                               scoring=rmsle_scorer,
                               return_train_score=True, verbose=verbose)
    random_search.fit(X, y)
    random_res = compile_results_search(random_search)
    
    return random_search, random_res

def compile_results_search(search):
    cvres = search.cv_results_
    search_res = pd.DataFrame(zip(cvres["mean_test_score"], cvres["params"]),
                 columns=['rmsle','params'])
    search_res.sort_values('rmsle', inplace=True, ascending=True)
    return search_res

def feature_importances(model,features):
    print(estimator_name(model))
    feat_imp = pd.DataFrame(data=model.feature_importances_,
                                   index=features, columns=['importance'])
    feat_imp.sort_values(by='importance', inplace=True,ascending=False)
    print(str(feat_imp)+'\n')
    return feat_imp
            
def save_skmodel_in_dir(model,name,dataset_type,folder='models'):
   if os.path.isdir(os.path.join(*folder)):
       file_name = dataset_type.value + '_' + name + ".pkl"
       joblib.dump(model, os.path.join(*folder,file_name))
   else:
       print('folder not found')
    