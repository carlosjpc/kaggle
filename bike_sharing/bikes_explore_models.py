#%% IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

#%%
def estimator_name(estimator):
    return estimator.__class__.__name__

def train_estimators(estimators):
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

def cross_validation(estimators, cv=5):
    result = []
    for estimator in estimators:
        scores = cross_val_score(estimator, X_train, y_train, cv=cv) #scoring="neg_mean_squared_error"
        rmse_scores = (scores).mean() #np.sqrt
        result.append(rmse_scores)
    df = pd.DataFrame(data=result, index=list(map(estimator_name,estimators)), columns=['mean_error'])
    return df.sort_values(by=['mean_error'], axis=0, ascending=False)

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

#%%
X_train, X_test, y_train, y_test = joblib.load("dataset/XY.pkl")

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
estimators_trained, train_results = train_estimators(estimators)
print('MODELS SCORES \n' +str(train_results))

#%%
cross_results = cross_validation(estimators,cv=5)
print('CROSS VALIDATION SCORES \n' +str(cross_results))

#%%
param_grid = [
    {'n_estimators': [300, 500], 'max_features': ['auto',15, 17]},
  ]

rfg_gs, rfg_gs_res = grid_search(RandomForestRegressor(),param_grid,X_train,y_train, verbose=1)
print(rfg_gs_res)

#%%
random_forest_best = RandomForestRegressor(max_features=15, n_estimators=300)
random_forest_best.fit(X_train,y_train)

#%%
print("{} score: {:.2f}".format(estimator_name(random_forest_best),random_forest_best.score(X_test, y_test)))
# er = VotingRegressor([('lr', r1), ('rf', r2)])