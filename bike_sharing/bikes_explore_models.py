import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import joblib

from scipy.stats import uniform, reciprocal, probplot
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

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
tf.random.set_seed(42)
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

def test_mse(model, X_test, y_test):
    y_pred = model.predict(X_test)
    sub = y_pred-y_test
    perc=abs(sub)/y_test*100
    print("\n {} MSE: {:.4f} ~= {:.1f}%".format(estimator_name(model),mean_squared_error(y_pred, y_test),perc.mean()))
    return mean_squared_error(y_pred, y_test)

def feature_importances(model):
    print(estimator_name(model))
    feat_imp = pd.DataFrame(data=random_forest_best.feature_importances_,
                                   index=X_train.columns, columns=['importance'])
    feat_imp.sort_values(by='importance', inplace=True,ascending=False)
    print(str(feat_imp)+'\n')
    return feat_imp
#%%
X_train, X_test, y_train, y_test, X_valid, y_valid = joblib.load("dataset/XY.pkl")
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
print('\n MODELS SCORES \n' +str(train_results))

#%%
cross_results = cross_validation(estimators,cv=5)
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
joblib.dump(random_forest_best, "models/random_forest_best.pkl")

#%% DecisionTreeRegressor
decision_tree_best = DecisionTreeRegressor(max_depth=15, max_features=16)
decision_tree_best.fit(X_train,y_train)
joblib.dump(decision_tree_best, "models/decision_tree_best.pkl")

#%% ExtraTreesRegressor
extra_forest_best = ExtraTreesRegressor(max_features=17, max_depth=25, n_estimators=350)
extra_forest_best.fit(X_train,y_train)
joblib.dump(extra_forest_best, "models/extra_forest_best.pkl")

#%% FEATURES
feature_importances_rfr = feature_importances(random_forest_best)
feature_importances_dtr = feature_importances(decision_tree_best)
feature_importances_etr = feature_importances(extra_forest_best)

#%% TEST
test_mse(random_forest_best, X_test, y_test)
test_mse(decision_tree_best, X_test, y_test)
test_mse(extra_forest_best, X_test, y_test)

#%% ENSEMBLE
Voter = VotingRegressor([('rfr', random_forest_best), 
                         ('etr',extra_forest_best)])

Voter.fit(X_train,y_train)
test_mse(Voter,X_test, y_test)
joblib.dump(extra_forest_best, "models/forest_voter.pkl")


