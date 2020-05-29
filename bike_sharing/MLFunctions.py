import numpy as np
from util import display_scores
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

  
def trainAndPrint(model, X, y, name):
    model.fit(X, y)
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    print("")
    print(name," MSE: ","{:.2f}".format(rmse))
    print(name, " RMSE: ", "{:.2f}".format(model.score(X,y)))
    return model

def testAndPrint(model, X, y, name):
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    print("")
    print(name," MSE: ","{:.2f}".format(rmse))
    print(name, " RMSE: ", "{:.2f}".format(model.score(X,y)))
    return y_predictions

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

def test_model(model, X, y):
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    return y_predictions, rmse


def rmse_from_predictions(model, X, y, name=""):
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    print("")
    print(name,"RMSE : ","{:.2f}".format(rmse))
    print(name, "SCORE: ", "{:.2f}".format(model.score(X,y)))
    return rmse, y_predictions


