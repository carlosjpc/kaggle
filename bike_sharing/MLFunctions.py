from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.metrics import mean_squared_error
from util import display_scores
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DeleteAttributes(BaseEstimator, TransformerMixin):
    def __init__(self, cols_ix = []): # no *args or **kargs
        self.cols_ix = cols_ix
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        print(type(X))
        for col_ix in self.cols_ix:
            print(col_ix)
        return X
    
class EncoderAndDeleteCol(BaseEstimator, TransformerMixin):
    def __init__(self, cols_ix = []):
        self.cols_ix = cols_ix
        self.cols_ix.sort(reverse=True)
        
    def transform(self, X):
        ohe = OneHotEncoder()
        ohe.fit_transform(X)
        self.categories_ = ohe.categories_
        self.categories_[0] = np.delete(self.categories_[0],self.cols_ix)
        return np.delete(ohe.fit_transform(X).toarray(),self.cols_ix,1)
        
    def fit(self, X, y=None, **fit_params):
        return self
  
def trainAndPrint(model, X, y, name):
    model.fit(X, y)
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    print("")
    print(name," MSE: ","{:.2f}".format(rmse))
    print(name, " RMSE: ", "{:.2f}".format(model.score(X,y)))
    return model

def cross_val_scoresAndPrint(model, X, y, scoring="neg_mean_squared_error", cv=10):
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    return scores

def testAndPrint(model, X, y, name):
    y_predictions = model.predict(X)
    mse = mean_squared_error(y, y_predictions)
    rmse = np.sqrt(mse)
    print("")
    print(name," MSE: ","{:.2f}".format(rmse))
    print(name, " RMSE: ", "{:.2f}".format(model.score(X,y)))
    return y_predictions


