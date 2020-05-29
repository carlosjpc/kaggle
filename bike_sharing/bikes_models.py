#%% IMPORTS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

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


#%% TRAIN MODELS
random_forest = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
extra_trees_clf = ExtraTreesRegressor(n_estimators=100, random_state=42, verbose=2)
svm_clf = LinearSVR(random_state=42, verbose=2)
mlp_clf = MLPClassifier(random_state=42, verbose=2)


#%%
estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

for estimator in estimators:
    print(estimator.score(X_val, y_val))
# print([estimator.score(X_val, y_val) for estimator in estimators])

r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=10, random_state=1)
X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
y = np.array([2, 6, 12, 20, 30, 42])
er = VotingRegressor([('lr', r1), ('rf', r2)])