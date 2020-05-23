import os
import pandas as pd
import numpy as np
import seaborn as sns
from util import print2, compareCategoryProportionsSamples
from util import getCorrVector,display_scores
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from MLFunctions import CombinedAttributesAdder, EncoderAndDeleteCol, trainAndPrint, cross_val_scoresAndPrint, DeleteAttributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib.dates import DateFormatter
import joblib

plt.style.use('seaborn')
DISPLAY_WIDTH = 400

# Load Available Data
bikes = pd.read_csv('dataset/train.csv', parse_dates = ['datetime'])
dt_col = "datetime"

# Disect Datetime
# bikes["hour"]      = bikes[dt_col].dt.hour
# bikes['dayOfWeek'] = bikes[dt_col].dt.dayofweek
# bikes['month']     = bikes[dt_col].dt.month
# bikes['year']      = bikes[dt_col].dt.year

target_name = "totalRides"
target_related_names = ["registered","casual"] 
bikes = bikes.rename(columns={"count": target_name})

# Delete data that will not be available in test data
for target_related_name in target_related_names:
    if target_related_name in bikes.columns:
        bikes.drop(target_related_name, axis=1, inplace=True)
        

print2(bikes.head(),10)
print(bikes.info())
print2(bikes.describe())

corrTarget, mostCorrelatedVarNames = getCorrVector(bikes,target_name)
print(corrTarget)

# Stratified Shuffle for test and train data according to temp
bikes["temp_cat"] = pd.cut(bikes["temp"],
                                bins=[0., 10, 20, 30, np.inf],
                                labels=[0, 10, 20, 30])  

print(bikes["temp_cat"].value_counts())
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(bikes, bikes["temp_cat"]):
    strat_train_set = bikes.loc[train_index]
    strat_test_set = bikes.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("temp_cat", axis=1, inplace=True)
    
# Plot total Rides vs time
ax1 = plt.subplots()
ax1 = sns.lineplot(x = dt_col,  y = target_name, data = bikes, label=target_name, alpha = 0.7)
ax1.legend()
ax1.set_xlabel('Date')
ax1.set_ylabel('no.')
date_form = DateFormatter("%m-%Y")
ax1.xaxis.set_major_formatter(date_form)
for item in ax1.get_xticklabels():
    item.set_rotation(45)
plt.show()

# Deal only with train data
bikesOriginal = bikes.copy()
bikes = strat_train_set.drop(target_name, axis=1)
bikes_labels = strat_train_set[target_name].copy()

bikes_numeric = bikes.drop(["season","weather","workingday"], axis=1)
num_attribs = list(bikes_numeric)
cat_attribs = ["season","weather","workingday"]

imputer = SimpleImputer(strategy="median")

imputer.fit(bikes_numeric)
bikes_numeric_filled = imputer.transform(bikes_numeric)


def full_pipeline(df):
    return df_tr

# num_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy="median")),
#         ('attribs_adder', CombinedAttributesAdder()),
#         ('std_scaler', StandardScaler()),
#     ])

# full_pipeline = ColumnTransformer([
#         ("num", num_pipeline, num_attribs),
#         ("cat", OneHotEncoder() , cat_attribs), #OneHotEncoder()
#     ])

# housing_prepared = full_pipeline.fit_transform(housing)
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])

# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()
# X_test_prepared = full_pipeline.transform(X_test)

# # # Train Model
# # lin_reg = LinearRegression()
# # lin_reg = trainAndPrint(lin_reg, housing_prepared, housing_labels, "LinReg")
# # lin_reg_scores = cross_val_scoresAndPrint(lin_reg, housing_prepared, housing_labels)
# # joblib.dump(lin_reg, "lin_reg.pkl")

# # tree_reg = DecisionTreeRegressor()
# # tree_reg = trainAndPrint(lin_reg, housing_prepared, housing_labels, "TreeReg")
# # tree_reg_scores = cross_val_scoresAndPrint(tree_reg, housing_prepared, housing_labels)
# # joblib.dump(tree_reg, "tree_reg.pkl")

# # forest_reg = RandomForestRegressor()
# # forest_reg = trainAndPrint(forest_reg, housing_prepared, housing_labels, "ForestReg")
# # forest_reg_scores = cross_val_scoresAndPrint(forest_reg, housing_prepared, housing_labels)
# # joblib.dump(forest_reg, "forest_reg.pkl")

# # save files
# joblib.dump(full_pipeline, "full_pipeline.pkl")
# joblib.dump(housing_labels, "housing_labels.pkl")
# joblib.dump(housing_prepared, "housing_prepared.pkl")
# joblib.dump(num_attribs, "num_attribs.pkl")
# joblib.dump(cat_one_hot_attribs, "cat_one_hot_attribs.pkl")
# joblib.dump(X_test_prepared,"X_test_prepared.pkl")
# joblib.dump(y_test,"y_test.pkl")
# joblib.dump(strat_test_set,"strat_test_set.pkl")