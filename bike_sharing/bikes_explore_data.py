import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from util import print2
from util import ascendingCorrelation
from scipy import stats
import statsmodels.api as sm
from AnalyzeDistributions import evaluate_distributions, evaluate_and_plot_dist
from scipyDist import fit_scipy_distributions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


plt.style.use('seaborn')
DISPLAY_WIDTH = 400

#%% LOAD DATA
bikes = pd.read_csv('dataset/train.csv', parse_dates = ['datetime'])
dt_col = "datetime"

# Additional Columns
bikes["hour"]      = bikes[dt_col].dt.hour
bikes['dayOfWeek'] = bikes[dt_col].dt.dayofweek
bikes['month']     = bikes[dt_col].dt.month
bikes['year']      = bikes[dt_col].dt.year

# Names and renaming
target_name = "totalRides"
target_related_names = ["registered","casual"] 
bikes = bikes.rename(columns={"count": target_name})

# General Info
print2(bikes.head(),10,10)
print(bikes.info())
print2(bikes.describe().T,10)


#%% CORRELATIONS
corr_matrix, corr_target, corr_var_names = ascendingCorrelation(bikes,target_name)
print("TARGET CORRELATIONS \n" + str(corr_target))
mask = np.triu(corr_matrix)
ax = sns.heatmap(corr_matrix,cmap="YlGnBu", mask=mask)

#%% PLOTS

# Plot total Rides vs time
ax = plt.subplots()
ax = sns.lineplot(x = dt_col,  y = target_name, data = bikes, label=target_name, alpha = 0.7)
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('no.')
date_form = DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)
for item in ax.get_xticklabels():
    item.set_rotation(45)
plt.show()

# Pair plots
ax = sns.pairplot(bikes[corr_var_names[:6]])
plt.show()

# # Distribution analysis
# ax = plt.subplots()
# ax = sns.distplot(bikes[target_name]);
# plt.show()

# ax = plt.subplot()
# ax = sns.distplot(bikes['temp'])
# plt.show()

#%% SCALED DATA
sc=StandardScaler() 
bikes_float = bikes.drop(dt_col, axis=1)
sc.fit(bikes_float.to_numpy())
bikes_scaled =sc.transform(bikes_float.to_numpy())
# y_std = y_std.flatten()
# size = len(yy)
    
#%% PRINCIAL COMPONENT ANALYSIS
n_components = 2
pca = PCA(n_components=n_components)
pca.fit(bikes_scaled)
col_names =[]
for i in range(n_components):
    col_names.append("pca_"+str(i+1))
    
bikes_pca = pd.DataFrame(data = pca.transform(bikes_scaled), columns=col_names)

ax = sns.pairplot(bikes_pca)
plt.show()

#%% DISTRIBUTION ANALYSIS
short_list_dist_names = ['alpha', 'cauchy', 'cosine', 'laplace', 'levy','levy_l','norm']
medium_list_dist_names = ['levy','levy_l','norm', 'laplace', 'ksone', 'kstwobign', 'norm', 'alpha', 'anglit','beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull']

# Change assignment to determine distributions to be analized
dist_names = medium_list_dist_names 

# Get the top distributions from the previous phase
number_distributions_to_plot = 3
   
# Calculate Distributions
target_dist = fit_scipy_distributions(bikes[target_name], 100, dist_names)
best_temp_dist = target_dist['Distribution'].iloc[0:number_distributions_to_plot]
# Plot distributions
fit_scipy_distributions(bikes[target_name], 100, best_temp_dist,  plot_dist=True)

# Distribution analysis for pca 1 
temp_dist = fit_scipy_distributions(bikes_pca['pca_1'], 100, dist_names)
best_temp_dist = temp_dist['Distribution'].iloc[0:number_distributions_to_plot]
fit_scipy_distributions(bikes_pca['pca_1'], 100, best_temp_dist, plot_dist=True)

# Distribution analysis for pca 2
temp_dist = fit_scipy_distributions(bikes_pca['pca_2'], 100, dist_names)
best_temp_dist = temp_dist['Distribution'].iloc[0:number_distributions_to_plot]
fit_scipy_distributions(bikes_pca['pca_2'], 100, best_temp_dist,  plot_dist=True)

#%% D
# ax = plt.subplot()
# parameters = evaluate_and_plot_dist(bikes["temp"], dist_names, bins=30)
# plt.show()

# # Store distribution paraemters in a dataframe
# dist_parameters = pd.DataFrame()
# dist_parameters['Distribution'] = (
#         dist_results['Distribution'].iloc[0:number_distributions_to_plot])
# dist_parameters['Distribution parameters'] = parameters

# # Print parameter results
# print ('\nDistribution parameters:')
# print ('------------------------')

# for index, row in dist_parameters.iterrows():
#     print ('\nDistribution:', row[0])
#     print ('Parameters:', row[1] )

