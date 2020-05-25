import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from util import print2
from util import ascendingCorrelation, plot_pca_with_hue
from scipy import stats
import statsmodels.api as sm
from scipyDist import fit_scipy_distributions, plot_distributions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats._continuous_distns import _distn_names


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

# Delete data that will not be available in test data
for target_related_name in target_related_names:
    if target_related_name in bikes.columns:
        bikes.drop(target_related_name, axis=1, inplace=True)
        

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
ax = sns.pairplot(bikes[corr_var_names[:4]])
plt.show()

#%% SCALED DATA
sc=StandardScaler() 
bikes_float = bikes.drop(dt_col, axis=1)
sc.fit(bikes_float.to_numpy())
bikes_scaled =sc.transform(bikes_float.to_numpy())
    
#%% PRINCIAL COMPONENT ANALYSIS
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(bikes_scaled)
col_names =[]
for i in range(n_components):
    col_names.append("pca_"+str(i+1))
bikes_pca = pd.DataFrame(data = pca.transform(bikes_scaled), columns=col_names)

plot_pca_with_hue(bikes_pca,hue=bikes[target_name])
plot_pca_with_hue(bikes_pca,hue=bikes['hour'], rot=.4)
plot_pca_with_hue(bikes_pca,hue=bikes['temp'], rot=-.4)

# Taking too much time
# ax = sns.pairplot(bikes_pca)
# plt.show()

#%% DISTRIBUTION ANALYSIS
short_list_dist_names = ['alpha', 'beta','cauchy', 'cosine', 'laplace', 'levy','levy_l','norm','burr','chi']
medium_list_dist_names = ['levy','levy_l','norm', 'laplace', 'ksone', 'kstwobign', 'alpha', 'anglit','beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull']
long_list_dist_names = _distn_names
# Change assignment as wished
dist_names = short_list_dist_names 

number_distributions_to_plot = 3
   
# Calculate Distributions    
target_dist = fit_scipy_distributions(bikes[target_name], 100, dist_names)
plot_distributions(bikes[target_name],target_dist[:number_distributions_to_plot],100)

# Distribution analysis for pca components
for pca_component in range(n_components):
    pca_dist = fit_scipy_distributions(bikes_pca['pca_'+str(pca_component+1)], 100, dist_names)
    plot_distributions(bikes_pca['pca_'+str(pca_component+1)],pca_dist[:number_distributions_to_plot],100)

#%% Distribution PCA 3 with 2 distributions
# Divide data for distribution with >1 frequency
split_threshold = 1
pca_3a = bikes_pca['pca_3'][bikes_pca['pca_3']<split_threshold]
pca_3b = bikes_pca['pca_3'][bikes_pca['pca_3']>=split_threshold]

pca_3a_dist = fit_scipy_distributions(pca_3a, 100, dist_names)
pca_3b_dist = fit_scipy_distributions(pca_3b, 100, dist_names)
both_dist_pca3 =pd.concat([pca_3a_dist.iloc[:1],pca_3b_dist.iloc[:1]]) 

plot_distributions(bikes_pca['pca_3'],both_dist_pca3,100)

