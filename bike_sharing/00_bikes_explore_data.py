#%% IMPORT
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from util import print2, pca_col_names
from util import ascendingCorrelation, plot_pca_with_hue
from scipy import stats
import statsmodels.api as sm
from scipy_dist_fn import fit_scipy_distributions, plot_distributions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats._continuous_distns import _distn_names
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import time

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
bikes['time']      = pd.Series(map(pd.Timestamp.timestamp,bikes['datetime']))

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

#%%
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=bikes,y=target_name,orient="v",ax=axes[0][0])
sns.boxplot(data=bikes,y=target_name,x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=bikes,y=target_name,x="hour",orient="v",ax=axes[1][0])
sns.boxplot(data=bikes,y=target_name,x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")

#%% REMOVE OUTLIERS
bikesWithoutOutliers = bikes[np.abs(bikes[target_name]-bikes[target_name].mean())<=(3*bikes[target_name].std())]
print ("Shape Of The Before Ouliers: ",bikes.shape)
print ("Shape Of The After Ouliers: ",bikesWithoutOutliers.shape)


#%%
fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(12,20)
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

monthAggregated = pd.DataFrame(bikes.groupby("month")[target_name].mean()).reset_index()
monthSorted = monthAggregated.sort_values(by=target_name,ascending=False)
sns.barplot(data=monthSorted,x="month",y="totalRides",ax=ax1)
ax1.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")

hourAggregated = pd.DataFrame(bikes.groupby(["hour","season"],sort=True)[target_name].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated[target_name],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')

hourAggregated = pd.DataFrame(bikes.groupby(["hour","dayOfWeek"],sort=True)[target_name].mean()).reset_index()
sns.pointplot(x=hourAggregated["hour"], y=hourAggregated[target_name],hue=hourAggregated["dayOfWeek"], data=hourAggregated, join=True,ax=ax3)
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')


#%% SCALED DATA
sc=StandardScaler() 
bikes_float = bikes.drop([dt_col, target_name], axis=1)
sc.fit(bikes_float.to_numpy())
bikes_scaled =sc.transform(bikes_float.to_numpy())
    
#%% PRINCIAL COMPONENT ANALYSIS
n_components = 3
pca = PCA(n_components=n_components)
pca.fit(bikes_scaled)
bikes_pca = pd.DataFrame(data=pca.transform(bikes_scaled), 
                         columns=pca_col_names(n_components))

plot_pca_with_hue(bikes_pca,hue=bikes[target_name])
plot_pca_with_hue(bikes_pca,hue=bikes['hour'], rot=.4)
plot_pca_with_hue(bikes_pca,hue=bikes['temp'], rot=-.4)

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

# Calculate Distributions    
target_dist = fit_scipy_distributions(bikesWithoutOutliers[target_name], 100, dist_names)
plot_distributions(np.log(bikesWithoutOutliers[target_name]),target_dist[:number_distributions_to_plot],100)

# Distribution analysis for pca components
for pca_component in range(n_components):
    pca_dist = fit_scipy_distributions(bikes_pca['pca_'+str(pca_component+1)], 100, dist_names)
    plot_distributions(bikes_pca['pca_'+str(pca_component+1)],pca_dist[:number_distributions_to_plot],100)

#%% Distribution PCA 3 with 2 distributions
# Divide data for distribution with >1 frequency
split_threshold = 0.9
pca_a = bikes_pca['pca_1'][bikes_pca['pca_1']<split_threshold]
pca_b = bikes_pca['pca_1'][bikes_pca['pca_1']>=split_threshold]

pca_a_dist = fit_scipy_distributions(pca_a, 100, dist_names)
pca_b_dist = fit_scipy_distributions(pca_b, 100, dist_names)
both_dist_pca =pd.concat([pca_a_dist.iloc[:1],pca_b_dist.iloc[:1]]) 

plot_distributions(bikes_pca['pca_1'],both_dist_pca,100)



# %% Random Forest Model To Predict 0's In Windspeed
from sklearn.ensemble import RandomForestRegressor

dataWind0 = bikes[bikes["windspeed"]==0]
dataWindNot0 = bikes[bikes["windspeed"]!=0]
rfModel_wind = RandomForestRegressor()
windColumns = ["season","weather","humidity","month","temp","year","atemp"]
rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])

wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])
dataWind0["windspeed"] = wind0Values
data = dataWindNot0.append(dataWind0)
data.sort_index(inplace=True)

