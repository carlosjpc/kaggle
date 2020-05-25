from AnalyzeDistributions import evaluate_distributions, evaluate_and_plot_dist
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from Distributions2 import best_fit_distribution
from BestDistributions import BestDistribution

plt.style.use('seaborn')

# Load data and select first column

from sklearn import datasets
data_set = datasets.load_breast_cancer()
# y=data_set.data[:,0]

data_set = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
y=data_set

y_df = pd.DataFrame(y, columns=['Data'])

# dist_results = evaluate_distributions(y_df['Data'], 50, ['levy_stable','geninvgauus','ksone'])

# best_distribution, best_params = best_fit_distribution(y_df['Data'])

# # Get the top distributions from the previous phase
# number_distributions_to_plot = 3
# dist_names = dist_results['Distribution'].iloc[0:number_distributions_to_plot]

# parameters = evaluate_and_plot_dist(y_df['Data'], dist_names, bins=100)

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