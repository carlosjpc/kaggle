import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats._continuous_distns import _distn_names
import scipy
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

dist_names = _distn_names
print(dist_names)
if 'levy_stable' in dist_names:
    dist_names.remove('levy_stable')
# Ovewrite distribution names for less analysis
dist_names = ['beta','expon', 'gamma', 'lognorm', 'norm',
                  'pearson3', 'triang', 'uniform', 'weibull_min', 'weibull_max']

dist_names = ['ksone', 'kstwobign', 'norm', 'alpha', 'anglit','beta', 'betaprime', 'bradford', 'burr', 'burr12', 'fisk', 'cauchy', 'chi', 'chi2', 'cosine', 'dgamma', 'dweibull']


def evaluate_distributions(data,bins=50,skip_dists=None):
    
    for dist in skip_dists:
        if dist in dist_names:
            dist_names.remove(dist)
            
    
    sc=StandardScaler() 
    yy =  data.to_numpy().reshape(-1,1)
    sc.fit(yy)
    y_std =sc.transform(yy)
    y_std = y_std.flatten()
    size = len(yy)
   
    # Set up empty lists to stroe results
    chi_square = []
    p_values = []
    
    # Set up bins for chi-square test
    # Observed data will be approximately evenly distrubuted aross all bins
    percentile_bins = np.linspace(0,100,bins)
    percentile_cutoffs = np.percentile(y_std, percentile_bins)
    observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)
    
    i=0
    # Loop through candidate distributions
    for distribution in dist_names:
        i+=1
        print("Dist ("+str(i)+"/"+str(len(dist_names))+") : " + distribution)
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(y_std)
        
        # Obtain the KS test P statistic, round it to 5 decimal places
        p = scipy.stats.kstest(y_std, distribution, args=param)[1]
        p = np.around(p, 5)
        p_values.append(p)    
        
        # Get expected counts in percentile bins
        # This is based on a 'cumulative distrubution function' (cdf)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
        
        # calculate chi-squared
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square.append(ss)
            
    # Collate results and sort by goodness of fit (best at top)
    
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    results.sort_values(['chi_square'], inplace=True)
        
    # Report results
    
    print ('\nDistributions sorted by goodness of fit:')
    print ('----------------------------------------')
    print (results)
    
    return results

def evaluate_and_plot_dist(data, dist_names, bins=100):
    
    y = data.to_numpy()
    x = np.arange(len(y))
    
    # Divide the observed data into bins
    number_of_bins = bins
    bin_cutoffs = np.linspace(np.percentile(y,0), np.percentile(y,99),number_of_bins)
    
    # Create the plot
    h = plt.hist(y, bins = bin_cutoffs, color='0.75')

    # Create an empty list to stroe fitted distribution parameters
    parameters = []
    
    # Loop through the distributions ot get line fit and paraemters   
    for dist_name in dist_names:
        # Set up distribution and store distribution paraemters
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        parameters.append(param)
        
        # Get line for each distribution (and scale to match observed data)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
        pdf_fitted *= scale_pdf
        
        # Add the line to the plot
        plt.plot(pdf_fitted, label=dist_name)
        
        # Set the plot x axis to contain 99% of the data
        # This can be removed, but sometimes outlier data makes the plot less clear
        plt.xlim(0,np.percentile(y,99))
    
    # Add legend and display plot
    
    plt.legend()
    plt.show()
    return parameters