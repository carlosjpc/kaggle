import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
# from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import scipy
from scipy.stats._continuous_distns import _distn_names
import warnings
warnings.filterwarnings("ignore")

def fit_scipy_distributions(data, bins=100, dist_names=_distn_names):
    label = data.name
    data = data.to_numpy()
    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(data), bins=bins)
    
    # Some details about the histogram
    bin_width = x[1]-x[0]
    N = len(data)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0 # go from bin edges to bin middles

    # loop through the distributions and store the sum of squared errors
    i=0
    sses = []
    parameters=[]
    for dist_name in dist_names:
        i+=1
        # print("Dist ("+str(i)+"/"+str(len(dist_names))+"): "+dist_name)
        dist = getattr(scipy.stats, dist_name)
        params = dist.fit(np.array(data))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf * bin_width * N # to go from pdf back to counts need to un-normalise the pdf

        sse = np.sum((y - pdf_scaled)**2)
        sses.append(sse)
        parameters.append(params)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['SSE'] = sses
    results['parameters'] = parameters
    results.sort_values(['SSE'], inplace=True)
    
    return results

def plot_distributions(data, dist_info, bins=100):
    dist_names = dist_info.iloc[:,0]
    parameters = dist_info.iloc[:,2]
    label = data.name
    data = data.to_numpy()
    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(data), bins=bins)
    
    # Some details about the histogram
    bin_width = x[1]-x[0]
    N = len(data)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0 # go from bin edges to bin middles
     
    fig, ax = plt.subplots()
    h = ax.hist(np.array(data), bins = bins, color = 'w')

    for d in range(len(dist_names)):
        dist_name = dist_names.iloc[d]
        params = parameters.iloc[d]
        # print("Dist ("+str(d)+"/"+str(len(dist_names))+"): "+dist_name)
        dist = getattr(scipy.stats, dist_name)
        # params = dist.fit(np.array(data))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf * bin_width * N # to go from pdf back to counts need to un-normalise the pdf

        ax.plot(x_mid, pdf_scaled, label = dist_name)
    
    plt.legend(loc=1)
    ax.set_xlabel(label)
    ax.set_ylabel('count')
    plt.show()



if __name__ == '__main__':
    y = st.norm.rvs(loc=7, scale=13, size=10000, random_state=0)
    sses, best_name, best_params = fit_scipy_distributions(y, bins = 100)
    sses, best_name, best_params = fit_scipy_distributions(y, 100, plot_best_fit=False, plot_all_fits=True)