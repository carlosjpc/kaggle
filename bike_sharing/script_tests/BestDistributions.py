import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
from scipy.stats import iqr


def solve_n_bins(x):
    """
    Uses the Freedman Diaconis Rule for generating the number of bins required
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    Bin Size = 2 IQR(x) / (n)^(1/3)
    """
    from scipy.stats import iqr

    x = np.asarray(x)
    hat = 2 * iqr(x) / (len(x) ** (1 / 3))

    if hat == 0:
        return int(np.sqrt(len(x)))
    else:
        return int(np.ceil((x.max() - x.min()) / hat))


class BestDistribution(object):
    """
    An exhaustive test of all the distributions and returns which distribution best fits the data.
    """
    VERY_SHORT = 'very short'
    SHORT = 'short'
    ALL = 'all'

    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
        st.fisk,
        st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
        st.invgauss,
        st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
        st.levy_stable,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
        st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
        st.tukeylambda,
        st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]

    DISTRIBUTIONS_SHORT = [
        st.beta, st.cauchy, st.chi, st.chi2, st.expon, st.exponnorm, st.gamma, st.logistic, st.lognorm, st.norm,
        st.pareto, st.powerlaw, st.rayleigh, st.t, st.uniform, st.wald
    ]

    DISTRIBUTIONS_VERY_SHORT = [st.beta, st.gamma, st.rayleigh, st.norm, st.pareto]

    # TODO There must be a better way
    DISTRIBUTIONS_DICT = {
        ALL: DISTRIBUTIONS,
        SHORT: DISTRIBUTIONS_SHORT,
        VERY_SHORT: DISTRIBUTIONS_VERY_SHORT
    }

    def __init__(self):
        self.best_distribution, self.best_params, self.runs_df = [None] * 3

    def fit(self, data, possible_distributions=None):
        # Check possible_distributions
        if type(possible_distributions) == str:
            distribution_list = self.DISTRIBUTIONS_DICT.get(possible_distributions, default=None)
            if distribution_list is None:
                raise ValueError('most be either {}'.format(self.DISTRIBUTIONS_DICT.keys()))

        elif type(possible_distributions) == list:
            distribution_list = possible_distributions

        else:
            distribution_list = self.DISTRIBUTIONS_SHORT if len(data) > 1000 else self.DISTRIBUTIONS_VERY_SHORT

        # Save run performance
        runs = []

        # Get histogram of original data
        y, x = np.histogram(data, bins=solve_n_bins(data), normed=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for distribution in tqdm(distribution_list):
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(data)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    runs.append([distribution.name, sse])

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                # Catch any error scipy throws
                pass

        self.runs_df = pd.DataFrame(data=runs, columns=['name', 'sse'])
        self.best_distribution, self.best_params = best_distribution, best_params
        return best_distribution.name

    def get_pdf(self, size=1e6):
        """
        Generate distributions's Probability Distribution Function
        """
        # Separate parts of parameters
        arg = self.best_params[:-2]
        loc = self.best_params[-2]
        scale = self.best_params[-1]

        # Get sane start and end points of distribution
        start = self.best_distribution.ppf(0.01, *arg, loc=loc, scale=scale) if arg else self.best_distribution.ppf(0.01, loc=loc, scale=scale)
        end = self.best_distribution.ppf(0.99, *arg, loc=loc, scale=scale) if arg else self.best_distribution.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = self.best_distribution.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        return pdf