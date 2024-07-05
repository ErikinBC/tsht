"""
Can we show how to get an unbiased estimate of the SD for the exponential dist?
"""

# https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
import numpy as np
import pandas as pd
from time import time
from scipy.stats import expon, kurtosis
from utils import calculate_summary_stats
from parameters import alpha

def calc_C_n(n: int, kappa:float) -> float:
    C_n_num = 8*n*(n-1)
    C_n_dem = C_n_num - (n-1)*(kappa - 3) - 2*n
    C_n = C_n_num / C_n_dem
    return C_n


#########################################
# --- (1) COMPARE ADJUSTMNET FACTOR --- #

nsim = 100000
n_finite = (10**np.linspace(np.log10(10), np.log10(100), 8)).astype(int)
lamb = [1, 2, 3]
n_lamb = len(lamb)
dist_exp = expon(scale=lamb)
holder_var_i = np.zeros([nsim, len(n_finite), n_lamb])
stime = time()
for i in range(nsim):
    # Draw sample variances for a given finite data size
    sim_var_i = np.vstack([dist_exp.rvs(size=[n, n_lamb], random_state=i+1).var(axis=0, ddof=1) for n in n_finite])
    holder_var_i[i] = sim_var_i
    if (i + 1) % 250 == 0:
        dtime = time() - stime
        nleft = nsim - (i + 1)
        rate = (i + 1) / dtime
        seta = nleft / rate
        print(f'iterations to go: {nleft} (ETA = {seta:.0f} seconds)\n')
# Summarize the results for a dataframe
di_expon = {'alpha':alpha, 'colnames':lamb, 'idxnames':n_finite, 'var_name':'lamb', 'value_name':'n'}
res_expon_var = calculate_summary_stats(holder_var_i, **di_expon)
res_expon_sd = calculate_summary_stats(holder_var_i**0.5, **di_expon)
# Apply exponential adjustmnet
kappa_expon = 9
C_n_expon = calc_C_n(n_finite, kappa_expon)
C_n_expon = np.expand_dims(C_n_expon, axis=[0, 2])
res_expon_adj = calculate_summary_stats(holder_var_i**0.5 * C_n_expon, **di_expon)

res_expon = pd.concat(objs = [res_expon_var.assign(msr='Variance', adjusted=False),
                                res_expon_sd.assign(msr='Standard Deviation', adjusted=False),
                                res_expon_adj.assign(msr='Standard Deviation', adjusted=True)])
# res_expon.drop(columns = ['lb', 'ub', 'sd'], inplace=True)


##########################################
# --- (2) ESTIMATE ADJUSTMENT FACTOR --- #


dist_exp.stats('v')
dist_exp.stats('k')

calc_C_n(n = 30, kappa=kappa_expon)


#############################################
# --- (X) PLOTTING --- #

import plotnine as pn
from utils import gg_save
from parameters import dir_figs


gg_res_expon = (pn.ggplot(res_expon, pn.aes(x='n', y='mu', color='adjusted')) + 
                pn.theme_bw() + pn.geom_line() + 
                pn.facet_wrap('~msr+lamb', scales='free', labeller=pn.label_both, ncol=n_lamb, nrow=2) + 
                # pn.geom_ribbon(pn.aes(ymin='lb', ymax='ub'),fill='grey', alpha=0.3) + 
                pn.scale_color_discrete(name = 'Adjusted') + 
                pn.scale_x_log10() + 
                pn.labs(x='Sample size', y='Value'))
gg_save('expon_sim.png', dir_figs, gg_res_expon, 8, 6)


