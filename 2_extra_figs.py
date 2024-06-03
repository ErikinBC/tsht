"""
GENERATE EXTRA FIGURES AROUND THE SNTN DISTRIBUTION AND TSHT

python3 3_extra_figs.py
"""

# External modules
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar
import plotnine as pn
from mizani.formatters import percent_format
# Internal modulas
from utils import gg_save
from sntn.dists import nts
from funs_simulations import two_stage_mean_estimation
from parameters import dir_figs, \
                        nsim_plots, atol_figs, \
                        seed, alpha, \
                        mu_sims, theta_high_sims, theta_low_sims, \
                        sigma2_sims, gamma2_high_sims, gamma2_low_sims, \
                        n_sims, m_sims, k_sims

# Precalculate other terms
sigma_n = np.sqrt(sigma2_sims / n_sims)
sigma_n_high = np.sqrt(gamma2_high_sims / n_sims)
sigma_n_low = np.sqrt(gamma2_low_sims / n_sims)
gamma_m = np.sqrt(sigma2_sims / m_sims)
gamma_n_high = np.sqrt(gamma2_high_sims / m_sims)
gamma_n_low = np.sqrt(gamma2_low_sims / m_sims)
Delta_default = k_sims * sigma_n

# Unconditional mean should just be -(mu + Delta - theta) / sigma_m
Delta_m = Delta_default / gamma_m
Delta_n = Delta_default / sigma_n
var_H0 = m_sims / n_sims
var_HA = (sigma_n / gamma_m) ** 2
var_HA_high = (sigma_n / gamma_n_high) ** 2
var_HA_low = (sigma_n / gamma_n_low) ** 2
mu_H0 = -Delta_m   # E(s_Delta | H0) = -Delta/sigma_m
sd_H0 = np.sqrt(1 + var_H0)
mu_HA_high = (theta_high_sims - mu_sims - Delta_default) / gamma_m
mu_HA_low = (theta_low_sims - mu_sims - Delta_default) / gamma_m

# Print default parameters for transparency
print(f'alpha = {alpha}\n'
      f'mu = {mu_sims}\n'
      f'theta_high = {theta_high_sims:.3f}\n'
      f'theta_low = {theta_low_sims:.3f}\n'
      f'n = {n_sims}\n'
      f'm = {m_sims}\n'
      f'sigma2 = {sigma2_sims}\n'
      f'gamma2_high = {gamma2_high_sims:.3f}\n'
      f'gamma2_low = {gamma2_low_sims:.3f}\n'
      f'k = {k_sims}\n'
      f'Delta = k*sigma_n = {Delta_default:.3f}\n'
      f'Delta_m = {Delta_m:.3f}\n'
      f'Delta_n = {Delta_n:.3f}\n'
      )


# Pre-define some NTS argument
di_dist_hatH0 = {'mu1':0, 'tau21':1, 'a':0, 'b':np.infty}
di_dist_hatHA = {**di_dist_hatH0, **{'a':-np.infty, 'b':0}}

# Dictionaries to clean up labels in the plots
di_HA = {'-1':'$H_A: \\theta<\\mu$', 
        '1':'$H_A: \\theta>\\mu$', 
         '0':'$H_0: \\theta=\\mu$'}
di_H0hat = {'All': 'Unconditional',
            'True': '$\\hat{H}_0$',
            'False': '$\\hat{H}_A$'}
di_var_param = {'H0':'$\\sigma = \\gamma$', 
                'HA_high':'$\\sigma < \\gamma$', 
                'HA_low':'$\\sigma > \\gamma$'}
di_rv = {'H0': '$s_\\Delta | \\hat{H}_0, H_0$',
          'HA': '$s_\\Delta | \\hat{H}_A, H_0$',
          'mu': '$\\hat\\mu_\\Delta - \\mu$'}

# Points to plot over
nseq = 2000
xseq = np.linspace(-5, +5, nseq)

# Plotting colors
di_colz_H0hat = {'True':'red', 'All':'black', 'False':'blue'}
ord_H0hat = list(di_colz_H0hat.keys())
colz_H0hat = list(di_colz_H0hat.values())

di_colz_var_param = {'HA_low':'blue', 'H0':'black', 'HA_high':'red', }
ord_var_param = list(di_colz_var_param.keys())
colz_var_param = list(di_colz_var_param.values())


################################################
# --- (1) CHECK SIMULATIONS AGAINST THEORY --- #

# ~~ Make sure results as expected before creating plots ~~ #
# Re-usable arguments
di_sim_base = {'mu':mu_sims, 
                'sigma2':sigma2_sims, 'gamma2':sigma2_sims,
                'n':n_sims, 'm':m_sims, 
                'k':k_sims, 'nsim':nsim_plots, 
                'alpha':alpha, 'seed':seed, 'ret_raw':True}

# Run the different simulation versions
sim_H0 = two_stage_mean_estimation(**{**di_sim_base, **{'theta':mu_sims}})
sim_HA1 = two_stage_mean_estimation(**{**di_sim_base, **{'theta':theta_high_sims, 'seed':seed+1}})
sim_HA2 = two_stage_mean_estimation(**{**di_sim_base, **{'theta':theta_low_sims, 'seed':seed+1}})
df_sim = pd.concat(objs=[sim_H0, sim_HA1, sim_HA2]).reset_index(drop=True)
df_sim = df_sim.assign(HA = lambda x: np.sign(x['theta'] - x['mu']).astype(int))
df_sim.drop(columns=['theta','mu'], inplace=True)
df_sim.rename(columns = {'H0hat':'H0hat'}, inplace=True)

# Check that the mean aligns with theory
mu_H_theory = pd.DataFrame({'HA':[0,1,-1], 'theory':[mu_H0, mu_HA_high, mu_HA_low]})
res_mu_H = df_sim.groupby('HA')['s_Delta'].mean().reset_index().merge(mu_H_theory)
np.testing.assert_allclose(res_mu_H['s_Delta'], res_mu_H['theory'], atol=atol_figs)

# Set up all the different distributions: H0, HA, hat{H}{0,A}|H{0,A}
di_dist_hatH0_H0 = {**di_dist_hatH0, **{'tau22':var_H0}}
di_dist_hatHA_H0 = {**di_dist_hatHA, **{'tau22':var_H0}}
dist_H0 = norm(loc=mu_H0, scale=sd_H0)
dist_hatH0_H0 = nts(**{**di_dist_hatH0_H0, **{'mu2':mu_H0}})
dist_hatHA_H0 = nts(**{**di_dist_hatHA_H0, **{'mu2':mu_H0}})
dist_HA_high = norm(loc=mu_HA_high, scale=sd_H0)
dist_hatH0_HA_high = nts(**{**di_dist_hatH0_H0, **{'mu2':mu_HA_high}})
dist_hatHA_HA_high = nts(**{**di_dist_hatHA_H0, **{'mu2':mu_HA_high}})
dist_HA_low = norm(loc=mu_HA_low, scale=sd_H0)
dist_hatH0_HA_low = nts(**{**di_dist_hatH0_H0, **{'mu2':mu_HA_low}})
dist_hatHA_HA_low = nts(**{**di_dist_hatHA_H0, **{'mu2':mu_HA_low}})

# Calculate their means
vecmu_H0 = [dist_H0.mean(), dist_hatH0_H0.mean()[0], dist_hatHA_H0.mean()[0]]
vecmu_HA_high = [dist_HA_high.mean(), dist_hatH0_HA_high.mean()[0], dist_hatHA_HA_high.mean()[0]]
vecmu_HA_low = [dist_HA_low.mean(), dist_hatH0_HA_low.mean()[0], dist_hatHA_HA_low.mean()[0]]
vec_H0hat = ['All',True,False]
vec_H = [0, +1, -1]
mu_Hhat_theory = pd.DataFrame({'HA':np.repeat(vec_H, 3), 
              'H0hat':np.tile(vec_H0hat, 3), 
              'theory':np.concatenate((vecmu_H0, vecmu_HA_high, vecmu_HA_low))})

# Compare to the conditional
df_sim_H = pd.concat(objs=[df_sim, df_sim.assign(H0hat='All')]).reset_index(drop=True)
df_sim_H['H0hat'] = df_sim_H['H0hat'].astype(str)
res_sim_H = df_sim_H.groupby(['HA','H0hat'])['s_Delta'].mean().reset_index().merge(mu_Hhat_theory)
np.testing.assert_allclose(res_sim_H['s_Delta'], res_sim_H['theory'], atol=atol_figs)


###################################
# --- (2) MEANS ARE NOT EQUAL --- #

# Rejection threshold
crit_val = dist_hatH0_H0.ppf(alpha)[0]

# Merge all into a big list
lst_dists = [{'HA':0, 'H0hat':'All', 'dist':dist_H0},
             {'HA':0, 'H0hat':'True', 'dist':dist_hatH0_H0},
             {'HA':0, 'H0hat':'False', 'dist':dist_hatHA_H0},
             {'HA':1, 'H0hat':'All', 'dist':dist_HA_high},
             {'HA':1, 'H0hat':'True', 'dist':dist_hatH0_HA_high},
             {'HA':1, 'H0hat':'False', 'dist':dist_hatHA_HA_high},
             {'HA':-1, 'H0hat':'All', 'dist':dist_HA_low},
             {'HA':-1, 'H0hat':'True', 'dist':dist_hatH0_HA_low},
             {'HA':-1, 'H0hat':'False', 'dist':dist_hatHA_HA_low}]

# Calculate the pdf of each
holder_dists = []
for di in lst_dists:
    holder_dists.append(pd.DataFrame({'HA':di['HA'], 'H0hat':di['H0hat'], 'x':xseq, 
                                      'pdf':di['dist'].pdf(xseq).flatten()}))
res_dist = pd.concat(holder_dists).reset_index(drop=True)
res_dist['H0hat'] = pd.Categorical(res_dist['H0hat'].astype(str), ord_H0hat)

# Calculate the rejection rate
reject_area = res_dist[res_dist['x'] <= crit_val]
reject_pct = reject_area.groupby(['HA','H0hat'],observed=False).\
    apply(lambda g: pd.Series({'pct':np.trapz(g['pdf'], x=g['x']), 'y':g['pdf'].max()}),include_groups=False).\
        reset_index()

# Plot #1: Assume H0 holds
# Name of the legend
legname_dist = 'Distribution'

res_dist_H0 = res_dist.query('HA == 0').copy()
reject_area_H0 = reject_area.query('HA == 0').copy()
reject_pct_H0 = reject_pct.query('HA == 0').copy()

gg_dist_H0 = (pn.ggplot(res_dist_H0, pn.aes(x='x', y='pdf', color='H0hat')) + 
              pn.theme_bw() + pn.geom_line() + 
              pn.geom_area(pn.aes(fill='H0hat'), data=reject_area_H0, alpha=0.15, position=pn.position_identity()) + 
              pn.facet_wrap('~HA', labeller=pn.labeller(cols=lambda x: di_HA[x])) + 
              pn.geom_text(pn.aes(x=crit_val-0.5, y='y*1.05', label='100*pct'),data=reject_pct_H0, format_string='{:.0f}%',size=9) + 
              pn.scale_color_manual(values=colz_H0hat, name=legname_dist, labels=lambda x: [di_H0hat[z] for z in x]) + 
              pn.scale_fill_manual(values=colz_H0hat, name=legname_dist, labels=lambda x: [di_H0hat[z] for z in x]) + 
              pn.geom_vline(xintercept=crit_val, linetype='--') + 
              pn.labs(y='Density', x='$s_\\Delta$'))
gg_save('dist_H0.png', dir_figs, gg_dist_H0, 7, 4)

# Plot #2: Assume HA holds (high and low)
res_dist_HA = res_dist.query('HA != 0').copy()
reject_area_HA = reject_area.query('HA != 0').copy()
reject_pct_HA = reject_pct.query('HA != 0').copy()

gg_dist_HA = (pn.ggplot(res_dist_HA, pn.aes(x='x', y='pdf', color='H0hat')) + 
              pn.theme_bw() + pn.geom_line() + 
              pn.geom_area(pn.aes(fill='H0hat'), data=reject_area_HA, alpha=0.15, position=pn.position_identity()) + 
              pn.facet_wrap('~HA', labeller=pn.labeller(cols=lambda x: di_HA[x]), ncol=2) + 
              pn.geom_text(pn.aes(x=crit_val-0.5, y='y*1.05', label='100*pct'),data=reject_pct_HA, format_string='{:.0f}%',size=9) + 
              pn.scale_color_manual(values=colz_H0hat, name=legname_dist, labels=lambda x: [di_H0hat[z] for z in x]) + 
              pn.scale_fill_manual(values=colz_H0hat, name=legname_dist, labels=lambda x: [di_H0hat[z] for z in x]) + 
              pn.geom_vline(xintercept=crit_val, linetype='--') + 
              pn.labs(y='Density', x='$s_\\Delta$'))
gg_save('dist_HA.png', dir_figs, gg_dist_HA, 11, 4)




#######################################
# --- (3) VARIANCES ARE NOT EQUAL --- #

# Create the three different variances (same means)
di_var_H0 = {'mu2':mu_H0, 'tau22':var_H0}
di_var_HA_high = {'mu2':mu_H0, 'tau22':var_HA_high}
di_var_HA_low = {'mu2':mu_H0, 'tau22':var_HA_low}
di_dist_hatH0_varH0 = {**di_dist_hatH0, **di_var_H0}
di_dist_hatH0_varHA_high = {**di_dist_hatH0, **di_var_HA_high}
di_dist_hatH0_varHA_low = {**di_dist_hatH0, **di_var_HA_low}
di_dist_hatHA_varH0 = {**di_dist_hatHA, **di_var_H0}
di_dist_hatHA_varHA_high = {**di_dist_hatHA, **di_var_HA_high}
di_dist_hatHA_varHA_low = {**di_dist_hatHA, **di_var_HA_low}

# Merge all into a big list
lst_dists = [{'var_param':'H0', 'H0hat':'True', 'dist':nts(**di_dist_hatH0_varH0)},
             {'var_param':'HA_high', 'H0hat':'True', 'dist':nts(**di_dist_hatH0_varHA_high)},
             {'var_param':'HA_low', 'H0hat':'True', 'dist':nts(**di_dist_hatH0_varHA_low)},
             {'var_param':'H0', 'H0hat':'False', 'dist':nts(**di_dist_hatHA_varH0)},
             {'var_param':'HA_high', 'H0hat':'False', 'dist':nts(**di_dist_hatHA_varHA_high)},
             {'var_param':'HA_low', 'H0hat':'False', 'dist':nts(**di_dist_hatHA_varHA_low)}
             ]

# Calculate the critival value for each distribution
dat_critv = {di['var_param']:di['dist'].ppf(alpha)[0] for di in lst_dists if di['H0hat']=='True'}
dat_critv = pd.DataFrame.from_dict(dat_critv, orient='index').\
    rename(columns={0:'crit_val'}).rename_axis('var_param').reset_index()

# Add the critical value to xseq (to ensure merge)
xseq_var = np.linspace(-10, +10, 5000)
xseq_var = np.sort(np.unique(np.concatenate((xseq_var, dat_critv['crit_val'].values))))

# Calculate the pdf of each
holder_dists = []
for di in lst_dists:
    tmp_df = pd.DataFrame({'x':xseq_var, 'pdf':di['dist'].pdf(xseq_var).flatten()})
    tmp_df = tmp_df.assign(**{k:v for k,v in di.items() if k != 'dist'})
    holder_dists.append(tmp_df.copy())
    del tmp_df
res_dist = pd.concat(holder_dists).reset_index(drop=True)
res_dist['H0hat'] = pd.Categorical(res_dist['H0hat'].astype(str), ord_H0hat).remove_unused_categories()
res_dist['var_param'] = pd.Categorical(res_dist['var_param'], ord_var_param)

# Calculate the rejection rate
reject_area = res_dist.merge(dat_critv).query('x <= crit_val')
reject_pct = reject_area.groupby(['var_param','H0hat'],observed=False).\
    apply(lambda g: pd.Series({'pct':np.trapz(g['pdf'], x=g['x']), 'y':g['pdf'].max()}),include_groups=False).\
        reset_index()
reject_pct = reject_pct.merge(dat_critv)
reject_pct = reject_pct.assign(h=lambda x: x['y'], x=lambda x: x['crit_val'])
reject_pct['H0hat'] = pd.Categorical(reject_pct['H0hat'].astype(str), ord_H0hat).remove_unused_categories()
reject_pct['var_param'] = pd.Categorical(reject_pct['var_param'], ord_var_param)
reject_area['H0hat'] = pd.Categorical(reject_area['H0hat'].astype(str), ord_H0hat).remove_unused_categories()
reject_area['var_param'] = pd.Categorical(reject_area['var_param'], ord_var_param)

# Do the plot of all three RVs
legname_var_param = 'Parameters: '  # Name of the legend

gg_var_param = (pn.ggplot(res_dist.query('pdf.abs() > 1e-10'), pn.aes(x='x', y='pdf', color='var_param')) + 
              pn.theme_bw() + pn.geom_line() + 
              pn.labs(y='Density', x='$s_\\Delta$') + 
              pn.scale_color_manual(values=colz_var_param, name=legname_var_param, labels=lambda x: [di_var_param[z] for z in x]) + 
              pn.scale_fill_manual(values=colz_var_param, name=legname_var_param, labels=lambda x: [di_var_param[z] for z in x]) + 
              pn.geom_segment(pn.aes(x='crit_val', xend='crit_val', y=0, yend='y',color='var_param'),data=reject_pct,linetype='--') + 
              pn.geom_area(pn.aes(fill='var_param'), data=reject_area, alpha=0.15, position=pn.position_identity()) + 
              pn.geom_text(pn.aes(x='x', y='h', label='100*pct'),data=reject_pct, format_string='{:.0f}%',size=8) + 
              pn.theme(legend_position='bottom') + 
              pn.facet_wrap('~H0hat', labeller=pn.labeller(cols=lambda x: di_H0hat[x])))
gg_save('var_param.png', dir_figs, gg_var_param, 7, 4)


####################################
# --- (4)  POWER EQUIVALENT N? --- #

def find_Delta_power(d, target, p, s_m, tau22):
    dist_H0 = nts(mu1=0, tau21=1, mu2=-d/s_m, tau22=tau22, a=0, b=+np.infty)
    dist_HA = nts(mu1=0, tau21=1, mu2=-d/s_m, tau22=tau22, a=-np.infty, b=0)
    power = dist_HA.cdf(dist_H0.ppf(p)[0])[0]
    err = power - target
    return err

# Hard code some parameters
m1 = 25
n1 = 16
n2 = 100
Delta1 = 0.10
sigma21 = 4
gamma_m1 = np.sqrt(sigma21 / m1)
sigma_n1 = np.sqrt(sigma21 / n1)
sigma_n2 = np.sqrt(sigma21 / n2)
var1 = m1 / n1
var2 = m1 / n2
mu1 = -Delta1 / gamma_m1

#  Set up the first distributions
dist_H01 = nts(**{**di_dist_hatH0, **{'mu2':mu1,'tau22':var1}})
dist_HA1 = nts(**{**di_dist_hatHA, **{'mu2':mu1,'tau22':var1}})
dist_H02 = nts(**{**di_dist_hatH0, **{'mu2':mu1,'tau22':var2}})
dist_HA2 = nts(**{**di_dist_hatHA, **{'mu2':mu1,'tau22':var2}})
# Calculate power and empirical null probabilities
power1 = dist_HA1.cdf(dist_H01.ppf(alpha))[0]
power2 = dist_HA2.cdf(dist_H02.ppf(alpha))[0]
# Find the Delta that matches the power of the first one
Delta3 = root_scalar(f = find_Delta_power, args = (power1, alpha, gamma_m1, var2), method = 'brentq', bracket=(0, 1)).root
mu3 = -Delta3 / gamma_m1
dist_H03 = nts(**{**di_dist_hatH0, **{'mu2':mu3,'tau22':var2}})
dist_HA3 = nts(**{**di_dist_hatHA, **{'mu2':mu3,'tau22':var2}})
np.testing.assert_allclose(dist_HA1.cdf(dist_H01.ppf(alpha)), dist_HA3.cdf(dist_H03.ppf(alpha)))
# Define the hat{mu}_Delta distributions
dist_mu1 = norm(loc=Delta1, scale = sigma_n1)
dist_mu2 = norm(loc=Delta1, scale = sigma_n2)
dist_mu3 = norm(loc=Delta3, scale = sigma_n2)
prob_HA1 = norm.cdf(Delta1 / sigma_n1)
prob_HA2 = norm.cdf(Delta1 / sigma_n2)
prob_HA3 = norm.cdf(Delta3 / sigma_n2)


# Create the mapping dict
di_param = {'n1d1': f"$n={n1}, \\Delta={Delta1:.2f}$",
            'n2d1': f"$n'={n2}, \\Delta={Delta1:.2f}$",
            'n2d3': f"$n'={n2}, \\Delta'={Delta3:.2f}$"}


# Set up the color/legend order
di_colz_param = {'n1d1':'red', 'n2d1':'black', 'n2d3':'blue'}
ord_param = list(di_colz_param.keys())
colz_param = list(di_colz_param.values())
ord_rv = ['mu', 'H0', 'HA']

# Merge all into a big list
lst_dists = [{'rv':'H0', 'param':'n1d1', 'dist':dist_H01},
             {'rv':'HA', 'param':'n1d1', 'dist':dist_HA1},
             {'rv':'H0', 'param':'n2d1', 'dist':dist_H02},
             {'rv':'HA', 'param':'n2d1', 'dist':dist_HA2},
             {'rv':'H0', 'param':'n2d3', 'dist':dist_H03},
             {'rv':'HA', 'param':'n2d3', 'dist':dist_HA3},
             {'rv':'mu', 'param':'n1d1', 'dist':dist_mu1},
             {'rv':'mu', 'param':'n2d1', 'dist':dist_mu2},
             {'rv':'mu', 'param':'n2d3', 'dist':dist_mu3}]

# Calculate the pdf of each
holder_dists = []
for di in lst_dists:
    holder_dists.append(pd.DataFrame({'rv':di['rv'], 'param':di['param'], 'x':xseq, 
                                      'pdf':di['dist'].pdf(xseq).flatten()}))
res_dist = pd.concat(holder_dists).reset_index(drop=True)

# Set up the xlimits
dat_xlim = pd.DataFrame({'rv':['H0', 'HA', 'mu'],
                        'x_lb':[-3, -5, -2], 
                        'x_ub':[5, 3, 2]})
res_dist = res_dist.merge(dat_xlim).query('x >= x_lb & x <= x_ub')
res_dist.reset_index(drop=True, inplace=True)
res_dist['param'] = pd.Categorical(res_dist['param'], ord_param)
res_dist['rv'] = pd.Categorical(res_dist['rv'], ord_rv)

# Add on the critival value
dat_critv = pd.DataFrame({'rv':'HA', 
                          'param':['n1d1', 'n2d1', 'n2d3'], 
                          'crit_val':[dist_H01.ppf(alpha)[0], dist_H02.ppf(alpha)[0], dist_H03.ppf(alpha)[0]]})
dat_critv = pd.concat(objs=[dat_critv.assign(rv='H0'), dat_critv, dat_critv.assign(rv='mu', crit_val=0)]).reset_index(drop=True)
dat_critv['rv'] = pd.Categorical(dat_critv['rv'], ord_rv)
dat_critv['param'] = pd.Categorical(dat_critv['param'], ord_param)
k = 2
dat_critv = dat_critv.assign(x=lambda x: x['crit_val'].round(k)).merge(res_dist[['rv','param','x','pdf']].round(k), on=['rv','param','x']).drop_duplicates(subset=['rv','param','x']).reset_index(drop=True).rename(columns={'pdf':'y'}).drop(columns='x')

# Calculate the rejection rate
reject_area = res_dist.merge(dat_critv).query('x <= crit_val').reset_index(drop=True)
reject_pct = reject_area.groupby(['rv','param'],observed=False).\
    apply(lambda g: pd.Series({'pct':np.trapz(g['pdf'], x=g['x']), 'y':g['pdf'].max()}),include_groups=False).\
        dropna().reset_index()
reject_pct = reject_pct.merge(dat_critv.drop(columns='y'))
x_di_rv = {'H0':+0.3, 'HA':+0.5, 'mu':0}
reject_pct = reject_pct.assign(x=lambda x: x['crit_val'] + x['rv'].astype(str).map(x_di_rv) * (x['param'].cat.codes+1))
reject_pct = reject_pct.assign(x=lambda x: np.where(x['rv'] == 'mu', -0.5, x['crit_val'] ))
reject_pct = reject_pct.assign(h=lambda x: x['y'] + np.where(x['rv'] == 'H0', 0.02, -0.05) *(x['param'].cat.codes+1))

# Do the plot of all three RVs
legname_n_Delta = 'Parameters: '  # Name of the legend

gg_n_Delta = (pn.ggplot(res_dist, pn.aes(x='x', y='pdf', color='param')) + 
              pn.theme_bw() + pn.geom_line() + 
              pn.labs(y='Density', x='Distribution') + 
              pn.scale_color_manual(values=colz_param, name=legname_n_Delta, labels=lambda x: [di_param[z] for z in x]) + 
              pn.scale_fill_manual(values=colz_param, name=legname_n_Delta, labels=lambda x: [di_param[z] for z in x]) + 
              pn.geom_segment(pn.aes(x='crit_val', xend='crit_val', y=0, yend='y',color='param'),data=dat_critv,linetype='--') + 
              pn.geom_area(pn.aes(fill='param'), data=reject_area, alpha=0.15, position=pn.position_identity()) + 
              pn.geom_text(pn.aes(x='x', y='h', label='100*pct'),data=reject_pct, format_string='{:.0f}%',size=8) + 
              pn.theme(legend_position='bottom') + 
              pn.facet_wrap('~rv', scales='free', labeller=pn.labeller(cols=lambda x: di_rv[x])))
gg_save('n_Delta.png', dir_figs, gg_n_Delta, 10, 4)


##########################################
# --- (5) POWER: COMPARATIVE STATICS --- #

# Create an outer grid of parameters
m_seq = np.arange(50, 1000+1, 150).astype(int)
n_seq = m_seq.copy()
Delta_seq = np.array([0.01, 0.1, 0.25, 0.5])
sigma2_seq = np.array([0.5, 2, 4, 8, 16])
df_params = pd.DataFrame(np.vstack(pd.core.reshape.util.cartesian_product([m_seq, n_seq, Delta_seq, sigma2_seq])).T, columns =['m','n','Delta','sigma2'])
df_params = df_params.assign(gamma_m = lambda x: np.sqrt(x['sigma2'] / x['m']),
                             m_n = lambda x: x['m'] / x['n'])
df_params = df_params.assign(mu = lambda x: -x['Delta'] / x['gamma_m'])
df_params['Delta'] = df_params['Delta'].round(2)
df_params['sigma2'] = df_params['sigma2'].round(2)
df_params[['n','m']] = df_params[['n','m']].astype(int)

# Create the vectorized distributions
di_mu2_tau22 = {'mu2':df_params['mu'], 'tau22':df_params['m_n']}
dist_vec_hatH0 = nts(**{**di_dist_hatH0, **di_mu2_tau22})
dist_vec_hatHA = nts(**{**di_dist_hatHA, **di_mu2_tau22})

# Rejection threshold
vec_crit_val = dist_vec_hatH0.ppf(alpha, method='fast')
vec_power_HA = np.squeeze(dist_vec_hatHA.cdf(vec_crit_val))
df_params['power'] = vec_power_HA.copy()

df_params.groupby('m')['power'].mean()
df_params.groupby('n')['power'].mean()
df_params.groupby('Delta')['power'].mean()

# Make some plots
lbl_Delta = lambda x: '$\\Delta: $' + x
lbl_sigma2 = lambda x: '$\\sigma^2: $' + x
lbl_n = lambda x: '$n: $' + x
lbl_m = lambda x: '$m: $' + x

tmp_df = df_params.assign(n=lambda x: pd.Categorical(x['n'].astype(str), x['n'].unique().astype(str)))
gg_power_m = (pn.ggplot(tmp_df, pn.aes(x='m', color='n',y='power')) + 
          pn.theme_bw() + pn.geom_line() + 
          pn.scale_y_continuous(labels=percent_format()) + 
          pn.labs(y='Power', x='m') + 
          pn.facet_grid('sigma2 ~ Delta',labeller=pn.labeller(cols=lbl_Delta, rows=lbl_sigma2)))
gg_save('power_curve_m.png', dir_figs, gg_power_m, 8, 8)

tmp_df = df_params.assign(m=lambda x: pd.Categorical(x['m'].astype(str), x['m'].unique().astype(str)))
gg_power_n = (pn.ggplot(tmp_df, pn.aes(x='n', color='m',y='power')) + 
          pn.theme_bw() + pn.geom_line() + 
          pn.scale_y_continuous(labels=percent_format()) + 
          pn.labs(y='Power', x='n') + 
          pn.facet_grid('sigma2 ~ Delta',labeller=pn.labeller(cols=lbl_Delta, rows=lbl_sigma2)))
gg_save('power_curve_n.png', dir_figs, gg_power_n, 8, 8)


tmp_df = df_params.assign(n=lambda x: pd.Categorical(x['n'].astype(str), x['n'].unique().astype(str)),
                          m=lambda x: pd.Categorical(x['m'].astype(str), x['m'].unique().astype(str)))
gg_power_sigma2 = (pn.ggplot(tmp_df, pn.aes(x='sigma2', color='m',y='power')) + 
          pn.theme_bw() + pn.geom_line() + 
          pn.scale_y_continuous(labels=percent_format()) + 
          pn.labs(y='Power', x='$\\sigma^2$') + 
          pn.facet_grid('Delta ~ n',labeller=pn.labeller(cols=lbl_n, rows=lbl_Delta)))
gg_save('power_curve_sigma2.png', dir_figs, gg_power_sigma2, 12, 7)

tmp_df = df_params.assign(n=lambda x: pd.Categorical(x['n'].astype(str), x['n'].unique().astype(str)),
                          m=lambda x: pd.Categorical(x['m'].astype(str), x['m'].unique().astype(str)))
gg_power_Delta = (pn.ggplot(tmp_df, pn.aes(x='Delta', color='m',y='power')) + 
          pn.theme_bw() + pn.geom_line() + 
          pn.scale_y_continuous(labels=percent_format()) + 
          pn.labs(y='Power', x='$\\Delta$') + 
          pn.facet_grid('sigma2 ~ n',labeller=pn.labeller(cols=lbl_n, rows=lbl_sigma2)))
gg_save('power_curve_Delta.png', dir_figs, gg_power_Delta, 12, 7)



print('~~~ End of 2_extra_figs.py ~~~')