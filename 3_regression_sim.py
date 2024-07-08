"""
Uses two real-world regression datasets to test the TSHT approach

python3 3_regression_sim.py
"""

# Load modules
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
from scipy.stats import norm
# Internal
from sntn.dists import nts
from funs_models import elnet_wrapper
from funs_simulations import dgp_sparse_yX
from utils import vprint, gg_save, generate_std_adjustments
from parameters import seed, alpha, nsim_regression, \
                        num_imputer_strategy, cat_imputer_strategy, \
                        alphas, di_elnet, \
                        n_train, n_test, m_trial, k_regression, \
                        di_resid_fun, metrics, di_metrics, \
                        s, p, sigma2_u, SigmaX, b0, \
                        path_reg_results, dir_figs
from parameters import verbose_regression as verbose

# How often to aggregate and print results
n_checkpoint = 5

##############################
# --- (1) SET UP CLASSES --- #

# (i) Elastic net
mdl = elnet_wrapper(
                    alphas = alphas, 
                    random_state=seed, 
                    num_imputer_strategy = num_imputer_strategy, 
                    cat_imputer_strategy = cat_imputer_strategy,
                    **di_elnet
                    )

# (ii) DGP (as long as Sigma is fixed, we don't need to re-initialize the class)
data_generator = dgp_sparse_yX(s = s,
                               p = p,
                               sigma2_u=sigma2_u,
                               Sigma_X=SigmaX,
                               b0 = b0, 
                               seed = seed)


##################################
# --- (2) RUN THE SIMULATION --- #

# Total samples to split
n = n_train + n_test + m_trial
# Variance will be constant
tau22 = m_trial / n_test

# Columns to add for every loop
cols_test_keep = [
                'metric', 'approach', 
                'power', 'prob_H0hat', 
                'mu_Delta', 'c_alpha', 
                'sigma_m', 'err_test'
                ]
    
# Run simulation loop
i_train = n_train
i_test = n_train+n_test
i_trial = n_train+n_test+m_trial
holder_loop = []
stime = time()
for i in range(nsim_regression):
    # (i) Index the different splits
    indices = np.random.permutation(n)
    idx_train = indices[:i_train]
    idx_test = indices[i_train:i_test]
    idx_trial = indices[i_test:i_trial]
    
    # (ii) Draw data
    yy_i, xx_i = data_generator.gen_yX(n = n, seed = seed + i)
    yy_train_i, xx_train_i = yy_i[idx_train], xx_i[idx_train]
    yy_test_i, xx_test_i = yy_i[idx_test], xx_i[idx_test]
    yy_trial_i, xx_trial_i = yy_i[idx_trial], xx_i[idx_trial]
    
    # (iii) Fit model
    mdl.fit(X = xx_train_i, y = yy_train_i, verbose=False)
    # Calculate the risk for a given coefficient
    beta_hat = np.concatenate( ([mdl.model.intercept_], mdl.model.coef_))
    data_generator.calc_risk(beta_hat=beta_hat, has_intercept=True)
    
    # (iv) Get test set standard deviations
    yhat_test_i = mdl.predict(X = xx_test_i)
    di_err_sigma_test_i = dict.fromkeys(di_resid_fun)
    for metric, resid_fun in di_resid_fun.items():
        # Generate the errors
        error_test = resid_fun(yy_test_i, yhat_test_i)
        error_test_mu = error_test.mean()
        res_std_test = generate_std_adjustments(error_test, random_state=seed)
        res_std_test.insert(0, 'metric', metric)
        res_std_test.insert(res_std_test.shape[1], 'err_test', error_test_mu)
        di_err_sigma_test_i[metric] = res_std_test
    dat_test_i = pd.concat(objs = list(di_err_sigma_test_i.values()))
    dat_test_i.reset_index(drop=True, inplace=True)
    dat_test_i['sigma_n'] = dat_test_i['sigma'] / np.sqrt(n_test)
    dat_test_i['sigma_m'] = dat_test_i['sigma'] / np.sqrt(m_trial)

    # (v) Set up empirical null
    Delta_i = k_regression * dat_test_i['sigma_n']
    dat_test_i['mu_Delta'] = dat_test_i['err_test'] + Delta_i
    dat_test_i['prob_H0hat'] = norm.cdf(-Delta_i / dat_test_i['sigma_n'])
    mu2_i = -Delta_i / dat_test_i['sigma_m']
    dist_H0hat = nts(mu1=0, tau21=1, mu2=mu2_i, tau22=tau22, a=0, b=np.infty)
    dist_HAhat = nts(mu1=0, tau21=1, mu2=mu2_i, tau22=tau22, a=-np.infty, b=0)
    dat_test_i['c_alpha'] = dist_H0hat.ppf(alpha).flatten()
    dat_test_i['power'] = dist_HAhat.cdf(dat_test_i['c_alpha'])
        
    # (vi) Run the trial
    yhat_trial_i = mdl.predict(X = xx_trial_i)
    holder_err_trial = []
    for metric, resid_fun in di_resid_fun.items():
        # Generate the errors
        error_trial = resid_fun(yy_trial_i, yhat_trial_i)
        error_trial_mu = error_trial.mean()
        dat_err_trial = pd.DataFrame({'err_trial':error_trial_mu}, index=[metric])
        holder_err_trial.append(dat_err_trial)
    dat_trial_i = pd.concat(objs=holder_err_trial).rename_axis('metric')
    dat_trial_i.reset_index(inplace=True)
    dat_trial_i = dat_trial_i.merge(dat_test_i[cols_test_keep])
    # Get the test statistic (in theory, gamma_m could be updated...)
    dat_trial_i = dat_trial_i.assign(s_Delta = lambda x: (x['err_trial'] - x['mu_Delta']) / x['sigma_m'])
    # Compare to critical value
    dat_trial_i = dat_trial_i.assign(reject = lambda x: x['s_Delta'] < x['c_alpha'])

    # (vii) Compare to the "simulation" OOS
    dat_oos_i = pd.Series({k: getattr(data_generator, f'{k}_oracle') for k in metrics})
    dat_oos_i = dat_oos_i.rename_axis('metric').reset_index()
    dat_oos_i.rename(columns = {0: 'mu_oracle'}, inplace=True)
    dat_trial_i = dat_trial_i.merge(dat_oos_i)
    # Calculate whether the empirical null is True
    dat_trial_i['H0hat'] = dat_trial_i['mu_oracle'] > dat_trial_i['mu_Delta']
    
    # (vii) Record the results 
    dat_trial_i.insert(0, 'sim', i+1)
    holder_loop.append(dat_trial_i)
    
    # Check time
    if (i + 1) % n_checkpoint == 0:
        dtime = time() - stime
        nleft = nsim_regression - (i + 1)
        rate = (i + 1) / dtime
        seta = nleft / rate
        vprint(f'Final alpha = {mdl.model.alpha:.2f}', verbose)
        vprint(f'Empirical error = {dat_trial_i["err_test"].mean():.3f}\n'
               f'empirical H_0 = {dat_trial_i["h0hat"].mean():.3f}\n', 
               verbose)
        vprint(
                f'Expected P(hatH_0|H_0) = {100*dat_trial_i["prob_H0hat"].mean():.1f}%\n'
                f'Critical value: {dat_trial_i["c_alpha"].mean():.4f}\n'
                f'Expected power = {100*dat_trial_i["power"].mean():.1f}%',
            verbose)
        vprint('\n------------\n'
                f'iterations to go: {nleft} (ETA = {seta:.0f} seconds)\n'
                '-------------\n\n\n', verbose=True)
    
    # Rebase/remove learneged classes
    mdl.preprocessor = None
    mdl.model = None

# Merge the loop results
res_reg = pd.concat(holder_loop).reset_index(drop=True)
# res_reg.rename(columns = {'h0hat':'H0hat'}, inplace=True)
# .rename_axis('metric').reset_index()
# Save for later
print(f'Expected H0 rate = {100*res_reg["prob_H0hat"].mean():.1f}%\n'
      f'Actual = {100*res_reg["H0hat"].mean():.1f}%')
print(res_reg.groupby(['H0hat','metric','approach'])[['reject']].agg({'mean','sum','count'}))
res_reg.to_csv(path_reg_results, index = False)
# res_reg = pd.read_csv(path_reg_results)


################################
# --- (3) RESULTS PLOTTING --- #

from utils import add_binom_CI
from mizani.formatters import percent_format

# Clean up text
di_H0hat = {
            True: 'Type-I error', 
            False: 'Power', 
            'reject': '$P(\hat{H}_0 | H_0)$',
            'All': 'Reject $\hat{H}_0$',
            }

# Aggregate performance
res_reject = pd.concat(objs=[res_reg, res_reg.assign(H0hat='All')]).\
                groupby(['H0hat','metric',])['reject'].\
                    agg({'mean','sum','count'})
res_reject = pd.concat(objs = [res_reject.reset_index(),
        res_reg.groupby('metric')['H0hat'].agg({'mean','sum','count'}).reset_index().assign(H0hat='reject')])
res_reject
res_reject.rename(columns = {'mean':'pct', 'sum':'k', 'count':'n'}, inplace=True)
res_reject.reset_index(inplace=True)
res_reject['H0hat'] = pd.Categorical(res_reject['H0hat'],list(di_H0hat)).map(di_H0hat)
res_reject['metric'] = res_reject['metric'].replace(di_metrics)
# Add on binomial confidence intervals
res_reject = add_binom_CI(df=res_reject, cn_den='n', cn_num='k', alpha=alpha)

# Calculate the horizontal expected lines
probH0 = res_reg['probH0'].mean()
probHA = 1 - probH0
betaH0 = 1 - res_reg['power'].mean()
expected_reject = alpha*probH0 + (1-betaH0)*probHA

dat_levels = pd.DataFrame({'H0hat':[False, True, 'All', 'reject'], 
                           'level':[res_reg['power'].mean(), alpha, expected_reject, probH0]})
dat_levels['H0hat'] = pd.Categorical(dat_levels['H0hat'],list(di_H0hat)).map(di_H0hat)


# Plot it
gg_reg_reject = (pn.ggplot(res_reject, pn.aes(x='metric', y='pct')) + 
                 pn.theme_bw() + 
                 pn.geom_point(color='black') + 
                 pn.geom_linerange(pn.aes(ymin='lb', ymax='ub'),color='black') + 
                 pn.facet_wrap('~H0hat', scales='free_y') + 
                 pn.labs(x='Metric', y='Percentage (%)') + 
                 pn.geom_hline(pn.aes(yintercept='level'), data=dat_levels, linetype='--',color='blue') + 
                 pn.scale_y_continuous(labels=percent_format()) + 
                 pn.theme(axis_text_x=pn.element_text(angle=90))
                 )
gg_save('regression_results.png', dir_figs, gg_reg_reject, 8, 6.5)



print('~~~ End of 3_regression.py ~~~')