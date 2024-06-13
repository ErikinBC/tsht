"""
Uses two real-world regression datasets to test the TSHT approach

python3 3_regression.py
"""

# Load modules
import numpy as np
import pandas as pd
import plotnine as pn
from time import time
from scipy.stats import binomtest, norm
# Internal
from sntn.dists import nts
from funs_models import elnet_wrapper
from funs_simulations import dgp_sparse_yX
from utils import vprint, gg_save
from parameters import seed, alpha, nsim_regression, \
                        num_imputer_strategy, cat_imputer_strategy, \
                        alphas, di_elnet, \
                        n_train, n_test, m_trial, k_regression, \
                        di_resid_fun, metrics, di_metrics, \
                        s, p, sigma2_u, SigmaX, b0, \
                        path_reg_results, dir_figs
from parameters import verbose_regression as verbose


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

# Run simulation loop
holder_loop = []
stime = time()
for i in range(nsim_regression):
    # (i) Index the different splits
    indices = np.random.permutation(n)
    i_train = n_train
    i_test = n_train+n_test
    i_trial = n_train+n_test+m_trial
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
    
    # (iv) Get test set variance
    yhat_test_i = mdl.predict(X = xx_test_i)
    di_err_sigma_test_i = {k: pd.Series(v(yy_test_i, yhat_test_i)).agg({'mean','std'}) 
                           for k,v in di_resid_fun.items()}
    di_err_sigma_test_i = {k:{'mean':v['mean'],
                              'std':n_test/(n_test-1)*v['std']} 
                              for k,v in di_err_sigma_test_i.items()}
    dat_test_i = pd.DataFrame(di_err_sigma_test_i).T
    dat_test_i.rename(columns={'mean':'err', 'std':'sigma'}, inplace=True)
    dat_test_i['sigma_n'] = dat_test_i['sigma'] / np.sqrt(n_test)
    dat_test_i['sigma_m'] = dat_test_i['sigma'] / np.sqrt(m_trial)

    # (v) Set up empirical null
    Delta_i = k_regression * dat_test_i['sigma_n']
    h0hat_err_i = dat_test_i['err'] + Delta_i
    prob_H0hat_i = np.mean(norm.cdf(-Delta_i / dat_test_i['sigma_n']))
    mu2_i = -Delta_i / dat_test_i['sigma_m']
    dist_H0hat = nts(mu1=0, tau21=1, mu2=mu2_i, tau22=tau22, a=0, b=np.infty)
    dist_HAhat = nts(mu1=0, tau21=1, mu2=mu2_i, tau22=tau22, a=-np.infty, b=0)
    c_alpha = dist_H0hat.ppf(alpha).flatten()[0]
    exp_power = np.mean(dist_HAhat.cdf(c_alpha))
    
    
    # (vi) Run the trial
    yhat_trial_i = mdl.predict(X = xx_trial_i)
    di_err_sigma_trial_i = {k: pd.Series(v(yy_trial_i, yhat_trial_i)).agg({'mean','std'}) 
                            for k,v in di_resid_fun.items()}
    di_err_sigma_trial_i = {k:{'mean':v['mean'],
                            'std':m_trial/(m_trial-1)*v['std']} 
                            for k,v in di_err_sigma_trial_i.items()}
    dat_trial_i = pd.DataFrame(di_err_sigma_trial_i).T
    dat_trial_i.rename(columns={'mean':'err', 'std':'sigma'}, inplace=True)
    # Get the test statistic (in theory, gamma_m could be updated...)
    s_Delta_i = (dat_trial_i['err'] - h0hat_err_i) / dat_test_i['sigma_m']
    # Calculate the p-value
    pval_i = dist_H0hat.cdf(s_Delta_i)
    reject_i = s_Delta_i < c_alpha

    # (vii) Compare to the "simulation" OOS
    err_oos_i = pd.Series({k: getattr(data_generator, f'{k}_oracle') for k in metrics})
    H0hat_i = err_oos_i >= h0hat_err_i

    # (vii) Record the results 
    res_i = pd.DataFrame({'sim':i,
                            'probH0':prob_H0hat_i, 'c_alpha':c_alpha, 'power':exp_power, 
                            'Delta':Delta_i, 's_Delta':s_Delta_i, 
                            'err_test':dat_test_i['err'], 'err_trial':dat_trial_i['err'], 'err_oos':err_oos_i,
                            'sigma_test':dat_test_i['sigma'], 'sigma_trial':dat_trial_i['sigma'],
                            'pval':pval_i, 'reject':reject_i, 'H0hat':H0hat_i})
    holder_loop.append(res_i)
    
    # Check time
    if (i + 1) % 5 == 0:
        dtime = time() - stime
        nleft = nsim_regression - (i + 1)
        rate = (i + 1) / dtime
        seta = nleft / rate
        vprint(f'Final alpha = {mdl.model.alpha:.2f}', verbose)
        vprint(f'Empirical error = {dat_test_i.err.round(3).to_list()}, empirical H_0 = {h0hat_err_i.round(3).to_list()}', verbose)
        vprint(f'Expected P(hatH_0|H_0) = {100*prob_H0hat_i:.1f}%\n'
            f'Critical value: {c_alpha:.3f}, Expected power = {100*exp_power:.1f}%',
            verbose)
        vprint('\n------------\n'
                f'iterations to go: {nleft} (ETA = {seta:.0f} seconds)\n'
                '-------------\n\n\n', verbose=True)
    
    # Rebase/remove learneged classes
    mdl.preprocessor = None
    mdl.model = None

# Merge the loop results
res_reg = pd.concat(holder_loop).rename_axis('metric').reset_index()
# Save for later
print(f'Expected H0 rate = {res_reg.probH0.mean()*100:.1f}%, actual = {res_reg.H0hat.mean()*100:.1f}%')
print(res_reg.groupby(['H0hat','metric',])[['reject']].agg({'mean','sum','count'}))
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