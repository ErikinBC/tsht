"""
Simulate two-stage hypothesis testing for Gaussian mean for the both a known and unknown variance. Code can be run as follows

Examples
========

Call with the default parameters
>>> python3 -W ignore::RuntimeWarning 1_simulations.py

Run with custom parameters
>>> python3 1_simulations.py --mu 1 --theta_high 5 --theta_low -3 --n 96 --m 158 --sigma2 2.7 --k 0.5 --alpha 0.1 --seed 1234 --nsim 10000 --nruns 10
"""

import argparse

def main(mu: float, 
         sigma2: float, 
         theta_low: float, 
         theta_high: float, 
         gamma2_high: float, 
         gamma2_low: float, 
         n: int, 
         m: int, 
         k: float, 
         alpha: float, 
         nsim: int,
         seed: int
        ) -> None:
    """
    Run the two sample Gaussian simulation

    Arguments
    =========
    mu: float
        The location parameter for the X_i r.v.
    sigma2: float
        The variance for X_i
    theta_low: float
        The location parameter for the Z_i r.v. (when E(Z) < E(X))
    theta_high: float
        The location parameter for the Z_i r.v. (when E(Z) > E(X))
    gamma2_high: float
        The variance parameter for the Z_i r.v. (when Var(Z) > Var(X))
    gamma2_low: float
        The variance parameter for the Z_i r.v. (when Var(Z) < Var(X))
    n: int
        The size of the X sample
    m: int
        The size of the Z sample
    k: int
        The number of standard deviations to take (k * sigma_n)
    alpha: float
        Type-I error rate
    nsim: int
        The size of the simulation
    seed: float | None = None
        Reproducability seed
    
    Returns
    =======
    ....
    
    The reject_rate plot shows the distribution of the type-I and II errors compared to the expected level. When the variance is unknown, the expected level is the average of expected type-II errors, which naturally comes about since a different c_alpha will be compared to a different SNTN distribution under hat{H}_A.
    """

    # Input checks
    assert theta_high > theta_low, 'make sure theta_high > theta_low'
    assert theta_high > mu, 'make sure theta_high > mu'
    assert theta_low < mu, 'make sure theta_low < mu'
    assert gamma2_high > sigma2, 'make sure sigma2_high > sigma2'
    assert gamma2_low < sigma2, 'make sure sigma2_low < sigma2'

    # External modules
    import numpy as np
    import pandas as pd
    # Internal modules
    from utils import ret_prod_df
    from parameters import path_simulations
    from funs_simulations import two_stage_mean_estimation

    # Set up arguments that will always be identical
    di_args = {
               'mu':mu, 'sigma2':sigma2,
               'n':n, 'm':m, 'k':k, 'nsim':nsim,
               'alpha':alpha, 'seed':seed, 'verbose':False
               }

    ###############################################
    # ---- (1) COMPARE DIFFERENT MEAN COMBOS ---- #
    
    # Set up the actual mean
    di_mu_baseline = [
        {'theta':mu, 'fix_mu_H0':True},
        {'theta':theta_high, 'fix_mu_H0':True},
        {'theta':theta_low, 'fix_mu_H0':True}
        ]
    
    # Loop over HA for (mu, theta)
    holder_baseline = []
    for di_mu in di_mu_baseline:
        print(f'di_mu = {di_mu}')
        # (i) known variance, equal variance (baseline)
        di_mu_eqvar_known = {**di_args, **di_mu, **{'gamma2':sigma2, 'fix_sigma_H0':True, 'estimate_sig2':False}}
        sim_mu_eqvar_known = two_stage_mean_estimation(**di_mu_eqvar_known).assign(**di_mu_eqvar_known)
        
        # (ii) Set up the comparitive statics
        cn_bool_mu = ['fix_sigma_H0', 'estimate_sig2']
        di_prod_mu = {'gamma2':[gamma2_high, gamma2_low], 
                        'fix_sigma_H0':[True, False],
                        'estimate_sig2':[True, False]}
        df_prod_mu = ret_prod_df(di_prod_mu)
        df_prod_mu[cn_bool_mu] = df_prod_mu[cn_bool_mu].astype(bool)

        # (iii) Loop over each
        holder_mu = []
        for tmp_di in df_prod_mu.to_dict(orient='records'):
            tmp_args = {**di_args, **di_mu, **tmp_di}
            tmp_sim = two_stage_mean_estimation(**tmp_args).assign(**tmp_args)
            holder_mu.append(tmp_sim)
        res_mu = pd.concat(objs=[sim_mu_eqvar_known, pd.concat(holder_mu)])
        res_mu.reset_index(drop=True, inplace=True)
        # Append
        holder_baseline.append(res_mu)
    # Combine final results
    res_sim = pd.concat(holder_baseline).reset_index(drop=True)
    # Save for later
    res_sim.to_csv(path_simulations, index=False)


    ############################
    # --- (2) PLOT RESULTS --- #

    # Load modules and tidy strings
    import plotnine as pn
    from mizani.formatters import percent_format
    from utils import makeifnot, gg_save
    from parameters import dir_figs, fn_fig_sims
    # Set up folder
    makeifnot(dir_figs)

    # Clean up plotting headers
    di_H0hat = {'power': 'Power', 
                'fpr': 'Type-I error'}
    di_H0_mu = {'mu_eq': '$H_0: \\mu = \\theta$',
               'mu_gt': '$H_A: \\mu > \\theta$',
               'mu_lt': '$H_A: \\mu < \\theta$'}
    di_H0_sigma = {'sigma_eq': '$\\sigma = \\gamma$',
                'sigma_gt': '$\\sigma > \\gamma$',
                'sigma_lt': '$\\sigma < \\gamma$'}

    # Get relevant columns
    col_plotting = ['H0hat', 'level', 'prop_reject', 
                    'mu', 'theta', 'sigma2', 'gamma2', 
                    'fix_mu_H0', 'fix_sigma_H0', 'estimate_sig2']
    dat_plotting = res_sim[col_plotting].copy()
    dat_plotting = dat_plotting.assign(H0_mu = lambda x: np.sign(x['mu'] - x['theta']),
                                       H0_sigma = lambda x: np.sign(x['sigma2'] - x['gamma2']))
    dat_plotting['H0_mu'] = dat_plotting['H0_mu'].map({0:'mu_eq', 1:'mu_gt', -1:'mu_lt'})
    dat_plotting['H0_sigma'] = dat_plotting['H0_sigma'].map({0:'sigma_eq', 1:'sigma_gt', -1:'sigma_lt'})
    # Calculate the differene relative to the level
    dat_plotting = dat_plotting.assign(gg=lambda x: x['estimate_sig2'].astype(str) + x['fix_sigma_H0'].astype(str))
    di_gg = {
            'FalseTrue': '$\\sigma \\text{ fixed}$',
            'FalseFalse': '$\\sigma, \\gamma$',
            'TrueTrue': '$\\hat\\sigma \\text{ fixed}$',
            'TrueFalse': '$\\hat\\sigma, \\hat\\gamma$',
            }
    dat_plotting['gg'] = pd.Categorical(dat_plotting['gg'], list(di_gg))

    lbl_H0hat = lambda x: di_H0hat[x]
    lbl_H0_mu = lambda x: di_H0_mu[x]
    lbl_H0_sigma = lambda x: [di_H0_sigma[z] for z in x]
    lbl_gg = lambda x: [di_gg[z] for z in x]
    dat_plotting.loc[0]
    legname = 'Variance'
    posd = pn.position_dodge(0.5)
    dat_plotting.loc[1]
    colz = {'FalseTrue':'red', 'FalseFalse':'red', 'TrueTrue':'blue', 'TrueFalse':'blue'}
    shapz = {'FalseTrue':'o', 'FalseFalse':'^', 'TrueTrue':'o', 'TrueFalse':'^'}
    gg_mu_sims = (pn.ggplot(dat_plotting, pn.aes(x='H0_sigma', y='prop_reject-level', color='gg', shape='gg')) + 
          pn.theme_bw() + 
          pn.geom_point(position = posd, size=2) + 
          pn.facet_grid('H0hat~H0_mu', labeller=pn.labeller(cols=lbl_H0_mu, rows=lbl_H0hat)) + 
          pn.scale_color_manual(values=colz,name=legname, labels=lbl_gg) + 
          pn.scale_shape_manual(values=shapz,name=legname, labels=lbl_gg) +
          pn.labs(x='$H_A: (\\sigma, \\gamma)$', y='Deviation to expected level') + 
          pn.scale_y_continuous(labels=percent_format()) + 
          pn.geom_hline(yintercept=0, linetype='--') + 
          pn.scale_x_discrete(labels = lbl_H0_sigma) + 
          pn.theme(axis_text_x=pn.element_text(angle=90)))
    gg_save(fn_fig_sims, dir_figs, gg_mu_sims, 9, 5)
    

    # def plt_pvalues(df: pd.DataFrame, fn: str, fold: str, h: int, w: int,
    #                 facet_term: str) -> None:
    #     """
    #     How does the distribution of p-values look, where the p-values come from the CDF of the binomial distribution compared to the expected proportion in the type-I and type-II errors
    #     """
    #     txt_pvalues = df.\
    #         groupby(cn_gg).\
    #         apply(lambda x: pd.DataFrame({'lt': (x['pval'] < alpha/2).mean(),
    #                                     'gt': (x['pval'] > 1-alpha/2).mean()},index=[0]),
    #             include_groups=False).\
    #         assign(eq=lambda x: 1 - (x['lt'] + x['gt'])).\
    #         melt(ignore_index=False, var_name='xpos', value_name='pct').\
    #         assign(x=lambda z: z['xpos'].map({'lt':0, 'gt':1, 'eq':0.5})).\
    #         reset_index().drop(columns=f'level_{len(cn_gg)}')

    #     gg_pvalues = (pn.ggplot(df, pn.aes(x='pval')) + 
    #                         pn.theme_bw() + 
    #                         pn.geom_histogram(pn.aes(y='stat(width*density)'), fill='grey', color='blue', alpha=0.5, breaks=list(np.arange(0,1.05,0.05)), binwidth=None, boundary=None) + 
    #                         pn.facet_grid(f'{facet_term}~H0hat',scales='free',labeller=pn.labeller(cols=lambda x: di_H0hat[x], rows=lambda x: di_facet_term[x])) + 
    #                         pn.labs(x='One-sided p-value (binomial proportion)', y='Frequency') + 
    #                         pn.scale_x_continuous(labels=percent_format()) + 
    #                         pn.geom_text(pn.aes(label='100*pct',x='x',y=0.07),format_string='{:.1f}%',data=txt_pvalues, size=8, color='red') + 
    #                         pn.geom_vline(xintercept=[alpha/2, 1-alpha/2],size=2,color='red'))
    #     gg_save(fn, fold, gg_pvalues, w, h)



if __name__ == "__main__":
    # Load the defaults from parameters
    from parameters import seed, nsim_sims, alpha, \
                            mu_sims, theta_high_sims, theta_low_sims, \
                            n_sims, m_sims, sigma2_sims, k_sims, \
                            gamma2_high_sims, gamma2_low_sims
    

    parser = argparse.ArgumentParser(description='Script for running the two-sample Gaussian distribution')
    parser.add_argument('--mu', type=float, default=mu_sims, help='Mean of the X (first) sample')
    parser.add_argument('--theta_high', type=float, default=theta_high_sims, help='Mean of the Z (second) sample when H_0 is false (i.e. theta > mu)')
    parser.add_argument('--theta_low', type=float, default=theta_low_sims, help='Mean of the Z (second) sample when H_0 is false (i.e. theta < mu)')
    parser.add_argument('--n', type=int, default=n_sims, help='Size of X')
    parser.add_argument('--m', type=int, default=m_sims, help='Size of Z')
    parser.add_argument('--sigma2', type=float, default=sigma2_sims, help='Variance of X_i (possibly Z_i)')
    parser.add_argument('--gamma2_high', type=float, default=gamma2_high_sims, help='Variance of Z_i when H_0 is false')
    parser.add_argument('--gamma2_low', type=float, default=gamma2_low_sims, help='Variance of Z_i when H_0 is false')
    parser.add_argument('--k', type=float, default=k_sims, help='How many (normalized) standardized deviations to set Delta')
    parser.add_argument('--alpha', type=float, default=alpha, help='Type-I error rate control')
    parser.add_argument('--seed', type=int, default=seed, help='Reproducability seed value')
    parser.add_argument('--nsim', type=int, default=nsim_sims, help='Number of simulations')


    args = parser.parse_args()
    main(mu = args.mu, 
         sigma2 = args.sigma2, 
         theta_low = args.theta_low, 
         theta_high = args.theta_high, 
         gamma2_high = args.gamma2_high, 
         gamma2_low = args.gamma2_low, 
         n = args.n, 
         m = args.m, 
         k = args.k, 
         alpha = args.alpha, 
         seed = args.seed, 
         nsim = args.nsim
        )
    
    print('~~~ End of 1_simulations.py ~~~')