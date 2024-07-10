"""
Some parameters to be used across scripts
"""

import os
import numpy as np

# Folder names
dir_figs = 'figures'
dir_data = 'data'

# File names
fn_simulations = 'mu_sims.csv'
path_simulations = os.path.join(dir_data, fn_simulations)
fn_fig_sims = 'mu_sims.png'
fn_reg_results = 'reg_results.csv'
path_reg_results = os.path.join(dir_data, fn_reg_results)

# Type-I error target
alpha = 0.1

# Reproducability seed
seed = 1234

# Plotting simulation
nsim_plots = 100000
atol_figs = 0.01    # Comparing simulation means to theory

# Unit testing
nsim_unittest = 1000000
rtol_unittest = 0.011

# 0_datasets
path_url_age = 'https://raw.githubusercontent.com/Moradnejad/AgeDataset/main/AgeDataset.rar'
path_raw_age = os.path.join(dir_data, 'AgeDataset.csv')
path_clean_age = os.path.join(dir_data, 'tidy_age.csv')
colname_y_age = 'death_age'
colname_x_age = ['gender', 'country', 'birth_year', 'occupation', 'death_manner']
di_rename_age = {'Gender':'gender', 
             'Country': 'country', 
             'Birth year': 'birth_year', 
             'Occupation': 'occupation', 
             'Age of death': 'death_age',
             'Death year': 'death_year',
             'Manner of death': 'death_manner'}


# 1_simulations parameters
nsim_sims = 500000
mu_sims = 0.9
sigma2_sims = 2.7
gamma2_high_sims = sigma2_sims * 4
gamma2_low_sims = sigma2_sims / 4
n_sims = 96
m_sims = 158
k_sims = 0.5
sigma_n = (sigma2_sims / n_sims)**0.5
theta_high_sims = mu_sims + k_sims*sigma_n
theta_low_sims = mu_sims - k_sims*sigma_n


# 3_regression
nsim_regression = 100
verbose_regression = True
alphas = [0.0, 0.50, 1.0]
num_imputer_strategy = 'median'
cat_imputer_strategy = 'most_frequent'
k_regression = 1.0
n_train = 250
n_test = 249
m_trial = 251
s = 5
p = 25
sigma2_u = 5
b0 = 1
SigmaX = np.diag(np.ones(p))
di_elnet = {'n_lambda': 25, 
            'min_lambda_ratio': 1e-3, 
            'n_splits': 3,
            'scoring': 'r2'}
# The different residual functions
di_resid_fun = {
    'mse': lambda y, ypred: (y - ypred)**2, 
    'mae': lambda y, ypred: np.abs(y - ypred), 
}
metrics = list(di_resid_fun)
di_metrics = {
    'mse':'MSE', 
    'mae':'MAE'
}


# 4_rwd
num_cat_min = 10  # For one-hot encoding, minimum number of values we'll need to see
m_sim = 1000
n_train_age = 5000
n_test_age = 111115
n_trial_age = 5002
