"""
TEST TSHT ON RWD

python3 4_rwd.py
"""

# External modules
import numpy as np
import pandas as pd
from time import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# Internal modules
from parameters import path_clean_age, num_cat_min, m_sim, \
    di_resid_fun, di_metrics, \
    n_train_age, n_test_age, n_trial_age
from utils import FrameCountVectorizer, \
                vprint, \
                find_string_columns, find_numeric_columns, select_k_vals_m_times


#################################
# --- (0) RE-USABLE ENCODER --- #

# Set up encoder for "token1; token2; ..."
semicolon_string_encoder = FrameCountVectorizer(input='content', 
                decode_error='ignore',
                token_pattern = None,
                lowercase=True,
                strip_accents='ascii',
                tokenizer=lambda x: x.split('; '), 
                min_df=num_cat_min,
                binary=True)

# String encoder with normalization
categorical_string_pipeline = Pipeline([
    ('encoder', semicolon_string_encoder), 
    ('scaler', StandardScaler())
])

# Standard normalizations
numeric_pipeline = Pipeline([
    ('scaler', StandardScaler())
])


###########################
# --- (1) AGE DATASET --- #

# (i) Load the dataset
df_age = pd.read_csv(path_clean_age)
y_age = df_age['y'].copy()
df_age.drop(columns='y', inplace=True)
cols_str_age = find_string_columns(df_age)
cols_num_age = find_numeric_columns(df_age) 
print(f'Found {len(cols_str_age)} string columns for age = {cols_str_age}')
print(f'Found {len(cols_num_age)} string columns for age = {cols_num_age}')
n_oos_age = df_age.shape[0] - (n_train_age + n_test_age + n_trial_age)
print(f'A total of {n_oos_age:,} rows to estimate oracle for age dataset')


# Set up the preprocessor the age dataset
preprocessor_age = ColumnTransformer([
    ('num', numeric_pipeline, cols_num_age),
    ('cat', categorical_string_pipeline, cols_str_age)
])

# (ii) Do some simulations
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
mdl = Ridge()
nsim = 25

idx_age = pd.Series(np.arange(df_age.shape[0]))
idx_train_age = n_train_age
idx_test_age = idx_train_age + n_test_age
idx_trial_age = idx_test_age + n_trial_age
sd_iter_age = np.exp(np.linspace(np.log(2), np.log(n_test_age*0.9), 20)).astype(int)
holder_age = []
stime = time()
for i in range(nsim):
    # Create the random splits'
    np.random.seed(i+1)
    idx_shuffle_i = idx_age.sample(frac=1, replace=False)
    idx_train_i = idx_shuffle_i[:idx_train_age].values
    idx_test_i = idx_shuffle_i[idx_train_age:idx_test_age].values
    idx_trial_i = idx_shuffle_i[idx_test_age:idx_trial_age].values
    # Fit some model
    preprocessor_age.fit(X = df_age.iloc[idx_train_i])
    xx_train_i = preprocessor_age.transform(X = df_age.iloc[idx_train_i])
    yy_train_i = y_age.iloc[idx_train_i].values
    mdl.fit(X = xx_train_i, y = yy_train_i)
    xx_test_i = preprocessor_age.transform(X = df_age.iloc[idx_test_i])
    yy_test_i = y_age.iloc[idx_test_i].values
    # Estimate variance for absolute and squared error
    eta_test_ii = mdl.predict(xx_test_i)
    # Estimate the variance in the residual
    holder_metric = []
    for metric, fun in di_resid_fun.items():
        resid_test_i = fun(yy_test_i, eta_test_ii)
        sd_i = np.array([select_k_vals_m_times(resid_test_i, k=k, m=m_sim).std(ddof=1, axis=1).mean() for k in sd_iter_age])
        res_i = pd.DataFrame({'metric':metric, 'sim':i, 'n_test':sd_iter_age, 'sd':sd_i})
        holder_metric.append(res_i)
    res_i = pd.concat(holder_metric).reset_index(drop=True)
    holder_age.append(res_i)
    # Print time to go
    dtime = time() - stime
    nleft = nsim - (i + 1)
    rate = (i + 1) / dtime
    seta = nleft / rate
    vprint('\n------------\n'
                f'iterations to go: {nleft} (ETA = {seta:.0f} seconds)\n'
                '-------------\n\n\n', verbose=True)
# Merge
res_age = pd.concat(holder_age)


########################
# --- (X) PLOTTING --- #

import plotnine as pn
from utils import gg_save
from parameters import dir_figs
from utils import gg_save

tmp_mu = res_age.groupby(['metric', 'n_test'])['sd'].mean().reset_index()
gg_age_sd = (pn.ggplot(res_age, pn.aes(x='n_test', y='sd')) + 
             pn.theme_bw() + 
             pn.facet_wrap('~metric', labeller=pn.labeller(cols=lambda x: di_metrics[x]), scales='free') + 
             pn.geom_line(pn.aes(group = 'sim'), color='grey', alpha=0.5) + 
             pn.geom_line(color='black', size=1.5, data=tmp_mu) + 
             pn.scale_x_log10() + 
             pn.labs(x='Number of test samples', y='Average STD'))
gg_save('age_sd.png', dir_figs, gg_age_sd, 10, 4)


