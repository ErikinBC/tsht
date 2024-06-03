"""
scratch work goes here

!!!disable smart send!!!
!!!parameterize the simulation save file and then hash it to compare!!!
"""

import numpy as np
from scipy.stats import norm

seed = 1234
nsim = 10000

# (1) Confirm pooled variance is the same when mu!=theta, but sigma_X=sigma_Y
n = 97
m = 64
sigma_x = 2
sigma_z = 2
sigma_n = sigma_x / np.sqrt(n)
gamma_m = sigma_z / np.sqrt(m)
mu = 1
theta = 1

# Draw some data
dist_X = norm(loc=mu, scale=sigma_x)
dist_Z = norm(loc=theta, scale=sigma_z)


# dist_XZbar = norm(loc=[mu, theta], scale=[sigma_n_oracle, gamma_m_oracle])
# mu_hat, theta_hat = dist_XZbar.rvs([nsim, 2], random_state=seed).T
# dof = n + m - 1
# sigma2_hat = sigma2 * stats.chi2(df=dof).rvs(nsim, random_state=seed) / dof

