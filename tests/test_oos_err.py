"""
Give tests to make sure code is working as expected

python3 -m pytest tests/test_oos_err.py -s -k 'test_indep'
"""

# External
import gc
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
# Internal
from parameters import seed
from funs_simulations import dgp_sparse_yX
from sklearn.metrics import mean_absolute_error, mean_squared_error

di_metrics = {'MSE': mean_squared_error,
              'MAE': mean_absolute_error}

"""
MAKE SURE MSE(BETA_HAT | X) & MSE(BETA_HAT) align with expectations
"""

# Set up the parameters
atol = 0.005
n_sim = 250000
n = 200
s = 5
p = 10
sigma2_u = 5
Sigma_X = np.diag(np.ones(p))
# b0 = 0.0

params_b0 = [ (0.0), (0.5), (1.0) ]

@pytest.mark.parametrize("b0", params_b0)
def test_indep(b0: float) -> None:
    """
    Tests MSE/MAE for uncorrelated columns of X
    """
    print(f'test_indep: b0 = {b0}')
    # (i) Generate the data
    data_generator = dgp_sparse_yX(s = s, p = p, sigma2_u = sigma2_u, b0=b0, Sigma_X=Sigma_X)
    yy, xx = data_generator.gen_yX(n=n, seed=seed+1)
    
    # (ii) Fit the linear regression model
    mdl = LinearRegression().fit(xx, yy)
    beta_hat = np.concatenate(([mdl.intercept_], mdl.coef_))
    eta_hat = mdl.predict(xx)
    
    # (iii) Get empirical error conditional on X
    Y_condX = data_generator.gen_yX(n=n_sim, seed=seed+2, X=xx)
    # E[e_i(theta) | X] = sigma2_u + ||beta - theta||^2_2 (includes intercept...)
    mse_x_emp = np.mean((Y_condX - np.atleast_2d(eta_hat).T)**2)
    mae_x_emp = np.mean(np.abs(Y_condX - np.atleast_2d(eta_hat).T))
    del Y_condX  # Free up space
    gc.collect()

    # (iv) Get empirical over intergrated over X
    yy_oos, xx_oos = data_generator.gen_yX(n=n_sim, seed=seed+3)
    eta_oos = mdl.predict(xx_oos)
    del xx_oos
    # E[e_i(theta) | X] = sigma2_u + 1/n * sum_i (beta-theta)' X_i'X_i (beta - theta)
    mse_emp = mean_squared_error(yy_oos, eta_oos)
    mae_emp = mean_absolute_error(yy_oos, eta_oos)
    del yy_oos, eta_oos
    gc.collect()
    print(f'mse_x_emp={mse_x_emp:.5f}\n mse_emp={mse_emp:.5f}')
    print(f'mae_x_emp={mae_x_emp:.5f}\n mae_emp={mae_emp:.5f}')

    # (v) Call calc_risk method the class method
    data_generator.calc_risk(beta_hat=beta_hat, has_intercept=True, X=xx)
    np.testing.assert_allclose(mse_emp, data_generator.mse_oracle, atol=atol)
    np.testing.assert_allclose(mse_x_emp, data_generator.mse_x_oracle, atol=atol)
    np.testing.assert_allclose(mae_emp, data_generator.mae_oracle, atol=atol)
    np.testing.assert_allclose(mae_x_emp, data_generator.mae_x_oracle, atol=atol)

