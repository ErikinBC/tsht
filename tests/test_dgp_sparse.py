"""
Checks that the DGP sparse accurately represents the true MSE/MAE of the distribution
"""

# External modules
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import toeplitz
from sklearn.linear_model import LinearRegression
# Internal modules
from funs_simulations import dgp_sparse_yX
from parameters import nsim_unittest, rtol_unittest, seed

# Set up the DGP parameters
params_dgp = {'s': 2, 
              'p': 5, 
              'sigma2_u': 4, 
              'seed': seed}
params_train = {'n': 1000,
               'seed': seed}
params_oos = {'n': nsim_unittest+1,
               'seed': seed}

p_seq = np.linspace(1, 0, num=params_dgp['p'])
ar_covar = toeplitz(p_seq, p_seq)
# Define the different covariance matrices to test
#   input: what goes into the dgp_sparse_yX class
#   output: what we expect the self.Sigma_X to look like
di_SigmaX = {
            'float': {'input': 1.7, 
                       'output': np.diag(0.7+np.ones(params_dgp['p']))}, 
            'vector': {'input': 1+np.arange(params_dgp['p']), 
                        'output': np.diag(1+np.arange(params_dgp['p']))},
            'matrix': {'input': ar_covar, 
                        'output':ar_covar}
            }
# Set up the pytest parameters
SigmaX_params = pytest.mark.parametrize("Sigma_X", list(di_SigmaX.values()))


class TestDGPSparseYX:
    """Checks that the class works as expected"""
    @SigmaX_params
    def test_SigmaX(self, Sigma_X):
        """Does the covariance matrix look as expected?"""
        dgp = dgp_sparse_yX(**params_dgp, Sigma_X=Sigma_X['input'])
        assert np.all(dgp.Sigma_X == Sigma_X['output']), 'Sigma_X does not look as expected'

    @SigmaX_params
    def test_gen_yX_output(self, Sigma_X, alpha: float = 0.05):
        """Does the generated data roughly adhere to what was expected?"""
        # Generate the data
        dgp = dgp_sparse_yX(**params_dgp, Sigma_X=Sigma_X['input'])
        result = dgp.gen_yX(**params_train)
        # Shape/type checks
        assert isinstance(result, tuple), "Expected output to be a tuple"
        assert len(result) == 2, "Expected tuple length to be 2"
        assert isinstance(result[0], np.ndarray), "Expected first element to be a numpy array"
        assert isinstance(result[1], np.ndarray), "Expected second element to be a numpy array"
        assert result[0].shape == (params_train['n'],), 'y is not the expected size'
        assert result[1].shape == (params_train['n'], params_dgp['p']), 'X is not the expected size'
        # Variance checks
        sighat2_y = result[0].var(ddof=1)
        sighat2_X = result[1].var(axis=0, ddof=1)
        dof = params_train['n'] - 1
        chi2_quants = stats.chi2(df=dof).ppf([1-alpha/2, alpha/2])
        confint_sigma2_y = dof*sighat2_y / chi2_quants
        assert (dgp.y_var >= confint_sigma2_y[0]) & (dgp.y_var <= confint_sigma2_y[1]), \
            'confidence interval does not contain population parameter!'
        confint_sigma2_X = np.atleast_2d(dof*sighat2_X) / np.atleast_2d(chi2_quants).T
        X_j_vars = np.diag(Sigma_X['output'])
        assert np.all((confint_sigma2_X[0] <= X_j_vars) & (confint_sigma2_X[1] >= X_j_vars)), \
            'confidence internval does not contain X_j variance == 1'
        
    # (i) NEED TO CHECK WITH INTERCEPT (MAKE THIS A DECORATOR!)
    @SigmaX_params
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_calc_risk(self, Sigma_X, fit_intercept: bool):
        """
        Checks whether the calc_risk function gets the right mean and variance of the error function
        """
        # Generate the training data
        dgp_gen = dgp_sparse_yX(**params_dgp, Sigma_X=Sigma_X['input'])
        y_train, x_train = dgp_gen.gen_yX(**params_train)
        # Fit linear regression model
        linreg = LinearRegression(fit_intercept=fit_intercept)
        linreg.fit(x_train, y_train)
        del y_train, x_train
        # Introduce some more noise in the intercept
        if fit_intercept:
            linreg.intercept_ = stats.norm().rvs(1, random_state=1)[0]
        # Calculate the risk metrics
        if fit_intercept:
            bhat = np.concatenate(([linreg.intercept_], linreg.coef_))
        else:
            bhat = linreg.coef_
        dgp_gen.calc_risk(beta_hat=bhat, has_intercept=fit_intercept)
        # Generate "out of sample" data
        y_oos, x_oos = dgp_gen.gen_yX(**params_oos)
        eta_oos = linreg.predict(x_oos)
        error_oos = y_oos - eta_oos
        del y_oos, x_oos
        # Calculate the empirical mean and variance of the squared and absolute errors
        mse_sim_mu = np.mean( (error_oos)**2 )
        mae_sim_mu = np.mean( np.abs(error_oos) )
        mse_sim_var = np.var( (error_oos)**2 , ddof=1)
        mae_sim_var = np.var( np.abs(error_oos), ddof=1)
        
        # Compare theory to empirical
        dat_emp = pd.DataFrame({'metric':['mse','mae'], 
                    'risk':[mse_sim_mu, mae_sim_mu],
                    'variance':[mse_sim_var, mae_sim_var]})
        dat_comp = dgp_gen.oracle.\
            melt('metric', ['risk','variance'], value_name='oracle').\
            merge(dat_emp.melt('metric', ['risk','variance'], value_name='emp'))
        np.testing.assert_allclose(dat_comp['oracle'], dat_comp['emp'], rtol=rtol_unittest)


params_theory = [
    (-10.0, 0.01, 1),
    (-1.0, 0.5, 2),
    (1.0, 2.0, 10),
    (4.0, 9.2, 1234)
]
@pytest.mark.parametrize("mu, sigma, seed", params_theory)
def test_theory(mu: int, sigma: int, seed: int) -> None:
    """Compares theoretical to empirical mean of the assumed distributional forms"""
    print(f'mu = {mu:.3f}, sigma = {sigma:.3f}, seed = {seed}')
    # Baseline normal distribution
    dist_gauss = stats.norm(loc=mu, scale=sigma)
    dist_chi2 = stats.ncx2(df=1, nc=(mu / sigma) ** 2, loc=0, scale=sigma**2)
    dist_folded = stats.foldnorm(c=np.abs(mu/sigma), loc=0, scale=sigma)

    # Draw data
    x = dist_gauss.rvs(size=nsim_unittest, random_state=seed)
    # Calculate empirical moments
    mu_x = np.mean(x)
    sigma2_x = np.var(x, ddof=1)
    mu_chi2 = np.mean(x**2)
    sigma2_chi2 = np.var(x**2, ddof=1)
    mu_folded = np.mean(np.abs(x))
    sigma2_folded = np.var(np.abs(x), ddof=1)
    # Compare to theory
    np.testing.assert_allclose(mu_x, dist_gauss.mean(), rtol=rtol_unittest)
    np.testing.assert_allclose(sigma2_x, dist_gauss.var(), rtol=rtol_unittest)
    np.testing.assert_allclose(mu_chi2, dist_chi2.mean(), rtol=rtol_unittest)
    np.testing.assert_allclose(sigma2_chi2, dist_chi2.var(), rtol=rtol_unittest)
    np.testing.assert_allclose(mu_folded, dist_folded.mean(), rtol=rtol_unittest)
    np.testing.assert_allclose(sigma2_folded, dist_folded.var(), rtol=rtol_unittest)


if __name__ == '__main__':
    # Will only execute pytest if code called directly, otherwise will treat as an import
    pytest.main()