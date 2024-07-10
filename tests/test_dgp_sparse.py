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
params_gen = {'n': 100,
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

    def test_gen_yX_output(self, Sigma_X=1):
        """Does the generated data roughly adhere to what was expected?"""
        # Generate the data
        dgp = dgp_sparse_yX(**params_dgp, Sigma_X=Sigma_X)
        result = dgp.gen_yX(**params_gen)
        # Variance checks

        # Shape/type checks
        assert isinstance(result, tuple), "Expected output to be a tuple"
        assert len(result) == 2, "Expected tuple length to be 2"
        assert isinstance(result[0], np.ndarray), "Expected first element to be a numpy array"
        assert isinstance(result[1], np.ndarray), "Expected second element to be a numpy array"
        


# (i) NEED TO CHECK WITH INTERCEPT (MAKE THIS A DECORATOR!)

# ! REPEAT FOR THE DGP ! #
def test_regression(
                    nsim: int = nsim_unittest,
                    ) -> None:
    # Create generator
    dgp_gen = dgp_sparse_yX(**params_dgp)
    # Draw data and fit linear regression model
    yy, xx = dgp_gen.gen_yX(n=params_dgp['n'], seed=params_dgp['seed'])
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(xx, yy)
    print(np.mean( (yy - linreg.predict(xx))**2 ))
    linreg.coef_ = np.repeat(0, dgp_gen.beta.shape[0])
    print(np.mean( (yy - linreg.predict(xx))**2 ))
    # breakpoint()
    # Calculate the coefficient
    bhat = linreg.coef_
    dgp_gen.calc_risk(beta_hat=bhat, has_intercept=False)

    # Generate "out of sample" data
    breakpoint()
    yy, xx = dgp_gen.gen_yX(n=nsim, seed=params_dgp['seed'])
    # error_oos = dgp_gen.gen_yX(n=nsim, seed=seed, y_only=True)
    # error_oos -= linreg.predict(dgp_gen.gen_yX(n=nsim, seed=seed, X_only=True))
    error_oos = yy - linreg.predict(xx)
    del xx
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
    print(dat_comp)
    breakpoint()
    np.testing.assert_allclose(dat_comp['oracle'], dat_comp['emp'], rtol=0.01)

    assert True



params_theory = [
    (-10.0, 0.01, 1),
    (-1.0, 0.5, 2),
    (1.0, 2.0, 10),
    (4.0, 9.2, 1234)
]
@pytest.mark.parametrize("mu, sigma, seed", params_theory)
def test_theory(mu: int, sigma: int, seed: int) -> None:
    """
    Compares theoretical to empirical mean
    """
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
    pytest.main()