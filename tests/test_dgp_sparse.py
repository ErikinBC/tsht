"""
Checks that the DGP sparse accurately represents the true MSE/MAE of the distribution

python3 -m pytest tests/test_dgp_sparse.py -s
"""

# External modules
import pytest
import numpy as np
from scipy import stats
# Internal modules
from parameters import nsim_unittest, rtol_unittest

params = [
    (-10.0, 0.01, 1),
    (-1.0, 0.5, 2),
    (1.0, 2.0, 10),
    (4.0, 9.2, 1234)
]
@pytest.mark.parametrize("mu, sigma, seed", params)
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
    x.std()
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

