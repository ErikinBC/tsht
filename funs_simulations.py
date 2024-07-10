# External modules
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import issymmetric
from typing import Tuple
from scipy.stats import norm, bernoulli, multivariate_normal
# Internal warnings
from sntn.dists import nts
from utils import check01, vprint, ScaledChi2, is_psd_cholesky


class dgp_sparse_yX():
    def __init__(self, 
                 s: int, 
                 p: int, 
                 sigma2_u: float,
                 Sigma_X: float | np.ndarray | None = 1.0, 
                 b0: int = 1.0, 
                 beta: np.ndarray | None = None,
                 seed: int | None = None, 
                 normalize_Sigma_X: bool = False,
                 ):
        """
        Class to help create labels and features in a sparse linear regression data generating process. DGP is:

        y = X'beta + u,
        X ~ MVN(0, Sigma_X)
        u ~ N(0, variance)
        |beta={b: b != 0}| = s
        beta_s ~ Rademacher(0.5) * b0

        Args
        ====
        s: int
            The total number of coefficients with non-zero signal
        p: int
            The total number of coefficients
        sigma2_u: float
            The variance of the residual 
        b0: int = 1
            (Optional) The (absolute) size of the coefficient, defaults to 1
        beta: np.ndarray | None = None
            (Optional) If specified, the coefficient vector to be used (otherwise will be generated)
        Sigma_X: float | np.ndarray | None = None
            (Optional) The covariance matrix of X, if not specified, will be randomly generated, if float, will assume diagonal, if 1-d array, will assume diagnonal as well
        seed: int | None = None
            (Optional) Will seed the results of beta and Sigma_X
        normalize_Sigma_X: bool, optional
            Should we normalize the values of Sigma_X so the largest variance is one?
        """
        # (i) Do input checks
        assert all([z > 0 for z in [s, p, sigma2_u]]), 'n, p, and sigma2 need to be strictly positive'
        assert s < p, 's must be less than p'

        # (ii) Generate the covariance matrix for X
        if Sigma_X is None:
            # Generate a (p,p) covariance matrix
            Sigma_X = norm().rvs([p,p], random_state=seed)
            Sigma_X = Sigma_X.dot(Sigma_X.T)
        else:
            # User has provided a covariance matrix, now we need to check it
            Sigma_X = np.array(Sigma_X)
            n_dims = len(Sigma_X.shape)
            assert n_dims <= 2, f'Sigma_X must either be a float, a (p,) array or a (p,p) array'
            if n_dims == 0:
                Sigma_X = np.diag(np.repeat(Sigma_X, p))
            elif n_dims == 1:
                # Convert into a 
                Sigma_X = np.diag(Sigma_X)
            else:
                assert Sigma_X.shape[1] == Sigma_X.shape[0], 'Sigma_X must be a square matrix'
        assert Sigma_X.shape[1] == p, 'Sigma_X must have the same dimensions as p'
        assert issymmetric(Sigma_X), 'Sigma_X must be symmetric'
        # Note PSD ensures off diagnonals cannot be larger (in abs terms) than diagonals
        assert is_psd_cholesky(Sigma_X), 'Sigma_X must be positive semi-definite'
        # Normalize if request
        if normalize_Sigma_X:
            Sigma_X /= np.max(np.abs(Sigma_X))
        # Determine if the covariance is diagonal only
        self.diagonal_Sigma = np.all(Sigma_X[~np.eye(p, dtype=bool)] == 0)

        # (iii) Generate coefficients
        if beta is not None:
            beta = np.atleast_1d(beta)
            assert beta.shape[0] == p, 'beta needs to be the same length as p'
        else:
            # Draw beta from a b0 * Rademacher(1/2) distribution
            beta = bernoulli(p=0.5).rvs(s, random_state=seed) 
            beta = np.where(beta == 0, -b0, +b0)
            beta = np.concatenate((beta, np.repeat(0, p - s)))

        # Assign as attributes        
        self.p = p
        self.s = s
        self.k = p - s
        self.Sigma_X = Sigma_X
        self.b0 = b0
        self.beta = beta
        self.sigma2_u = sigma2_u
        self.dist_u = norm(loc=0, scale = np.sqrt(sigma2_u))
        mu = np.repeat(0, p)
        self.dist_X = multivariate_normal(mean = mu, cov = Sigma_X)
        self.intercept = 0
        # Calculate variance of X'beta ~ N(0, beta'Sigma beta)
        if self.diagonal_Sigma:
            self.eta_var = np.sum( np.diag(Sigma_X) * beta**2 )
        else:
            self.eta_var = Sigma_X.dot(beta).dot(beta)
        self.y_var = self.eta_var + self.sigma2_u


    def _check_X(self, X: np.ndarray) -> None:
        """Make sure X is the right shape"""
        assert len(X.shape) == 2, 'X needs to be a matrix'
        assert X.shape[1] == self.p, f'if you provide X, it needs to have {self.p} columns'

    def gen_yX(self, 
               n: int, 
               X: np.ndarray | None = None,
               seed: int | None = None,
               y_only: bool = False,
               X_only: bool = False,
               ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Draw from the data generating process

        n:int
            The sample size, or the number of draws for X
        X: np.ndarray | None = None
            (Optional) Pre-specify the X to use, in which case hard code n == len(X), and it will return a y of length n
        seed: int | None = None
            (Optional) Seeding all scipy.dists calls (u, X)
        """
        assert n > 0, 'n needs to be a positive int'
        if X is None:  # Draw everything from scratch
            X = self.dist_X.rvs(size = n, random_state = seed)
            u = self.dist_u.rvs(size = n, random_state = seed)
            eta = X.dot(self.beta) + self.intercept
            y = eta + u
            if y_only:
                return y
            elif X_only:
                return X
            else:
                return y, X
        else:  # Draw from ~ | X
            self._check_X(X)
            U = self.dist_u.rvs(size = (X.shape[0], n), random_state=seed)
            Eta = X.dot(np.atleast_2d(self.beta).T) + self.intercept
            Y = Eta + U
            return Y


    def calc_risk(self,
                 beta_hat: np.ndarray,
                 has_intercept: bool = True,
                #  X: np.ndarray | None = None,
                 ) -> None:
        """
        Calculates the MSE and MAE for a given linear regression model.

        The distribution of the residuals is:
            e = u + X'(beta - bhat)
            e ~ N(0, variance_u + |beta - hat{beta}|^2_2)
        e^2 ~ nc_chi2(df=1, nc=(mu / sigma)**2), scale=
            
        Args
        ====
        beta_hat: np.ndarray
            The estimated coefficients
        has_intercept: bool = True
            Whether beta_hat has an intercept included in the first position
        X: np.ndarray | None = None
            Should the conditional values also be calculated for X?
        """
        # (i) Input checks
        if has_intercept:
            beta_hat0 = beta_hat[0]
            beta_hat = beta_hat[1:]
        else:
            beta_hat0 = 0
        assert beta_hat.shape[0] == self.beta.shape[0], f'With has_intercept={has_intercept}, there should be {self.p}+1 parameters in beta_hat, not: {beta_hat.shape[0]+1}'
        beta_err = self.beta - beta_hat
        l2_beta_err = np.sum(beta_err**2)
        
        # (ii) Calculate risk averaging over X
        total_var = self.sigma2_u + l2_beta_err
        total_sd = np.sqrt(total_var)
        mu = self.intercept - beta_hat0
        dist_chi2 = stats.ncx2(df=1, nc=mu**2 / total_var, loc=0, scale=total_var)
        dist_folded = stats.foldnorm(c=np.abs(mu/total_sd), loc=0, scale=total_sd)
        mse_mu = dist_chi2.mean()
        mae_mu = dist_folded.mean()

        # Calculate the variance of the residuals
        mse_var = dist_chi2.var()
        mae_var = dist_folded.var()
        # Store if a DataFrame
        self.oracle = pd.DataFrame({'conditional': False,
                                    'metric':['mse', 'mae'],
                                    'risk': [mse_mu, mae_mu],
                                    'variance': [mse_var, mae_var]})
        
        # # (iii) Calculate risk conditional on X (optional)
        # self.mse_x_oracle = None
        # self.mae_x_oracle = None
        # if X is not None:
        #     self._check_X(X)
        #     n_X = len(X)
        #     if has_intercept:
        #         iX = np.hstack((np.ones([n_X,1]), X))
        #     else:
        #         iX = X.copy()
        #     err_X = beta_err.dot(iX.T.dot(iX)).dot(beta_err)
        #     eta_err = iX.dot(beta_err)
        #     # (i) Calculate expectations
        #     # MSE is residual variance plus 
        #     self.mse_x_oracle = self.sigma2_u + err_X / n_X
        #     # Calculate the folded normal expectation
        #     self.mae_x_oracle = np.sqrt(self.sigma2_u * 2 / np.pi) * np.exp(-eta_err**2/(2*self.sigma2_u)) + eta_err*(1 - 2*norm.cdf(-eta_err / np.sqrt(self.sigma2_u)))
        #     self.mae_x_oracle = np.sum(self.mae_x_oracle) / n_X
        #     # Append on
        #     oracle_x = pd.DataFrame({'conditional': False,
        #                             'metric':['mse', 'mae'],
        #                             'risk': [mse_mu, mae_mu],
        #                             'variance': []})
        #     self.oracle = pd.concat(objs = [self.oracle, oracle_x])
        #     self.oracle.reset_index(drop=True, inplace=True)



def two_stage_mean_estimation(
                    mu: float, 
                    theta: float, 
                    sigma2: float,
                    gamma2: float,
                    n: int, 
                    m: int, 
                    k: int, 
                    nsim: int,            
                    alpha: float, 
                    fix_mu_H0: bool = False,
                    fix_sigma_H0: bool = False,
                    estimate_sig2: bool = False,
                    # pool_var: bool = False,
                    ret_raw: bool = False,
                    seed: float | None = None,
                    verbose: bool = False
                ) -> pd.DataFrame:
    """
    Function to carry out simulations for a two-stage hypothesis test bounding the mean from a second sample. Assume that we have two samples X=(X_1, ..., X_n) and Z=(Z_1, ..., Z_m), X_i ~ N(mu, sigma2), Z_i ~ N(theta, gamma2)

    Inputs
    ======
    mu: float
        The location parameter for the X_i r.v.
    theta: float
        The location parameter for the Z_j r.v.
    sigma2: float
        The variance parameter for the X_i r.v.
    gamma2: float
        The variance parameter for the Z_j r.v.
    n: int
        The size of the X sample
    m: int
        The size of the Z sample
    k: int
        The number of standard deviations to take (i.e. sigma_n; possibly estimated)
    nsim: int
        The size of the simulation
    alpha: float=0.1
        Type-I error rate
    fix_mu_H0: bool = True
        Should mu = theta be assumed in H0?
    fix_sigma_H0: bool = True
        shuuld sigma = gamma be assumed in H0?
    estimate_sig2: bool = False
        Should the variance term be estimated?
    ret_raw: bool = False
        Should the raw simulation results be returned? (i.e. no aggregation)
    seed: float | None = None
        Reproducability seed
    verbose: bool = False
        Should some of the intermediate results be printed
    
    Returns
    =======
    A DataFrame containing the simulation experiments with columns:
        H0hat: Whether the empirical null of theta >= mu_Delta is true
        n_reject: The number of rejected empirical null hypotheses
        n_test: The number of hypothesis tests
        prop_reject: The proportion of rejected empirical nulls
        level: The expected level (based on theoretical distribution)
        pval: Probability we would object 'prop_reject' given a binomial distribion with 'level'

    Example
    =======
    >>> two_stage_mean_estimation(mu=1, theta=1.5, sigma2=3.5, gamma2=2.5, n=50, m=75, k=1, alpha=0.1, nsim=10000, seed=123)
    H0hat  n_test  prop_reject  n_reject     level      pval
    0      power    8443     0.701765      5925  0.703175  0.784680
    1        fpr    1557     0.105331       164  0.100000  0.454186
        
    """

    # (i) Input checks and transformations
    assert all([x >= 0 for x in [n, m, k, sigma2, gamma2, seed]]), 'n, m, k, seed, sigma2 need to be strictly positive'
    assert all([isinstance(x, int) for x in [n, m]]), 'n and m need to be ints'
    check01(alpha, inclusive=False, run_assert=True)
    dof_n = n - 1
    dof_m = m - 1
    # var_diff = sigma2 != gamma2
    # if pool_var:
    #     dof_n += m  # Add on the Z samples
    #     assert not var_diff, 'If pool_var==True, then make sure sigma2 == gamma2'

    # (ii) Calculate oracle terms
    sigma_n_oracle = np.sqrt(sigma2 / n)
    gamma_m_oracle = np.sqrt(gamma2 / m)
    sigma_m_oracle = np.sqrt(sigma2 / m)
    Delta = k * sigma_n_oracle  # technically couldn't be known in advance, but in practice this is just a constant so doesn't matter for simulation results
    prob_H0hat_oracle = stats.norm.cdf(-(mu + Delta - theta) / sigma_n_oracle)
    vprint(f'Oracle of empirical null being true = {100*prob_H0hat_oracle:0.1f}% with right threshold', verbose)
    # Actual distributions
    dist_Xbar = stats.norm(loc=mu, scale=sigma_n_oracle)
    dist_Zbar = stats.norm(loc=theta, scale=gamma_m_oracle)
    dist_sigma2hat = ScaledChi2(variance=sigma2, dof=dof_n)
    dist_gamma2hat = ScaledChi2(variance=gamma2, dof=dof_m)
    # SNTN distribution (oracle)
    di_dist_base = {'mu1':0, 'tau21':1}
    di_H0_ub = {'a':0, 'b':np.infty}
    di_H0_lb = {'a':-np.infty, 'b':0}
    mu2_oracle = -(mu + Delta - theta) / gamma_m_oracle
    tau22_oracle = (sigma_n_oracle / gamma_m_oracle) ** 2
    di_tnorm_oracle = {'mu2':mu2_oracle, 'tau22':tau22_oracle}
    di_dist_H0hat_oracle = {**di_dist_base, **di_tnorm_oracle, **di_H0_ub}
    di_dist_HAhat_oracle = {**di_dist_base, **di_tnorm_oracle, **di_H0_lb}
    dist_H0hat_oracle = nts(**di_dist_H0hat_oracle)
    dist_HAhat_oracle = nts(**di_dist_HAhat_oracle)
    c_alpha_oracle = dist_H0hat_oracle.ppf(alpha).squeeze()
    power_oracle = dist_HAhat_oracle.cdf(c_alpha_oracle).mean()

    # (iii) Draw the samples
    # Draw the variances
    if estimate_sig2:
        # Drawn around a chi2 distribution
        sigma2_hat = dist_sigma2hat.rvs(nsim, seed=seed+1)
        gamma2_hat = dist_gamma2hat.rvs(nsim, seed=seed+2)
        sigma_n = np.sqrt( sigma2_hat / n )
        sigma_m = np.sqrt( sigma2_hat / m )
        gamma_m = np.sqrt( gamma2_hat / m )
    else:
        # Variance are drawn as "oracle"
        sigma_n = sigma_n_oracle
        gamma_m = gamma_m_oracle
        sigma_m = sigma_m_oracle
    # Draw the sample means
    mu_hat = dist_Xbar.rvs(nsim, random_state=seed+3)
    theta_hat = dist_Zbar.rvs(nsim, random_state=seed+4)
    # Calculate the s_Delta test statistic
    mu_hat_Delta = mu_hat + Delta
    s_Delta = (theta_hat - mu_hat_Delta) / gamma_m
    df_sim = pd.DataFrame({'mu_hat_Delta':mu_hat_Delta, 
                           'theta_hat':theta_hat, 
                           's_Delta':s_Delta,
                           'H0hat': theta >= mu_hat_Delta  # Note that this is an oracle property
                           })
    # Check that oracle type-I and type-II error align
    oracle_typeI = (df_sim.query('H0hat')['s_Delta'] < c_alpha_oracle).mean()
    oracle_typeII = 1 - (df_sim.query('~H0hat')['s_Delta'] < c_alpha_oracle).mean()
    vprint(f'Expected type-I = {100*alpha:0.1f}%, actual = {100*oracle_typeI:0.1f}%', verbose)
    vprint(f'Expected type-II = {100*(1-power_oracle):0.1f}%, actual = {100*oracle_typeII:0.1f}%', verbose)
    
    # (iv) Construct the test distributions    
    if fix_mu_H0 and fix_sigma_H0:
        # We're assuming that mu=theta and sigma=gamma, whether or not that's true
        mu2 = -Delta / sigma_m  # mu - theta cancels out, sigma is used in place of gamma
        tau22 = m / n  # sigma/gamma cancel out
    elif fix_mu_H0 and not fix_sigma_H0:
        # We're assuming that mu=theta, but allowing gamma and sigma to vary and be estimated after seeing Z
        mu2 = -Delta / gamma_m  # mu - theta cancel out, gamma is allowed to be be different than sigma
        tau22 = (sigma_n / gamma_m) ** 2
    elif not fix_mu_H0 and fix_sigma_H0:
        # We somehow know mu!=theta, and forcing sigma = gamma
        mu2 = -(mu + Delta - theta) / sigma_m  # sigma is used in place of gamma
        tau22 = m / n  # sigma and gamma cancel out
    else: # not fix_mu_H0 and not fix_sigma_H0:
        # We somehow know mu!=theta, and allowing gamma and sigma to vary and (possibly) be estimated after seeing Z
        mu2 = -(mu + Delta - theta) / gamma_m  # all terms allowed
        tau22 = (sigma_n / gamma_m) ** 2  # all terms allowed
    di_tnorm = {'mu2':mu2, 'tau22':tau22}
    di_dist_H0hat = {**di_dist_base, **di_tnorm, **di_H0_ub}
    di_dist_HAhat = {**di_dist_base, **di_tnorm, **di_H0_lb}
    dist_H0hat = nts(**di_dist_H0hat)
    dist_HAhat = nts(**di_dist_HAhat)
    # Calculate P(hat{H}_0) 
    prob_H0hat_exp = np.mean(stats.norm.cdf(mu2))
    prob_HAhat_exp = 1 - prob_H0hat_exp
    # Set up the critical value expected to deliver a given power
    c_alpha_exp = dist_H0hat.ppf(alpha).squeeze()
    power_exp = dist_HAhat.cdf(c_alpha_oracle).mean()
    vprint(f'\nfix_mu_H0={fix_mu_H0}, fix_sigma_H0={fix_sigma_H0}\n'
           f'c_alpha: oracle={c_alpha_oracle:.3f}, expected={np.mean(c_alpha_exp):.3f}\n'
           f'power: oracle={power_oracle:.3f}, expected={power_exp:.3f}\n',
           verbose)
    
    # (v) Run the test
    df_sim['reject'] = df_sim['s_Delta'] < c_alpha_exp
    df_sim['pval'] = dist_H0hat.cdf(df_sim['s_Delta'])
    # Return raw data if needed
    if ret_raw:  
        di_params = {'mu':mu, 'theta':theta, 'sigma2':sigma2, 'gamma2':gamma2, 
                     'n':n, 'm':m, 'alpha':alpha}
        df_sim = df_sim.assign(**di_params)
        return df_sim
    
    # (vi) Calculate coverage
    res_sim = df_sim.groupby('H0hat')['reject'].\
        agg({'count', 'mean', 'sum'}).\
        rename(index={False:'power', True:'fpr'}).\
        reset_index()
    # Rename columns
    di_col_rename = {'count':'n_test', 'mean':'prop_reject', 'sum':'n_reject', }
    res_sim.rename(columns = di_col_rename, inplace=True)
    # Add on what was expected (freq = P(hat{H_{0,A}}))
    dat_typeI_II = pd.DataFrame({'H0hat':['fpr', 'power'], 'level':[alpha, power_exp], 
                                 'freq_exp':[prob_H0hat_exp, prob_HAhat_exp]})
    res_sim = res_sim.merge(dat_typeI_II)
    res_sim = res_sim.assign(freq_act=lambda x: x['n_test'] / nsim)
    
    # Return dataframe
    return res_sim