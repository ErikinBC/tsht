import os
import glob
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint as prop_CI

def makeifnot(path):
    """Make a folder if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def vprint(stmt:str, verbose:bool) -> None:
    """Print if verbose"""
    if verbose:
        print(stmt)


def unzip_folder(folder: str, wildcard: str = '*.zip'):
    """Find the zip file in the data directory"""
    zip_files = glob.glob(os.path.join(folder, wildcard))

    # Unzip the files
    for zip_filename in zip_files:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(folder)
        os.remove(zip_filename)  # Remove the zip file after extracting


def remove_path(path: str) -> None:
    """Remove a file or directory at the given path."""
    if os.path.isfile(path):
        os.remove(path)  # Remove the file
        print(f"Removed file: {path}")
    elif os.path.isdir(path):
        shutil.rmtree(path)  # Remove the directory and all its contents
        print(f"Removed directory: {path}")
    else:
        print(f"No such file or directory: {path}")


def ret_prod_df(di: dict) -> pd.DataFrame:
    """
    Returns the combinations of all terms in a dictionary
    """
    assert isinstance(di, dict), 'di must be a dict'
    res = pd.core.reshape.util.cartesian_product(list(di.values()))
    res = pd.DataFrame(np.vstack(res).T, columns = list(di.keys()))
    return res

def is_psd_cholesky(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def add_binom_CI(df:pd.DataFrame, cn_den:str, cn_pct:None | str=None, cn_num:None | str=None, alpha:float=0.05, method='beta') -> pd.DataFrame:
    """
    Add on the binomial proportion CI to an existing DataFrame. User must specify either the percent of successes (cn_pct) or number of successes (cn_num)

    Params
    ======
    cn_den:             Column with the number of observations (denominator)
    cn_pct:             Column with the percent of successes
    cn_num:             Column giving the number of successes (numerator)
    alpha:              Type-I error rate
    method:             See proportion_confint(method=...)
    """
    assert (cn_pct is not None) or (cn_num is not None), 'At least one of cn_{pct,num} must be specified'
    num_tries = df[cn_den].copy()
    if cn_pct is not None:  # count can be unknown
        assert isinstance(cn_pct, str), 'if cn_pct is not None, then it must be a string'
        pct_success = df[cn_pct].copy()
        num_successes = pct_success * num_tries
    else:  # percent can be unknwoen
        assert isinstance(cn_num, str), 'if cn_prop is None, then cn_count needs to be a string'
        num_successes = df[cn_num].copy()
        pct_success = num_successes / num_tries
    tmp_CI = prop_CI(count=num_successes, nobs=num_tries, alpha=alpha, method=method)
    tmp_CI = pd.DataFrame(np.vstack(tmp_CI).T, columns=['lb','ub'])
    res = pd.concat(objs=[df, tmp_CI], axis=1)
    assert np.all( (res['lb'] <= pct_success) & (res['ub'] >= pct_success) ), 'Woops! [lb,ub] do not bracket the percent of successes!'
    return res



class ScaledChi2:
    def __init__(self, variance, dof):
        """
        Initialize the Scaled Chi-squared distribution with given parameters.
        
        :param variance: scale factor (multiplies the chi-squared distribution)
        :param dof_nm: degrees of freedom of the chi-squared distribution
        :param seed: random seed for reproducibility
        """
        self.variance = variance
        self.dof = dof
        self.dist = chi2(df=self.dof)
        self.variance_dof = dof / variance

    def rvs(self, nsim:int, seed: int | None = None):
        """
        Generate random variates of the scaled chi-squared distribution.
        
        :param nsim: number of simulations
        :return: array of random variates
        """
        return self.dist.rvs(size=nsim, random_state=seed) / self.variance_dof

    def pdf(self, x):
        """
        Compute the probability density function at x.
        
        :param x: values at which to calculate the PDF
        :return: PDF values
        """
        return self.dist.pdf(x * self.variance_dof) / self.variance

    def cdf(self, x):
        """
        Compute the cumulative distribution function at x.
        
        :param x: values at which to calculate the CDF
        :return: CDF values
        """
        return self.dist.cdf(x * self.variance_dof)

    def ppf(self, q):
        """
        Compute the percent point function (inverse of CDF) at q.
        
        :param q: quantile values
        :return: quantile function values
        """
        return self.dist.ppf(q) / self.variance_dof



def broadcast_for_ineq(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """
    Broadcasts two arrays so that y >= x can be calcualted

    Parameters
    ----------
    x: np.ndarray
        An (k1, ..., kd) array
    y: np.ndarray
        An (j1, ..., jd) array
    
    Returns
    -------
    z: np.ndarray
        An (k1, ..., kd, j1, ..., jd) array
    """
    # Ensure ability to broadcast
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    # Reshape x and y to match the desired output shape
    ndim_x = len(x.shape)
    ndim_y = len(y.shape)
    x_new_shape = x.shape + (1,) * ndim_y
    y_new_shape = (1,) * ndim_x + y.shape
    x_reshaped = np.reshape(x, x_new_shape)
    y_reshaped = np.reshape(y, y_new_shape)

    # Perform the element-wise comparison
    z = y_reshaped >= x_reshaped
    return z


def try_flatten(x):
    """Flatten an array if only a single dimension is not one"""
    xs = np.array(x.shape)
    ns1 = np.sum(xs > 1)
    if ns1 == 1:
        x = x.flatten()
    return x

def rvec(x):
    """Convert to a row vector"""
    if isinstance(x, list):
        x = np.array(x)
    if len(x.shape) == 1:
        return np.atleast_2d(x)
    else:
        return x

def cvec(x):
    """Return as a column vector if 2d or less"""
    if len(x.shape) <= 2:
        z = rvec(x)
        if z.shape[0] == 1:
            z = z.T
        return z
    else:
        return x

def check01(x, inclusive:bool=False, run_assert:bool=True) -> bool | None:
    """Checks that float is between (0-1) (inclusive==False) or [0,1] (inclusive==True)"""
    if inclusive:
        stmt = f'Not all x was between [0,1]: {x}'
        check = (x >= 0) & (x <= 1)
    else:
        stmt = f'Not all x was between (0,1): {x}'
        check = (x > 0) & (x < 1)
    if run_assert:
        assert np.all(check), stmt
    else:
        return check


def check_pos(x:np.ndarray, inclusive:bool=True, run_assert:bool=True):
    """Checks that x>0 or (x>=0 if inclusive==True)"""
    if inclusive:
        stmt = f'Not all x >= 0: {x}'
        check = x >= 0
    else:
        stmt = f'Not all x > 0: {x}'
        check = x > 0
    if run_assert:
        assert np.all(check), stmt
    else:
        return check


def check_pos_int(x:int, inclusive:bool=True, run_assert:bool=True):
    """Checks that x>0 or (x>=0 if inclusive==True) and is an integer"""
    check1 = isinstance(x, int)
    if run_assert:
        check_pos(x, inclusive, run_assert)
        assert check1, f'x needs to be an int: {type(x)}'
    else:
        check2 = check_pos(x, inclusive, run_assert)
        check = check1 and check2
        return check


def check_binary(x):
    """Check that array is all zeros or ones"""
    ux = np.unique(x)
    check = all([z in [0,1] for z in ux])
    return check


# Save plotnine objects, delete existing file
def gg_save(fn, fold, gg, width, height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height, limitsize=False)

