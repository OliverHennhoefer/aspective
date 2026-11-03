"""Square-root lasso (Belloni, Chernozhukov & Wang, Biometrika 2011)."""

from numbers import Integral, Real

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class SqrtLasso(RegressorMixin, BaseEstimator):
    """Square-root lasso with its pivotal, data-free penalty.

    The lasso's best penalty is proportional to the noise level ``sigma``,
    which is unknown, so in practice it is tuned by cross-validation. The
    square-root lasso minimizes

        ||y - X beta||_2 / sqrt(n) + (lambda / n) ||beta||_1,

    the root of the mean squared error instead of the mean squared error.
    Its optimal penalty does not depend on ``sigma``: Belloni, Chernozhukov
    and Wang set ``lambda = c sqrt(n) Phi^-1(1 - a / (2 p))`` with ``c = 1.1``
    and ``a = 0.05``, and proved the same near-oracle rates as the lasso
    with known ``sigma``. No cross-validation is needed, so a fit costs a
    few lasso fits instead of a whole cross-validated path.

    Solved by the scaled-lasso iteration of Sun and Zhang (2012), which
    alternates between a lasso fit at penalty ``sigma * lambda / n`` and
    the noise estimate ``sigma = ||y - X beta|| / sqrt(n)``; features are
    standardized and coefficients returned on the original scale.

    Prefer it over :class:`~sklearn.linear_model.LassoCV` when CV is
    expensive or unreliable (small samples, dependent rows) and you want
    a penalty with a theoretical guarantee.

    Parameters
    ----------
    c : float, default=1.1
    a : float in (0, 1), default=0.05
        The penalty level controls false inclusions at about this rate.
    max_iter : int, default=100
        Iterations of the scaled-lasso updates.
    tol : float, default=1e-6

    Attributes
    ----------
    alpha_ : float
        Final lasso penalty, ``sigma_ * lambda / n``, on the standardized scale.
    sigma_ : float
        Estimated noise standard deviation.
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    support_ : ndarray of int
    n_iter_ : int

    References
    ----------
    A. Belloni, V. Chernozhukov and L. Wang. Square-Root Lasso: Pivotal
    Recovery of Sparse Signals via Conic Programming. Biometrika,
    98(4):791-806, 2011.

    T. Sun and C.-H. Zhang. Scaled Sparse Linear Regression. Biometrika,
    99(4):879-898, 2012.
    """

    _parameter_constraints = {
        "c": [Interval(Real, 0, None, closed="neither")],
        "a": [Interval(Real, 0, 1, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, c=1.1, a=0.05, max_iter=100, tol=1e-6):
        self.c = c
        self.a = a
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        n, p = X.shape
        x_mean, y_mean = X.mean(axis=0), y.mean()
        sd = X.std(axis=0)
        scale = np.where(sd > 0, sd, 1.0)
        Z = (X - x_mean) / scale
        yc = y - y_mean
        lam_over_n = self.c * norm.ppf(1 - self.a / (2 * p)) / np.sqrt(n)
        sigma = max(np.sqrt(np.mean(yc**2)), 1e-12)
        lasso = Lasso(fit_intercept=False, warm_start=True, max_iter=10_000)
        for it in range(1, self.max_iter + 1):
            lasso.set_params(alpha=sigma * lam_over_n).fit(Z, yc)
            new_sigma = np.sqrt(np.mean((yc - Z @ lasso.coef_) ** 2))
            done = abs(new_sigma - sigma) <= self.tol * max(sigma, 1e-12)
            sigma = max(new_sigma, 1e-12)
            if done:
                break
        self.n_iter_ = it
        self.sigma_ = float(sigma)
        self.alpha_ = float(sigma * lam_over_n)
        self.coef_ = lasso.coef_ / scale
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        self.support_ = np.flatnonzero(self.coef_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X @ self.coef_ + self.intercept_
