"""MCP and SCAD penalized least squares (Zhang, AoS 2010; Fan & Li, JASA 2001)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import check_cv
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

_DEFAULT_GAMMA = {"mcp": 3.0, "scad": 3.7}


def _threshold(z, lam, penalty, gamma):
    """Univariate solution for a standardized feature (Breheny & Huang, 2011)."""
    a = abs(z)
    soft = max(a - lam, 0.0)
    if penalty == "mcp":
        value = soft / (1 - 1 / gamma) if a <= gamma * lam else a
    else:  # scad
        if a <= 2 * lam:
            value = soft
        elif a <= gamma * lam:
            value = max(a - gamma * lam / (gamma - 1), 0.0) / (1 - 1 / (gamma - 1))
        else:
            value = a
    return np.copysign(value, z)


def _path(G, xty, alphas, penalty, gamma, max_iter, tol):
    """Coordinate descent along a decreasing path, warm-started; returns (n_alphas, p)."""
    p = len(xty)
    usable = np.diag(G) > 1e-10
    beta = np.zeros(p)
    corr = xty.copy()  # X^T (y - X beta) / n
    coefs = np.empty((len(alphas), p))
    n_iter = 0
    for a_idx, lam in enumerate(alphas):
        for n_iter in range(1, max_iter + 1):
            # One full sweep, then sweeps over the active set until it settles.
            max_change = 0.0
            for j in np.flatnonzero(usable):
                new = _threshold(corr[j] + beta[j], lam, penalty, gamma)
                delta = new - beta[j]
                if delta != 0.0:
                    corr -= delta * G[:, j]
                    beta[j] = new
                    max_change = max(max_change, abs(delta))
            if max_change < tol:
                break
            for _ in range(max_iter):
                active = np.flatnonzero(beta)
                inner = 0.0
                for j in active:
                    new = _threshold(corr[j] + beta[j], lam, penalty, gamma)
                    delta = new - beta[j]
                    if delta != 0.0:
                        corr -= delta * G[:, j]
                        beta[j] = new
                        inner = max(inner, abs(delta))
                if inner < tol:
                    break
        coefs[a_idx] = beta
    return coefs, n_iter


def _standardize(X, y):
    x_mean, y_mean = X.mean(axis=0), y.mean()
    sd = X.std(axis=0)
    scale = np.where(sd > 1e-12 * max(1.0, np.abs(X).max(initial=0.0)), sd, 0.0)
    Z = (X - x_mean) / np.where(scale > 0, scale, 1.0)
    Z[:, scale == 0] = 0.0
    return Z, y - y_mean, x_mean, y_mean, scale


def _fit_path(X, y, alphas, penalty, gamma, max_iter, tol):
    Z, yc, x_mean, y_mean, scale = _standardize(X, y)
    n = len(Z)
    std_coefs, n_iter = _path(Z.T @ Z / n, Z.T @ yc / n, alphas, penalty, gamma, max_iter, tol)
    coefs = std_coefs / np.where(scale > 0, scale, 1.0)
    intercepts = y_mean - coefs @ x_mean
    return coefs, intercepts, n_iter


class _NonconvexMixin:
    def _gamma(self):
        gamma = _DEFAULT_GAMMA[self.penalty] if self.gamma is None else self.gamma
        floor = 1.0 if self.penalty == "mcp" else 2.0
        if gamma <= floor:
            raise ValueError(f"gamma must be greater than {floor:g} for penalty={self.penalty!r}.")
        return gamma

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


_DOC = """
    The lasso penalizes every coefficient by the same amount, so it shrinks
    large, clearly relevant coefficients as much as small ones: the
    estimates are biased, and to keep that bias small the lasso has to
    choose a small penalty that lets in many noise variables. The SCAD
    (Fan & Li, 2001) and MCP (Zhang, 2010) penalties start like the lasso
    near zero but level off, so coefficients larger than ``gamma * alpha``
    are not shrunk at all. Both have the oracle property: with high
    probability they select the true variables and estimate them as well
    as least squares on the true support would. MCP is the "minimax
    concave" penalty, the least concave one that achieves this.

    Fitted by coordinate descent on standardized features, warm-started
    along a decreasing path of penalties, as in the ncvreg package
    (Breheny & Huang, 2011). Coefficients are returned on the original
    scale. Prefer these over the lasso for sparse problems with a few
    strong effects; ``gamma`` near its lower bound approaches best-subset
    selection and is less stable, large ``gamma`` approaches the lasso.
"""

_REFS = """
    References
    ----------
    C.-H. Zhang. Nearly Unbiased Variable Selection under Minimax Concave
    Penalty. The Annals of Statistics, 38(2):894-942, 2010.

    J. Fan and R. Li. Variable Selection via Nonconcave Penalized
    Likelihood and its Oracle Properties. Journal of the American
    Statistical Association, 96(456):1348-1360, 2001.

    P. Breheny and J. Huang. Coordinate Descent Algorithms for Nonconvex
    Penalized Regression, with Applications to Biological Feature
    Selection. The Annals of Applied Statistics, 5(1):232-253, 2011.
"""


class NonconvexPenaltyRegression(_NonconvexMixin, RegressorMixin, BaseEstimator):
    __doc__ = """Least squares with the MCP or SCAD penalty at a fixed strength.
""" + _DOC + """
    Parameters
    ----------
    alpha : float, default=0.1
        Penalty strength ``lambda`` on the standardized scale.
    penalty : {"mcp", "scad"}, default="mcp"
    gamma : float or None, default=None
        Concavity; None uses 3 for MCP and 3.7 for SCAD.
    max_iter : int, default=1000
    tol : float, default=1e-6

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    support_ : ndarray of int
    n_iter_ : int
        Coordinate-descent sweeps at the final penalty.
""" + _REFS

    _parameter_constraints = {
        "alpha": [Interval(Real, 0, None, closed="left")],
        "penalty": [StrOptions({"mcp", "scad"})],
        "gamma": [Interval(Real, 0, None, closed="neither"), None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, alpha=0.1, penalty="mcp", gamma=None, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.penalty = penalty
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        gamma = self._gamma()
        # Approach the requested penalty along a short path, for a better local optimum.
        Z, yc, *_ = _standardize(X, y)
        alpha_max = np.abs(Z.T @ yc).max() / len(Z)
        alphas = np.geomspace(max(alpha_max, self.alpha), max(self.alpha, 1e-12), 10) \
            if alpha_max > self.alpha > 0 else np.array([self.alpha])
        coefs, intercepts, self.n_iter_ = _fit_path(X, y, alphas, self.penalty, gamma,
                                                    self.max_iter, self.tol)
        self.coef_, self.intercept_ = coefs[-1], float(intercepts[-1])
        self.support_ = np.flatnonzero(self.coef_)
        return self


class NonconvexPenaltyRegressionCV(_NonconvexMixin, RegressorMixin, BaseEstimator):
    __doc__ = """MCP or SCAD penalized least squares with the penalty chosen by CV.
""" + _DOC + """
    The penalty is chosen from a path of ``n_alphas`` values, log-spaced
    from the smallest that sets every coefficient to zero down to
    ``eps`` times it (``eps`` defaults to 0.001, or 0.05 when there are
    more features than samples, as in ncvreg), by K-fold cross-validated
    mean squared error.

    Parameters
    ----------
    penalty : {"mcp", "scad"}, default="mcp"
    gamma : float or None, default=None
        Concavity; None uses 3 for MCP and 3.7 for SCAD.
    n_alphas : int, default=50
    eps : float or None, default=None
    cv : int, cross-validation generator or iterable, default=5
    max_iter : int, default=1000
    tol : float, default=1e-6

    Attributes
    ----------
    alpha_ : float
    alphas_ : ndarray of shape (n_alphas,)
    mse_path_ : ndarray of shape (n_alphas, n_folds)
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    support_ : ndarray of int
    n_iter_ : int
        Coordinate-descent sweeps at the final penalty.
""" + _REFS

    _parameter_constraints = {
        "penalty": [StrOptions({"mcp", "scad"})],
        "gamma": [Interval(Real, 0, None, closed="neither"), None],
        "n_alphas": [Interval(Integral, 2, None, closed="left")],
        "eps": [Interval(Real, 0, 1, closed="neither"), None],
        "cv": ["cv_object"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, penalty="mcp", gamma=None, n_alphas=50, eps=None, cv=5, max_iter=1000,
                 tol=1e-6):
        self.penalty = penalty
        self.gamma = gamma
        self.n_alphas = n_alphas
        self.eps = eps
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        gamma = self._gamma()
        n, p = X.shape
        Z, yc, *_ = _standardize(X, y)
        alpha_max = max(np.abs(Z.T @ yc).max() / n, 1e-12)
        eps = (1e-3 if n > p else 5e-2) if self.eps is None else self.eps
        self.alphas_ = np.geomspace(alpha_max, alpha_max * eps, self.n_alphas)
        folds = list(check_cv(self.cv).split(X, y))
        mse = np.empty((self.n_alphas, len(folds)))
        for f, (train, test) in enumerate(folds):
            coefs, intercepts, _ = _fit_path(X[train], y[train], self.alphas_, self.penalty, gamma,
                                          self.max_iter, self.tol)
            pred = X[test] @ coefs.T + intercepts
            mse[:, f] = np.mean((pred - y[test, None]) ** 2, axis=0)
        self.mse_path_ = mse
        best = int(np.argmin(mse.mean(axis=1)))
        self.alpha_ = float(self.alphas_[best])
        coefs, intercepts, self.n_iter_ = _fit_path(X, y, self.alphas_[: best + 1], self.penalty, gamma,
                                      self.max_iter, self.tol)
        self.coef_, self.intercept_ = coefs[-1], float(intercepts[-1])
        self.support_ = np.flatnonzero(self.coef_)
        return self
