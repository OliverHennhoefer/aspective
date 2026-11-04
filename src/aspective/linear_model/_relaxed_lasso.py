"""Relaxed Lasso (Meinshausen, Computational Statistics & Data Analysis 2007)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, lasso_path
from sklearn.model_selection import check_cv
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


def _relaxed_fit(X, y, alpha, phi, fit_intercept, max_iter, tol):
    """Select with Lasso(alpha), then refit the selected set with Lasso(phi * alpha)."""
    selector = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
    selector.fit(X, y)
    support = np.flatnonzero(selector.coef_)
    coef = np.zeros(X.shape[1])
    n_iter = selector.n_iter_
    if support.size == 0:
        intercept = float(np.mean(y)) if fit_intercept else 0.0
        return coef, intercept, support, n_iter
    if phi == 0:
        refit = LinearRegression(fit_intercept=fit_intercept)
    elif phi == 1:
        refit = selector
    else:
        refit = Lasso(alpha=phi * alpha, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol)
    if refit is not selector:
        refit.fit(X[:, support], y)
        coef[support] = refit.coef_
        n_iter = max(n_iter, getattr(refit, "n_iter_", 0))
    else:
        coef = selector.coef_.copy()
    intercept = float(np.ravel(refit.intercept_)[0]) if fit_intercept else 0.0
    return coef, intercept, support, n_iter


class RelaxedLasso(RegressorMixin, BaseEstimator):
    """Relaxed Lasso with fixed penalties.

    The lasso uses one penalty both to select variables and to shrink them,
    so a penalty strong enough to drop noise variables also over-shrinks the
    true ones. The relaxed lasso separates the two: ``Lasso(alpha)`` selects
    the support, then the selected variables are refitted with the weaker
    penalty ``phi * alpha``. ``phi=1`` is the lasso and ``phi=0`` is least
    squares on the lasso support.

    Use :class:`RelaxedLassoCV` to choose ``alpha`` and ``phi`` by
    cross-validation.

    Parameters
    ----------
    alpha : float, default=1.0
        Selection penalty, as in :class:`sklearn.linear_model.Lasso`.
    phi : float in [0, 1], default=0.5
        Relaxation factor for the refit.
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    tol : float, default=1e-4

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    support_ : ndarray of int
        Indices of the selected features.
    n_iter_ : int
        Largest number of coordinate descent iterations of the two fits.

    References
    ----------
    N. Meinshausen. Relaxed Lasso. Computational Statistics & Data Analysis,
    52(1):374-393, 2007.

    T. Hastie, R. Tibshirani and R. Tibshirani. Best Subset, Forward Stepwise
    or Lasso? Analysis and Recommendations Based on Extensive Comparisons.
    Statistical Science, 35(4):579-592, 2020.
    """

    _parameter_constraints = {
        "alpha": [Interval(Real, 0, None, closed="neither")],
        "phi": [Interval(Real, 0, 1, closed="both")],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, alpha=1.0, phi=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.phi = phi
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True)
        self.coef_, self.intercept_, self.support_, self.n_iter_ = _relaxed_fit(
            X, y, self.alpha, self.phi, self.fit_intercept, self.max_iter, self.tol
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X @ self.coef_ + self.intercept_


class RelaxedLassoCV(RegressorMixin, BaseEstimator):
    """Relaxed Lasso with ``alpha`` and ``phi`` chosen by cross-validation.

    Parameters
    ----------
    n_alphas : int, default=50
        Number of selection penalties on a log grid below ``alpha_max``.
    eps : float, default=1e-3
        Ratio of the smallest to the largest ``alpha`` on the grid.
    phis : array-like, default=(0, 0.25, 0.5, 0.75, 1)
        Relaxation factors to try.
    cv : int, cross-validation generator or iterable, default=5
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    tol : float, default=1e-4

    Attributes
    ----------
    alpha_ : float
    phi_ : float
    alphas_ : ndarray of shape (n_alphas,)
    mse_path_ : ndarray of shape (n_alphas, n_phis)
        Mean cross-validated squared error for every pair.
    coef_, intercept_, support_, n_iter_ :
        Refit on the full data with the chosen pair.

    References
    ----------
    N. Meinshausen. Relaxed Lasso. Computational Statistics & Data Analysis,
    52(1):374-393, 2007.
    """

    _parameter_constraints = {
        "n_alphas": [Interval(Integral, 1, None, closed="left")],
        "eps": [Interval(Real, 0, 1, closed="neither")],
        "phis": ["array-like"],
        "cv": ["cv_object"],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_alphas=50,
        eps=1e-3,
        phis=(0.0, 0.25, 0.5, 0.75, 1.0),
        cv=5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
    ):
        self.n_alphas = n_alphas
        self.eps = eps
        self.phis = phis
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True)
        phis = np.asarray(self.phis, dtype=float)
        if phis.ndim != 1 or phis.size == 0 or np.any((phis < 0) | (phis > 1)):
            raise ValueError("phis must be a non-empty 1-d sequence of values in [0, 1].")

        yc = y - y.mean() if self.fit_intercept else y
        Xc = X - X.mean(axis=0) if self.fit_intercept else X
        alpha_max = np.max(np.abs(Xc.T @ yc)) / len(y)
        if alpha_max == 0:
            alpha_max = 1.0
        self.alphas_ = np.geomspace(alpha_max, alpha_max * self.eps, self.n_alphas)

        errors = np.zeros((self.n_alphas, phis.size))
        folds = list(check_cv(self.cv).split(X, y))
        for train, test in folds:
            for a, alpha in enumerate(self.alphas_):
                for f, phi in enumerate(phis):
                    coef, intercept, _, _ = _relaxed_fit(
                        X[train], y[train], alpha, phi, self.fit_intercept, self.max_iter, self.tol
                    )
                    resid = y[test] - X[test] @ coef - intercept
                    errors[a, f] += np.mean(resid**2)
        self.mse_path_ = errors / len(folds)
        a, f = np.unravel_index(np.argmin(self.mse_path_), self.mse_path_.shape)
        self.alpha_, self.phi_ = float(self.alphas_[a]), float(phis[f])
        self.coef_, self.intercept_, self.support_, self.n_iter_ = _relaxed_fit(
            X, y, self.alpha_, self.phi_, self.fit_intercept, self.max_iter, self.tol
        )
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X @ self.coef_ + self.intercept_
