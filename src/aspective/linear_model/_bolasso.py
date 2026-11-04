"""Bolasso: bootstrapped Lasso for consistent variable selection (Bach, ICML 2008)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class Bolasso(RegressorMixin, BaseEstimator):
    """Bootstrap-intersected Lasso with a least-squares refit.

    The lasso only selects the true variables consistently under the strict
    irrepresentable condition. Bach (2008) showed that running the lasso on
    bootstrap replicates and keeping the variables selected in all of them
    gives consistent selection under much weaker conditions. The selected
    set is then refitted by ordinary least squares.

    ``threshold=1`` is the strict intersection of the paper; a value like 0.9
    is the "soft" Bolasso-S variant, which is more robust in practice.

    Parameters
    ----------
    alpha : float or None, default=None
        Lasso penalty used on every replicate. When None it is chosen once on
        the full data by :class:`sklearn.linear_model.LassoCV`.
    n_bootstrap : int, default=128
        Number of bootstrap replicates (Bach used 128).
    threshold : float in (0, 1], default=1.0
        Minimum selection frequency for a variable to be kept.
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    alpha_ : float
        Penalty that was used.
    selection_frequencies_ : ndarray of shape (n_features,)
        Fraction of replicates in which each variable was selected.
    support_ : ndarray of int
        Indices of the kept variables.
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    n_iter_ : int
        Largest number of coordinate descent iterations over the replicates.

    References
    ----------
    F. R. Bach. Bolasso: Model Consistent Lasso Estimation through the
    Bootstrap. In Proceedings of the 25th International Conference on Machine
    Learning (ICML), 2008.
    """

    _parameter_constraints = {
        "alpha": [Interval(Real, 0, None, closed="neither"), None],
        "n_bootstrap": [Interval(Integral, 1, None, closed="left")],
        "threshold": [Interval(Real, 0, 1, closed="right")],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        alpha=None,
        n_bootstrap=128,
        threshold=1.0,
        fit_intercept=True,
        max_iter=1000,
        random_state=None,
    ):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.threshold = threshold
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n, p = X.shape
        if self.alpha is None:
            cv = min(5, n) if n >= 2 else 2
            self.alpha_ = float(
                LassoCV(fit_intercept=self.fit_intercept, max_iter=self.max_iter, cv=cv)
                .fit(X, y)
                .alpha_
            )
        else:
            self.alpha_ = float(self.alpha)

        lasso = Lasso(alpha=self.alpha_, fit_intercept=self.fit_intercept, max_iter=self.max_iter)
        hits = np.zeros(p)
        self.n_iter_ = 0
        for _ in range(self.n_bootstrap):
            idx = rng.randint(0, n, n)
            hits += lasso.fit(X[idx], y[idx]).coef_ != 0
            self.n_iter_ = max(self.n_iter_, lasso.n_iter_)
        self.selection_frequencies_ = hits / self.n_bootstrap
        self.support_ = np.flatnonzero(self.selection_frequencies_ >= self.threshold - 1e-12)

        self.coef_ = np.zeros(p)
        if self.support_.size:
            ols = LinearRegression(fit_intercept=self.fit_intercept).fit(X[:, self.support_], y)
            self.coef_[self.support_] = ols.coef_
            self.intercept_ = float(ols.intercept_) if self.fit_intercept else 0.0
        else:
            self.intercept_ = float(np.mean(y)) if self.fit_intercept else 0.0
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X @ self.coef_ + self.intercept_
