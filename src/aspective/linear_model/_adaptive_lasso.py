"""Adaptive Lasso (Zou, JASA 2006)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import check_cv
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


def _initial_coef(X, y, initial, fit_intercept):
    if initial == "ols":
        model = LinearRegression(fit_intercept=fit_intercept)
    else:
        model = RidgeCV(alphas=np.logspace(-3, 3, 13), fit_intercept=fit_intercept)
    return model.fit(X, y).coef_


def _adaptive_weights(coef, gamma):
    # A zero initial coefficient gets an infinite penalty: the feature is dropped.
    with np.errstate(divide="ignore"):
        return np.abs(coef) ** -gamma


class _AdaptiveLassoMixin:
    def _rescaled(self, X, weights):
        keep = np.isfinite(weights)
        return X[:, keep] / weights[keep], keep

    def _set_coef(self, inner_coef, inner_intercept, weights, keep):
        coef = np.zeros(len(weights))
        coef[keep] = inner_coef / weights[keep]
        self.coef_ = coef
        self.intercept_ = float(inner_intercept)
        self.support_ = np.flatnonzero(coef)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X @ self.coef_ + self.intercept_


class AdaptiveLasso(_AdaptiveLassoMixin, RegressorMixin, BaseEstimator):
    """Adaptive Lasso with a fixed penalty.

    The lasso penalizes every coefficient equally, so to drop the noise
    variables it must also shrink the large, true effects, and it selects the
    right variables only under a restrictive irrepresentable condition. The
    adaptive lasso uses a weighted penalty ``alpha * sum_j w_j |beta_j|`` with
    ``w_j = 1 / |beta_init_j| ** gamma`` from an initial root-n consistent fit.
    Variables that look large are barely penalized, small ones heavily. Zou
    proved that it has the oracle property: it selects the true variables
    with probability tending to one and estimates them as efficiently as if
    the true model were known. It is solved as a lasso on rescaled inputs.

    Use :class:`AdaptiveLassoCV` to choose ``alpha`` and ``gamma``.

    Parameters
    ----------
    alpha : float, default=1.0
        Penalty, as in :class:`sklearn.linear_model.Lasso`.
    gamma : float, default=1.0
        Exponent of the adaptive weights.
    initial : {"ridge", "ols"}, default="ridge"
        Initial estimator for the weights. Zou (2006) used least squares and
        recommended ridge regression when the inputs are collinear; the ridge
        penalty is chosen by efficient leave-one-out CV.
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    tol : float, default=1e-4

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    support_ : ndarray of int
    weights_ : ndarray of shape (n_features,)
        The adaptive penalty weights (infinite for a dropped feature).
    n_iter_ : int
        Coordinate descent iterations of the lasso fit.

    References
    ----------
    H. Zou. The Adaptive Lasso and Its Oracle Properties. Journal of the
    American Statistical Association, 101(476):1418-1429, 2006.
    """

    _parameter_constraints = {
        "alpha": [Interval(Real, 0, None, closed="neither")],
        "gamma": [Interval(Real, 0, None, closed="neither")],
        "initial": [StrOptions({"ridge", "ols"})],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, alpha=1.0, gamma=1.0, initial="ridge", fit_intercept=True,
                 max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.gamma = gamma
        self.initial = initial
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True)
        self.weights_ = _adaptive_weights(_initial_coef(X, y, self.initial, self.fit_intercept),
                                          self.gamma)
        Xw, keep = self._rescaled(X, self.weights_)
        if Xw.shape[1] == 0:
            self._set_coef(np.zeros(0), y.mean() if self.fit_intercept else 0.0, self.weights_, keep)
            self.n_iter_ = 0
            return self
        lasso = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                      max_iter=self.max_iter, tol=self.tol).fit(Xw, y)
        self.n_iter_ = lasso.n_iter_
        self._set_coef(lasso.coef_, lasso.intercept_, self.weights_, keep)
        return self


class AdaptiveLassoCV(_AdaptiveLassoMixin, RegressorMixin, BaseEstimator):
    """Adaptive Lasso with ``alpha`` and ``gamma`` chosen by cross-validation.

    For every ``gamma`` the weights come from the initial fit on the full
    training data, as in Zou (2006); then the lasso path on the rescaled
    inputs is cross-validated with the same folds for every ``gamma``, and
    the pair with the lowest error is refitted on all the data.

    Parameters
    ----------
    gammas : array-like, default=(0.5, 1, 2)
        Exponents to try; Zou (2006) used 0.5, 1 and 2.
    n_alphas : int, default=100
    eps : float, default=1e-3
    cv : int, cross-validation generator or iterable, default=5
    initial : {"ridge", "ols"}, default="ridge"
    fit_intercept : bool, default=True
    max_iter : int, default=1000
    tol : float, default=1e-4

    Attributes
    ----------
    alpha_ : float
    gamma_ : float
    cv_mse_ : ndarray of shape (n_gammas,)
        Best mean cross-validated squared error for each gamma.
    coef_, intercept_, support_, weights_, n_iter_ :
        Refit on all the data with the chosen pair.

    References
    ----------
    H. Zou. The Adaptive Lasso and Its Oracle Properties. Journal of the
    American Statistical Association, 101(476):1418-1429, 2006.
    """

    _parameter_constraints = {
        "gammas": ["array-like"],
        "n_alphas": [Interval(Integral, 1, None, closed="left")],
        "eps": [Interval(Real, 0, 1, closed="neither")],
        "cv": ["cv_object"],
        "initial": [StrOptions({"ridge", "ols"})],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, gammas=(0.5, 1.0, 2.0), n_alphas=100, eps=1e-3, cv=5, initial="ridge",
                 fit_intercept=True, max_iter=1000, tol=1e-4):
        self.gammas = gammas
        self.n_alphas = n_alphas
        self.eps = eps
        self.cv = cv
        self.initial = initial
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True)
        gammas = np.asarray(self.gammas, dtype=float)
        if gammas.ndim != 1 or gammas.size == 0 or np.any(gammas <= 0):
            raise ValueError("gammas must be a non-empty 1-d sequence of positive values.")
        folds = list(check_cv(self.cv).split(X, y))
        init = _initial_coef(X, y, self.initial, self.fit_intercept)

        best = None
        self.cv_mse_ = np.full(gammas.size, np.inf)
        for g, gamma in enumerate(gammas):
            weights = _adaptive_weights(init, gamma)
            Xw, keep = self._rescaled(X, weights)
            if Xw.shape[1] == 0:
                continue
            model = LassoCV(alphas=self.n_alphas, eps=self.eps, cv=folds,
                            fit_intercept=self.fit_intercept, max_iter=self.max_iter,
                            tol=self.tol).fit(Xw, y)
            self.cv_mse_[g] = model.mse_path_.mean(axis=1).min()
            if best is None or self.cv_mse_[g] < best[0]:
                best = (self.cv_mse_[g], gamma, model, weights, keep)

        if best is None:  # every initial coefficient was exactly zero
            self.gamma_, self.alpha_ = float(gammas[0]), np.inf
            self.weights_ = _adaptive_weights(init, gammas[0])
            self._set_coef(np.zeros(0), y.mean() if self.fit_intercept else 0.0,
                           self.weights_, np.zeros(X.shape[1], bool))
            self.n_iter_ = 0
            return self
        _, self.gamma_, model, self.weights_, keep = best
        self.alpha_ = float(model.alpha_)
        self.n_iter_ = model.n_iter_
        self._set_coef(model.coef_, model.intercept_, self.weights_, keep)
        return self
