"""Supervised principal components (Bair, Hastie, Paul & Tibshirani, JASA 2006)."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import check_cv
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


def _feature_scores(X, y):
    """Absolute standardized univariate regression coefficients, as in Bair et al."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    norms = np.sqrt(np.einsum("ij,ij->j", Xc, Xc))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.abs(Xc.T @ yc) / (norms * max(np.sqrt(yc @ yc), 1e-300))
    return np.nan_to_num(score, nan=0.0)


def _fit_spc(X, y, n_keep, n_components, order):
    selected = np.sort(order[:n_keep])
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xs = X[:, selected] - x_mean[selected]
    _, sing, vt = np.linalg.svd(Xs, full_matrices=False)
    k = int(min(n_components, np.sum(sing > 1e-10 * max(sing[0], 1e-300)) if sing.size else 0))
    loadings = vt[:k].T
    scores = Xs @ loadings
    theta = np.linalg.lstsq(scores, y - y_mean, rcond=None)[0] if k else np.zeros(0)
    coef = np.zeros(X.shape[1])
    coef[selected] = loadings @ theta
    return coef, y_mean - x_mean @ coef, selected, k


class SupervisedPCARegressor(RegressorMixin, BaseEstimator):
    """Supervised principal components regression.

    Principal components regression takes the leading directions of
    variation of all features, whether or not they relate to the response,
    so with many noise features the first components can miss the signal.
    Supervised PCA first screens the features by their univariate
    association with the response (the absolute standardized regression
    coefficient), keeps those above a threshold, and only then computes
    principal components of the kept features and regresses the response on
    the first few. The threshold and the number of components are chosen by
    cross-validation. Bair et al. showed it is consistent under a latent
    factor model where ordinary PCR is not.

    Prefer it over PCR or partial least squares when there are many more
    features than samples and the response depends on a latent factor that
    a subset of features measure, as in gene-expression data. The threshold
    grid is expressed as the number of top-ranked features kept.

    Parameters
    ----------
    n_thresholds : int, default=20
        Number of candidate feature-set sizes, log-spaced from 1 to all.
    n_components : int or array-like of int, default=(1, 2, 3)
        Number(s) of principal components to try; the paper used 1 to 3.
    cv : int, cross-validation generator or iterable, default=5

    Attributes
    ----------
    n_features_kept_ : int
    n_components_ : int
    support_ : ndarray of int
        The features that passed the screen.
    coef_ : ndarray of shape (n_features,)
        The implied linear coefficients on the original features.
    intercept_ : float
    cv_mse_ : ndarray of shape (n_sizes, n_components)

    References
    ----------
    E. Bair, T. Hastie, D. Paul and R. Tibshirani. Prediction by Supervised
    Principal Components. Journal of the American Statistical Association,
    101(473):119-137, 2006.
    """

    _parameter_constraints = {
        "n_thresholds": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "cv": ["cv_object"],
    }

    def __init__(self, n_thresholds=20, n_components=(1, 2, 3), cv=5):
        self.n_thresholds = n_thresholds
        self.n_components = n_components
        self.cv = cv

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        comps = np.atleast_1d(np.asarray(self.n_components))
        if comps.ndim != 1 or comps.size == 0 or np.any(comps < 1):
            raise ValueError("n_components must be a non-empty sequence of positive integers.")
        p = X.shape[1]
        sizes = np.unique(np.round(np.geomspace(1, p, min(self.n_thresholds, p))).astype(int))
        self.sizes_ = sizes

        errors = np.zeros((len(sizes), len(comps)))
        folds = list(check_cv(self.cv).split(X, y))
        for train, test in folds:
            order = np.argsort(-_feature_scores(X[train], y[train]), kind="stable")
            for a, size in enumerate(sizes):
                for b, k in enumerate(comps):
                    coef, intercept, _, _ = _fit_spc(X[train], y[train], size, k, order)
                    errors[a, b] += np.mean((y[test] - X[test] @ coef - intercept) ** 2)
        self.cv_mse_ = errors / len(folds)
        a, b = np.unravel_index(np.argmin(self.cv_mse_), self.cv_mse_.shape)
        order = np.argsort(-_feature_scores(X, y), kind="stable")
        self.coef_, intercept, self.support_, self.n_components_ = _fit_spc(
            X, y, sizes[a], comps[b], order)
        self.intercept_ = float(intercept)
        self.n_features_kept_ = int(sizes[a])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X @ self.coef_ + self.intercept_
