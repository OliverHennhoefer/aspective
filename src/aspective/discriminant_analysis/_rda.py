"""Regularized Discriminant Analysis (Friedman, JASA 1989)."""

from numbers import Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class RegularizedDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Regularized Discriminant Analysis.

    Each class covariance is shrunk toward the pooled covariance (``lam``) and
    the result toward a scaled identity (``gamma``). ``lam=0, gamma=0`` is QDA,
    ``lam=1, gamma=0`` is LDA, and values in between often beat both when the
    sample is small relative to the number of features.

    Following Friedman (1989), with scatter matrices ``S_k`` and ``S = sum S_k``::

        Sigma_k(lam)        = ((1 - lam) S_k + lam S) / ((1 - lam) n_k + lam n)
        Sigma_k(lam, gamma) = (1 - gamma) Sigma_k(lam) + gamma tr(Sigma_k(lam)) / p I

    Parameters
    ----------
    lam : float in [0, 1], default=0.5
        Shrinkage of the class covariances toward the pooled covariance.
    gamma : float in [0, 1], default=0.0
        Shrinkage toward a multiple of the identity.
    priors : array-like of shape (n_classes,), default=None
        Class priors. Estimated from the training data when None.
    tol : float, default=1e-10
        Eigenvalues below ``tol`` times the largest eigenvalue are clipped,
        which keeps the model defined when a covariance is singular.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    priors_ : ndarray of shape (n_classes,)
    means_ : ndarray of shape (n_classes, n_features)
    covariances_ : ndarray of shape (n_classes, n_features, n_features)
        Regularized covariance of each class.

    References
    ----------
    J. H. Friedman. Regularized Discriminant Analysis. Journal of the American
    Statistical Association, 84(405):165-175, 1989.
    """

    _parameter_constraints = {
        "lam": [Interval(Real, 0, 1, closed="both")],
        "gamma": [Interval(Real, 0, 1, closed="both")],
        "priors": ["array-like", None],
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, lam=0.5, gamma=0.0, priors=None, tol=1e-10):
        self.lam = lam
        self.gamma = gamma
        self.priors = priors
        self.tol = tol

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("The number of classes has to be greater than one; got 1 class.")
        n, p = X.shape

        if self.priors is None:
            self.priors_ = counts / n
        else:
            priors = np.asarray(self.priors, dtype=float)
            if priors.shape != (n_classes,) or np.any(priors < 0):
                raise ValueError("priors must be non-negative with one entry per class.")
            self.priors_ = priors / priors.sum()

        self.means_ = np.vstack([X[y_idx == k].mean(axis=0) for k in range(n_classes)])
        scatters = np.empty((n_classes, p, p))
        for k in range(n_classes):
            centered = X[y_idx == k] - self.means_[k]
            scatters[k] = centered.T @ centered
        pooled = scatters.sum(axis=0)

        lam, gamma = self.lam, self.gamma
        self.covariances_ = np.empty_like(scatters)
        self._rotations = np.empty_like(scatters)
        self._scalings = np.empty((n_classes, p))
        for k in range(n_classes):
            denom = (1 - lam) * counts[k] + lam * n
            cov = ((1 - lam) * scatters[k] + lam * pooled) / max(denom, np.finfo(float).tiny)
            cov = (1 - gamma) * cov + gamma * np.trace(cov) / p * np.eye(p)
            self.covariances_[k] = cov
            eigval, eigvec = np.linalg.eigh(cov)
            floor = max(self.tol * eigval.max(), np.finfo(float).tiny)
            self._scalings[k] = np.maximum(eigval, floor)
            self._rotations[k] = eigvec
        return self

    def _joint_log_likelihood(self, X):
        jll = np.empty((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            z = (X - self.means_[k]) @ self._rotations[k] / np.sqrt(self._scalings[k])
            jll[:, k] = (
                -0.5 * np.sum(z**2, axis=1)
                - 0.5 * np.sum(np.log(self._scalings[k]))
                + np.log(self.priors_[k])
            )
        return jll

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        jll = self._joint_log_likelihood(X)
        if len(self.classes_) == 2:
            return jll[:, 1] - jll[:, 0]
        return jll

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
