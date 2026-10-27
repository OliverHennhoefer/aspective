"""Vanilla baselines that scikit-learn does not provide in the needed form."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data


class ScottKDE(BaseEstimator):
    """Plain Gaussian KDE with the bandwidth rule and subsampling of ``RobustKDEDetector``.

    It is the robust KDE with all kernel weights fixed at ``1/m``, so a
    comparison between the two isolates the effect of the robust weights.
    ``score_samples`` returns the log density, as in scikit-learn.
    """

    def __init__(self, max_samples=None, random_state=None):
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        n, d = X.shape
        centers = X
        if self.max_samples is not None and self.max_samples < n:
            centers = X[check_random_state(self.random_state).choice(n, self.max_samples, replace=False)]
        m = centers.shape[0]
        self.bandwidth_ = float(max(np.mean(X.std(axis=0)), 1e-12) * m ** (-1.0 / (d + 4)))
        self.centers_ = centers
        return self

    def score_samples(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        h, (m, d) = self.bandwidth_, self.centers_.shape
        log_norm = -0.5 * d * np.log(2 * np.pi * h**2) - np.log(m)
        out = np.empty(X.shape[0])
        for s in range(0, X.shape[0], 2048):
            sq = cdist(X[s : s + 2048], self.centers_, "sqeuclidean")
            out[s : s + 2048] = logsumexp(-sq / (2 * h**2), axis=1) + log_norm
        return out
