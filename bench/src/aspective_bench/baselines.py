"""Vanilla models that scikit-learn does not ship, for use as predecessors.

Kept minimal and dependency-free so the benchmark does not need PyOD.
"""

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors


class KNNDistanceDetector(OutlierMixin, BaseEstimator):
    """The k-NN distance detector (Ramaswamy, Rastogi & Shim, SIGMOD 2000).

    Scores a point by its distance to its ``n_neighbors``-th nearest
    training point, as PyOD's ``KNN(method="largest")``. ``score_samples``
    follows scikit-learn: higher means more normal.
    """

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.k_ = min(self.n_neighbors, len(X))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_).fit(X)
        return self

    def score_samples(self, X):
        dist, _ = self.nn_.kneighbors(np.asarray(X, dtype=float))
        return -dist[:, -1]
