"""Discriminant adaptive nearest neighbours (Hastie & Tibshirani, IEEE TPAMI 1996)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


def _inv_sqrt(matrix, floor):
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.maximum(vals, floor)
    return (vecs / np.sqrt(vals)) @ vecs.T


class DANNClassifier(ClassifierMixin, BaseEstimator):
    """Discriminant adaptive nearest-neighbour classification (DANN).

    k-nearest-neighbour classification uses the same round neighbourhood
    everywhere. Near a class boundary that wastes neighbours on directions
    in which the class does not change. DANN adapts the metric to each
    query point: it takes the ``n_neighbors_metric`` nearest training points,
    computes their pooled within-class covariance ``W`` and between-class
    covariance ``B``, and measures distance with

        Sigma = W^-1/2 (W^-1/2 B W^-1/2 + epsilon I) W^-1/2,

    which stretches the neighbourhood along the local decision boundary and
    shrinks it across. The query is then classified by a majority vote of
    its ``n_neighbors`` nearest points in that metric. This is a local LDA:
    ``epsilon`` rounds the neighbourhood so it never becomes an infinite
    strip. Hastie and Tibshirani found it beat k-NN, CART and LDA on most
    simulated and real problems they tried.

    Inputs are standardized before distances are taken. Prefer DANN over
    k-NN when only some directions matter locally; it costs one small
    eigendecomposition and a pass over the training data per query.

    Parameters
    ----------
    n_neighbors : int, default=5
        Neighbours that vote, in the adapted metric.
    n_neighbors_metric : int or None, default=None
        Neighbourhood used to estimate the local metric; None uses
        ``max(n_samples // 5, 50)``, the paper's choice, capped at n.
    epsilon : float, default=1.0
        Softening of the metric; the paper recommends 1.
    n_iter : int, default=1
        Times the metric is re-estimated in the neighbourhood it defines.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_neighbors_metric_ : int

    References
    ----------
    T. Hastie and R. Tibshirani. Discriminant Adaptive Nearest Neighbor
    Classification. IEEE Transactions on Pattern Analysis and Machine
    Intelligence, 18(6):607-616, 1996.
    """

    _parameter_constraints = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "n_neighbors_metric": [Interval(Integral, 2, None, closed="left"), None],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, n_neighbors=5, n_neighbors_metric=None, epsilon=1.0, n_iter=1):
        self.n_neighbors = n_neighbors
        self.n_neighbors_metric = n_neighbors_metric
        self.epsilon = epsilon
        self.n_iter = n_iter

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, self._y = np.unique(y, return_inverse=True)
        n = len(X)
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        self._Z = (X - self._mean) / self._sd
        km = max(n // 5, 50) if self.n_neighbors_metric is None else self.n_neighbors_metric
        self.n_neighbors_metric_ = int(min(km, n))
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors_metric_).fit(self._Z)
        return self

    def _metric(self, idx):
        Z, y = self._Z[idx], self._y[idx]
        d = Z.shape[1]
        center = Z.mean(axis=0)
        W = np.zeros((d, d))
        B = np.zeros((d, d))
        for c in np.unique(y):
            Zc = Z[y == c]
            share = len(Zc) / len(Z)
            mc = Zc.mean(axis=0)
            dev = Zc - mc
            W += dev.T @ dev / len(Z)
            B += share * np.outer(mc - center, mc - center)
        scale = max(np.trace(W) / d, 1e-12)
        W_half = _inv_sqrt(W, 1e-6 * scale)
        Bstar = W_half @ B @ W_half
        return W_half @ (Bstar + self.epsilon * np.eye(d)) @ W_half

    def _vote_proba(self, Z):
        n = len(self._Z)
        k = min(self.n_neighbors, n)
        km = self.n_neighbors_metric_
        _, start = self._nn.kneighbors(Z)
        proba = np.zeros((len(Z), len(self.classes_)))
        for q in range(len(Z)):
            idx = start[q]
            diff = self._Z - Z[q]
            for it in range(self.n_iter):
                sigma = self._metric(idx)
                dist = np.einsum("ij,jk,ik->i", diff, sigma, diff)
                if it + 1 < self.n_iter:
                    idx = np.argpartition(dist, km - 1)[:km]
            nearest = np.argpartition(dist, k - 1)[:k] if k < n else np.arange(n)
            proba[q] = np.bincount(self._y[nearest], minlength=len(self.classes_)) / k
        return proba

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._vote_proba((X - self._mean) / self._sd)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
