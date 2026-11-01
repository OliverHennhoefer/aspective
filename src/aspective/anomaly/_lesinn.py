"""LeSiNN: least similar nearest neighbours (Pang, Ting & Albrecht, ICDM 2015)."""

from numbers import Integral

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector


class LeSiNNDetector(BaseDetector):
    """Outlier detection by nearest neighbours in tiny random subsamples.

    The classic k-nearest-neighbour detector scores a point by its distance
    to its nearest neighbours in the whole data set: it needs all pairwise
    distances, and clusters of outliers mask each other because they are
    each other's neighbours. LeSiNN draws ``n_estimators`` random subsamples
    of only ``max_samples`` points (8 by default) and scores a point by its
    distance to the nearest point of each subsample, averaged over the
    subsamples. A tiny subsample rarely contains the outliers of a cluster,
    which removes the masking, and scoring is linear in the data size.
    Pang, Ting and Albrecht found it as accurate as or better than
    Isolation Forest and full k-NN on a broad benchmark, at a fraction of
    the cost; the idea extends the single-sample detector of Sugiyama and
    Borgwardt (NeurIPS 2013) to an ensemble.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points. A training point is not compared with itself
    when it is part of a subsample.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of subsamples (``t`` in the paper).
    max_samples : int, default=8
        Subsample size (``psi`` in the paper), capped at n.
    contamination : float in (0, 0.5], default=0.1
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    subsamples_ : ndarray of shape (n_estimators, max_samples, n_features)

    References
    ----------
    G. Pang, K. M. Ting and D. Albrecht. LeSiNN: Detecting Anomalies by
    Identifying Least Similar Nearest Neighbours. In Proceedings of the 2015
    IEEE International Conference on Data Mining Workshop (ICDMW), pages
    623-630, 2015.

    M. Sugiyama and K. M. Borgwardt. Rapid Distance-Based Outlier Detection
    via Sampling. In Advances in Neural Information Processing Systems 26
    (NeurIPS), 2013.
    """

    def __init__(self, n_estimators=50, max_samples=8, contamination=0.1, random_state=None):
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        for name in ("n_estimators", "max_samples"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if len(X) < 2:
            raise ValueError("LeSiNNDetector needs at least 2 samples.")
        self._set_n_classes(y)
        rng = check_random_state(self.random_state)
        n = len(X)
        psi = min(self.max_samples, n - 1)
        members = np.array([rng.choice(n, psi, replace=False) for _ in range(self.n_estimators)])
        self.subsamples_ = X[members]
        total = np.zeros(n)
        for sample_idx in members:
            dist = pairwise_distances(X, X[sample_idx])
            dist[sample_idx, np.arange(psi)] = np.inf  # a member is not its own neighbour
            total += dist.min(axis=1)
        self.decision_scores_ = total / self.n_estimators
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "subsamples_")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        total = sum(pairwise_distances(X, s).min(axis=1) for s in self.subsamples_)
        return total / len(self.subsamples_)
