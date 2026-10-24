"""Isolation Distributional Kernel detector (Ting, Xu, Washio & Zhou, KDD 2020)."""

from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector


class IDKDetector(BaseDetector):
    """Point anomaly detection with the Isolation Distributional Kernel.

    The Isolation Kernel is data dependent: each of ``n_estimators``
    partitionings is the Voronoi diagram of ``max_samples`` points drawn from
    the training data, and two points are similar when they fall in the same
    cells. Its feature map is a sparse one-hot vector per partitioning, so the
    kernel mean embedding of the whole training set is just the average of
    those vectors.

    A point is scored by its similarity to the embedding of the training
    distribution. Points in sparse regions share cells with few training
    points and get low similarity. Fitting and scoring are linear in the
    number of samples.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points, and ``predict`` returns 0 for inliers and 1 for
    outliers.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of partitionings (``t`` in the paper).
    max_samples : int, default=16
        Points per partitioning (``psi`` in the paper). Smaller values give a
        smoother, wider kernel; the paper tunes it over powers of two.
    contamination : float in (0, 0.5], default=0.1
        Expected share of outliers, used to set ``threshold_``.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Anomaly scores of the training data (higher is more anomalous).
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    centers_ : ndarray of shape (n_estimators, max_samples, n_features)
    embedding_ : ndarray of shape (n_estimators, max_samples)
        Kernel mean embedding of the training data.

    References
    ----------
    K. M. Ting, B.-C. Xu, T. Washio and Z.-H. Zhou. Isolation Distributional
    Kernel: A New Tool for Kernel based Anomaly Detection. In Proceedings of
    the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining
    (KDD), 2020.
    """

    def __init__(self, n_estimators=200, max_samples=16, contamination=0.1, random_state=None):
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def _cells(self, X):
        """Index of the nearest center in every partitioning, shape (n, t)."""
        t, psi, _ = self.centers_.shape
        flat = self.centers_.reshape(t * psi, -1)
        sq_norms = np.sum(flat**2, axis=1)
        cells = np.empty((X.shape[0], t), dtype=np.intp)
        chunk = max(1, int(5e6 // (t * psi)))
        for s in range(0, X.shape[0], chunk):
            block = X[s : s + chunk]
            dist = sq_norms[None, :] - 2 * block @ flat.T
            cells[s : s + chunk] = dist.reshape(len(block), t, psi).argmin(axis=2)
        return cells

    def _similarity(self, X):
        cells = self._cells(X)
        return self.embedding_[np.arange(self.n_estimators)[None, :], cells].mean(axis=1)

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        if not isinstance(self.n_estimators, Integral) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(self.max_samples, Integral) or self.max_samples < 2:
            raise ValueError("max_samples must be an integer of at least 2.")
        self._set_n_classes(y)
        rng = check_random_state(self.random_state)
        n = X.shape[0]
        psi = min(self.max_samples, n)
        self.centers_ = np.stack(
            [X[rng.choice(n, psi, replace=False)] for _ in range(self.n_estimators)]
        )
        cells = self._cells(X)
        self.embedding_ = np.zeros((self.n_estimators, psi))
        for i in range(self.n_estimators):
            self.embedding_[i] = np.bincount(cells[:, i], minlength=psi) / n
        self.decision_scores_ = -self._similarity(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["centers_", "embedding_"])
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        return -self._similarity(X)
