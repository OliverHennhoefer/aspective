"""Local Outlier Probabilities (Kriegel, Kroeger, Schubert & Zimek, CIKM 2009)."""

from numbers import Integral, Real

import numpy as np
from scipy.special import erf
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector

_TINY = 1e-12


class LoOPDetector(BaseDetector):
    """Local Outlier Probabilities.

    LOF compares a point's density with its neighbours', but its scores
    have no fixed scale: an LOF of 2 can be normal in one data set and
    extreme in another, so a threshold does not transfer. LoOP keeps the
    local comparison and turns it into a probability. The density of a
    point is estimated from the quadratic mean of its distances to its
    ``n_neighbors`` nearest neighbours (a "probabilistic distance" of
    ``lambda_`` standard deviations), the probabilistic local outlier
    factor ``PLOF`` compares it with the neighbours' average, and the
    score ``LoOP = max(0, erf(PLOF / (nPLOF * sqrt(2))))`` normalizes PLOF by
    its spread over the training data. Scores lie in [0, 1], are near 0 for
    points inside dense regions and near 1 for clear outliers, and mean the
    same thing in every data set.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points. Training scores use each training point's
    neighbours without itself; new points are scored against the training
    data.

    Parameters
    ----------
    n_neighbors : int, default=20
        Context size ``k``; the paper used 10 to 30, like LOF's ``MinPts``.
    lambda_ : float, default=3.0
        Significance of the probabilistic distance; 3 corresponds to about
        99.7% under a normal model, the paper's suggestion.
    contamination : float in (0, 0.5], default=0.1

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    n_plof_ : float
        The normalization factor ``nPLOF`` from the training data.

    References
    ----------
    H.-P. Kriegel, P. Kroeger, E. Schubert and A. Zimek. LoOP: Local Outlier
    Probabilities. In Proceedings of the 18th ACM Conference on Information
    and Knowledge Management (CIKM), pages 1649-1652, 2009.
    """

    def __init__(self, n_neighbors=20, lambda_=3.0, contamination=0.1):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.lambda_ = lambda_

    def _pdist(self, dist):
        return self.lambda_ * np.sqrt(np.mean(dist**2, axis=1))

    def _probabilities(self, pdist, idx):
        plof = pdist / np.maximum(self._train_pdist[idx].mean(axis=1), _TINY) - 1.0
        return np.maximum(0.0, erf(plof / (max(self.n_plof_, _TINY) * np.sqrt(2)))), plof

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float64, np.float32])
        if not isinstance(self.n_neighbors, Integral) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if not isinstance(self.lambda_, Real) or self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if len(X) < 2:
            raise ValueError("LoOPDetector needs at least 2 samples.")
        self._set_n_classes(y)
        self.n_neighbors_ = min(self.n_neighbors, len(X) - 1)
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors_).fit(X)
        dist, idx = self._nn.kneighbors()
        self._train_pdist = self._pdist(dist)
        plof = self._train_pdist / np.maximum(self._train_pdist[idx].mean(axis=1), _TINY) - 1.0
        self.n_plof_ = float(self.lambda_ * np.sqrt(np.mean(plof**2)))
        self.decision_scores_, _ = self._probabilities(self._train_pdist, idx)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "n_plof_")
        X = validate_data(self, X, reset=False, dtype=[np.float64, np.float32])
        dist, idx = self._nn.kneighbors(X)
        return self._probabilities(self._pdist(dist), idx)[0]
