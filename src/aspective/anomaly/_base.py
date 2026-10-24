"""Shared base class for the anomaly detectors.

The detectors follow PyOD's ``BaseDetector`` contract. When PyOD is installed
they subclass it directly, so they work with PyOD's ensembles, thresholding
and benchmark tooling. Without PyOD a small stand-in with the same contract is
used, so the package does not require it.
"""

import numpy as np
from sklearn.base import BaseEstimator

try:
    from pyod.models.base import BaseDetector as _PyODBase
except ImportError:  # pragma: no cover - exercised only without pyod
    _PyODBase = None


class _FallbackDetector(BaseEstimator):
    """Minimal re-implementation of PyOD's ``BaseDetector`` contract.

    Fitted detectors expose ``decision_scores_`` (higher is more anomalous),
    ``threshold_`` and ``labels_`` (0 inlier, 1 outlier).
    """

    def __init__(self, contamination=0.1):
        if not 0.0 < contamination <= 0.5:
            raise ValueError(f"contamination must be in (0, 0.5], got {contamination!r}")
        self.contamination = contamination

    def _set_n_classes(self, y):
        self._classes = 2
        return self

    def _process_decision_scores(self):
        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(int).ravel()
        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)
        return self

    def predict(self, X, return_confidence=False):
        if return_confidence:
            raise NotImplementedError("return_confidence needs PyOD; install aspective[pyod].")
        scores = self.decision_function(X)
        return (scores > self.threshold_).astype(int).ravel()

    def predict_proba(self, X, method="linear", return_confidence=False):
        if method != "linear" or return_confidence:
            raise NotImplementedError("Only method='linear' is available without PyOD.")
        train = self.decision_scores_
        test = self.decision_function(X)
        span = train.max() - train.min()
        if span == 0:
            span = 1.0
        outlier = np.clip((test - train.min()) / span, 0, 1)
        return np.column_stack([1 - outlier, outlier])

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_


BaseDetector = _PyODBase if _PyODBase is not None else _FallbackDetector
