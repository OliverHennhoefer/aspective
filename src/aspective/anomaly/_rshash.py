"""RS-Hash subspace outlier detection (Sathe & Aggarwal, ICDM 2016)."""

from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector


class RSHashDetector(BaseDetector):
    """Outlier detection by randomized subspace grids.

    Each of ``n_estimators`` components draws a sample of the data, a random
    subspace and a random grid with random offsets, then counts how many
    sample points fall in each grid cell. A point whose cells are nearly
    empty across many subspaces is an outlier. The score of a component is
    ``log2(1 + count)``; scores are averaged over components. Training points
    that were part of a component's sample do not count themselves.

    The paper keeps the counts in a count-min sketch to bound memory; this
    implementation counts cells exactly, which gives the same scores without
    hash collisions.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points, and ``predict`` returns 0 for inliers and 1 for
    outliers.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of ensemble components (``m`` in the paper).
    max_samples : int, default=1000
        Sample size per component (``s`` in the paper), capped at n.
    contamination : float in (0, 0.5], default=0.1
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)

    References
    ----------
    S. Sathe and C. C. Aggarwal. Subspace Outlier Detection in Linear Time
    with Randomized Hashing. In Proceedings of the 16th IEEE International
    Conference on Data Mining (ICDM), 2016.
    """

    def __init__(self, n_estimators=100, max_samples=1000, contamination=0.1, random_state=None):
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def _normalize(self, X):
        return (X - self._min) / self._range

    def _cell_keys(self, X, comp):
        dims, shift, width = comp["dims"], comp["shift"], comp["width"]
        grid = np.floor((X[:, dims] + shift) / width).astype(np.int64)
        return grid

    def _component_scores(self, X, comp, exclude=None):
        cells = self._cell_keys(X, comp)
        keys, counts = comp["keys"], comp["counts"]
        # Look up each row's cell among the sampled cells via a structured view.
        view = np.ascontiguousarray(cells).view([("", cells.dtype)] * cells.shape[1]).ravel()
        pos = np.searchsorted(keys, view)
        pos_clipped = np.minimum(pos, len(keys) - 1)
        found = keys[pos_clipped] == view
        c = np.where(found, counts[pos_clipped], 0).astype(float)
        if exclude is not None:
            c[exclude] -= 1
        return np.log2(1 + c)

    def fit(self, X, y=None):
        X = validate_data(self, X)
        for name in ("n_estimators", "max_samples"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self._set_n_classes(y)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        self._min = X.min(axis=0)
        span = X.max(axis=0) - self._min
        self._range = np.where(span > 0, span, 1.0)
        Xn = self._normalize(X)

        s = min(self.max_samples, n)
        root = np.sqrt(s)
        low, high = (1 / root, 1 - 1 / root) if s > 4 else (0.25, 0.75)
        self.components_ = []
        train_scores = np.zeros(n)
        for _ in range(self.n_estimators):
            width = rng.uniform(low, high)
            base = max(2.0, 1 / width)
            log_s = np.log(s) / np.log(base)
            r_low = int(np.floor(1 + 0.5 * log_s))
            r_high = max(r_low, int(np.floor(log_s)))
            r = int(np.clip(rng.randint(r_low, r_high + 1), 1, d))
            dims = rng.choice(d, r, replace=False)
            shift = rng.uniform(0, width, r)
            sample = rng.choice(n, s, replace=False)
            cells = np.floor((Xn[sample][:, dims] + shift) / width).astype(np.int64)
            view = np.ascontiguousarray(cells).view([("", cells.dtype)] * r).ravel()
            keys, counts = np.unique(view, return_counts=True)
            comp = {"dims": dims, "shift": shift, "width": width, "keys": keys, "counts": counts}
            self.components_.append(comp)
            in_sample = np.zeros(n, dtype=bool)
            in_sample[sample] = True
            train_scores += self._component_scores(Xn, comp, exclude=in_sample)

        self.decision_scores_ = -train_scores / self.n_estimators
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "components_")
        X = validate_data(self, X, reset=False)
        Xn = self._normalize(X)
        total = sum(self._component_scores(Xn, comp) for comp in self.components_)
        return -total / len(self.components_)
