"""SCiForest: isolation with split-selection criterion (Liu, Ting & Zhou, ECML PKDD 2010)."""

from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector
from ._isolation import ObliqueIsolationTree, average_path_length


def _best_sd_gain(proj):
    """Best split of every column of ``proj`` (m, tau) by Liu et al.'s averaged gain.

    Returns (gain, threshold) per column, gain = (sd - (sd_left + sd_right) / 2) / sd.
    """
    m = len(proj)
    s = np.sort(proj, axis=0)
    csum, csq = np.cumsum(s, axis=0), np.cumsum(s * s, axis=0)
    tot, totsq = csum[-1], csq[-1]
    k = np.arange(1, m)[:, None]  # left sizes 1..m-1
    lsum, lsq = csum[:-1], csq[:-1]
    rsum, rsq = tot - lsum, totsq - lsq
    lvar = np.maximum(lsq / k - (lsum / k) ** 2, 0.0)
    rvar = np.maximum(rsq / (m - k) - (rsum / (m - k)) ** 2, 0.0)
    sd = np.sqrt(np.maximum(totsq / m - (tot / m) ** 2, 0.0))
    avg = (np.sqrt(lvar) + np.sqrt(rvar)) / 2
    gain = np.where(sd > 0, (sd - avg) / np.where(sd > 0, sd, 1.0), -np.inf)
    distinct = s[1:] > s[:-1]  # only split between different values
    gain = np.where(distinct, gain, -np.inf)
    best = np.argmax(gain, axis=0)
    cols = np.arange(proj.shape[1])
    threshold = (s[best, cols] + s[best + 1, cols]) / 2
    return gain[best, cols], threshold


class SCiForestDetector(BaseDetector):
    """Isolation forest with split selection for clustered anomalies (SCiForest).

    Isolation Forest picks split points at random. Anomalies that form a
    small, dense cluster are then hard to isolate: the random cut rarely
    falls between the cluster and the normal data, and once inside the
    cluster its points look like a normal region. SCiForest chooses each
    split instead: at every node it draws ``n_hyperplanes`` random
    hyperplanes on ``n_attributes`` features (coefficients uniform in
    [-1, 1], each divided by the feature's standard deviation in the node)
    and keeps the hyperplane and threshold that maximize the averaged
    standard-deviation gain, which favours cuts that separate a small,
    distant group. At test time a point whose projection lies outside the
    range the node saw during training is isolated at that node. Liu, Ting
    and Zhou found it detected clustered and scattered anomalies better
    than Isolation Forest, LOF and ORCA.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points (``2 ** (-E[h(x)] / c(psi))``).

    Parameters
    ----------
    n_estimators : int, default=100
    max_samples : int, default=256
        Subsample size ``psi`` per tree, capped at n.
    n_hyperplanes : int, default=10
        Candidate hyperplanes per node (``tau`` in the paper).
    n_attributes : int, default=2
        Features per hyperplane (``q`` in the paper), capped at n_features.
    contamination : float in (0, 0.5], default=0.1
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    estimators_ : list of trees

    References
    ----------
    F. T. Liu, K. M. Ting and Z.-H. Zhou. On Detecting Clustered Anomalies
    Using SCiForest. In Machine Learning and Knowledge Discovery in
    Databases (ECML PKDD), LNCS 6322, pages 274-290, 2010.
    """

    def __init__(self, n_estimators=100, max_samples=256, n_hyperplanes=10, n_attributes=2,
                 contamination=0.1, random_state=None):
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_hyperplanes = n_hyperplanes
        self.n_attributes = n_attributes
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        for name in ("n_estimators", "max_samples", "n_hyperplanes", "n_attributes"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        n, d = X.shape
        self._set_n_classes(y)
        rng = check_random_state(self.random_state)
        psi = min(self.max_samples, n)
        height = int(np.ceil(np.log2(max(psi, 2))))
        q = min(self.n_attributes, d)
        tau = self.n_hyperplanes

        def choose_split(Xn, rng):
            sd = Xn.std(axis=0)
            varying = np.flatnonzero(sd > 0)
            if len(varying) == 0:
                return None
            normals = np.zeros((d, tau))
            for t in range(tau):
                attrs = rng.choice(varying, min(q, len(varying)), replace=False)
                normals[attrs, t] = rng.uniform(-1, 1, size=len(attrs)) / sd[attrs]
            gain, threshold = _best_sd_gain(Xn @ normals)
            best = int(np.argmax(gain))
            if not np.isfinite(gain[best]):
                return None
            return normals[:, best], float(threshold[best])

        self.estimators_ = [
            ObliqueIsolationTree(X[rng.choice(n, psi, replace=False)], choose_split, height, rng,
                                 use_range=True)
            for _ in range(self.n_estimators)
        ]
        self._c = float(average_path_length(np.array([psi]))[0]) or 1.0
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        depth = np.mean([tree.path_length(X) for tree in self.estimators_], axis=0)
        return 2.0 ** (-depth / self._c)
