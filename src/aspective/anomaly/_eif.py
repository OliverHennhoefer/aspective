"""Extended Isolation Forest (Hariri, Carrasco Kind & Brunner, IEEE TKDE 2021)."""

from numbers import Integral

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector
from ._isolation import ObliqueIsolationTree, average_path_length


class ExtendedIsolationForestDetector(BaseDetector):
    """Isolation Forest with random oblique cuts (Extended Isolation Forest).

    Isolation Forest isolates points with axis-parallel cuts. Its anomaly
    scores therefore carry artefacts: regions that line up with a dense
    cluster along one axis, but are far from all data, look normal, and
    score maps show rectangular "ghost" bands. Extended Isolation Forest
    cuts each node with a hyperplane of random orientation, a normal
    vector with standard-normal entries through a point drawn uniformly
    from the node's bounding box. ``extension_level`` sets how many
    coordinates of the normal are non-zero (0 is Isolation Forest, the
    default ``n_features - 1`` is fully oblique). Hariri et al. showed the
    artefacts disappear and found equal or better ROC AUC than Isolation
    Forest on the ODDS benchmark data.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points (the usual ``2 ** (-E[h(x)] / c(psi))``).

    Parameters
    ----------
    n_estimators : int, default=100
    max_samples : int, default=256
        Subsample size ``psi`` per tree, capped at n.
    extension_level : int or None, default=None
        Non-zero coordinates of each normal minus one; None is
        ``n_features - 1``.
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
    S. Hariri, M. Carrasco Kind and R. J. Brunner. Extended Isolation
    Forest. IEEE Transactions on Knowledge and Data Engineering,
    33(4):1479-1489, 2021.

    F. T. Liu, K. M. Ting and Z.-H. Zhou. Isolation Forest. In Proceedings
    of the 8th IEEE International Conference on Data Mining (ICDM), 2008.
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        extension_level=None,
        contamination=0.1,
        random_state=None,
    ):
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        for name in ("n_estimators", "max_samples"):
            value = getattr(self, name)
            if not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        n, d = X.shape
        level = d - 1 if self.extension_level is None else self.extension_level
        if not isinstance(level, Integral) or not 0 <= level <= d - 1:
            raise ValueError(f"extension_level must be an integer in [0, {d - 1}].")
        self._set_n_classes(y)
        rng = check_random_state(self.random_state)
        psi = min(self.max_samples, n)
        height = int(np.ceil(np.log2(max(psi, 2))))

        def choose_split(Xn, rng):
            lo, hi = Xn.min(axis=0), Xn.max(axis=0)
            if np.all(hi <= lo):
                return None
            normal = rng.normal(size=d)
            zero = rng.choice(d, d - level - 1, replace=False)
            normal[zero] = 0.0
            point = rng.uniform(lo, hi)
            return normal, float(point @ normal)

        self.estimators_ = [
            ObliqueIsolationTree(X[rng.choice(n, psi, replace=False)], choose_split, height, rng)
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
