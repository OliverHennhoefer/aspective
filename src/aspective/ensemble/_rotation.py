"""Random Rotation Ensembles (Blaser & Fryzlewicz, JMLR 2016)."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


def random_rotation(n_features, random_state=None):
    """Draw a rotation matrix uniformly from the Haar measure on SO(n).

    Uses the QR decomposition of a Gaussian matrix with the sign correction of
    Mezzadri (2007), then flips one column if needed so the determinant is +1.
    """
    rng = check_random_state(random_state)
    q, r = np.linalg.qr(rng.standard_normal((n_features, n_features)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class _BaseRandomRotation(BaseEstimator):
    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "scaling": [StrOptions({"minmax", "standard", "rank", "none"})],
        "bootstrap": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, n_estimators=100, scaling="minmax", bootstrap=True, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.scaling = scaling
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _fit_scaler(self, X):
        # Rotations mix features, so they must be on comparable scales first.
        if self.scaling == "minmax":
            low, high = X.min(axis=0), X.max(axis=0)
            self._center = (low + high) / 2
            self._scale = np.where(high > low, (high - low) / 2, 1.0)
        elif self.scaling == "standard":
            self._center = X.mean(axis=0)
            std = X.std(axis=0)
            self._scale = np.where(std > 0, std, 1.0)
        elif self.scaling == "rank":
            self._train_sorted = np.sort(X, axis=0)
        else:
            self._center = np.zeros(X.shape[1])
            self._scale = np.ones(X.shape[1])

    def _scale_X(self, X):
        if self.scaling != "rank":
            return (X - self._center) / self._scale
        # Map each value to its empirical CDF position in the training data, in [-1, 1].
        n = self._train_sorted.shape[0]
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = self._train_sorted[:, j]
            lo = np.searchsorted(col, X[:, j], side="left")
            hi = np.searchsorted(col, X[:, j], side="right")
            out[:, j] = (lo + hi) / (2 * n)
        return 2 * out - 1

    def _fit_members(self, X, y):
        self._validate_params()
        rng = check_random_state(self.random_state)
        self._fit_scaler(X)
        Xs = self._scale_X(X)
        n, p = Xs.shape
        self.rotations_ = []
        self.estimators_ = []
        for _ in range(self.n_estimators):
            rotation = random_rotation(p, rng)
            idx = rng.randint(0, n, n) if self.bootstrap else np.arange(n)
            member = clone(self._default_estimator() if self.estimator is None else self.estimator)
            if "random_state" in member.get_params():
                member.set_params(random_state=rng.randint(np.iinfo(np.int32).max))
            member.fit(Xs[idx] @ rotation, y[idx])
            self.rotations_.append(rotation)
            self.estimators_.append(member)
        return self

    def _rotated(self, X):
        Xs = self._scale_X(X)
        return (Xs @ rotation for rotation in self.rotations_)


class RandomRotationClassifier(ClassifierMixin, _BaseRandomRotation):
    """Ensemble in which every member sees its own random rotation of the data.

    Axis-aligned trees need many splits to follow a boundary that is oblique
    to the feature axes. Rotating the (scaled) feature space uniformly at
    random before fitting each member gives the ensemble oblique boundaries
    and extra diversity. With the default base estimator this is the
    "rotated random forest" of Blaser & Fryzlewicz (2016).

    Parameters
    ----------
    estimator : estimator, default=None
        Base estimator. Defaults to a decision tree with
        ``max_features="sqrt"``, i.e. a random forest member.
    n_estimators : int, default=100
    scaling : {"minmax", "standard", "rank", "none"}, default="minmax"
        Scaling applied before rotating. The paper recommends scaling so no
        feature dominates the rotation; "rank" is the most robust to outliers.
    bootstrap : bool, default=True
        Fit each member on a bootstrap sample.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    estimators_ : list of fitted estimators
    rotations_ : list of ndarray of shape (n_features, n_features)

    References
    ----------
    R. Blaser and P. Fryzlewicz. Random Rotation Ensembles. Journal of Machine
    Learning Research, 17(4):1-26, 2016.
    """

    def _default_estimator(self):
        return DecisionTreeClassifier(max_features="sqrt")

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        return self._fit_members(X, y_idx)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        proba = np.zeros((X.shape[0], len(self.classes_)))
        for member, Xr in zip(self.estimators_, self._rotated(X)):
            if hasattr(member, "predict_proba"):
                member_proba = member.predict_proba(Xr)
                proba[:, member.classes_.astype(int)] += member_proba
            else:
                proba[np.arange(X.shape[0]), member.predict(Xr).astype(int)] += 1
        return proba / len(self.estimators_)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RandomRotationRegressor(RegressorMixin, _BaseRandomRotation):
    """Regression ensemble in which every member sees its own random rotation.

    See :class:`RandomRotationClassifier` for the method. The default base
    estimator is a regression tree with ``max_features=1/3``, the usual
    random forest choice for regression.

    References
    ----------
    R. Blaser and P. Fryzlewicz. Random Rotation Ensembles. Journal of Machine
    Learning Research, 17(4):1-26, 2016.
    """

    def _default_estimator(self):
        return DecisionTreeRegressor(max_features=1 / 3)

    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True)
        return self._fit_members(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return np.mean([m.predict(Xr) for m, Xr in zip(self.estimators_, self._rotated(X))], axis=0)
