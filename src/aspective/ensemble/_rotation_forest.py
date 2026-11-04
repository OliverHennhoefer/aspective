"""Rotation Forest (Rodriguez, Kuncheva & Alonso, IEEE TPAMI 2006)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class RotationForestClassifier(ClassifierMixin, BaseEstimator):
    """Rotation Forest: decision trees on PCA-rotated feature subsets.

    Bagging and random forests make trees diverse by resampling rows and
    features, which costs each tree some accuracy. Rotation Forest instead
    gives every tree the full training set in a different coordinate
    system: the features are split at random into groups of
    ``n_features_per_subset``; for each group, principal components are
    computed on a bootstrap sample of ``sample_fraction`` of the rows of a
    random non-empty subset of the classes; and all components of all
    groups together form a block-diagonal rotation of the feature space.
    Axis-parallel splits in rotated coordinates are oblique splits in the
    original ones, so the trees stay accurate yet differ. Rodriguez et al.
    found it significantly more accurate than bagging, AdaBoost and random
    forests on 33 UCI data sets, and Bagnall et al. (2018) confirmed it as
    the strongest tree ensemble on 121 UCI data sets with continuous
    features.

    Features are standardized before rotation. Prefer Rotation Forest over
    a random forest for continuous features; it is slower to train, and
    with mainly categorical features its advantage shrinks.

    Parameters
    ----------
    n_estimators : int, default=100
    n_features_per_subset : int, default=3
        Size ``M`` of the feature groups; the last group is filled up with
        features drawn from the others, as in the reference implementation.
    sample_fraction : float in (0, 1], default=0.75
        Fraction of the selected rows used to compute each group's PCA.
    min_samples_leaf : int, default=1
        Of every tree (unpruned trees by default, as in the paper's C4.5).
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    estimators_ : list of DecisionTreeClassifier
    rotations_ : list of ndarray of shape (n_features, n_features)

    References
    ----------
    J. J. Rodriguez, L. I. Kuncheva and C. J. Alonso. Rotation Forest: A New
    Classifier Ensemble Method. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 28(10):1619-1630, 2006.

    A. Bagnall, A. Bostrom, G. Cawley, M. Flynn, J. Large and J. Lines. Is
    Rotation Forest the Best Classifier for Problems with Continuous
    Features? arXiv:1809.06705, 2018.
    """

    _parameter_constraints = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "n_features_per_subset": [Interval(Integral, 1, None, closed="left")],
        "sample_fraction": [Interval(Real, 0, 1, closed="right")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_estimators=100,
        n_features_per_subset=3,
        sample_fraction=0.75,
        min_samples_leaf=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.sample_fraction = sample_fraction
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _rotation(self, Z, y_idx, n_classes, rng):
        n, p = Z.shape
        M = min(self.n_features_per_subset, p)
        perm = rng.permutation(p)
        n_groups = int(np.ceil(p / M))
        col = 0
        groups = []
        for g in range(n_groups):
            group = perm[g * M : (g + 1) * M]
            if len(group) < M:  # fill the last group with features from the others
                extra = rng.choice(np.setdiff1d(perm, group), M - len(group), replace=False)
                group = np.concatenate([group, extra])
            groups.append(group)
        R = np.zeros((p, n_groups * M))
        for group in groups:
            classes = np.flatnonzero(rng.rand(n_classes) < 0.5)
            if len(classes) == 0:
                classes = rng.randint(n_classes, size=1)
            rows = np.flatnonzero(np.isin(y_idx, classes))
            size = max(2, int(round(self.sample_fraction * len(rows))))
            rows = rng.choice(rows, size, replace=True)
            sub = Z[np.ix_(rows, group)]
            sub = sub - sub.mean(axis=0)
            # All components, including those of zero variance, so the rotation keeps every
            # direction (a degenerate sample still gives an orthonormal basis). The M x M
            # scatter matrix avoids SVD's n x n left factor.
            _, vecs = np.linalg.eigh(sub.T @ sub)
            R[group, col : col + M] += vecs[:, ::-1]  # decreasing variance
            col += M
        return R

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        rng = check_random_state(self.random_state)
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        Z = (X - self._mean) / self._sd
        self.estimators_, self.rotations_ = [], []
        for _ in range(self.n_estimators):
            R = self._rotation(Z, y_idx, len(self.classes_), rng)
            tree = DecisionTreeClassifier(
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(np.iinfo(np.int32).max),
            )
            tree.fit(Z @ R, y_idx)
            self.estimators_.append(tree)
            self.rotations_.append(R)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        Z = (X - self._mean) / self._sd
        proba = np.zeros((len(X), len(self.classes_)))
        for tree, R in zip(self.estimators_, self.rotations_):
            proba[:, tree.classes_] += tree.predict_proba(Z @ R)
        return proba / len(self.estimators_)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
