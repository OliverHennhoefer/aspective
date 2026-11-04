"""LogitBoost (Friedman, Hastie & Tibshirani, Annals of Statistics 2000)."""

from numbers import Integral, Real

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class LogitBoostClassifier(ClassifierMixin, BaseEstimator):
    """LogitBoost: additive logistic regression by Newton steps.

    Friedman, Hastie and Tibshirani showed that AdaBoost fits an additive
    model by stagewise minimization of the exponential loss. LogitBoost
    fits the same kind of additive model to the multinomial log-likelihood
    instead, by quasi-Newton steps: at every round and for every class it
    fits a regression tree by weighted least squares to the working
    response ``z = (y* - p) / (p (1 - p))`` with weights ``p (1 - p)``, and
    adds the centred fits to the class scores. The log-likelihood grows
    only linearly in the margin of a badly misclassified point, where the
    exponential loss grows exponentially, so LogitBoost is less sensitive
    to label noise; and the class scores are calibrated log-odds. The
    paper found it at least as accurate as AdaBoost on UCI benchmarks, and
    better for multiclass problems.

    Prefer LogitBoost over AdaBoost when labels may be noisy or you need
    class probabilities. Working responses are capped at ``z_max`` for
    stability, as the paper recommends.

    Parameters
    ----------
    n_estimators : int, default=100
        Boosting rounds; each fits one tree per class (one in total for
        two classes).
    max_depth : int, default=3
        Depth of the regression trees.
    learning_rate : float, default=1.0
        Shrinkage of each round; the paper uses 1.
    z_max : float, default=4.0
        Cap on the absolute working response.
    random_state : int, RandomState instance or None, default=None
        Only used to break ties between equally good splits.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    estimators_ : list of list of DecisionTreeRegressor
        One list per round, with one tree per class (one for two classes).

    References
    ----------
    J. Friedman, T. Hastie and R. Tibshirani. Additive Logistic Regression:
    A Statistical View of Boosting. The Annals of Statistics,
    28(2):337-407, 2000.
    """

    _parameter_constraints = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "z_max": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
    }

    def __init__(
        self, n_estimators=100, max_depth=3, learning_rate=1.0, z_max=4.0, random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.z_max = z_max
        self.random_state = random_state

    def _tree(self, rng):
        return DecisionTreeRegressor(
            max_depth=self.max_depth, random_state=rng.randint(np.iinfo(np.int32).max)
        )

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        rng = check_random_state(self.random_state)
        n = len(X)
        self.estimators_ = []
        if K == 1:
            return self
        if K == 2:
            F = np.zeros(n)  # half log-odds, as in the paper's two-class algorithm
            target = y_idx.astype(float)
            for _ in range(self.n_estimators):
                p = 1 / (1 + np.exp(-2 * F))
                w = np.maximum(p * (1 - p), 1e-10)
                z = np.clip((target - p) / w, -self.z_max, self.z_max)
                tree = self._tree(rng).fit(X, z, sample_weight=w)
                F += self.learning_rate * 0.5 * tree.predict(X)
                self.estimators_.append([tree])
            return self
        Y = np.eye(K)[y_idx]
        F = np.zeros((n, K))
        for _ in range(self.n_estimators):
            P = softmax(F, axis=1)
            fits = np.empty((n, K))
            trees = []
            for k in range(K):
                w = np.maximum(P[:, k] * (1 - P[:, k]), 1e-10)
                z = np.clip((Y[:, k] - P[:, k]) / w, -self.z_max, self.z_max)
                tree = self._tree(rng).fit(X, z, sample_weight=w)
                fits[:, k] = tree.predict(X)
                trees.append(tree)
            F += self.learning_rate * (K - 1) / K * (fits - fits.mean(axis=1, keepdims=True))
            self.estimators_.append(trees)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        K = len(self.classes_)
        if K == 2:
            F = np.zeros(len(X))
            for (tree,) in self.estimators_:
                F += self.learning_rate * 0.5 * tree.predict(X)
            return 2 * F  # log-odds of the second class
        F = np.zeros((len(X), K))
        for trees in self.estimators_:
            fits = np.column_stack([t.predict(X) for t in trees])
            F += self.learning_rate * (K - 1) / K * (fits - fits.mean(axis=1, keepdims=True))
        return F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if F.ndim == 1:
            p = 1 / (1 + np.exp(-F))
            return np.column_stack([1 - p, p])
        return softmax(F, axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
