"""DART: dropouts meet multiple additive regression trees (Rashmi & Gilad-Bachrach, AISTATS 2015)."""

from numbers import Integral, Real

import numpy as np
from scipy.special import expit, logsumexp, softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

_DOC_PARAMS = """
    Parameters
    ----------
    n_estimators : int, default=100
        Boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage ``eta``. With dropped trees the new tree gets weight
        ``eta / (k + eta)`` and the ``k`` dropped trees are scaled by
        ``k / (k + eta)``, so ``eta = 1`` is the paper's normalization.
    drop_rate : float in [0, 1], default=0.1
        Probability that each existing round is dropped.
    max_depth : int, default=3
        Depth of the regression trees.
    min_samples_leaf : int, default=1
    random_state : int, RandomState instance or None, default=None
"""


class _DARTBase(BaseEstimator):
    _parameter_constraints = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "drop_rate": [Interval(Real, 0, 1, closed="both")],
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, n_estimators=100, learning_rate=0.1, drop_rate=0.1, max_depth=3,
                 min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _boost(self, X, n_outputs, init, gradients):
        """Shared DART loop. ``gradients(F)`` returns per-output (gradient, hessian) arrays,
        each of shape (n, n_outputs); leaf values are Newton steps sum(g) / sum(h)."""
        rng = check_random_state(self.random_state)
        n = len(X)
        eta = self.learning_rate
        self.estimators_, self._leaf_values = [], []
        weights = []
        contrib = []  # unweighted training predictions of every round, (n, n_outputs)
        F = np.tile(init, (n, 1))
        for _ in range(self.n_estimators):
            T = len(weights)
            dropped = np.flatnonzero(rng.rand(T) < self.drop_rate) if T else np.empty(0, int)
            F_drop = F - sum((weights[i] * contrib[i] for i in dropped), np.zeros_like(F))
            g, h = gradients(F_drop)
            trees, leaves, pred = [], [], np.empty((n, n_outputs))
            for k in range(n_outputs):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                ).fit(X, g[:, k])
                leaf = tree.apply(X)
                n_nodes = tree.tree_.node_count
                num = np.bincount(leaf, weights=g[:, k], minlength=n_nodes)
                den = np.bincount(leaf, weights=h[:, k], minlength=n_nodes)
                values = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
                trees.append(tree)
                leaves.append(values)
                pred[:, k] = values[leaf]
            k_drop = len(dropped)
            if k_drop:
                new_w = eta / (k_drop + eta)
                scale = k_drop / (k_drop + eta)
                for i in dropped:
                    weights[i] *= scale
                F = F_drop + scale * (F - F_drop)
            else:
                new_w = eta
            weights.append(new_w)
            contrib.append(pred)
            F = F + new_w * pred
            self.estimators_.append(trees)
            self._leaf_values.append(leaves)
        self.estimator_weights_ = np.array(weights)
        self._init = np.asarray(init, dtype=float)

    def _raw_predict(self, X):
        n_outputs = len(self._init)
        F = np.tile(self._init, (len(X), 1))
        for trees, leaves, w in zip(self.estimators_, self._leaf_values, self.estimator_weights_):
            for k in range(n_outputs):
                F[:, k] += w * leaves[k][trees[k].apply(X)]
        return F


class DARTRegressor(RegressorMixin, _DARTBase):
    __doc__ = """Gradient-boosted regression trees with dropout (DART).

    In gradient boosting the first trees capture most of the signal and the
    later ones only correct small residuals, so later trees matter little
    and the model is sensitive to their exact number ("over-specialization").
    DART borrows dropout from neural networks: before each round, every
    existing tree is dropped with probability ``drop_rate``; the new tree is
    fitted to the residuals of the remaining ensemble; then the new tree
    and the dropped ones are rescaled so that the ensemble's output keeps
    its scale. Rashmi and Gilad-Bachrach found DART more accurate than
    plain MART on regression, ranking and classification tasks, and it is
    a booster option in XGBoost and LightGBM.

    Prefer DART over :class:`~sklearn.ensemble.GradientBoostingRegressor`
    when later trees overfit; it is slower to predict incrementally since
    every round may rescale earlier trees. Squared loss.
""" + _DOC_PARAMS + """
    Attributes
    ----------
    estimators_ : list of list of DecisionTreeRegressor
    estimator_weights_ : ndarray of shape (n_estimators,)

    References
    ----------
    K. V. Rashmi and R. Gilad-Bachrach. DART: Dropouts meet Multiple
    Additive Regression Trees. In Proceedings of the 18th International
    Conference on Artificial Intelligence and Statistics (AISTATS), PMLR
    38:489-497, 2015.
    """

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float32)
        y = y.astype(np.float64)
        self._boost(X, 1, [y.mean()], lambda F: ((y - F[:, 0])[:, None], np.ones((len(y), 1))))
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float32)
        return self._raw_predict(X)[:, 0]


class DARTClassifier(ClassifierMixin, _DARTBase):
    __doc__ = """Gradient-boosted classification trees with dropout (DART).

    The classification counterpart of :class:`DARTRegressor`, with the
    logistic loss for two classes and the multinomial loss (one tree per
    class and round) otherwise. Each round fits regression trees to the
    negative gradient of the loss of the ensemble without the dropped
    rounds and sets leaf values by a Newton step, as scikit-learn's
    gradient boosting does; dropped rounds and the new one are then
    rescaled as in the paper.
""" + _DOC_PARAMS + """
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    estimators_ : list of list of DecisionTreeRegressor
    estimator_weights_ : ndarray of shape (n_estimators,)

    References
    ----------
    K. V. Rashmi and R. Gilad-Bachrach. DART: Dropouts meet Multiple
    Additive Regression Trees. In Proceedings of the 18th International
    Conference on Artificial Intelligence and Statistics (AISTATS), PMLR
    38:489-497, 2015.
    """

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float32)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("DARTClassifier needs at least 2 classes; the data has 1 class.")
        prior = np.clip(np.bincount(y_idx) / len(y_idx), 1e-6, 1 - 1e-6)
        if K == 2:
            t = y_idx.astype(float)

            def grads(F):
                p = expit(F[:, 0])
                return (t - p)[:, None], np.maximum(p * (1 - p), 1e-12)[:, None]

            self._boost(X, 1, [np.log(prior[1] / prior[0])], grads)
        else:
            Y = np.eye(K)[y_idx]

            def grads(F):
                P = softmax(F, axis=1)
                # Newton step of scikit-learn's multinomial gradient boosting.
                return Y - P, np.maximum(P * (1 - P), 1e-12) * K / (K - 1)

            self._boost(X, K, np.log(prior), grads)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float32)
        F = self._raw_predict(X)
        return F[:, 0] if F.shape[1] == 1 else F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if F.ndim == 1:
            p = expit(F)
            return np.column_stack([1 - p, p])
        return np.exp(F - logsumexp(F, axis=1, keepdims=True))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
