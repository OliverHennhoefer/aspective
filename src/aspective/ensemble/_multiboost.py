"""MultiBoost (Webb, Machine Learning 2000)."""

from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class MultiBoostClassifier(ClassifierMixin, BaseEstimator):
    """MultiBoost: AdaBoost committees combined by wagging.

    AdaBoost mainly reduces bias; bagging and its continuous variant,
    wagging (random instance weights), mainly reduce variance. MultiBoost
    gets both: it runs AdaBoost, but about every ``sqrt(n_estimators)``
    rounds it ends the current sub-committee and restarts from fresh random
    weights drawn from a continuous Poisson distribution. Restarts also
    happen when a member is perfect (error 0) or no better than chance,
    where AdaBoost would have to stop. All members vote with their AdaBoost
    weights. Webb found MultiBoost more accurate than both AdaBoost and
    bagging on most of 36 UCI data sets, at the same cost as AdaBoost and
    with sub-committees that can be trained in parallel.

    The weights follow the multiclass SAMME rule (Zhu et al., 2009), so
    with a single committee and no restarts MultiBoost is
    :class:`~sklearn.ensemble.AdaBoostClassifier`.

    Parameters
    ----------
    estimator : classifier or None, default=None
        Base learner with ``sample_weight`` support; None uses a decision
        tree of depth 3.
    n_estimators : int, default=100
        Committee size ``T``.
    n_subcommittees : int or None, default=None
        Number of wagged sub-committees; None uses ``floor(sqrt(T))``, the
        paper's choice.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    estimators_ : list of classifiers
    estimator_weights_ : ndarray of shape (n_members,)

    References
    ----------
    G. I. Webb. MultiBoosting: A Technique for Combining Boosting and
    Wagging. Machine Learning, 40(2):159-196, 2000.

    J. Zhu, H. Zou, S. Rosset and T. Hastie. Multi-class AdaBoost.
    Statistics and Its Interface, 2(3):349-360, 2009.
    """

    _parameter_constraints = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "n_subcommittees": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
    }

    def __init__(self, estimator=None, n_estimators=100, n_subcommittees=None, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.n_subcommittees = n_subcommittees
        self.random_state = random_state

    @staticmethod
    def _poisson_weights(n, rng):
        # Webb's continuous Poisson: -log(Uniform(1..999) / 1000), normalized.
        w = -np.log(rng.randint(1, 1000, size=n) / 1000.0)
        return w / w.sum()

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        n = len(X)
        rng = check_random_state(self.random_state)
        base = DecisionTreeClassifier(max_depth=3) if self.estimator is None else self.estimator
        T = self.n_estimators
        n_sub = self.n_subcommittees or max(1, int(np.floor(np.sqrt(T))))
        # Members after which a new sub-committee starts: I_i = ceil(i T / n_sub).
        boundaries = {int(np.ceil(i * T / n_sub)) for i in range(1, n_sub)}

        self.estimators_, weights = [], []
        w = np.full(n, 1.0 / n)
        attempts = 0
        while len(self.estimators_) < T and attempts < 2 * T:
            attempts += 1
            if len(self.estimators_) in boundaries and attempts > 1:
                boundaries.discard(len(self.estimators_))
                w = self._poisson_weights(n, rng)
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=rng.randint(np.iinfo(np.int32).max))
            est.fit(X, y_idx, sample_weight=w * n)
            miss = est.predict(X) != y_idx
            err = float(np.sum(w[miss]))
            if K == 1 or err <= 0:
                self.estimators_.append(est)
                weights.append(10.0 + np.log(max(K - 1, 1)))  # a perfect member gets a large vote
                w = self._poisson_weights(n, rng)
                continue
            if err >= 1 - 1.0 / K:  # no better than chance: discard, restart from random weights
                w = self._poisson_weights(n, rng)
                continue
            alpha = np.log((1 - err) / err) + np.log(K - 1)
            self.estimators_.append(est)
            weights.append(alpha)
            w = w * np.exp(alpha * miss)
            w = np.maximum(w / w.sum(), 1e-8)
            w /= w.sum()
        if not self.estimators_:  # every member was no better than chance: keep the last one
            self.estimators_.append(est)
            weights.append(1.0)
        self.estimator_weights_ = np.array(weights)
        return self

    def _votes(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        votes = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))
        for est, a in zip(self.estimators_, self.estimator_weights_):
            votes[rows, est.predict(X)] += a
        return votes / self.estimator_weights_.sum()

    def decision_function(self, X):
        """Normalized weighted votes; for two classes, the vote margin of the second."""
        votes = self._votes(X)
        return votes[:, 1] - votes[:, 0] if votes.shape[1] == 2 else votes

    def predict_proba(self, X):
        votes = self._votes(X)
        K = len(self.classes_)
        if K == 1:
            return np.ones((len(votes), 1))
        # SAMME.R-free mapping of normalized votes to probabilities, as in scikit-learn.
        scores = np.exp(votes / (K - 1))
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        votes = self._votes(X)
        return self.classes_[np.argmax(votes, axis=1)]
