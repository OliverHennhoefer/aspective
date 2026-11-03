"""k-dependence Bayesian classifier (Sahami, KDD 1996)."""

from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._discrete import _DiscreteEncoderMixin
from ._tan import _conditional_mutual_information


def _mutual_information_with_class(codes, y_idx, n_categories, n_classes):
    """I(X_i; Y) for every feature, from empirical frequencies."""
    n = len(codes)
    mi = np.zeros(codes.shape[1])
    for i in range(codes.shape[1]):
        joint = np.bincount(y_idx * n_categories[i] + codes[:, i],
                            minlength=n_classes * n_categories[i]).reshape(n_classes, -1) / n
        outer = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi[i] = np.nansum(joint * np.log(joint / outer))
    return mi


class KDBClassifier(_DiscreteEncoderMixin, ClassifierMixin, BaseEstimator):
    """k-dependence Bayesian classifier (KDB).

    Naive Bayes lets every feature depend on the class only. KDB lets each
    feature depend on the class and on up to ``k`` other features, which
    gives a dial between naive Bayes (``k=0``) and the full Bayesian
    network. The structure is found greedily without search: features are
    ordered by their mutual information with the class, ``I(X_i; Y)``, and
    each feature takes as parents the ``k`` features earlier in that order
    with the highest conditional mutual information ``I(X_i; X_j | Y)``.
    Sahami showed that going from ``k=0`` to ``k=1`` or ``2`` reduces the
    error of naive Bayes on most UCI data sets, and Webb et al. (2012)
    found KDB with ``k=1`` or ``2`` among the most accurate Bayesian network
    classifiers on large data.

    Prefer KDB over naive Bayes and :class:`TANClassifier` when there is
    enough data to estimate the larger tables: with ``k=2`` every feature
    has a table over the class and two parents. :class:`AODEClassifier`
    is the safer choice on small data sets.

    KDB works on discrete features. Set ``n_bins`` to discretize continuous
    features into quantile bins; with ``n_bins=None`` every distinct value is
    a category. Values not seen in training map to an extra, empty category.

    Parameters
    ----------
    k : int, default=2
        Largest number of feature parents per feature.
    n_bins : int or None, default=None
        Number of quantile bins per feature, or None for categorical input.
    alpha : float, default=1.0
        Additive (Laplace) smoothing of every conditional probability table.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    parents_ : list of ndarray
        Feature parents of each feature (the class is always a parent too).
    order_ : ndarray of shape (n_features,)
        Features sorted by mutual information with the class, decreasing.
    n_categories_ : ndarray of shape (n_features,)
        Number of categories per feature, including the unseen-value slot.

    References
    ----------
    M. Sahami. Learning Limited Dependence Bayesian Classifiers. In
    Proceedings of the Second International Conference on Knowledge
    Discovery and Data Mining (KDD), pages 335-338, 1996.

    G. I. Webb, J. R. Boughton, F. Zheng, K. M. Ting and H. Salem. Learning
    by Extrapolation from Marginal to Full-Multivariate Probability
    Distributions: Decreasingly Naive Bayesian Classification. Machine
    Learning, 86(2):233-272, 2012.
    """

    _parameter_constraints = {
        "k": [Interval(Integral, 0, None, closed="left")],
        "n_bins": [Interval(Integral, 2, None, closed="left"), None],
        "alpha": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, k=2, n_bins=None, alpha=1.0):
        self.k = k
        self.n_bins = n_bins
        self.alpha = alpha

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        codes = self._encode(X, fit=True)
        n, p = codes.shape
        n_classes = len(self.classes_)
        self.n_categories_ = np.array([len(c) + 1 for c in self._categories])

        mi = _mutual_information_with_class(codes, y_idx, self.n_categories_, n_classes)
        self.order_ = np.argsort(-mi, kind="stable")
        cmi = (_conditional_mutual_information(codes, y_idx, self.n_categories_, n_classes)
               if p > 1 and self.k > 0 else np.zeros((p, p)))
        self.parents_ = [np.empty(0, dtype=np.intp) for _ in range(p)]
        for pos, i in enumerate(self.order_):
            earlier = self.order_[:pos]
            if len(earlier) and self.k > 0:
                best = earlier[np.argsort(-cmi[i, earlier], kind="stable")[: self.k]]
                self.parents_[i] = np.sort(best)

        alpha = self.alpha
        class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        self._log_prior = np.log(class_counts + alpha) - np.log(n + alpha * n_classes)
        # One table per feature: log P(x_i | y, parents), the parents' values flattened.
        self._log_tables = []
        for i in range(p):
            parent_codes, n_parent = self._parent_codes(codes, i)
            counts = np.zeros((n_classes, n_parent, self.n_categories_[i]))
            np.add.at(counts, (y_idx, parent_codes, codes[:, i]), 1.0)
            totals = counts.sum(axis=2, keepdims=True)
            self._log_tables.append(
                np.log(counts + alpha) - np.log(totals + alpha * self.n_categories_[i])
            )
        return self

    def _parent_codes(self, codes, i):
        flat = np.zeros(len(codes), dtype=np.intp)
        size = 1
        for j in self.parents_[i]:
            flat = flat * self.n_categories_[j] + codes[:, j]
            size *= int(self.n_categories_[j])
        return flat, size

    def _joint_log_likelihood(self, X):
        codes = self._encode(X, fit=False)
        jll = np.tile(self._log_prior, (len(codes), 1))
        for i, table in enumerate(self._log_tables):
            parent_codes, _ = self._parent_codes(codes, i)
            jll += table[:, parent_codes, codes[:, i]].T
        return jll

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
