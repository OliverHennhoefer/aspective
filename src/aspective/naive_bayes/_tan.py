"""Tree-Augmented Naive Bayes (Friedman, Geiger & Goldszmidt, Machine Learning 1997)."""

from numbers import Integral, Real

import numpy as np
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._discrete import _DiscreteEncoderMixin


def _conditional_mutual_information(codes, y_idx, n_categories, n_classes):
    """I(X_i; X_j | Y) for every pair of features, from empirical frequencies."""
    n, p = codes.shape
    cmi = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            ci, cj = n_categories[i], n_categories[j]
            flat = (y_idx * ci + codes[:, i]) * cj + codes[:, j]
            joint = np.bincount(flat, minlength=n_classes * ci * cj).reshape(n_classes, ci, cj) / n
            p_y = joint.sum(axis=(1, 2), keepdims=True)
            p_iy = joint.sum(axis=2, keepdims=True)
            p_jy = joint.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = joint * np.log(joint * p_y / (p_iy * p_jy))
            cmi[i, j] = cmi[j, i] = np.nansum(terms)
    return cmi


class TANClassifier(_DiscreteEncoderMixin, ClassifierMixin, BaseEstimator):
    """Tree-Augmented Naive Bayes.

    Naive Bayes treats all features as independent given the class. TAN lets
    every feature depend on the class and on at most one other feature, and
    learns which one: it builds the maximum-weight spanning tree over the
    features, weighting each pair by its conditional mutual information
    ``I(X_i; X_j | Y)`` (Chow & Liu, 1968), and roots the tree at the first
    feature. This is the best tree-shaped Bayesian network in likelihood
    among all networks where the class is a parent of every feature, and it
    is found in time quadratic in the number of features with no search.

    Prefer TAN over naive Bayes when features are correlated within classes.
    :class:`AODEClassifier` averages over all one-parent structures instead of
    learning one; TAN is cheaper to predict with and gives one readable
    dependency tree (``parents_``).

    TAN works on discrete features. Set ``n_bins`` to discretize continuous
    features into quantile bins; with ``n_bins=None`` every distinct value is
    a category. Values not seen in training map to an extra, empty category.

    Parameters
    ----------
    n_bins : int or None, default=None
        Number of quantile bins per feature, or None for categorical input.
    alpha : float, default=1.0
        Additive (Laplace) smoothing of every conditional probability table.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    parents_ : ndarray of shape (n_features,)
        Index of each feature's parent feature in the tree, -1 for the root.
    n_categories_ : ndarray of shape (n_features,)
        Number of categories per feature, including the unseen-value slot.

    References
    ----------
    N. Friedman, D. Geiger and M. Goldszmidt. Bayesian Network Classifiers.
    Machine Learning, 29(2-3):131-163, 1997.

    C. K. Chow and C. N. Liu. Approximating Discrete Probability Distributions
    with Dependence Trees. IEEE Transactions on Information Theory,
    14(3):462-467, 1968.
    """

    _parameter_constraints = {
        "n_bins": [Interval(Integral, 2, None, closed="left"), None],
        "alpha": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, n_bins=None, alpha=1.0):
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

        self.parents_ = np.full(p, -1, dtype=np.intp)
        if p > 1:
            cmi = _conditional_mutual_information(codes, y_idx, self.n_categories_, n_classes)
            # Maximum spanning tree as a minimum spanning tree of shifted weights;
            # the shift keeps every pair connected even when its CMI is zero.
            weights = cmi.max() + 1.0 - cmi
            np.fill_diagonal(weights, 0.0)
            tree = minimum_spanning_tree(weights)
            tree = tree + tree.T
            order, predecessors = breadth_first_order(tree, 0, directed=False)
            self.parents_[order[1:]] = predecessors[order[1:]]

        alpha = self.alpha
        class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        self._log_prior = np.log(class_counts + alpha) - np.log(n + alpha * n_classes)
        # One table per feature: log P(x_j | y, x_parent), shape (classes, parent values, values).
        self._log_tables = []
        for j in range(p):
            parent = self.parents_[j]
            n_parent = 1 if parent < 0 else self.n_categories_[parent]
            parent_codes = np.zeros(n, dtype=np.intp) if parent < 0 else codes[:, parent]
            counts = np.zeros((n_classes, n_parent, self.n_categories_[j]))
            np.add.at(counts, (y_idx, parent_codes, codes[:, j]), 1.0)
            totals = counts.sum(axis=2, keepdims=True)
            self._log_tables.append(
                np.log(counts + alpha) - np.log(totals + alpha * self.n_categories_[j])
            )
        return self

    def _joint_log_likelihood(self, X):
        codes = self._encode(X, fit=False)
        jll = np.tile(self._log_prior, (len(codes), 1))
        for j, table in enumerate(self._log_tables):
            parent = self.parents_[j]
            parent_codes = np.zeros(len(codes), dtype=np.intp) if parent < 0 else codes[:, parent]
            jll += table[:, parent_codes, codes[:, j]].T
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
