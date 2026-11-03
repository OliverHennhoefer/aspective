"""Hidden naive Bayes (Zhang, Jiang & Su, AAAI 2005)."""

from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._discrete import _DiscreteEncoderMixin
from ._tan import _conditional_mutual_information


class HiddenNaiveBayesClassifier(_DiscreteEncoderMixin, ClassifierMixin, BaseEstimator):
    """Hidden naive Bayes (HNB).

    Naive Bayes treats the features as independent given the class. HNB
    gives every feature a hidden parent that blends the influence of all
    other features: the conditional probability of feature ``i`` is a
    mixture of one-dependence estimates,

        P(x_i | hp_i, y) = sum_{j != i} w_ij P(x_i | x_j, y),

    with weights proportional to the conditional mutual information
    ``I(X_i; X_j | Y)``, so strongly related features count most. There is
    no structure search, every table has only two dimensions besides the
    class, and training is a single pass plus the mutual-information
    matrix. Zhang, Jiang and Su found HNB significantly more accurate than
    naive Bayes, TAN and AODE on 36 UCI data sets.

    Prefer HNB over naive Bayes when features are correlated within
    classes and you want one model, not an average, with the same table
    sizes as :class:`AODEClassifier`. With one feature it is naive Bayes.

    HNB works on discrete features. Set ``n_bins`` to discretize continuous
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
    weights_ : ndarray of shape (n_features, n_features)
        Mixture weight of feature ``j`` in the hidden parent of feature
        ``i``; rows sum to one, the diagonal is zero.
    n_categories_ : ndarray of shape (n_features,)
        Number of categories per feature, including the unseen-value slot.

    References
    ----------
    H. Zhang, L. Jiang and J. Su. Hidden Naive Bayes. In Proceedings of the
    20th National Conference on Artificial Intelligence (AAAI), pages
    919-924, 2005.

    L. Jiang, H. Zhang and Z. Cai. A Novel Bayes Model: Hidden Naive Bayes.
    IEEE Transactions on Knowledge and Data Engineering, 21(10):1361-1371,
    2009.
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
        cats = self.n_categories_ = np.array([len(c) + 1 for c in self._categories])
        alpha = self.alpha

        class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        self._log_prior = np.log(class_counts + alpha) - np.log(n + alpha * n_classes)

        if p > 1:
            cmi = _conditional_mutual_information(codes, y_idx, cats, n_classes)
            np.fill_diagonal(cmi, 0.0)
            cmi = np.maximum(cmi, 0.0)
            totals = cmi.sum(axis=1, keepdims=True)
            uniform = (1.0 - np.eye(p)) / (p - 1)
            self.weights_ = np.where(totals > 0, cmi / np.where(totals > 0, totals, 1.0), uniform)
        else:
            self.weights_ = np.zeros((1, 1))

        # Naive Bayes tables, used when there is a single feature.
        self._nb_tables = []
        for i in range(p):
            counts = np.zeros((n_classes, cats[i]))
            np.add.at(counts, (y_idx, codes[:, i]), 1.0)
            self._nb_tables.append((counts + alpha) / (class_counts[:, None] + alpha * cats[i]))
        # tables[i][j]: P(x_i | x_j, y) with shape (classes, values of j, values of i).
        self._tables = []
        for i in range(p):
            row = []
            for j in range(p):
                if j == i or self.weights_[i, j] == 0:
                    row.append(None)
                    continue
                counts = np.zeros((n_classes, cats[j], cats[i]))
                np.add.at(counts, (y_idx, codes[:, j], codes[:, i]), 1.0)
                row.append((counts + alpha) / (counts.sum(axis=2, keepdims=True) + alpha * cats[i]))
            self._tables.append(row)
        return self

    def _joint_log_likelihood(self, X):
        codes = self._encode(X, fit=False)
        m, p = codes.shape
        jll = np.tile(self._log_prior, (m, 1))
        for i in range(p):
            if p == 1:
                prob = self._nb_tables[0][:, codes[:, 0]].T
            else:
                prob = np.zeros((m, len(self.classes_)))
                for j, table in enumerate(self._tables[i]):
                    if table is not None:
                        prob += self.weights_[i, j] * table[:, codes[:, j], codes[:, i]].T
            jll += np.log(prob)
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
