"""Averaged One-Dependence Estimators (Webb, Boughton & Wang, Machine Learning 2005)."""

from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class AODEClassifier(ClassifierMixin, BaseEstimator):
    """Averaged One-Dependence Estimators.

    Naive Bayes assumes all features are independent given the class. AODE
    instead averages over all models in which one feature (the super-parent)
    is a parent of every other feature::

        P(y, x) ~ sum_i  P(y, x_i) prod_{j != i} P(x_j | y, x_i)

    Only super-parents whose value was seen at least ``min_count`` times in
    training take part. When none qualifies the prediction falls back to
    naive Bayes. There is no model selection and training is a single pass
    over the data.

    AODE works on discrete features. Set ``n_bins`` to discretize continuous
    features into quantile bins; with ``n_bins=None`` every distinct value of a
    feature is treated as a category. Values not seen in training are mapped
    to an extra, empty category.

    Parameters
    ----------
    n_bins : int or None, default=None
        Number of quantile bins per feature, or None for categorical input.
    min_count : int, default=1
        Minimum training frequency of a feature value for it to act as a
        super-parent. Webb et al. (2005) used 30 for large datasets.
    alpha : float, default=1.0
        Additive (Laplace) smoothing.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_categories_ : ndarray of shape (n_features,)
        Number of categories per feature, including the unseen-value slot.

    References
    ----------
    G. I. Webb, J. R. Boughton and Z. Wang. Not So Naive Bayes: Aggregating
    One-Dependence Estimators. Machine Learning, 58(1):5-24, 2005.
    """

    _parameter_constraints = {
        "n_bins": [Interval(Integral, 2, None, closed="left"), None],
        "min_count": [Interval(Integral, 0, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, n_bins=None, min_count=1, alpha=1.0):
        self.n_bins = n_bins
        self.min_count = min_count
        self.alpha = alpha

    def _encode(self, X, fit):
        if self.n_bins is not None:
            if fit:
                self._binner = KBinsDiscretizer(
                    n_bins=self.n_bins, encode="ordinal", strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                )
                X = self._binner.fit_transform(X)
            else:
                X = self._binner.transform(X)
        if fit:
            self._categories = [np.unique(X[:, j]) for j in range(X.shape[1])]
        codes = np.empty(X.shape, dtype=np.intp)
        for j, cats in enumerate(self._categories):
            pos = np.searchsorted(cats, X[:, j])
            pos_clipped = np.minimum(pos, len(cats) - 1)
            seen = cats[pos_clipped] == X[:, j]
            codes[:, j] = np.where(seen, pos_clipped, len(cats))
        return codes

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        codes = self._encode(X, fit=True)
        n, p = codes.shape
        n_classes = len(self.classes_)

        # One global index per (feature, value); the last value of each
        # feature is the slot for values unseen in training.
        self.n_categories_ = np.array([len(c) + 1 for c in self._categories])
        self._offsets = np.concatenate([[0], np.cumsum(self.n_categories_)[:-1]])
        n_values = int(self.n_categories_.sum())
        flat = codes + self._offsets

        onehot = np.zeros((n, n_values))
        onehot[np.repeat(np.arange(n), p), flat.ravel()] = 1.0
        self._pair_counts = np.empty((n_classes, n_values, n_values))
        for k in range(n_classes):
            block = onehot[y_idx == k]
            self._pair_counts[k] = block.T @ block
        self._value_counts = onehot.sum(axis=0)
        self._class_counts = np.bincount(y_idx, minlength=n_classes).astype(float)
        self._n_samples = n
        return self

    def _joint_log_likelihood(self, X):
        codes = self._encode(X, fit=False)
        flat = codes + self._offsets
        n, p = flat.shape
        # Work in chunks: the gathered pair counts take n * n_classes * p**2 floats.
        chunk = max(1, int(2e7 // max(1, len(self.classes_) * p * p)))
        return np.vstack([self._jll_chunk(flat[s : s + chunk]) for s in range(0, n, chunk)])

    def _jll_chunk(self, flat):
        n, p = flat.shape
        n_classes = len(self.classes_)
        alpha = self.alpha
        card = self.n_categories_.astype(float)

        # counts[s, k, i, j] = N(y=k, x_i, x_j) for sample s; the diagonal holds N(y=k, x_i).
        counts = self._pair_counts[:, flat[:, :, None], flat[:, None, :]]
        counts = np.moveaxis(counts, 0, 1)
        parent = np.diagonal(counts, axis1=2, axis2=3)  # (n, k, p)

        log_parent = np.log(parent + alpha) - np.log(self._n_samples + alpha * n_classes * card)
        log_child = np.log(counts + alpha) - np.log(parent[..., None] + alpha * card[None, None, None, :])
        eye = np.eye(p, dtype=bool)
        log_child = np.where(eye, 0.0, log_child)
        spode = log_parent + log_child.sum(axis=3)  # (n, k, p): one score per super-parent

        qualifies = self._value_counts[flat] >= max(self.min_count, 1)  # (n, p)
        masked = np.where(qualifies[:, None, :], spode, -np.inf)
        jll = logsumexp(masked, axis=2)

        no_parent = ~qualifies.any(axis=1)
        if no_parent.any():
            class_counts = self._class_counts
            log_prior = np.log(class_counts + alpha) - np.log(self._n_samples + alpha * n_classes)
            nb = log_prior[None, :] + (
                np.log(parent[no_parent] + alpha)
                - np.log(class_counts[None, :, None] + alpha * card[None, None, :])
            ).sum(axis=2)
            jll[no_parent] = nb
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
