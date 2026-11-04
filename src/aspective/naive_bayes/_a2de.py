"""Averaged two-dependence estimators (Webb, Boughton, Zheng, Ting & Salem, Machine Learning 2012)."""

from itertools import combinations
from numbers import Integral, Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from ._discrete import _DiscreteEncoderMixin


class A2DEClassifier(_DiscreteEncoderMixin, ClassifierMixin, BaseEstimator):
    """Averaged two-dependence estimators (A2DE).

    AODE averages over the models in which one feature is a parent of all
    the others. A2DE takes the next step of the AnDE family: it averages
    over all models in which a *pair* of features are parents of all the
    others,

        P(y, x) ~ sum_{i < j}  P(y, x_i, x_j) prod_{k != i, j} P(x_k | y, x_i, x_j),

    using only parent pairs whose value combination was seen at least
    ``min_count`` times. Like AODE it needs no structure search and no
    model selection, and its tables are plain frequency counts. Webb et
    al. showed that A2DE has lower bias than AODE and, given enough data,
    lower error: it beat AODE, naive Bayes, TAN and KDB in zero-one loss on
    most of 62 UCI data sets, above all the larger ones.

    Prefer A2DE over :class:`AODEClassifier` on data sets with thousands of
    rows; on small ones its larger tables gain less. Prediction costs one
    pass over the training data per parent pair, so it grows with the
    square of the number of features.

    A2DE works on discrete features. Set ``n_bins`` to discretize
    continuous features into quantile bins; with ``n_bins=None`` every
    distinct value is a category. Values not seen in training map to an
    extra, empty category. Rows with no usable parent pair fall back to
    naive Bayes, and with fewer than two features A2DE is naive Bayes.

    Parameters
    ----------
    n_bins : int or None, default=None
        Number of quantile bins per feature, or None for categorical input.
    min_count : int, default=1
        Minimum training frequency of a parent pair's values.
    alpha : float, default=1.0
        Additive (Laplace) smoothing.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    n_categories_ : ndarray of shape (n_features,)
        Number of categories per feature, including the unseen-value slot.

    References
    ----------
    G. I. Webb, J. R. Boughton, F. Zheng, K. M. Ting and H. Salem. Learning
    by Extrapolation from Marginal to Full-Multivariate Probability
    Distributions: Decreasingly Naive Bayesian Classification. Machine
    Learning, 86(2):233-272, 2012.
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

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_, self._y = np.unique(y, return_inverse=True)
        self._codes = self._encode(X, fit=True)
        self.n_categories_ = np.array([len(c) + 1 for c in self._categories])
        n_classes = len(self.classes_)
        alpha = self.alpha
        counts = np.bincount(self._y, minlength=n_classes).astype(float)
        self._log_prior = np.log(counts + alpha) - np.log(len(self._y) + alpha * n_classes)
        # Naive Bayes tables for the fallback.
        self._nb = []
        for j, c in enumerate(self.n_categories_):
            table = np.zeros((n_classes, c))
            np.add.at(table, (self._y, self._codes[:, j]), 1.0)
            self._nb.append(np.log(table + alpha) - np.log(counts[:, None] + alpha * c))
        return self

    def _joint_log_likelihood(self, X):
        codes = self._encode(X, fit=False)
        m, p = codes.shape
        K = len(self.classes_)
        alpha, n = self.alpha, len(self._y)
        cats = self.n_categories_
        width = int(cats.max())
        nb = np.tile(self._log_prior, (m, 1))
        for j, table in enumerate(self._nb):
            nb += table[:, codes[:, j]].T
        if p < 2:
            return nb
        terms = []
        others_all = np.arange(p)
        for i, j in combinations(range(p), 2):
            ci, cj = int(cats[i]), int(cats[j])
            parent = (self._y * ci + self._codes[:, i]) * cj + self._codes[:, j]
            c3 = np.bincount(parent, minlength=K * ci * cj).reshape(K, ci, cj).astype(float)
            # Counts of (y, x_i, x_j, k, x_k) for every other feature k.
            flat = ((parent[:, None] * p + others_all[None, :]) * width + self._codes).ravel()
            c4 = np.bincount(flat, minlength=K * ci * cj * p * width).reshape(K, ci, cj, p, width)
            vi, vj = codes[:, i], codes[:, j]
            pair_count = c3[:, vi, vj].sum(axis=0)
            usable = pair_count >= max(self.min_count, 1)
            logp = (np.log(c3[:, vi, vj] + alpha) - np.log(n + alpha * K * ci * cj)).T  # (m, K)
            others = np.delete(others_all, [i, j])
            num = c4[:, vi[:, None], vj[:, None], others[None, :], codes[:, others]]  # (K, m, p-2)
            den = c3[:, vi, vj][:, :, None] + alpha * cats[others][None, None, :]
            logp += np.log((num + alpha) / den).sum(axis=2).T
            terms.append(np.where(usable[:, None], logp, -np.inf))
        stacked = np.stack(terms)  # (pairs, m, K)
        any_usable = np.isfinite(stacked[:, :, 0]).any(axis=0)
        with np.errstate(invalid="ignore"):
            jll = logsumexp(stacked, axis=0)
        jll[~any_usable] = nb[~any_usable]
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
