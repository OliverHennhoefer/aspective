"""Flexible naive Bayes with kernel density estimates (John & Langley, UAI 1995)."""

from numbers import Real

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class FlexibleNaiveBayes(ClassifierMixin, BaseEstimator):
    """Naive Bayes with a kernel density estimate per feature and class.

    Gaussian naive Bayes models each feature within each class by a single
    normal distribution, which fails for skewed, multimodal or discrete
    features. Flexible naive Bayes keeps the naive independence assumption
    but estimates each class-conditional density nonparametrically, as an
    average of Gaussian kernels centred on the training values. John and
    Langley proved this is consistent for any continuous density and found
    it more accurate than Gaussian naive Bayes on most of the UCI data sets
    they tried, and never much worse.

    Features are standardized with the training mean and standard
    deviation, and the kernel width of class ``c`` is ``1 / sqrt(n_c)`` on
    that scale as in the paper (``bandwidth="john-langley"``), or
    Silverman's rule per feature and class (``bandwidth="silverman"``).

    Prefer it over :class:`~sklearn.naive_bayes.GaussianNB` when features
    are far from normal. Prediction costs one pass over the training data
    per query, like a nearest-neighbour method.

    Parameters
    ----------
    bandwidth : {"john-langley", "silverman"} or float, default="john-langley"
        Kernel width on the standardized scale.
    min_bandwidth : float, default=1e-3
        Lower bound of the kernel width, so constant features stay finite.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    class_prior_ : ndarray of shape (n_classes,)
    bandwidth_ : ndarray of shape (n_classes, n_features)

    References
    ----------
    G. H. John and P. Langley. Estimating Continuous Distributions in
    Bayesian Classifiers. In Proceedings of the Eleventh Conference on
    Uncertainty in Artificial Intelligence (UAI), pages 338-345, 1995.
    """

    _parameter_constraints = {
        "bandwidth": [
            StrOptions({"john-langley", "silverman"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "min_bandwidth": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, bandwidth="john-langley", min_bandwidth=1e-3):
        self.bandwidth = bandwidth
        self.min_bandwidth = min_bandwidth

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        Z = (X - self._mean) / self._sd
        counts = np.bincount(y_idx, minlength=len(self.classes_))
        self.class_prior_ = counts / counts.sum()
        self._Z = [Z[y_idx == c] for c in range(len(self.classes_))]
        widths = []
        for Zc in self._Z:
            n_c = len(Zc)
            if self.bandwidth == "john-langley":
                h = np.full(Z.shape[1], 1.0 / np.sqrt(n_c))
            elif self.bandwidth == "silverman":
                h = 1.06 * Zc.std(axis=0) * n_c ** (-0.2)
            else:
                h = np.full(Z.shape[1], float(self.bandwidth))
            widths.append(np.maximum(h, self.min_bandwidth))
        self.bandwidth_ = np.array(widths)
        return self

    def _joint_log_likelihood(self, X):
        Z = (X - self._mean) / self._sd
        jll = np.empty((len(Z), len(self.classes_)))
        for c, Zc in enumerate(self._Z):
            h = self.bandwidth_[c]
            n_c = len(Zc)
            total = np.zeros(len(Z))
            batch = max(1, int(2e6 // max(n_c, 1)))
            for j in range(Z.shape[1]):
                for s in range(0, len(Z), batch):
                    u = (Z[s : s + batch, j, None] - Zc[None, :, j]) / h[j]
                    total[s : s + batch] += logsumexp(-0.5 * u * u, axis=1)
            norm = np.sum(np.log(h)) + Z.shape[1] * (0.5 * np.log(2 * np.pi) + np.log(n_c))
            jll[:, c] = np.log(self.class_prior_[c]) + total - norm
        return jll

    def predict_log_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        jll = self._joint_log_likelihood(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
