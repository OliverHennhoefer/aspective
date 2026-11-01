"""Mixture Discriminant Analysis (Hastie & Tibshirani, JRSS-B 1996)."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class MixtureDiscriminantAnalysis(ClassifierMixin, BaseEstimator):
    """Mixture Discriminant Analysis.

    LDA models each class as one Gaussian, all sharing a covariance matrix,
    so its decision boundaries are linear and a class made of several
    clusters is summarized by a single centroid. MDA models each class as a
    mixture of ``n_subclasses`` Gaussians, still with one covariance matrix
    shared by every subclass of every class, and fits the mixtures jointly by
    EM. Boundaries become piecewise smooth while the number of covariance
    parameters stays that of LDA, so MDA does not overfit like QDA does on
    small samples. ``n_subclasses=1`` is LDA.

    Prefer MDA over LDA when a class is multimodal or non-convex, for example
    a class that is the union of several groups. Tune ``n_subclasses`` by
    cross-validation; Hastie & Tibshirani used 2 to 5.

    EM starts from k-means within each class (the paper's initialization)
    and stops when the log-likelihood improves by less than ``tol``
    relative to its size.

    Parameters
    ----------
    n_subclasses : int, default=3
        Mixture components per class, capped at the class size.
    max_iter : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-5
        Relative log-likelihood tolerance for stopping EM.
    reg_covar : float, default=1e-6
        Added to the diagonal of the shared covariance, relative to its mean
        variance, to keep it invertible.
    random_state : int, RandomState instance or None, default=None
        Seeds the k-means initialization.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
    means_ : list of ndarray of shape (n_subclasses_k, n_features)
        Subclass centroids of each class.
    weights_ : list of ndarray of shape (n_subclasses_k,)
        Mixing proportions within each class.
    covariance_ : ndarray of shape (n_features, n_features)
        The shared covariance matrix.
    priors_ : ndarray of shape (n_classes,)
    n_iter_ : int

    References
    ----------
    T. Hastie and R. Tibshirani. Discriminant Analysis by Gaussian Mixtures.
    Journal of the Royal Statistical Society, Series B, 58(1):155-176, 1996.
    """

    _parameter_constraints = {
        "n_subclasses": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "reg_covar": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(self, n_subclasses=3, max_iter=100, tol=1e-5, reg_covar=1e-6, random_state=None):
        self.n_subclasses = n_subclasses
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _component_log_density(self, X, means, chol):
        """log N(x; mu, Sigma) for every row of X and every row of means."""
        diff = X[:, None, :] - means[None, :, :]
        solved = cho_solve(chol, diff.reshape(-1, X.shape[1]).T).T.reshape(diff.shape)
        maha = np.einsum("nkp,nkp->nk", diff, solved)
        log_det = 2 * np.sum(np.log(np.diag(chol[0])))
        return -0.5 * (maha + log_det + X.shape[1] * np.log(2 * np.pi))

    def _shared_covariance(self, X, y_idx, resp, means):
        n, p = X.shape
        cov = np.zeros((p, p))
        for k, (r, mu) in enumerate(zip(resp, means)):
            Xk = X[y_idx == k]
            for s in range(len(mu)):
                d = Xk - mu[s]
                cov += (d * r[:, s : s + 1]).T @ d
        cov /= n
        cov.flat[:: p + 1] += self.reg_covar * max(np.trace(cov) / p, 1e-12)
        return cov

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        rng = check_random_state(self.random_state)
        n_classes = len(self.classes_)
        n, p = X.shape
        counts = np.bincount(y_idx, minlength=n_classes)
        self.priors_ = counts / n
        # Center and scale for numerical stability; the model is affine-equivariant.
        self._shift = X.mean(axis=0)
        scale = X.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        Z = (X - self._shift) / self._scale

        resp, means = [], []
        for k in range(n_classes):
            Zk = Z[y_idx == k]
            r = min(self.n_subclasses, len(np.unique(Zk, axis=0)))
            if r == 1:
                labels = np.zeros(len(Zk), dtype=int)
            else:
                labels = KMeans(r, n_init=3, random_state=rng.randint(2**31 - 1)).fit_predict(Zk)
            onehot = np.eye(r)[labels]
            resp.append(onehot)
            means.append(onehot.T @ Zk / onehot.sum(axis=0)[:, None])
        weights = [r.mean(axis=0) for r in resp]

        prev = -np.inf
        self.n_iter_ = 0
        for it in range(self.max_iter):
            cov = self._shared_covariance(Z, y_idx, resp, means)
            chol = cho_factor(cov, lower=True)
            # E-step: subclass responsibilities within each class.
            loglik = 0.0
            for k in range(n_classes):
                Zk = Z[y_idx == k]
                logp = self._component_log_density(Zk, means[k], chol) + np.log(weights[k])
                norm = logsumexp(logp, axis=1, keepdims=True)
                loglik += norm.sum()
                resp[k] = np.exp(logp - norm)
            self.n_iter_ = it + 1
            if loglik - prev <= self.tol * abs(loglik):
                break
            prev = loglik
            # M-step: subclass weights and means; the covariance is updated at the loop top.
            for k in range(n_classes):
                Zk = Z[y_idx == k]
                mass = resp[k].sum(axis=0)
                keep = mass > 1e-10
                resp[k] = resp[k][:, keep]
                mass = mass[keep]
                weights[k] = mass / mass.sum()
                means[k] = resp[k].T @ Zk / mass[:, None]

        cov = self._shared_covariance(Z, y_idx, resp, means)
        self._chol = cho_factor(cov, lower=True)
        self._means_z = means
        self._log_weights = [np.log(w) for w in weights]
        self.means_ = [m * self._scale + self._shift for m in means]
        self.weights_ = weights
        self.covariance_ = cov * np.outer(self._scale, self._scale)
        return self

    def _joint_log_likelihood(self, X):
        Z = (X - self._shift) / self._scale
        jll = np.empty((len(Z), len(self.classes_)))
        for k, (mu, logw) in enumerate(zip(self._means_z, self._log_weights)):
            jll[:, k] = logsumexp(self._component_log_density(Z, mu, self._chol) + logw, axis=1)
        return jll + np.log(self.priors_)

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
