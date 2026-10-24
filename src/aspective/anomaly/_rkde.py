"""Robust kernel density estimation (Kim & Scott, JMLR 2012)."""

from numbers import Integral, Real

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector


def _psi_over_d(d, loss, a, b, c):
    """Weight function psi(d)/d of the robust loss (Kim & Scott, Sec. 3)."""
    d = np.maximum(d, 1e-12)
    if loss == "huber":
        return np.minimum(1.0, a / d)
    # Hampel: identity up to a, constant to b, linear descent to zero at c.
    w = np.ones_like(d)
    w = np.where((d > a) & (d <= b), a / d, w)
    w = np.where((d > b) & (d <= c), a * (c - d) / ((c - b) * d), w)
    return np.where(d > c, 0.0, w)


class RobustKDEDetector(BaseDetector):
    """Anomaly detection with a robust kernel density estimate.

    A kernel density estimate is a mean in the kernel's feature space, so a
    few contaminating points pull it toward themselves just as outliers pull
    a sample mean. Kim & Scott replace that mean by an M-estimator with a
    Huber or Hampel loss, fitted by kernelized iteratively reweighted least
    squares (KIRWLS). Points far from the bulk in feature space get smaller
    weights, so the density is closer to the clean distribution and the
    outliers stand out more clearly.

    The density is a weighted sum of Gaussian kernels on the training points,
    and the anomaly score is the negative log density.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points, and ``predict`` returns 0 for inliers and 1 for
    outliers.

    The Hampel loss is redescending: points far enough from the bulk get
    zero weight. When the data holds clusters of very different density,
    the iterations can drop a whole sparse cluster, which then scores as
    anomalous. Use ``loss="huber"`` in that case.

    Parameters
    ----------
    loss : {"huber", "hampel"}, default="hampel"
        Robust loss. Thresholds follow the paper: for Huber, ``a`` is the
        median of the feature-space distances under the plain KDE; for
        Hampel, ``a``, ``b`` and ``c`` are their 50th, 75th and 95th
        percentiles.
    bandwidth : float or "scott", default="scott"
        Gaussian kernel bandwidth. "scott" applies Scott's rule to the mean
        feature standard deviation.
    max_iter : int, default=100
    tol : float, default=1e-6
        Stop when the relative change of the objective falls below ``tol``.
    max_samples : int or None, default=None
        Fit on a random subsample of this size. The fit needs the full
        n-by-n kernel matrix, so subsample above a few thousand points.
    contamination : float in (0, 0.5], default=0.1
    random_state : int, RandomState instance or None, default=None
        Used only for subsampling.

    Attributes
    ----------
    weights_ : ndarray of shape (n_kernels,)
        Robust weights of the kernel centers; they sum to one.
    bandwidth_ : float
    n_iter_ : int
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)

    References
    ----------
    J. Kim and C. D. Scott. Robust Kernel Density Estimation. Journal of
    Machine Learning Research, 13:2529-2565, 2012.
    """

    def __init__(
        self, loss="hampel", bandwidth="scott", max_iter=100, tol=1e-6, max_samples=None,
        contamination=0.1, random_state=None,
    ):
        super().__init__(contamination=contamination)
        self.loss = loss
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.max_samples = max_samples
        self.random_state = random_state

    def _kernel(self, A, B):
        d = A.shape[1]
        h = self.bandwidth_
        norm = (2 * np.pi * h**2) ** (-d / 2)
        return norm * np.exp(-cdist(A, B, "sqeuclidean") / (2 * h**2))

    def fit(self, X, y=None):
        X = validate_data(self, X)
        if self.loss not in ("huber", "hampel"):
            raise ValueError(f"loss must be 'huber' or 'hampel', got {self.loss!r}.")
        if not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")
        self._set_n_classes(y)
        n, d = X.shape
        centers = X
        if self.max_samples is not None and self.max_samples < n:
            rng = check_random_state(self.random_state)
            centers = X[rng.choice(n, self.max_samples, replace=False)]
        m = centers.shape[0]

        if self.bandwidth == "scott":
            sigma = np.mean(X.std(axis=0))
            self.bandwidth_ = float(max(sigma, 1e-12) * m ** (-1.0 / (d + 4)))
        elif isinstance(self.bandwidth, Real) and self.bandwidth > 0:
            self.bandwidth_ = float(self.bandwidth)
        else:
            raise ValueError("bandwidth must be a positive float or 'scott'.")

        K = self._kernel(centers, centers)
        k0 = K[0, 0]

        def feature_distances(w):
            Kw = K @ w
            sq = k0 - 2 * Kw + w @ Kw
            return np.sqrt(np.maximum(sq, 0.0))

        w = np.full(m, 1.0 / m)
        dist = feature_distances(w)
        if self.loss == "huber":
            a, b, c = np.median(dist), None, None
        else:
            a, b, c = np.percentile(dist, [50, 75, 95])
            b, c = max(b, a + 1e-12), max(c, b + 2e-12)

        def objective(dist):
            if self.loss == "huber":
                return np.sum(np.where(dist <= a, dist**2 / 2, a * dist - a**2 / 2))
            rho = np.where(dist <= a, dist**2 / 2, a * dist - a**2 / 2)
            rho = np.where(
                (dist > b) & (dist <= c),
                -a / (2 * (c - b)) * (c - dist) ** 2 + a * (b + c - a) / 2,
                rho,
            )
            return np.sum(np.where(dist > c, a * (b + c - a) / 2, rho))

        previous = objective(dist)
        self.n_iter_ = 0
        for self.n_iter_ in range(1, self.max_iter + 1):
            phi = _psi_over_d(dist, self.loss, a, b, c)
            if phi.sum() <= 0:
                break
            w = phi / phi.sum()
            dist = feature_distances(w)
            current = objective(dist)
            if abs(previous - current) <= self.tol * max(abs(previous), 1e-12):
                break
            previous = current

        self.weights_ = w
        self._centers = centers
        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()
        return self

    def score_log_samples(self, X):
        """Log of the robust density estimate at each point, computed stably."""
        check_is_fitted(self, "weights_")
        X = validate_data(self, X, reset=False)
        d = X.shape[1]
        h = self.bandwidth_
        log_norm = -0.5 * d * np.log(2 * np.pi * h**2)
        with np.errstate(divide="ignore"):
            log_w = np.log(self.weights_)
        out = np.empty(X.shape[0])
        for s in range(0, X.shape[0], 2048):
            sq = cdist(X[s : s + 2048], self._centers, "sqeuclidean")
            out[s : s + 2048] = logsumexp(log_w[None, :] - sq / (2 * h**2), axis=1) + log_norm
        return out

    def score_samples(self, X):
        """Robust density estimate at each point."""
        return np.exp(self.score_log_samples(X))

    def decision_function(self, X):
        return -self.score_log_samples(X)
