"""Additive model with P-splines (Hastie & Tibshirani, 1986; Eilers & Marx, 1996)."""

from numbers import Integral, Real

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class PSplineGAMRegressor(RegressorMixin, BaseEstimator):
    """Additive model with penalized B-splines (P-splines), smoothness chosen by GCV.

    Linear regression fits one slope per feature. A generalized additive
    model (Hastie & Tibshirani, 1986) replaces each slope by a smooth
    function, ``y = b0 + f_1(x_1) + ... + f_p(x_p)``, which keeps the model
    additive and readable (one curve per feature) while capturing
    nonlinear effects. Following Eilers and Marx, each ``f_j`` is a cubic
    B-spline on ``n_knots`` equally spaced knots with a second-order
    difference penalty on neighbouring coefficients. The penalty shrinks
    every curve toward a straight line, so the strongest smoothing gives
    back ordinary linear regression within the training range. The penalty
    weight is chosen by generalized cross-validation from a log-spaced
    grid, all at the cost of one eigendecomposition.

    Features with fewer than ``min_unique`` distinct values enter
    linearly. Outside the training range each curve stays at its value at
    the boundary: a linearly extrapolated spline can reach values far
    outside anything seen in training on a single unusual point. One
    smoothing parameter is shared by all features (on standardized
    curves), as in the ``mgcv`` "one smoothing parameter" setting; use it
    for moderate numbers of features.

    Parameters
    ----------
    n_knots : int, default=10
        Equally spaced knots per feature (``n_knots + 2`` basis functions).
    alphas : array-like or None, default=None
        Candidate penalty weights; None uses 25 values from 1e-4 to 1e6,
        relative to the average curvature of the basis.
    min_unique : int, default=6
        Features with fewer distinct training values enter linearly.

    Attributes
    ----------
    alpha_ : float
        Selected penalty weight.
    edf_ : float
        Effective degrees of freedom of the fit.
    gcv_ : ndarray of shape (n_alphas,)
    coef_ : ndarray of shape (n_basis,)
    intercept_ : float
    spline_features_ : ndarray of int
        Features modelled by splines; the others are linear.

    References
    ----------
    T. Hastie and R. Tibshirani. Generalized Additive Models. Statistical
    Science, 1(3):297-310, 1986.

    P. H. C. Eilers and B. D. Marx. Flexible Smoothing with B-splines and
    Penalties. Statistical Science, 11(2):89-121, 1996.

    B. D. Marx and P. H. C. Eilers. Direct Generalized Additive Modeling
    with Penalized Likelihood. Computational Statistics & Data Analysis,
    28(2):193-209, 1998.
    """

    _parameter_constraints = {
        "n_knots": [Interval(Integral, 3, None, closed="left")],
        "alphas": ["array-like", None],
        "min_unique": [Interval(Integral, 2, None, closed="left")],
    }

    def __init__(self, n_knots=10, alphas=None, min_unique=6):
        self.n_knots = n_knots
        self.alphas = alphas
        self.min_unique = min_unique

    def _basis(self, X):
        blocks = []
        for j, spline in self._splines:
            blocks.append(spline.transform(X[:, [j]]))
        if len(self._linear):
            blocks.append(X[:, self._linear])
        if not blocks:
            return np.zeros((len(X), 0))
        return np.hstack(blocks) - self._center

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        n, p = X.shape
        self._splines, penalties, linear = [], [], []
        for j in range(p):
            if len(np.unique(X[:, j])) >= self.min_unique:
                spline = SplineTransformer(
                    n_knots=self.n_knots, degree=3, knots="uniform", extrapolation="constant"
                ).fit(X[:, [j]])
                self._splines.append((j, spline))
                k = spline.n_features_out_
                D = np.diff(np.eye(k), n=2, axis=0)
                penalties.append(D.T @ D)
            else:
                linear.append(j)
        self._linear = np.array(linear, dtype=np.intp)
        self.spline_features_ = np.array([j for j, _ in self._splines], dtype=np.intp)
        blocks = [s.transform(X[:, [j]]) for j, s in self._splines]
        if len(linear):
            blocks.append(X[:, self._linear])
        B = np.hstack(blocks) if blocks else np.zeros((n, 0))
        self._center = B.mean(axis=0) if B.shape[1] else np.zeros(0)
        B = B - self._center
        q = B.shape[1]
        y_mean = y.mean()
        yc = y - y_mean
        P = (
            linalg.block_diag(*penalties, np.zeros((len(linear), len(linear))))
            if q
            else np.zeros((0, 0))
        )
        if q == 0:
            self.coef_, self.intercept_, self.alpha_, self.edf_ = (
                np.zeros(0),
                float(y_mean),
                0.0,
                1.0,
            )
            self.gcv_ = np.zeros(0)
            return self
        G = B.T @ B
        # Each spline block spans the constant, which centring removed: a tiny ridge keeps
        # G positive definite without changing the fit noticeably.
        G = G + 1e-8 * max(np.trace(G) / q, 1e-12) * np.eye(q)
        scale = np.trace(G) / max(np.trace(P), 1e-300)
        L = linalg.cholesky(G, lower=True)
        Linv = linalg.solve_triangular(L, np.eye(q), lower=True)
        S = Linv @ P @ Linv.T * scale
        lam, U = linalg.eigh((S + S.T) / 2)
        lam = np.maximum(lam, 0.0)
        z = U.T @ (Linv @ (B.T @ yc))
        BW = B @ (Linv.T @ U)  # fitted values are BW @ (shrink * z), z = BW' yc
        # trace(BW diag(shrink) BW'): the ridge on G leaves the directions that centring
        # removed with near-zero norm, so they must not count as degrees of freedom.
        col_norm = np.einsum("ij,ij->j", BW, BW)
        alphas = (
            np.geomspace(1e-4, 1e6, 25) if self.alphas is None else np.asarray(self.alphas, float)
        )
        gcv = np.empty(len(alphas))
        for a, alpha in enumerate(alphas):
            shrink = 1.0 / (1.0 + alpha * lam)
            fitted = BW @ (shrink * z)
            edf = 1.0 + shrink @ col_norm
            rss = float(np.sum((yc - fitted) ** 2))
            gcv[a] = n * rss / max(n - edf, 1e-12) ** 2
        best = int(np.argmin(gcv))
        self.gcv_, self.alpha_ = gcv, float(alphas[best])
        shrink = 1.0 / (1.0 + self.alpha_ * lam)
        self.edf_ = float(1.0 + shrink @ col_norm)
        self.coef_ = Linv.T @ (U @ (shrink * z))
        self.intercept_ = float(y_mean)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._basis(X) @ self.coef_ + self.intercept_
