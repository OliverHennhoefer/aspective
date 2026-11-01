"""Multivariate LOESS (Cleveland & Devlin, JASA 1988)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


class LoessRegressor(RegressorMixin, BaseEstimator):
    """Locally weighted linear or quadratic regression (LOESS).

    k-nearest-neighbour regression predicts the plain average of the k
    closest training responses: it is biased wherever the regression
    function has a slope, above all near the edges of the data, and its
    prediction surface is a step function. LOESS fits, at every query
    point, a weighted least-squares line (``degree=1``) or quadratic
    (``degree=2``) to the nearest ``span`` fraction of the training data,
    with tricube weights ``(1 - (d / d_max) ** 3) ** 3`` that fall smoothly to
    zero at the farthest neighbour, and predicts its value at the query.
    The local fit removes the first-order bias of local averaging (Fan,
    1993), and the prediction surface is continuous.

    Inputs are standardized before distances are taken, as Cleveland &
    Devlin recommend for predictors on different scales. A tiny ridge
    term keeps each local fit solvable when a neighbourhood is degenerate.
    The robustness iterations of Cleveland (1979) are not included; this is
    R's ``loess(family = "gaussian")``.

    Prefer LOESS over k-NN regression for smooth signals in a few
    dimensions; tune ``span`` like k. Prediction costs one small weighted
    regression per query point.

    Parameters
    ----------
    span : float in (0, 1], default=0.3
        Fraction of the training data in each neighbourhood. At least
        ``min_neighbors`` points are always used.
    degree : {1, 2}, default=1
        Degree of the local polynomial.
    min_neighbors : int or None, default=None
        Smallest neighbourhood; None uses twice the local model's number of
        coefficients.
    ridge : float, default=1e-6
        Ridge penalty of each local fit, relative to its weighted scale.

    Attributes
    ----------
    n_neighbors_ : int
        Neighbourhood size used.

    References
    ----------
    W. S. Cleveland and S. J. Devlin. Locally Weighted Regression: An
    Approach to Regression Analysis by Local Fitting. Journal of the
    American Statistical Association, 83(403):596-610, 1988.

    W. S. Cleveland. Robust Locally Weighted Regression and Smoothing
    Scatterplots. Journal of the American Statistical Association,
    74(368):829-836, 1979.
    """

    _parameter_constraints = {
        "span": [Interval(Real, 0, 1, closed="right")],
        "degree": [Interval(Integral, 1, 2, closed="both")],
        "min_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "ridge": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, span=0.3, degree=1, min_neighbors=None, ridge=1e-6):
        self.span = span
        self.degree = degree
        self.min_neighbors = min_neighbors
        self.ridge = ridge

    def _design(self, Z):
        if self.degree == 1:
            return np.column_stack([np.ones(len(Z)), Z])
        return self._poly.transform(Z)

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        n, p = X.shape
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        Z = (X - self._mean) / self._sd
        if self.degree == 2:
            self._poly = PolynomialFeatures(2).fit(Z)
        n_coef = self._design(Z[:1]).shape[1]
        floor = 2 * n_coef if self.min_neighbors is None else self.min_neighbors
        self.n_neighbors_ = int(min(n, max(np.ceil(self.span * n), floor)))
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors_).fit(Z)
        self._Z, self._y = Z, y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        Z = (X - self._mean) / self._sd
        k = self.n_neighbors_
        out = np.empty(len(Z))
        n_coef = self._design(Z[:1]).shape[1]
        batch = max(1, int(4e6 // (k * n_coef)))
        for s in range(0, len(Z), batch):
            q = Z[s : s + batch]
            dist, idx = self._nn.kneighbors(q)
            d_max = dist[:, -1:]
            d_max = np.where(d_max > 0, d_max * (1 + 1e-12), 1.0)
            w = (1 - np.clip(dist / d_max, 0, 1) ** 3) ** 3
            w = np.maximum(w, 1e-12)  # the farthest neighbour keeps a sliver of weight
            # Centre each neighbourhood on its query point: the prediction is the intercept.
            local = self._Z[idx] - q[:, None, :]
            design = self._design(local.reshape(-1, local.shape[2])).reshape(len(q), k, n_coef)
            wd = design * w[..., None]
            gram = np.einsum("bki,bkj->bij", wd, design)
            rhs = np.einsum("bki,bk->bi", wd, self._y[idx])
            scale = np.trace(gram, axis1=1, axis2=2)[:, None, None] / n_coef
            gram = gram + self.ridge * scale * np.eye(n_coef)
            out[s : s + batch] = np.linalg.solve(gram, rhs[..., None])[:, 0, 0]
        return out
