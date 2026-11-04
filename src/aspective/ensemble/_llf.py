"""Local linear forests (Friedberg, Tibshirani, Athey & Wager, JCGS 2020)."""

from numbers import Integral, Real

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data


class LocalLinearForestRegressor(RegressorMixin, BaseEstimator):
    """Random forest weights with a local ridge regression at each query.

    A random forest prediction is a weighted average of training
    responses, where the weight of training point ``i`` for query ``x`` is
    how often, across trees, ``i`` shares a leaf with ``x`` (divided by the
    leaf size). Averaging cannot follow a slope: forests are biased near
    the edges of the data and wherever the signal is smooth but not flat.
    Local linear forests keep the forest's adaptive weights but fit, at
    each query, a weighted linear regression with a ridge penalty on the
    slopes,

        min  sum_i alpha_i(x) (y_i - mu - (x_i - x)' theta)^2 + lambda ||theta||^2,

    and predict ``mu``. Friedberg et al. proved asymptotic normality and
    found lower error than random forests, BART and gradient boosting on
    smooth signals and a range of real data sets.

    The forest is scikit-learn's random forest with leaves of at least
    ``min_samples_leaf`` points; every training point, not only the
    in-bag ones, contributes to the weights. Inputs are standardized for
    the ridge penalty, and the weights sum to one, so ``ll_lambda`` is on
    the scale of a local variance. Prediction costs one small weighted
    regression per query.

    Parameters
    ----------
    n_estimators : int, default=200
    min_samples_leaf : int, default=5
        Smallest leaf, as ``min.node.size`` in the grf package.
    max_features : int, float, "sqrt", "log2" or None, default=1.0
        Features tried per split, as in RandomForestRegressor.
    ll_lambda : float, default=0.1
        Ridge penalty on the local slopes; grf's default.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    forest_ : RandomForestRegressor

    References
    ----------
    R. Friedberg, J. Tibshirani, S. Athey and S. Wager. Local Linear
    Forests. Journal of Computational and Graphical Statistics,
    30(2):503-517, 2021 (online 2020).
    """

    _parameter_constraints = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0, 1, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "ll_lambda": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_estimators=200,
        min_samples_leaf=5,
        max_features=1.0,
        ll_lambda=0.1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.ll_lambda = ll_lambda
        self.random_state = random_state

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        self.forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        ).fit(X, y)
        self._mean = X.mean(axis=0)
        sd = X.std(axis=0)
        self._sd = np.where(sd > 0, sd, 1.0)
        self._Z = (X - self._mean) / self._sd
        self._y = y
        leaves = self.forest_.apply(X)  # (n, n_trees)
        n = len(X)
        # For each tree: sparse leaf-membership of the training points, scaled by 1 / leaf size.
        self._members = []
        for b in range(leaves.shape[1]):
            ids, inverse, counts = np.unique(leaves[:, b], return_inverse=True, return_counts=True)
            member = sparse.csr_matrix(
                (1.0 / counts[inverse], (inverse, np.arange(n))), shape=(len(ids), n)
            )
            self._members.append((ids, member))
        return self

    def _weights(self, X):
        leaves = self.forest_.apply(X)
        W = sparse.csr_matrix((len(X), len(self._y)))
        for b, (ids, member) in enumerate(self._members):
            pos = np.searchsorted(ids, leaves[:, b])
            W = W + member[pos]
        return W / len(self._members)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        W = self._weights(X).tocsr()
        Zq = (X - self._mean) / self._sd
        d = Zq.shape[1]
        penalty = self.ll_lambda * np.eye(d + 1)
        penalty[0, 0] = 0.0
        out = np.empty(len(X))
        for q in range(len(X)):
            row = W.getrow(q)
            idx, w = row.indices, row.data
            w = w / w.sum()
            D = np.column_stack([np.ones(len(idx)), self._Z[idx] - Zq[q]])
            wd = D * w[:, None]
            gram = wd.T @ D + penalty
            rhs = wd.T @ self._y[idx]
            beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            out[q] = beta[0]
        return out
