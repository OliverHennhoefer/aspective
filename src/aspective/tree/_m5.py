"""M5 model trees (Quinlan, AI 1992; Wang & Witten, ECML 1997)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


def _fit_linear(X, y, features):
    """Least squares with intercept on the given features; returns (intercept, coefs)."""
    if len(features) == 0:
        return float(y.mean()), np.zeros(0)
    design = np.column_stack([np.ones(len(X)), X[:, features]])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(beta[0]), beta[1:]


def _pessimistic(resid, v, multiplier=2.0):
    """M5' pessimistic error: RMS residual times (n + multiplier v) / (n - v), 10 if n <= v."""
    n = len(resid)
    factor = (n + multiplier * v) / (n - v) if n > v else 10.0
    return factor * float(np.sqrt(np.mean(resid**2)))


def _residuals(X, y, intercept, coef, features):
    return y - intercept - (X[:, features] @ coef if len(features) else 0.0)


class M5ModelTreeRegressor(RegressorMixin, BaseEstimator):
    """M5 model tree: a regression tree with linear models in the leaves.

    A regression tree predicts a constant in each leaf, so it approximates
    a smooth or linear trend by a staircase and needs many leaves to do so.
    M5 grows a tree that minimizes the variance of the response, as CART
    does, and then fits a linear regression at every node on the features
    tested below it. Each linear model is simplified by greedily dropping
    features while the pessimistic error estimate, ``(n + 2 v) / (n - v)``
    times the root mean squared residual with ``v`` parameters (the M5'
    form used by Weka), does not grow. Working upward, a subtree is replaced by its
    node's linear model when the model's estimated error is no larger than
    the subtree's. Predictions are smoothed along the path from the leaf
    to the root: at each step up, ``p <- (n p + k q) / (n + k)`` with ``q``
    the node's own model and ``n`` the training size of the node below,
    which removes the jumps between neighbouring leaves. Quinlan and Wang &
    Witten found model trees far more accurate than regression trees and
    comparable to neural networks, with much smaller trees; this is the
    M5' variant of Weka's M5P.

    Prefer it over a regression tree whenever the response varies
    smoothly with the features; it stays interpretable and extrapolates
    linearly within each leaf.

    Parameters
    ----------
    min_samples_leaf : int, default=4
        Smallest leaf of the grown tree (M5P's ``minNumInstances``).
    max_depth : int or None, default=None
        Depth limit of the grown tree, before pruning.
    smoothing : float, default=15.0
        Smoothing constant ``k``; 0 disables smoothing.
    prune : bool, default=True

    Attributes
    ----------
    tree_ : sklearn.tree._tree.Tree
        The grown (unpruned) tree structure.
    is_leaf_ : ndarray of bool, shape (n_nodes,)
        Nodes that are leaves after pruning.
    n_leaves_ : int

    References
    ----------
    J. R. Quinlan. Learning with Continuous Classes. In Proceedings of the
    5th Australian Joint Conference on Artificial Intelligence, pages
    343-348, 1992.

    Y. Wang and I. H. Witten. Inducing Model Trees for Continuous Classes.
    In Poster Papers of the 9th European Conference on Machine Learning
    (ECML), 1997.
    """

    _parameter_constraints = {
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "smoothing": [Interval(Real, 0, None, closed="left")],
        "prune": ["boolean"],
    }

    def __init__(self, min_samples_leaf=4, max_depth=None, smoothing=15.0, prune=True):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.smoothing = smoothing
        self.prune = prune

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        grower = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                       max_depth=self.max_depth, random_state=0).fit(X, y)
        tree = self.tree_ = grower.tree_
        left, right = tree.children_left, tree.children_right
        n_nodes = tree.node_count
        paths = grower.decision_path(X).tocsc()
        samples = [paths.indices[paths.indptr[k] : paths.indptr[k + 1]] for k in range(n_nodes)]
        self._n_node = np.array([len(s) for s in samples])

        self._intercepts = np.zeros(n_nodes)
        self._coefs = [np.zeros(0)] * n_nodes
        self._features = [np.zeros(0, dtype=np.intp)] * n_nodes
        self.is_leaf_ = left == -1
        n = len(X)
        n_params = np.zeros(n_nodes, dtype=int)
        residuals = {}
        tol = 1e-9 * max(float(np.std(y)), 1e-300)
        attrs = [set() for _ in range(n_nodes)]

        for node in reversed(range(n_nodes)):  # children have larger ids than their parent
            Xn, yn = X[samples[node]], y[samples[node]]
            if left[node] != -1:
                attrs[node] = {int(tree.feature[node])} | attrs[left[node]] | attrs[right[node]]
            features = np.array(sorted(attrs[node]), dtype=np.intp)
            intercept, coef = _fit_linear(Xn, yn, features)
            resid = _residuals(Xn, yn, intercept, coef, features)
            error = _pessimistic(resid, len(features) + 1)
            # Drop the feature with the smallest standardized coefficient while that helps.
            while len(features):
                sd = Xn[:, features].std(axis=0)
                drop = int(np.argmin(np.abs(coef) * sd))
                trial = np.delete(features, drop)
                t_int, t_coef = _fit_linear(Xn, yn, trial)
                t_resid = _residuals(Xn, yn, t_int, t_coef, trial)
                t_err = _pessimistic(t_resid, len(trial) + 1)
                if t_err > error:
                    break
                features, intercept, coef, resid, error = trial, t_int, t_coef, t_resid, t_err
            self._intercepts[node], self._coefs[node], self._features[node] = intercept, coef, features
            if left[node] == -1:
                n_params[node] = len(features) + 1
                residuals[node] = resid
                continue
            # The subtree's residuals on this node's rows, with its leaves' models.
            sub_resid = np.empty(n)
            for child in (left[node], right[node]):
                sub_resid[samples[child]] = residuals.pop(child)
            sub_resid = sub_resid[samples[node]]
            sub_params = n_params[left[node]] + n_params[right[node]]
            # A small tolerance, so exact fits are not kept apart by rounding noise.
            if self.prune and error <= _pessimistic(sub_resid, sub_params) + tol:
                self.is_leaf_[node] = True
                n_params[node] = len(features) + 1
                residuals[node] = resid
            else:
                n_params[node] = sub_params
                residuals[node] = sub_resid
        self._parent = np.full(n_nodes, -1)
        internal = np.flatnonzero(left != -1)
        self._parent[left[internal]] = internal
        self._parent[right[internal]] = internal
        self.n_leaves_ = int(self._count_leaves(0))
        return self

    def _count_leaves(self, node):
        if self.is_leaf_[node]:
            return 1
        tree = self.tree_
        return self._count_leaves(tree.children_left[node]) + self._count_leaves(tree.children_right[node])

    def _model(self, X, node):
        f = self._features[node]
        out = np.full(len(X), self._intercepts[node])
        if len(f):
            out += X[:, f] @ self._coefs[node]
        return out

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        tree = self.tree_
        # Route every row down to its leaf of the pruned tree.
        node = np.zeros(len(X), dtype=np.intp)
        active = ~self.is_leaf_[node]
        while active.any():
            idx = np.flatnonzero(active)
            cur = node[idx]
            go_left = X[idx, tree.feature[cur]] <= tree.threshold[cur]
            node[idx] = np.where(go_left, tree.children_left[cur], tree.children_right[cur])
            active[idx] = ~self.is_leaf_[node[idx]]
        pred = np.empty(len(X))
        for leaf in np.unique(node):
            rows = np.flatnonzero(node == leaf)
            p = self._model(X[rows], leaf)
            child = leaf
            parent = self._parent[child]
            k = self.smoothing
            while k > 0 and parent != -1:
                n = self._n_node[child]
                p = (n * p + k * self._model(X[rows], parent)) / (n + k)
                child, parent = parent, self._parent[parent]
            pred[rows] = p
        return pred
