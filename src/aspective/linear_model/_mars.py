"""Multivariate adaptive regression splines (Friedman, Annals of Statistics 1991)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


def _hinges(x, knots):
    """Columns max(0, x - t) and max(0, t - x) for every knot t: shape (n, 2 * n_knots)."""
    diff = x[:, None] - knots[None, :]
    return np.concatenate([np.maximum(diff, 0.0), np.maximum(-diff, 0.0)], axis=1)


class MARSRegressor(RegressorMixin, BaseEstimator):
    """Multivariate adaptive regression splines (MARS).

    A linear model in a basis of hinge functions ``max(0, x_j - t)`` and
    ``max(0, t - x_j)`` (and, with ``max_degree > 1``, their products) that
    are chosen from the data. A forward pass adds, at each step, the
    reflected pair of hinges (on a variable, at a knot, multiplying an
    existing term) that most reduces the residual sum of squares; a
    backward pass then removes terms one by one and keeps the subset with
    the best generalized cross-validation score

        GCV = RSS / n / (1 - C(M) / n)^2,   C(M) = M + penalty (M - 1) / 2.

    MARS keeps what linear regression and regression trees each do well:
    it is continuous and piecewise linear, picks variables and
    interactions automatically, and extrapolates linearly instead of by a
    constant. Friedman found it more accurate than recursive partitioning
    on smooth functions in moderate dimension; with ``max_degree=1`` it
    fits an additive model, the default of the R package earth.

    Candidate knots are ``n_knots`` interior quantiles of each feature
    (Friedman's ``minspan``/``endspan`` rules play the same role: they keep
    knots away from the extremes and from each other). The forward search
    updates the projections of every candidate onto the current basis
    incrementally, so a step costs one pass over the candidate hinges.

    Parameters
    ----------
    max_terms : int or None, default=None
        Largest number of terms (with the intercept) in the forward pass;
        None uses earth's ``min(200, max(20, 2 * n_features)) + 1``.
    max_degree : int, default=1
        Largest number of hinges multiplied in one term.
    penalty : float or None, default=None
        GCV cost per knot; None uses 2 for ``max_degree=1`` and 3 otherwise.
    n_knots : int, default=20
        Candidate knots per feature.
    threshold : float, default=0.001
        The forward pass stops when the best step raises R^2 by less.

    Attributes
    ----------
    terms_ : list of tuple
        Selected terms, each a tuple of hinges ``(feature, knot, sign)``
        (sign +1 for ``max(0, x - t)``); the empty tuple is the intercept.
    coef_ : ndarray of shape (n_terms,)
    gcv_ : float

    References
    ----------
    J. H. Friedman. Multivariate Adaptive Regression Splines. The Annals of
    Statistics, 19(1):1-67, 1991.

    S. Milborrow. earth: Multivariate Adaptive Regression Splines. R
    package, derived from mda:mars by T. Hastie and R. Tibshirani.
    """

    _parameter_constraints = {
        "max_terms": [Interval(Integral, 1, None, closed="left"), None],
        "max_degree": [Interval(Integral, 1, None, closed="left")],
        "penalty": [Interval(Real, 0, None, closed="left"), None],
        "n_knots": [Interval(Integral, 1, None, closed="left")],
        "threshold": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(self, max_terms=None, max_degree=1, penalty=None, n_knots=20, threshold=0.001):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.n_knots = n_knots
        self.threshold = threshold

    @staticmethod
    def _term_column(X, term):
        col = np.ones(len(X))
        for j, t, s in term:
            col = col * np.maximum(s * (X[:, j] - t), 0.0)
        return col

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        n, p = X.shape
        max_terms = (min(200, max(20, 2 * p)) + 1) if self.max_terms is None else self.max_terms
        max_terms = min(max_terms, n)
        levels = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        self._knots = [
            np.unique(np.quantile(X[:, j], levels, method="inverted_cdf")) for j in range(p)
        ]
        n_cand = [len(k) for k in self._knots]
        offsets = np.concatenate([[0], np.cumsum(2 * np.array(n_cand))])

        tss = float(np.sum((y - y.mean()) ** 2))
        terms = [()]
        q0 = np.ones(n) / np.sqrt(n)
        Q = [q0]
        Qy = [q0 @ y]
        # Per parent term: candidate column norms, inner products with y, and the
        # projections of every candidate onto the orthonormal basis built so far.
        parents = {}

        def candidates(parent_col, j):
            return parent_col[:, None] * _hinges(X[:, j], self._knots[j])

        def add_parent(idx):
            col = self._term_column(X, terms[idx])
            norms, cy, proj = [], [], []
            for j in range(p):
                C = candidates(col, j)
                norms.append(np.einsum("ij,ij->j", C, C))
                cy.append(y @ C)
                proj.append(np.array(Q) @ C)
            parents[idx] = (
                col,
                np.concatenate(norms),
                np.concatenate(cy),
                np.concatenate(proj, axis=1),
            )

        def extend_parents(q):
            for idx, (col, norms, cy, proj) in parents.items():
                row = np.concatenate([q @ candidates(col, j) for j in range(p)])
                parents[idx] = (col, norms, cy, np.vstack([proj, row]))

        add_parent(0)
        rss = tss
        while len(terms) + 1 < max_terms + 1 and len(terms) < max_terms:
            best = None
            qy = np.array(Qy)
            for idx, (col, norms, cy, proj) in parents.items():
                used = {j for j, _, _ in terms[idx]}
                uu = norms - np.einsum("ij,ij->j", proj, proj)
                ur = cy - qy @ proj
                for j in range(p):
                    if j in used:
                        continue
                    lo, k = offsets[j], n_cand[j]
                    a, b = slice(lo, lo + k), slice(lo + k, lo + 2 * k)
                    pa, pb = proj[:, a], proj[:, b]
                    uu_a, uu_b = uu[a], uu[b]
                    uw = -np.einsum("ij,ij->j", pa, pb)  # the two hinges never overlap
                    ra, rb = ur[a], ur[b]
                    tol_a = 1e-9 * np.maximum(norms[a], 1e-300)
                    tol_b = 1e-9 * np.maximum(norms[b], 1e-300)
                    ok_a, ok_b = uu_a > tol_a, uu_b > tol_b
                    single_a = np.where(ok_a, ra**2 / np.where(ok_a, uu_a, 1), 0.0)
                    single_b = np.where(ok_b, rb**2 / np.where(ok_b, uu_b, 1), 0.0)
                    det = uu_a * uu_b - uw**2
                    ok = ok_a & ok_b & (det > 1e-9 * uu_a * uu_b)
                    pair = np.where(
                        ok,
                        (uu_b * ra**2 - 2 * uw * ra * rb + uu_a * rb**2) / np.where(ok, det, 1),
                        0.0,
                    )
                    gain = np.where(ok, pair, np.maximum(single_a, single_b))
                    i = int(np.argmax(gain))
                    if best is None or gain[i] > best[0]:
                        best = (
                            gain[i],
                            idx,
                            j,
                            i,
                            bool(ok[i]),
                            bool(ok_a[i]),
                            bool(ok_b[i]),
                            single_a[i] >= single_b[i],
                        )
            if best is None or best[0] <= self.threshold * tss or tss == 0:
                break
            gain, idx, j, i, both, ok_a, ok_b, a_better = best
            t = self._knots[j][i]
            if both:
                signs = [1, -1]
            else:
                signs = [1] if (ok_a and (a_better or not ok_b)) else [-1]
            parent_term = terms[idx]
            for s in signs:
                if len(terms) >= max_terms:
                    break
                term = parent_term + ((j, float(t), s),)
                col = self._term_column(X, term)
                v = col - np.array(Q).T @ (np.array(Q) @ col)
                v = v - np.array(Q).T @ (np.array(Q) @ v)
                norm = np.linalg.norm(v)
                if norm <= 1e-10 * max(np.linalg.norm(col), 1e-300):
                    continue
                q = v / norm
                extend_parents(q)
                Q.append(q)
                Qy.append(q @ y)
                terms.append(term)
                if len(term) < self.max_degree:
                    add_parent(len(terms) - 1)
            rss = tss - float(np.sum(np.array(Qy[1:]) ** 2))

        self._prune(X, y, terms)
        return self

    def _prune(self, X, y, terms):
        n = len(X)
        penalty = (2.0 if self.max_degree == 1 else 3.0) if self.penalty is None else self.penalty
        B = np.column_stack([self._term_column(X, t) for t in terms])
        G = B.T @ B
        G += 1e-10 * np.trace(G) / len(G) * np.eye(len(G))
        by = B.T @ y
        yy = float(y @ y)
        active = list(range(len(terms)))
        A = np.linalg.inv(G)

        def gcv(rss, m):
            c = m + penalty * (m - 1) / 2
            return np.inf if c >= n else rss / n / (1 - c / n) ** 2

        beta = A @ by
        rss = max(yy - by @ beta, 0.0)
        best_gcv, best_set = gcv(rss, len(active)), list(active)
        while len(active) > 1:
            # Removing term k raises the RSS by beta_k^2 / A_kk (downdating the inverse).
            inc = beta[1:] ** 2 / np.diag(A)[1:]
            k = 1 + int(np.argmin(inc))
            rss = rss + inc[k - 1]
            A = A - np.outer(A[:, k], A[k, :]) / A[k, k]
            A = np.delete(np.delete(A, k, axis=0), k, axis=1)
            del active[k]
            beta = A @ by[active]
            score = gcv(max(rss, 0.0), len(active))
            if score <= best_gcv:
                best_gcv, best_set = score, list(active)
        self.terms_ = [terms[i] for i in best_set]
        Bs = B[:, best_set]
        self.coef_ = np.linalg.lstsq(Bs, y, rcond=None)[0]
        self.gcv_ = float(best_gcv)

    def _basis(self, X):
        return np.column_stack([self._term_column(X, t) for t in self.terms_])

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._basis(X) @ self.coef_
