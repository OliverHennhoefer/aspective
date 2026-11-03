"""Nearest-neighbour detectors: INFLO, AntiHub2 and LDOF.

All three score a training point against the other training points (a point
is not its own neighbour) and a new point against the whole training set.
"""

from numbers import Integral, Real

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted, validate_data

from ._base import BaseDetector

_TINY = 1e-12


class _NeighborDetector(BaseDetector):
    def _check_k(self, n):
        if not isinstance(self.n_neighbors, Integral) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        if n < 2:
            raise ValueError(f"{type(self).__name__} needs at least 2 samples.")
        return min(self.n_neighbors, n - 1)

    def _fit_neighbors(self, X):
        self._X = X
        self.n_neighbors_ = k = self._check_k(len(X))
        self._nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = self._nn.kneighbors(X)
        # Drop each point itself (or one of its exact duplicates, which is equivalent).
        self_pos = np.argmax(idx == np.arange(len(X))[:, None], axis=1)
        has_self = (idx == np.arange(len(X))[:, None]).any(axis=1)
        drop = np.where(has_self, self_pos, k)
        keep = np.ones_like(idx, dtype=bool)
        keep[np.arange(len(X)), drop] = False
        self._train_dist = dist[keep].reshape(len(X), k)
        self._train_idx = idx[keep].reshape(len(X), k)
        self._kdist = np.maximum(self._train_dist[:, -1], _TINY)

    def _query(self, X):
        dist, idx = self._nn.kneighbors(X, n_neighbors=self.n_neighbors_)
        return dist, idx

    def _reverse_counts(self, X):
        """For each row of X, the training points that would count it among their k nearest."""
        out = []
        for chunk in pairwise_distances_chunked(X, self._X):
            out.extend(list(chunk <= self._kdist[None, :]))
        return out


class INFLODetector(_NeighborDetector):
    """Influenced outlierness (INFLO): LOF with reverse neighbours.

    LOF compares a point's density with that of its k nearest neighbours.
    Where a sparse and a dense cluster meet, a point of the sparse cluster
    can have neighbours from the dense one, and LOF wrongly flags it.
    INFLO compares the point's density ``1 / k-distance`` with the mean
    density of its *influence space*: its k nearest neighbours together
    with its reverse nearest neighbours, the points that count it among
    their own k nearest. A point in a sparse cluster has reverse
    neighbours from its own cluster, which keeps its reference density
    honest. Jin et al. found INFLO identified meaningful outliers more
    reliably than LOF where clusters of different density are close.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points (INFLO near 1 is normal).

    Parameters
    ----------
    n_neighbors : int, default=20
    contamination : float in (0, 0.5], default=0.1

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    n_neighbors_ : int

    References
    ----------
    W. Jin, A. K. H. Tung, J. Han and W. Wang. Ranking Outliers Using
    Symmetric Neighborhood Relationship. In Advances in Knowledge Discovery
    and Data Mining (PAKDD), LNCS 3918, pages 577-593, 2006.
    """

    def __init__(self, n_neighbors=20, contamination=0.1):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        self._set_n_classes(y)
        self._fit_neighbors(X)
        n = len(X)
        density = 1.0 / self._kdist
        self._density = density
        # Influence space of training point p: kNN(p) and RNN(p) = {o : p in kNN(o)}.
        rows = np.repeat(np.arange(n), self.n_neighbors_)
        cols = self._train_idx.ravel()
        sums = np.zeros(n)
        counts = np.zeros(n)
        member = set(zip(rows.tolist(), cols.tolist()))
        np.add.at(sums, rows, density[cols])
        np.add.at(counts, rows, 1)
        # Reverse neighbours not already among the k nearest.
        for o, p in zip(rows.tolist(), cols.tolist()):
            if (p, o) not in member:
                sums[p] += density[o]
                counts[p] += 1
        self.decision_scores_ = (sums / counts) / density
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "_density")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        dist, idx = self._query(X)
        density = 1.0 / np.maximum(dist[:, -1], _TINY)
        reverse = self._reverse_counts(X)
        scores = np.empty(len(X))
        for i in range(len(X)):
            space = np.union1d(idx[i], np.flatnonzero(reverse[i]))
            scores[i] = self._density[space].mean() / density[i]
        return scores


class AntiHubDetector(_NeighborDetector):
    """AntiHub2: outliers as points that are rarely anyone's nearest neighbour.

    The k-NN distance detector loses contrast in high dimensions, where all
    distances become alike. Radovanović et al. showed that the number of
    times a point occurs among the k nearest neighbours of other points,
    its reverse-neighbour count ``N_k``, keeps its contrast: "anti-hubs"
    with low ``N_k`` are the outliers. AntiHub2 refines the count with
    those of the point's own neighbours, ``(1 - a) N_k(x) + a sum N_k(y)``
    over its k nearest ``y``, which breaks the many ties among small
    counts; ``a`` is chosen on the training data to make the lowest
    ``ratio`` share of scores as distinct as possible. The paper found it
    competitive with distance-based detectors overall and better in high
    dimensions.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points (the negated AntiHub2 count).

    Parameters
    ----------
    n_neighbors : int, default=20
    ratio : float in (0, 1], default=0.1
        Share of lowest scores whose distinctness selects ``a``.
    step : float in (0, 1], default=0.1
        Grid step for ``a`` in [0, 1].
    contamination : float in (0, 0.5], default=0.1

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    alpha_ : float
        Selected mixing weight ``a``.
    n_neighbors_ : int

    References
    ----------
    M. Radovanović, A. Nanopoulos and M. Ivanović. Reverse Nearest
    Neighbors in Unsupervised Distance-Based Outlier Detection. IEEE
    Transactions on Knowledge and Data Engineering, 27(5):1369-1382, 2015.
    """

    def __init__(self, n_neighbors=20, ratio=0.1, step=0.1, contamination=0.1):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.ratio = ratio
        self.step = step

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        for name in ("ratio", "step"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        self._set_n_classes(y)
        self._fit_neighbors(X)
        n = len(X)
        self._counts = np.bincount(self._train_idx.ravel(), minlength=n).astype(float)
        neighbor_sum = self._counts[self._train_idx].sum(axis=1)
        m = max(1, int(np.ceil(self.ratio * n)))
        best, best_disc = 0.0, -1.0
        for a in np.arange(0, 1 + 1e-9, self.step):
            a = min(float(a), 1.0)
            score = (1 - a) * self._counts + a * neighbor_sum
            disc = len(np.unique(np.sort(score)[:m])) / m
            if disc > best_disc:
                best, best_disc = a, disc
        self.alpha_ = best
        self.decision_scores_ = -((1 - best) * self._counts + best * neighbor_sum)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "alpha_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        _, idx = self._query(X)
        counts = np.array([r.sum() for r in self._reverse_counts(X)], dtype=float)
        neighbor_sum = self._counts[idx].sum(axis=1)
        return -((1 - self.alpha_) * counts + self.alpha_ * neighbor_sum)


class LDOFDetector(_NeighborDetector):
    """Local distance-based outlier factor (LDOF).

    The k-NN distance detector flags points far from their neighbours, but
    in data with clusters of different spread a point at the edge of a
    wide cluster is as far from its neighbours as an outlier next to a
    tight one. LDOF divides a point's mean distance to its k nearest
    neighbours by the mean distance between those neighbours: inside a
    cluster, of any spread, the ratio is about one or less, and a point
    that sits outside the group formed by its neighbours scores high.
    Zhang, Hutter and Jin found it more accurate than LOF and the k-NN
    detector for outliers near scattered clusters, and less sensitive to
    ``n_neighbors``.

    Follows PyOD's detector API: ``decision_function`` returns higher scores
    for more anomalous points.

    Parameters
    ----------
    n_neighbors : int, default=20
        Neighbours per point; at least 2 are needed for the inner distance.
    contamination : float in (0, 0.5], default=0.1

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
    threshold_ : float
    labels_ : ndarray of shape (n_samples,)
    n_neighbors_ : int

    References
    ----------
    K. Zhang, M. Hutter and H. Jin. A New Local Distance-Based Outlier
    Detection Approach for Scattered Real-World Data. In Advances in
    Knowledge Discovery and Data Mining (PAKDD), LNCS 5476, pages 813-822,
    2009.
    """

    def __init__(self, n_neighbors=20, contamination=0.1):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors

    def _ldof(self, dist, idx):
        outer = dist.mean(axis=1)
        k = idx.shape[1]
        if k < 2:
            return outer
        pts = self._X[idx]  # (m, k, d)
        sq = np.einsum("mkd,mkd->mk", pts, pts)
        gram = np.einsum("mkd,mld->mkl", pts, pts)
        pair = np.sqrt(np.maximum(sq[:, :, None] + sq[:, None, :] - 2 * gram, 0.0))
        inner = pair.sum(axis=(1, 2)) / (k * (k - 1))
        return outer / np.maximum(inner, _TINY)

    def _batched(self, dist, idx):
        out = np.empty(len(idx))
        step = 512
        for s in range(0, len(idx), step):
            out[s : s + step] = self._ldof(dist[s : s + step], idx[s : s + step])
        return out

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64)
        self._set_n_classes(y)
        self._fit_neighbors(X)
        self.decision_scores_ = self._batched(self._train_dist, self._train_idx)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        check_is_fitted(self, "n_neighbors_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        dist, idx = self._query(X)
        return self._batched(dist, idx)
