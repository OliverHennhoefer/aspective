"""Array-based isolation trees with oblique splits, shared by EIF and SCiForest."""

import numpy as np

EULER_GAMMA = 0.5772156649015329


def average_path_length(n):
    """c(n): average path length of an unsuccessful search in a binary search tree."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2 * (np.log(n[big] - 1) + EULER_GAMMA) - 2 * (n[big] - 1) / n[big]
    out[n == 2] = 1.0
    return out


class ObliqueIsolationTree:
    """One isolation tree whose nodes split on ``X @ normal <= offset``.

    ``choose_split(X, rng)`` returns ``(normal, offset)`` for the rows of a
    node, or None when the node cannot be split. With ``use_range`` the
    projections' range at every node is stored, and a point whose projection
    falls outside it is isolated at that node (SCiForest's range check).
    """

    def __init__(self, X, choose_split, height_limit, rng, use_range=False):
        normals, offsets, left, right, sizes, lows, highs, depths = [], [], [], [], [], [], [], []

        def new_node(size, depth):
            normals.append(None)
            offsets.append(0.0)
            left.append(-1)
            right.append(-1)
            sizes.append(size)
            lows.append(-np.inf)
            highs.append(np.inf)
            depths.append(depth)
            return len(sizes) - 1

        stack = [(np.arange(len(X)), new_node(len(X), 0))]
        while stack:
            rows, node = stack.pop()
            depth = depths[node]
            if depth >= height_limit or len(rows) <= 1:
                continue
            split = choose_split(X[rows], rng)
            if split is None:
                continue
            normal, offset = split
            proj = X[rows] @ normal
            go_left = proj <= offset
            if go_left.all() or not go_left.any():
                continue
            normals[node], offsets[node] = normal, offset
            if use_range:
                lows[node], highs[node] = proj.min(), proj.max()
            lnode = new_node(int(go_left.sum()), depth + 1)
            rnode = new_node(int((~go_left).sum()), depth + 1)
            left[node], right[node] = lnode, rnode
            stack.append((rows[go_left], lnode))
            stack.append((rows[~go_left], rnode))

        d = X.shape[1]
        self.normal = np.array([np.zeros(d) if v is None else v for v in normals])
        self.offset = np.array(offsets)
        self.left = np.array(left)
        self.right = np.array(right)
        self.depth = np.array(depths, dtype=float)
        self.leaf_correction = average_path_length(np.array(sizes))
        self.low = np.array(lows)
        self.high = np.array(highs)
        self.use_range = use_range

    def path_length(self, X):
        node = np.zeros(len(X), dtype=np.intp)
        length = np.full(len(X), np.nan)
        active = np.ones(len(X), dtype=bool)
        while active.any():
            idx = np.flatnonzero(active)
            cur = node[idx]
            leaf = self.left[cur] == -1
            if leaf.any():
                length[idx[leaf]] = self.depth[cur[leaf]] + self.leaf_correction[cur[leaf]]
                active[idx[leaf]] = False
            idx, cur = idx[~leaf], cur[~leaf]
            if not len(idx):
                break
            proj = np.einsum("ij,ij->i", X[idx], self.normal[cur])
            if self.use_range:
                outside = (proj < self.low[cur]) | (proj > self.high[cur])
                length[idx[outside]] = self.depth[cur[outside]]
                active[idx[outside]] = False
                idx, cur, proj = idx[~outside], cur[~outside], proj[~outside]
            node[idx] = np.where(proj <= self.offset[cur], self.left[cur], self.right[cur])
        return length
