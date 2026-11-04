"""Discretization shared by the Bayesian network classifiers."""

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class _DiscreteEncoderMixin:
    """Maps each feature to integer codes ``0 .. n_categories - 1``.

    With ``self.n_bins`` set, features are first cut into quantile bins. The
    last code of every feature is reserved for values not seen in training.
    """

    def _encode(self, X, fit):
        if self.n_bins is not None:
            if fit:
                self._binner = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode="ordinal",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                )
                X = self._binner.fit_transform(X)
            else:
                X = self._binner.transform(X)
        if fit:
            self._categories = [np.unique(X[:, j]) for j in range(X.shape[1])]
        codes = np.empty(X.shape, dtype=np.intp)
        for j, cats in enumerate(self._categories):
            pos = np.searchsorted(cats, X[:, j])
            pos_clipped = np.minimum(pos, len(cats) - 1)
            seen = cats[pos_clipped] == X[:, j]
            codes[:, j] = np.where(seen, pos_clipped, len(cats))
        return codes
