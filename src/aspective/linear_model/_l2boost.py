"""Componentwise L2 boosting (Buehlmann & Yu, JASA 2003; Buehlmann, Annals of Statistics 2006)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import check_cv
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data


def _boost_path(X, y, n_steps, learning_rate, X_eval=None):
    """Run componentwise L2 boosting; return final coefficients, the intercept and,
    if ``X_eval`` is given, its predictions after every step (row 0 is before any)."""
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    norms = np.einsum("ij,ij->j", Xc, Xc)
    usable = norms > 1e-12 * max(norms.max(initial=0.0), 1.0)
    gram = Xc.T @ Xc
    corr = Xc.T @ yc  # X^T (current residual), kept up to date through the Gram matrix
    coef = np.zeros(X.shape[1])
    preds = None
    if X_eval is not None:
        Xe = X_eval - x_mean
        preds = np.empty((n_steps + 1, len(X_eval)))
        preds[0] = y_mean
    safe_norms = np.where(usable, norms, 1.0)
    for m in range(n_steps):
        gain = np.where(usable, corr**2 / safe_norms, -1.0)
        j = int(np.argmax(gain))
        if gain[j] <= 0:  # residual is orthogonal to every feature: boosting has converged
            if preds is not None:
                preds[m + 1 :] = preds[m]
            break
        step = learning_rate * corr[j] / norms[j]
        coef[j] += step
        corr -= step * gram[:, j]
        if preds is not None:
            preds[m + 1] = preds[m] + step * Xe[:, j]
    intercept = y_mean - x_mean @ coef
    return coef, intercept, preds


class L2BoostRegressor(RegressorMixin, BaseEstimator):
    """Componentwise linear L2 boosting with the number of steps chosen by CV.

    Gradient boosting with squared loss where each base learner is a simple
    linear regression on one feature: at every step the feature that best
    fits the current residuals is chosen and its coefficient moves a small
    step ``learning_rate`` toward the least-squares fit. Stopping early
    shrinks and selects coefficients much like the lasso, while the
    coefficient paths are smoother and correlated features enter more
    gradually. Buehlmann (2006) proved consistency for high-dimensional
    sparse linear models with many more features than samples. The number
    of steps is the only tuning parameter and is chosen here by K-fold
    cross-validation over the whole path, which is computed incrementally
    in O(n_features) per step after one Gram matrix.

    This is the default linear model of the R package mboost (Buehlmann &
    Hothorn, 2007). Prefer it over the lasso when features are strongly
    correlated, when you want a sparse model whose coefficients are not
    over-shrunk at a single penalty, or as a fast, interpretable baseline.

    Parameters
    ----------
    max_iter : int, default=2000
        Largest number of boosting steps considered.
    learning_rate : float in (0, 1], default=0.1
        Step length ``nu``; Buehlmann & Yu recommend a small value like 0.1.
    cv : int, cross-validation generator or iterable, default=5
        How the number of steps is chosen; None uses ``max_iter`` steps.

    Attributes
    ----------
    n_iter_ : int
        Number of boosting steps used in the final model.
    cv_mse_ : ndarray of shape (max_iter + 1,)
        Mean cross-validated squared error after each number of steps.
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    support_ : ndarray of int
        Features that were selected at least once.

    References
    ----------
    P. Buehlmann and B. Yu. Boosting with the L2 Loss: Regression and
    Classification. Journal of the American Statistical Association,
    98(462):324-339, 2003.

    P. Buehlmann. Boosting for High-Dimensional Linear Models. The Annals of
    Statistics, 34(2):559-583, 2006.

    P. Buehlmann and T. Hothorn. Boosting Algorithms: Regularization,
    Prediction and Model Fitting. Statistical Science, 22(4):477-505, 2007.
    """

    _parameter_constraints = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, 1, closed="right")],
        "cv": ["cv_object", None],
    }

    def __init__(self, max_iter=2000, learning_rate=0.1, cv=5):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.cv = cv

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        if self.cv is None:
            self.n_iter_ = self.max_iter
        else:
            errors = np.zeros(self.max_iter + 1)
            folds = list(check_cv(self.cv).split(X, y))
            for train, test in folds:
                _, _, preds = _boost_path(
                    X[train], y[train], self.max_iter, self.learning_rate, X_eval=X[test]
                )
                errors += np.mean((preds - y[test]) ** 2, axis=1)
            self.cv_mse_ = errors / len(folds)
            self.n_iter_ = int(np.argmin(self.cv_mse_))
        self.coef_, self.intercept_, _ = _boost_path(X, y, self.n_iter_, self.learning_rate)
        self.intercept_ = float(self.intercept_)
        self.support_ = np.flatnonzero(self.coef_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X @ self.coef_ + self.intercept_
