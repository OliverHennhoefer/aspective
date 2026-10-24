"""Firth-penalized logistic regression with the FLIC and FLAC corrections.

Firth, Biometrika 1993; Puhr, Heinze, Nold, Lusa & Geroldinger,
Statistics in Medicine 2017.
"""

import warnings
from numbers import Integral, Real

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


def _log_likelihood(eta, y, w):
    # sum w * (y * eta - log(1 + exp(eta))), written stably
    return np.sum(w * (y * eta - np.logaddexp(0.0, eta)))


def _firth_newton(X, y, w, max_iter, tol, max_step, max_halfstep):
    """Penalized Newton-Raphson for Firth's logistic regression.

    Maximizes l(beta) + 0.5 log det I(beta) with the modified score
    U*(beta) = X^T W_case (y - p + h (0.5 - p)), where h is the diagonal
    of the weighted hat matrix (Heinze & Schemper, 2002).
    """
    n, p = X.shape
    beta = np.zeros(p)

    def penalized(beta):
        eta = X @ beta
        prob = expit(eta)
        v = w * prob * (1 - prob)
        info = X.T @ (v[:, None] * X)
        sign, logdet = np.linalg.slogdet(info)
        if sign <= 0:
            return -np.inf, prob, v, info
        return _log_likelihood(eta, y, w) + 0.5 * logdet, prob, v, info

    loglik, prob, v, info = penalized(beta)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        info_inv = np.linalg.pinv(info, hermitian=True)
        sqrt_v = np.sqrt(v)
        xs = X * sqrt_v[:, None]
        h = np.einsum("ij,jk,ik->i", xs, info_inv, xs)
        score = X.T @ (w * (y - prob) + h * (0.5 - prob))
        step = info_inv @ score
        largest = np.max(np.abs(step))
        if largest > max_step:
            step *= max_step / largest

        new_beta = beta + step
        new_loglik, new_prob, new_v, new_info = penalized(new_beta)
        for _ in range(max_halfstep):
            if new_loglik >= loglik:
                break
            step /= 2
            new_beta = beta + step
            new_loglik, new_prob, new_v, new_info = penalized(new_beta)

        converged = np.max(np.abs(new_beta - beta)) < tol
        beta, loglik, prob, v, info = new_beta, new_loglik, new_prob, new_v, new_info
        if converged:
            break
    else:
        warnings.warn(
            "Firth logistic regression did not converge; increase max_iter.", ConvergenceWarning
        )

    info_inv = np.linalg.pinv(info, hermitian=True)
    xs = X * np.sqrt(v)[:, None]
    h = np.einsum("ij,jk,ik->i", xs, info_inv, xs)
    return beta, loglik, h, n_iter


def _ml_newton(X, y, w, offset, max_iter, tol):
    """Plain weighted maximum likelihood with an offset (used by FLIC and FLAC)."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        prob = expit(offset + X @ beta)
        v = w * prob * (1 - prob)
        info = X.T @ (v[:, None] * X)
        step = np.linalg.pinv(info, hermitian=True) @ (X.T @ (w * (y - prob)))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


class FirthLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression with Firth's bias-reducing penalty.

    Firth's penalty (a Jeffreys prior) removes the first-order small-sample
    bias of the maximum likelihood estimate and gives finite coefficients even
    under complete separation, where ordinary logistic regression diverges.

    Firth's estimate pulls predicted probabilities toward 0.5, which hurts
    calibration when events are rare. Puhr et al. (2017) proposed two fixes:

    - ``correction="flic"`` keeps Firth's slopes and re-estimates the intercept
      by maximum likelihood, so the average predicted probability equals the
      observed event rate.
    - ``correction="flac"`` refits by maximum likelihood on the data augmented
      with pseudo-observations weighted by the hat values, plus an indicator
      for the pseudo-data.

    Parameters
    ----------
    correction : {"none", "flic", "flac"}, default="none"
        Intercept correction for better calibrated probabilities.
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-8
        Convergence threshold on the largest coefficient change.
    max_step : float, default=5.0
        Largest allowed change of any coefficient in one Newton step.
    max_halfstep : int, default=25
        Maximum step halvings per iteration.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    coef_ : ndarray of shape (1, n_features)
    intercept_ : ndarray of shape (1,)
    loglik_ : float
        Penalized log-likelihood of the Firth fit.
    n_iter_ : int

    References
    ----------
    D. Firth. Bias Reduction of Maximum Likelihood Estimates. Biometrika,
    80(1):27-38, 1993.

    R. Puhr, G. Heinze, M. Nold, L. Lusa and A. Geroldinger. Firth's logistic
    regression with rare events: accurate effect estimates and predictions?
    Statistics in Medicine, 36(14):2302-2317, 2017.
    """

    _parameter_constraints = {
        "correction": [StrOptions({"none", "flic", "flac"})],
        "fit_intercept": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "max_step": [Interval(Real, 0, None, closed="neither")],
        "max_halfstep": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self, correction="none", fit_intercept=True, max_iter=100, tol=1e-8,
        max_step=5.0, max_halfstep=25,
    ):
        self.correction = correction
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.max_step = max_step
        self.max_halfstep = max_halfstep

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        if type_of_target(y) != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {type_of_target(y)}. Wrap the model in OneVsRestClassifier for more classes."
            )
        self.classes_, y01 = np.unique(y, return_inverse=True)
        y01 = y01.astype(float)
        w = np.ones(len(y01))

        design = np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X
        beta, self.loglik_, h, self.n_iter_ = _firth_newton(
            design, y01, w, self.max_iter, self.tol, self.max_step, self.max_halfstep
        )

        if self.correction != "none" and not self.fit_intercept:
            raise ValueError("FLIC and FLAC need fit_intercept=True.")
        if self.correction == "flic":
            offset = X @ beta[1:]
            beta[0] = _ml_newton(np.ones((len(X), 1)), y01, w, offset, self.max_iter, self.tol)[0]
        elif self.correction == "flac":
            n = len(X)
            aug_X = np.vstack([design, design, design])
            aug_X = np.column_stack([aug_X, np.r_[np.zeros(n), np.ones(2 * n)]])
            aug_y = np.r_[y01, y01, 1 - y01]
            aug_w = np.r_[w, w * h / 2, w * h / 2]
            full = _ml_newton(aug_X, aug_y, aug_w, 0.0, self.max_iter, self.tol)
            beta = full[:-1]

        if self.fit_intercept:
            self.intercept_ = np.array([beta[0]])
            self.coef_ = beta[1:].reshape(1, -1)
        else:
            self.intercept_ = np.zeros(1)
            self.coef_ = beta.reshape(1, -1)
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        prob = expit(self.decision_function(X))
        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int)]
