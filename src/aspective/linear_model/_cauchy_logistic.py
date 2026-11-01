"""Logistic regression with weakly informative Cauchy priors (Gelman et al., AoAS 2008)."""

from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


def _log_prior_and_grad(beta, scale, df):
    """Log density (up to a constant) of independent Student-t priors, and its gradient."""
    z = beta / scale
    if np.isinf(df):
        return -0.5 * np.sum(z**2), -z / scale
    return (-0.5 * (df + 1) * np.sum(np.log1p(z**2 / df)),
            -(df + 1) * z / (scale * (df + z**2)))


class CauchyLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression with the Gelman et al. default prior.

    Gelman, Jakulin, Pittau and Su proposed a default prior for logistic
    regression that every analysis can use unchanged: put inputs on a common
    scale, then give each coefficient an independent Cauchy prior with
    centre 0 and scale 2.5 (the intercept gets scale 10). The prior says a
    change of one unit on the common scale rarely moves the probability from
    0.01 to 0.5, so it is weak, but its heavy tails leave large effects
    alone. Estimates are the posterior mode, which is always finite, also
    under separation, where maximum likelihood diverges and ridge shrinks
    every coefficient by the same rule.

    With ``scale_inputs=True`` (the paper's recommendation) inputs with more
    than two distinct values are centred and scaled to standard deviation
    0.5, and binary inputs are centred, so that coefficients of both kinds
    are comparable. ``coef_`` is reported on the original scale. Setting
    ``prior_df`` to another value gives the Student-t family the paper also
    discusses; ``prior_df=np.inf`` is a Gaussian prior.

    Prefer it over maximum-likelihood or ridge logistic regression for
    small or sparse data, separation, and when you want a default with no
    penalty to tune. :class:`FirthLogisticRegression` is the alternative
    that targets bias of the coefficients rather than a prior belief.

    Parameters
    ----------
    prior_scale : float, default=2.5
        Scale of the prior on each coefficient, on the common input scale.
    intercept_prior_scale : float, default=10.0
        Scale of the prior on the intercept (on centred inputs).
    prior_df : float, default=1.0
        Degrees of freedom of the Student-t prior; 1 is Cauchy.
    scale_inputs : bool, default=True
        Put inputs on the paper's common scale before fitting.
    max_iter : int, default=1000
    tol : float, default=1e-8
        Gradient tolerance of the L-BFGS solver.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
    coef_ : ndarray of shape (1, n_features)
    intercept_ : ndarray of shape (1,)
    n_iter_ : int

    References
    ----------
    A. Gelman, A. Jakulin, M. G. Pittau and Y.-S. Su. A Weakly Informative
    Default Prior Distribution for Logistic and Other Regression Models. The
    Annals of Applied Statistics, 2(4):1360-1383, 2008.
    """

    _parameter_constraints = {
        "prior_scale": [Interval(Real, 0, None, closed="neither")],
        "intercept_prior_scale": [Interval(Real, 0, None, closed="neither")],
        "prior_df": [Interval(Real, 0, None, closed="right")],  # right-closed admits np.inf
        "scale_inputs": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(self, prior_scale=2.5, intercept_prior_scale=10.0, prior_df=1.0,
                 scale_inputs=True, max_iter=1000, tol=1e-8):
        self.prior_scale = prior_scale
        self.intercept_prior_scale = intercept_prior_scale
        self.prior_df = prior_df
        self.scale_inputs = scale_inputs
        self.max_iter = max_iter
        self.tol = tol

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        self._validate_params()
        X, y = validate_data(self, X, y, dtype=np.float64)
        check_classification_targets(y)
        if type_of_target(y) != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the target "
                f"is {type_of_target(y)}. Wrap the model in OneVsRestClassifier for more classes."
            )
        self.classes_, y01 = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"Need samples of 2 classes, got {len(self.classes_)} class.")
        y01 = y01.astype(float)
        n, p = X.shape

        shift = X.mean(axis=0)
        scale = np.ones(p)
        if self.scale_inputs:
            sd = X.std(axis=0)
            n_distinct = np.array([len(np.unique(X[:, j])) for j in range(p)])
            scale = np.where((n_distinct > 2) & (sd > 0), 2 * sd, 1.0)
            # Binary inputs: centred, unit range.
            binary = n_distinct == 2
            span = X.max(axis=0) - X.min(axis=0)
            scale = np.where(binary & (span > 0), span, scale)
        Z = (X - shift) / scale
        priors = np.r_[self.intercept_prior_scale, np.full(p, self.prior_scale)]
        design = np.column_stack([np.ones(n), Z])

        def objective(beta):
            eta = design @ beta
            loglik = np.sum(y01 * eta + log_expit(-eta))
            grad = design.T @ (y01 - expit(eta))
            lp, lp_grad = _log_prior_and_grad(beta, priors, self.prior_df)
            return -(loglik + lp), -(grad + lp_grad)

        result = minimize(objective, np.zeros(p + 1), jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "gtol": self.tol})
        beta = result.x
        self.n_iter_ = int(result.nit)
        self.coef_ = (beta[1:] / scale)[None, :]
        self.intercept_ = np.array([beta[0] - np.sum(beta[1:] * shift / scale)])
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return X @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        p1 = expit(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict_log_proba(self, X):
        eta = self.decision_function(X)
        return np.column_stack([log_expit(-eta), log_expit(eta)])

    def predict(self, X):
        positive = self.decision_function(X) > 0
        return self.classes_[positive.astype(int)]
