"""scikit-learn's estimator checks for every supervised estimator."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from aspective.discriminant_analysis import MixtureDiscriminantAnalysis, RegularizedDiscriminantAnalysis
from aspective.linear_model import (
    AdaptiveLasso,
    AdaptiveLassoCV,
    Bolasso,
    CauchyLogisticRegression,
    FirthLogisticRegression,
    L2BoostRegressor,
    RelaxedLasso,
    RelaxedLassoCV,
    SupervisedPCARegressor,
)
from aspective.naive_bayes import AODEClassifier, TANClassifier
from aspective.neighbors import LoessRegressor

ESTIMATORS = [
    RegularizedDiscriminantAnalysis(),
    RegularizedDiscriminantAnalysis(lam=0.0, gamma=0.1),
    AODEClassifier(),
    AODEClassifier(n_bins=5),
    FirthLogisticRegression(),
    FirthLogisticRegression(correction="flic"),
    FirthLogisticRegression(correction="flac"),
    RelaxedLasso(alpha=0.1),
    RelaxedLassoCV(n_alphas=5, cv=3),
    Bolasso(alpha=0.1, n_bootstrap=8, random_state=0),
    TANClassifier(),
    TANClassifier(n_bins=4),
    MixtureDiscriminantAnalysis(random_state=0),
    MixtureDiscriminantAnalysis(n_subclasses=1),
    CauchyLogisticRegression(),
    CauchyLogisticRegression(prior_df=float("inf"), scale_inputs=False),
    AdaptiveLasso(alpha=0.1),
    AdaptiveLassoCV(cv=3),
    L2BoostRegressor(max_iter=200, cv=3),
    SupervisedPCARegressor(cv=3),
    LoessRegressor(),
    LoessRegressor(degree=2, span=0.5),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    check(estimator)
