"""scikit-learn's estimator checks for every supervised estimator."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from aspective.discriminant_analysis import MixtureDiscriminantAnalysis, RegularizedDiscriminantAnalysis
from aspective.ensemble import (
    DARTClassifier,
    DARTRegressor,
    LocalLinearForestRegressor,
    LogitBoostClassifier,
    MultiBoostClassifier,
    RotationForestClassifier,
)
from aspective.linear_model import (
    AdaptiveLasso,
    AdaptiveLassoCV,
    Bolasso,
    CauchyLogisticRegression,
    FirthLogisticRegression,
    L2BoostRegressor,
    MARSRegressor,
    NonconvexPenaltyRegression,
    NonconvexPenaltyRegressionCV,
    PSplineGAMRegressor,
    RelaxedLasso,
    RelaxedLassoCV,
    SqrtLasso,
)
from aspective.naive_bayes import (
    A2DEClassifier,
    AODEClassifier,
    FlexibleNaiveBayes,
    HiddenNaiveBayesClassifier,
    KDBClassifier,
    TANClassifier,
)
from aspective.neighbors import DANNClassifier, LoessRegressor
from aspective.tree import M5ModelTreeRegressor

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
    LoessRegressor(),
    LoessRegressor(degree=2, span=0.5),
    KDBClassifier(),
    KDBClassifier(k=1, n_bins=4),
    HiddenNaiveBayesClassifier(),
    HiddenNaiveBayesClassifier(n_bins=4),
    FlexibleNaiveBayes(),
    FlexibleNaiveBayes(bandwidth="silverman"),
    DANNClassifier(),
    RotationForestClassifier(n_estimators=5, random_state=0),
    LogitBoostClassifier(n_estimators=10, random_state=0),
    MultiBoostClassifier(n_estimators=10, random_state=0),
    DARTClassifier(n_estimators=10, random_state=0),
    DARTRegressor(n_estimators=10, random_state=0),
    LocalLinearForestRegressor(n_estimators=10, random_state=0),
    M5ModelTreeRegressor(),
    MARSRegressor(),
    MARSRegressor(max_degree=2, max_terms=11),
    NonconvexPenaltyRegression(),
    NonconvexPenaltyRegression(penalty="scad"),
    NonconvexPenaltyRegressionCV(cv=3, n_alphas=10),
    SqrtLasso(),
    A2DEClassifier(),
    A2DEClassifier(n_bins=4),
    PSplineGAMRegressor(),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    check(estimator)
