"""scikit-learn's estimator checks for every supervised estimator."""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis
from aspective.ensemble import RandomRotationClassifier, RandomRotationRegressor
from aspective.linear_model import Bolasso, FirthLogisticRegression, RelaxedLasso, RelaxedLassoCV
from aspective.naive_bayes import AODEClassifier

ESTIMATORS = [
    RegularizedDiscriminantAnalysis(),
    RegularizedDiscriminantAnalysis(lam=0.0, gamma=0.1),
    AODEClassifier(),
    AODEClassifier(n_bins=5),
    FirthLogisticRegression(),
    FirthLogisticRegression(correction="flic"),
    FirthLogisticRegression(correction="flac"),
    RandomRotationClassifier(n_estimators=5, random_state=0),
    RandomRotationRegressor(n_estimators=5, random_state=0),
    RandomRotationClassifier(estimator=DecisionTreeClassifier(), n_estimators=3, scaling="rank", random_state=0),
    RandomRotationRegressor(estimator=DecisionTreeRegressor(), n_estimators=3, scaling="standard", random_state=0),
    RelaxedLasso(alpha=0.1),
    RelaxedLassoCV(n_alphas=5, cv=3),
    Bolasso(alpha=0.1, n_bootstrap=8, random_state=0),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    check(estimator)
