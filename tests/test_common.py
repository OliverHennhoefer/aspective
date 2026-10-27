"""scikit-learn's estimator checks for every supervised estimator."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis
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
    RelaxedLasso(alpha=0.1),
    RelaxedLassoCV(n_alphas=5, cv=3),
    Bolasso(alpha=0.1, n_bootstrap=8, random_state=0),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check):
    check(estimator)
