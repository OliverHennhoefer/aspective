"""The benchmark every aspective estimator must pass.

To add a model, append a ``Benchmark`` to ``BENCHMARKS``: the candidate, the
vanilla model(s) it improves on, a suite and a metric. Then run

    python -m benchmarks run <name>
    python -m benchmarks report

and commit ``benchmarks/results/<name>.json`` and ``benchmarks/RESULTS.md``.
``tests/test_benchmark_gate.py`` fails while any public estimator has no
benchmark, a stale result, or a failing verdict. See ``benchmarks/README.md``.

Predecessors are configured like the candidate's own building blocks (same
number of trees, same discretization, same bandwidth rule), so a comparison
measures the paper's idea and not a tuning difference.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.svm import OneClassSVM

from aspective.anomaly import IDKDetector, RobustKDEDetector, RSHashDetector
from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis
from aspective.ensemble import RandomRotationClassifier, RandomRotationRegressor
from aspective.linear_model import Bolasso, FirthLogisticRegression, RelaxedLassoCV
from aspective.naive_bayes import AODEClassifier

from .baselines import ScottKDE
from .protocol import Benchmark

N_TREES = 200
KDE_CENTERS = 2000


def scaled(estimator):
    return make_pipeline(StandardScaler(), estimator)


def ml_logistic():
    return scaled(LogisticRegression(penalty=None, max_iter=10_000))


def binned_nb():
    binner = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile",
                              quantile_method="averaged_inverted_cdf")
    return make_pipeline(binner, CategoricalNB(min_categories=5))


def firth(correction="none"):
    return lambda: scaled(FirthLogisticRegression(correction=correction))


# Shared factories, so the same baseline is computed once and reused.
ISOLATION_FOREST = ("Isolation Forest", lambda: scaled(IsolationForest(n_estimators=N_TREES, random_state=0)))
LOF = ("LOF", lambda: scaled(LocalOutlierFactor(novelty=True)))
KDE = ("Gaussian KDE", lambda: scaled(ScottKDE(max_samples=KDE_CENTERS, random_state=0)))
OCSVM = ("One-class SVM", lambda: scaled(OneClassSVM(gamma="scale", nu=0.5)))
LASSO_CV = ("LassoCV", lambda: scaled(LassoCV(cv=5)))

BENCHMARKS = [
    Benchmark(
        name="rda",
        candidate=("RDA (tuned)", lambda: GridSearchCV(
            RegularizedDiscriminantAnalysis(),
            {"lam": [0, 0.25, 0.5, 0.75, 1], "gamma": [0, 0.05, 0.25, 0.5]}, cv=3)),
        predecessors={"LDA": LinearDiscriminantAnalysis, "QDA": QuadraticDiscriminantAnalysis},
        references={"LDA (Ledoit-Wolf)": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")},
        suite="pmlb-classification",
        metric="accuracy",
        notes="RDA picks (lambda, gamma) by inner 3-fold CV, as in Friedman (1989); lambda=1, "
              "gamma=0 is LDA and lambda=0, gamma=0 is QDA.",
    ),
    Benchmark(
        name="aode",
        candidate=("AODE (5 bins)", lambda: AODEClassifier(n_bins=5)),
        predecessors={"Naive Bayes (5 bins)": binned_nb},
        references={"Gaussian NB": GaussianNB},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization, as in Webb et al. (2005).",
    ),
    Benchmark(
        name="firth",
        candidate=("Firth", firth()),
        predecessors={"ML logistic": ml_logistic},
        suite="pmlb-binary",
        metric="brier",
        notes="Full-size training folds; features standardized for every model.",
    ),
    Benchmark(
        name="firth-small-n",
        candidate=("Firth", firth()),
        predecessors={"ML logistic": ml_logistic},
        references={"Ridge logistic (C=1)": lambda: scaled(LogisticRegression())},
        suite="pmlb-binary",
        metric="brier",
        train_size=100,
        notes="Ten stratified splits with 100 training rows, the small-sample regime Firth (1993) "
              "targets, where maximum likelihood often meets separation.",
    ),
    Benchmark(
        name="firth-flic-small-n",
        candidate=("Firth + FLIC", firth("flic")),
        predecessors={"ML logistic": ml_logistic},
        references={"Firth": firth()},
        suite="pmlb-binary",
        metric="brier",
        train_size=100,
        notes="Puhr et al. (2017) add FLIC/FLAC to fix Firth's bias of predicted probabilities "
              "toward one half.",
    ),
    Benchmark(
        name="firth-flac-small-n",
        candidate=("Firth + FLAC", firth("flac")),
        predecessors={"ML logistic": ml_logistic},
        references={"Firth": firth()},
        suite="pmlb-binary",
        metric="brier",
        train_size=100,
    ),
    Benchmark(
        name="rotation-classifier",
        candidate=("Rotated RF", lambda: RandomRotationClassifier(n_estimators=N_TREES, random_state=0)),
        predecessors={"Random Forest": lambda: RandomForestClassifier(n_estimators=N_TREES, random_state=0)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Same number of trees and the same member trees (max_features='sqrt').",
    ),
    Benchmark(
        name="rotation-regressor",
        candidate=("Rotated RF", lambda: RandomRotationRegressor(n_estimators=N_TREES, random_state=0)),
        predecessors={"Random Forest": lambda: RandomForestRegressor(
            n_estimators=N_TREES, max_features=1 / 3, random_state=0)},
        suite="pmlb-regression",
        metric="r2",
        notes="Same number of trees and the same member trees (max_features=1/3).",
    ),
    Benchmark(
        name="relaxed-lasso",
        candidate=("Relaxed LassoCV", lambda: scaled(RelaxedLassoCV(cv=5))),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
    ),
    Benchmark(
        name="bolasso",
        candidate=("Bolasso", lambda: scaled(Bolasso(random_state=0))),
        predecessors=dict([LASSO_CV]),
        references={"Bolasso-S (0.9)": lambda: scaled(Bolasso(threshold=0.9, random_state=0))},
        suite="pmlb-regression",
        metric="r2",
        notes="Bolasso is mainly a variable-selection method; this checks that its predictions "
              "keep up with the lasso.",
    ),
    Benchmark(
        name="idk",
        candidate=("IDK", lambda: scaled(IDKDetector(random_state=0))),
        predecessors=dict([KDE, OCSVM]),
        references=dict([ISOLATION_FOREST, LOF]),
        suite="adbench-classical",
        metric="roc_auc",
        notes="IDK replaces the Gaussian kernel of KDE and the one-class SVM by the data-dependent "
              "Isolation Kernel (Ting et al., 2020).",
    ),
    Benchmark(
        name="rshash",
        candidate=("RS-Hash", lambda: scaled(RSHashDetector(random_state=0))),
        predecessors=dict([ISOLATION_FOREST]),
        references=dict([LOF]),
        suite="adbench-classical",
        metric="roc_auc",
    ),
    Benchmark(
        name="robust-kde",
        candidate=("Robust KDE", lambda: scaled(RobustKDEDetector(max_samples=KDE_CENTERS, random_state=0))),
        predecessors=dict([KDE]),
        references=dict([ISOLATION_FOREST]),
        suite="adbench-classical",
        metric="roc_auc",
        notes="Both use the same Scott bandwidth and the same 2000 kernel centers, so only the "
              "robust weights differ.",
    ),
]

BY_NAME = {b.name: b for b in BENCHMARKS}

# Benchmarks that fail the gate and are tracked as open work. The gate test
# expects them to fail (strict xfail), so fixing one forces its removal here.
# Do not add new models to this list: a new model must pass to be merged.
PENDING = {
    "firth-small-n": "inconclusive vs ML logistic on Brier score at n=100",
    "firth-flic-small-n": "inconclusive vs ML logistic on Brier score at n=100",
    "firth-flac-small-n": "inconclusive vs ML logistic on Brier score at n=100",
    "rotation-classifier": "inconclusive vs random forest",
    "rotation-regressor": "significantly worse than random forest",
    "idk": "inconclusive vs Gaussian KDE and one-class SVM at default max_samples=16",
    "robust-kde": "inconclusive vs plain Gaussian KDE",
}

# Public estimators whose benchmark is another estimator's, with the reason.
COVERED_BY = {
    "RelaxedLasso": "relaxed-lasso: RelaxedLassoCV is RelaxedLasso with (alpha, phi) chosen by CV",
}
