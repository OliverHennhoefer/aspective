"""The benchmark every aspective estimator must pass.

To add a model, append a ``Benchmark`` to ``BENCHMARKS``: the candidate, the
vanilla model(s) it improves on, a suite and a metric. Then run

    uv run aspective-bench run <name>
    uv run aspective-bench report

and commit ``bench/results/<name>.json`` and ``bench/RESULTS.md``.
``bench/tests/test_gate.py`` fails while any public estimator has no
benchmark, a stale result, or a failing verdict. See ``bench/README.md``.

Predecessors are configured like the candidate's own building blocks (same
number of trees, same discretization, same bandwidth rule), so a comparison
measures the paper's idea and not a tuning difference.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from aspective.anomaly import ExtendedIsolationForestDetector, LeSiNNDetector, RSHashDetector
from aspective.discriminant_analysis import MixtureDiscriminantAnalysis, RegularizedDiscriminantAnalysis
from aspective.ensemble import (
    LocalLinearForestRegressor,
    LogitBoostClassifier,
    MultiBoostClassifier,
    RotationForestClassifier,
)
from aspective.linear_model import (
    AdaptiveLassoCV,
    Bolasso,
    CauchyLogisticRegression,
    FirthLogisticRegression,
    L2BoostRegressor,
    MARSRegressor,
    NonconvexPenaltyRegressionCV,
    PSplineGAMRegressor,
    RelaxedLassoCV,
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

from .protocol import Benchmark

N_TREES = 200
N_ROUNDS = 200  # boosting rounds
DEPTH = 3  # depth of boosted trees


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


# LOESS and k-NN get five neighbourhood sizes each, so neither is tuned more.
def loess():
    return GridSearchCV(scaled(LoessRegressor()),
                        {"loessregressor__span": [0.02, 0.05, 0.1, 0.25, 0.5]}, cv=3)


def knn_regression():
    return GridSearchCV(scaled(KNeighborsRegressor()),
                        {"kneighborsregressor__n_neighbors": [3, 5, 10, 20, 40]}, cv=3)


# Shared factories, so the same baseline is computed once and reused.
NB_5_BINS = ("Naive Bayes (5 bins)", binned_nb)
ML_LOGISTIC = ("ML logistic", ml_logistic)
ISOLATION_FOREST = ("Isolation Forest", lambda: scaled(IsolationForest(n_estimators=N_TREES, random_state=0)))
LOF = ("LOF", lambda: scaled(LocalOutlierFactor(novelty=True)))
LASSO_CV = ("LassoCV", lambda: scaled(LassoCV(cv=5)))
ADABOOST = ("AdaBoost (depth 3)", lambda: AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=DEPTH), n_estimators=N_ROUNDS, random_state=0))


def cart_regression():
    return GridSearchCV(DecisionTreeRegressor(random_state=0),
                        {"min_samples_leaf": [1, 5, 10, 20, 50]}, cv=3)


CART_TUNED = ("CART (tuned leaf size)", cart_regression)

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
        predecessors=dict([NB_5_BINS]),
        references={"Gaussian NB": GaussianNB},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization, as in Webb et al. (2005).",
    ),
    Benchmark(
        name="firth",
        candidate=("Firth", firth()),
        predecessors=dict([ML_LOGISTIC]),
        suite="pmlb-binary",
        metric="brier",
        notes="Full-size training folds; features standardized for every model.",
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
        name="rshash",
        candidate=("RS-Hash", lambda: scaled(RSHashDetector(random_state=0))),
        predecessors=dict([ISOLATION_FOREST]),
        references=dict([LOF]),
        suite="adbench-classical",
        metric="roc_auc",
    ),
    Benchmark(
        name="tan",
        candidate=("TAN (5 bins)", lambda: TANClassifier(n_bins=5)),
        predecessors=dict([NB_5_BINS]),
        references={"AODE (5 bins)": lambda: AODEClassifier(n_bins=5)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization and Laplace smoothing.",
    ),
    Benchmark(
        name="mda",
        candidate=("MDA (tuned)", lambda: GridSearchCV(
            MixtureDiscriminantAnalysis(random_state=0), {"n_subclasses": [1, 2, 3, 5]}, cv=3)),
        predecessors={"LDA": LinearDiscriminantAnalysis},
        references={"QDA": QuadraticDiscriminantAnalysis},
        suite="pmlb-classification",
        metric="accuracy",
        notes="MDA picks the number of subclasses per class by inner 3-fold CV from 1 to 5, as "
              "Hastie & Tibshirani (1996) did; one subclass is LDA.",
    ),
    Benchmark(
        name="cauchy-logistic",
        candidate=("Cauchy prior", CauchyLogisticRegression),
        predecessors=dict([ML_LOGISTIC]),
        references={"Ridge logistic (C=1)": lambda: scaled(LogisticRegression(max_iter=10_000)),
                    "Firth": firth()},
        suite="pmlb-binary",
        metric="brier",
        notes="The Cauchy model applies the paper's own input scaling (sd 0.5, binary inputs "
              "centred); the other models get standardized inputs.",
    ),
    Benchmark(
        name="adaptive-lasso",
        candidate=("Adaptive LassoCV", lambda: scaled(AdaptiveLassoCV(cv=5))),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="gamma in {0.5, 1, 2} and alpha chosen by 5-fold CV; ridge initial estimator.",
    ),
    Benchmark(
        name="l2boost",
        candidate=("L2Boost (CV)", lambda: scaled(L2BoostRegressor(cv=5))),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="Componentwise linear least squares, learning rate 0.1, number of steps (up to "
              "2000) by 5-fold CV, as in mboost.",
    ),
    Benchmark(
        name="loess",
        candidate=("LOESS (tuned span)", loess),
        predecessors={"k-NN (tuned k)": knn_regression},
        suite="pmlb-regression",
        metric="r2",
        notes="Local linear fits with tricube weights. Both tune their neighbourhood size over "
              "five values by inner 3-fold CV on standardized inputs.",
    ),
    Benchmark(
        name="lesinn",
        candidate=("LeSiNN", lambda: scaled(LeSiNNDetector(random_state=0))),
        predecessors=dict([ISOLATION_FOREST]),
        references=dict([LOF]),
        suite="adbench-classical",
        metric="roc_auc",
        notes="The paper's defaults: 50 subsamples of 8 points.",
    ),
    # --- third wave -----------------------------------------------------------
    Benchmark(
        name="kdb",
        candidate=("KDB k=2 (5 bins)", lambda: KDBClassifier(k=2, n_bins=5)),
        predecessors=dict([NB_5_BINS]),
        references={"TAN (5 bins)": lambda: TANClassifier(n_bins=5)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization and Laplace smoothing.",
    ),
    Benchmark(
        name="hnb",
        candidate=("Hidden NB (5 bins)", lambda: HiddenNaiveBayesClassifier(n_bins=5)),
        predecessors=dict([NB_5_BINS]),
        references={"AODE (5 bins)": lambda: AODEClassifier(n_bins=5)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization and Laplace smoothing.",
    ),
    Benchmark(
        name="a2de",
        candidate=("A2DE (5 bins)", lambda: A2DEClassifier(n_bins=5)),
        predecessors=dict([NB_5_BINS]),
        references={"AODE (5 bins)": lambda: AODEClassifier(n_bins=5)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both use the same 5-bin quantile discretization and Laplace smoothing.",
    ),
    Benchmark(
        name="flexible-nb",
        candidate=("Flexible NB", FlexibleNaiveBayes),
        predecessors={"Gaussian NB": GaussianNB},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Kernel width 1/sqrt(n_c) on standardized features, the paper's rule.",
    ),
    Benchmark(
        name="dann",
        candidate=("DANN (k=5)", lambda: DANNClassifier(n_neighbors=5)),
        predecessors={"k-NN (k=5)": lambda: scaled(KNeighborsClassifier(n_neighbors=5))},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both vote with 5 neighbours on standardized inputs; DANN estimates its metric "
              "from max(n/5, 50) neighbours with epsilon = 1, as in the paper.",
    ),
    Benchmark(
        name="rotation-forest",
        candidate=("Rotation Forest", lambda: RotationForestClassifier(n_estimators=N_TREES,
                                                                       random_state=0)),
        predecessors={"Random forest": lambda: RandomForestClassifier(n_estimators=N_TREES,
                                                                       random_state=0)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="Both grow 200 unpruned trees; groups of 3 features, as in the paper.",
    ),
    Benchmark(
        name="logitboost",
        candidate=("LogitBoost (depth 3)", lambda: LogitBoostClassifier(
            n_estimators=N_ROUNDS, max_depth=DEPTH, random_state=0)),
        predecessors=dict([ADABOOST]),
        references={"Gradient boosting (depth 3)": lambda: GradientBoostingClassifier(
            n_estimators=N_ROUNDS, max_depth=DEPTH, random_state=0)},
        suite="pmlb-classification",
        metric="accuracy",
        notes="200 rounds of depth-3 trees for every model, no shrinkage for LogitBoost and "
              "AdaBoost (SAMME), as in the paper.",
    ),
    Benchmark(
        name="multiboost",
        candidate=("MultiBoost (depth 3)", lambda: MultiBoostClassifier(
            DecisionTreeClassifier(max_depth=DEPTH), n_estimators=N_ROUNDS, random_state=0)),
        predecessors=dict([ADABOOST]),
        suite="pmlb-classification",
        metric="accuracy",
        notes="Same depth-3 trees and SAMME weights; MultiBoost restarts with wagging weights "
              "about every sqrt(200) = 14 members.",
    ),
    Benchmark(
        name="local-linear-forest",
        candidate=("Local linear forest", lambda: LocalLinearForestRegressor(
            n_estimators=N_TREES, min_samples_leaf=5, random_state=0)),
        predecessors={"Random forest (leaf 5)": lambda: RandomForestRegressor(
            n_estimators=N_TREES, min_samples_leaf=5, random_state=0)},
        suite="pmlb-regression",
        metric="r2",
        notes="The same forest (200 trees, leaves of at least 5) supplies the weights; ridge "
              "penalty 0.1 on the local slopes, grf's default.",
    ),
    Benchmark(
        name="m5",
        candidate=("M5 model tree", M5ModelTreeRegressor),
        predecessors=dict([CART_TUNED]),
        references=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="M5 with its defaults (leaves of 4, pruning, smoothing k = 15). CART picks its "
              "minimum leaf size from five values by inner 3-fold CV.",
    ),
    Benchmark(
        name="mars",
        candidate=("MARS (additive)", MARSRegressor),
        predecessors=dict([LASSO_CV, CART_TUNED]),
        suite="pmlb-regression",
        metric="r2",
        notes="Degree 1 (earth's default), terms pruned by GCV with penalty 2. MARS generalizes "
              "both stepwise linear regression and recursive partitioning, so it must keep up "
              "with both.",
    ),
    Benchmark(
        name="pspline-gam",
        candidate=("P-spline GAM", PSplineGAMRegressor),
        predecessors=dict([LASSO_CV]),
        references={"MARS (additive)": MARSRegressor},
        suite="pmlb-regression",
        metric="r2",
        notes="Cubic P-splines on 10 equally spaced knots per feature, one smoothing "
              "parameter chosen by GCV; the strongest smoothing is linear regression.",
    ),
    Benchmark(
        name="mcp",
        candidate=("MCP (CV)", lambda: NonconvexPenaltyRegressionCV(penalty="mcp", cv=5)),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="gamma = 3, penalty chosen by 5-fold CV from 50 values, as ncvreg does.",
    ),
    Benchmark(
        name="scad",
        candidate=("SCAD (CV)", lambda: NonconvexPenaltyRegressionCV(penalty="scad", cv=5)),
        predecessors=dict([LASSO_CV]),
        suite="pmlb-regression",
        metric="r2",
        notes="gamma = 3.7, penalty chosen by 5-fold CV from 50 values, as ncvreg does.",
    ),
    Benchmark(
        name="extended-iforest",
        candidate=("Extended IF", lambda: scaled(ExtendedIsolationForestDetector(
            n_estimators=N_TREES, random_state=0))),
        predecessors=dict([ISOLATION_FOREST]),
        suite="adbench-classical",
        metric="roc_auc",
        notes="Fully extended cuts, 200 trees of 256 points like the Isolation Forest.",
    ),
]

BY_NAME = {b.name: b for b in BENCHMARKS}

# Public estimators whose benchmark is another estimator's, with the reason.
COVERED_BY = {
    "RelaxedLasso": "relaxed-lasso: RelaxedLassoCV is RelaxedLasso with (alpha, phi) chosen by CV",
    "AdaptiveLasso": "adaptive-lasso: AdaptiveLassoCV is AdaptiveLasso with (alpha, gamma) chosen by CV",
    "NonconvexPenaltyRegression": "mcp, scad: NonconvexPenaltyRegressionCV is it with alpha chosen by CV",
}
