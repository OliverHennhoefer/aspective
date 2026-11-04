"""Every model aspective has considered, and what became of it.

This is the source of ``docs/models.md`` (written by ``aspective-bench
report``). It records the estimators in the package, those the benchmark
gate removed, those dropped before a benchmark could run, shortlisted ones
not built yet, and those left out because they already have a maintained
home or are out of scope. Results of included models are read from the
committed benchmark results, so the page never disagrees with the gate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    name: str
    task: str  # "classification", "regression" or "anomaly detection"
    paper: str
    improves_on: str
    status: str  # key of STATUSES
    benchmark: str = ""  # included models: name of their benchmark
    note: str = ""  # everything else: result or reason, and where the code is


STATUSES = {
    "included": "In aspective",
    "removed": "Removed by the benchmark gate",
    "dropped": "Dropped before benchmarking",
    "not built": "Shortlisted, not built yet",
    "left out": "Left out: a maintained home exists, or out of scope",
}

CLF, REG, AD = "classification", "regression", "anomaly detection"

ENTRIES = [
    # In the package. Results come from bench/results/<benchmark>.json.
    Entry("RegularizedDiscriminantAnalysis", CLF, "Friedman, JASA 1989", "LDA, QDA", "included", "rda"),
    Entry("MixtureDiscriminantAnalysis", CLF, "Hastie & Tibshirani, JRSS-B 1996", "LDA", "included", "mda"),
    Entry("AODEClassifier", CLF, "Webb, Boughton & Wang, Machine Learning 2005", "Naive Bayes", "included", "aode"),
    Entry("A2DEClassifier", CLF, "Webb, Boughton, Zheng, Ting & Salem, Machine Learning 2012", "Naive Bayes",
          "included", "a2de"),
    Entry("TANClassifier", CLF, "Friedman, Geiger & Goldszmidt, Machine Learning 1997", "Naive Bayes",
          "included", "tan"),
    Entry("KDBClassifier", CLF, "Sahami, KDD 1996", "Naive Bayes", "included", "kdb"),
    Entry("HiddenNaiveBayesClassifier", CLF, "Zhang, Jiang & Su, AAAI 2005", "Naive Bayes", "included", "hnb"),
    Entry("FlexibleNaiveBayes", CLF, "John & Langley, UAI 1995", "Gaussian naive Bayes", "included", "flexible-nb"),
    Entry("DANNClassifier", CLF, "Hastie & Tibshirani, IEEE TPAMI 1996", "k-NN", "included", "dann"),
    Entry("RotationForestClassifier", CLF, "Rodríguez, Kuncheva & Alonso, IEEE TPAMI 2006", "Random forest",
          "included", "rotation-forest"),
    Entry("LogitBoostClassifier", CLF, "Friedman, Hastie & Tibshirani, Annals of Statistics 2000", "AdaBoost",
          "included", "logitboost"),
    Entry("MultiBoostClassifier", CLF, "Webb, Machine Learning 2000", "AdaBoost", "included", "multiboost"),
    Entry("FirthLogisticRegression", CLF, "Firth, Biometrika 1993; Puhr et al., Statistics in Medicine 2017",
          "Logistic regression", "included", "firth"),
    Entry("CauchyLogisticRegression", CLF, "Gelman, Jakulin, Pittau & Su, Annals of Applied Statistics 2008",
          "Logistic regression", "included", "cauchy-logistic"),
    Entry("LoessRegressor", REG, "Cleveland & Devlin, JASA 1988", "k-NN regression", "included", "loess"),
    Entry("LocalLinearForestRegressor", REG, "Friedberg, Tibshirani, Athey & Wager, JCGS 2020", "Random forest",
          "included", "local-linear-forest"),
    Entry("M5ModelTreeRegressor", REG, "Quinlan 1992; Wang & Witten, ECML 1997", "CART", "included", "m5"),
    Entry("RelaxedLasso, RelaxedLassoCV", REG, "Meinshausen, CSDA 2007", "Lasso", "included", "relaxed-lasso"),
    Entry("AdaptiveLasso, AdaptiveLassoCV", REG, "Zou, JASA 2006", "Lasso", "included", "adaptive-lasso"),
    Entry("NonconvexPenaltyRegressionCV (MCP)", REG, "Zhang, Annals of Statistics 2010", "Lasso", "included", "mcp"),
    Entry("NonconvexPenaltyRegressionCV (SCAD)", REG, "Fan & Li, JASA 2001", "Lasso", "included", "scad"),
    Entry("Bolasso", REG, "Bach, ICML 2008", "Lasso", "included", "bolasso"),
    Entry("L2BoostRegressor", REG, "Bühlmann & Yu, JASA 2003; Bühlmann, Annals of Statistics 2006", "Lasso",
          "included", "l2boost"),
    Entry("MARSRegressor", REG, "Friedman, Annals of Statistics 1991", "Linear regression, CART", "included", "mars"),
    Entry("PSplineGAMRegressor", REG, "Hastie & Tibshirani, Statistical Science 1986; Eilers & Marx, "
          "Statistical Science 1996", "Linear regression", "included", "pspline-gam"),
    Entry("RSHashDetector", AD, "Sathe & Aggarwal, ICDM 2016", "Isolation Forest", "included", "rshash"),
    Entry("LeSiNNDetector", AD, "Pang, Ting & Albrecht, ICDM Workshops 2015", "Isolation Forest", "included",
          "lesinn"),
    Entry("ExtendedIsolationForestDetector", AD, "Hariri, Carrasco Kind & Brunner, IEEE TKDE 2021",
          "Isolation Forest", "included", "extended-iforest"),

    # Removed by the gate. The code is in the git history at the commit named.
    Entry("RandomRotationClassifier", CLF, "Blaser & Fryzlewicz, JMLR 2016", "Random forest", "removed",
          note="Inferior on the larger suites: median accuracy −0.006, 11 severe losses against 1 severe "
               "win (earlier inconclusive with min-max and rank scaling). Code: `2fecda7`."),
    Entry("RandomRotationRegressor", REG, "Blaser & Fryzlewicz, JMLR 2016", "Random forest", "removed",
          note="Inferior: median R² −0.023, worse on 15 of 25 datasets. Code: `2fecda7`."),
    Entry("IDKDetector (Isolation Distributional Kernel)", AD, "Ting, Xu, Washio & Zhou, KDD 2020",
          "Gaussian KDE, one-class SVM", "removed",
          note="Inconclusive on the larger suites vs Gaussian KDE (median ROC AUC −0.012) and one-class SVM "
               "(−0.003); the hypersphere kernel was inconclusive or worse at ψ = 16, 64, 256. Code: `2fecda7`."),
    Entry("RobustKDEDetector", AD, "Kim & Scott, JMLR 2012", "Gaussian KDE", "removed",
          note="Inconclusive on the larger suites: median ROC AUC +0.000, p = 0.08; also with the paper's "
               "median nearest-neighbour bandwidth. Code: `2fecda7`."),
    Entry("SupervisedPCARegressor", REG, "Bair, Hastie, Paul & Tibshirani, JASA 2006", "PCR", "removed",
          note="Inferior: median R² −0.010, worse on 13 of 25; built for many more features than samples. "
               "Code: `2aa9e33`."),
    Entry("LoOPDetector", AD, "Kriegel, Kröger, Schubert & Zimek, CIKM 2009", "LOF", "removed",
          note="Inconclusive on the larger suites: median ROC AUC −0.012, worse on 25 of 46. Code: `2aa9e33`."),
    Entry("DARTClassifier", CLF, "Rashmi & Gilad-Bachrach, AISTATS 2015", "Gradient boosting", "removed",
          note="Inferior on the larger suites: median accuracy −0.005, worse on 28 of 68. Code: `b2bf66c`."),
    Entry("DARTRegressor", REG, "Rashmi & Gilad-Bachrach, AISTATS 2015", "Gradient boosting", "removed",
          note="Inconclusive on the larger suites: median R² −0.001, worse on 12 of 31. Code: `b2bf66c`."),
    Entry("SqrtLasso", REG, "Belloni, Chernozhukov & Wang, Biometrika 2011", "LassoCV", "removed",
          note="Inferior: the pivotal penalty over-shrinks, median R² −0.012, worse on 14 of 25. Code: `b2bf66c`."),
    Entry("SCiForestDetector", AD, "Liu, Ting & Zhou, ECML PKDD 2010", "Isolation Forest", "removed",
          note="Inconclusive on the larger suites: median ROC AUC −0.002, worse on 20 of 46. Code: `b2bf66c`."),
    Entry("INFLODetector", AD, "Jin, Tung, Han & Wang, PAKDD 2006", "LOF", "removed",
          note="Inferior: median ROC AUC −0.019, worse on 21 of 37. Code: `b2bf66c`."),
    Entry("AntiHubDetector (AntiHub2)", AD, "Radovanović, Nanopoulos & Ivanović, IEEE TKDE 2015",
          "k-NN distance detector", "removed",
          note="Inferior: median ROC AUC −0.078, worse on 28 of 37; built for very high-dimensional data. "
               "Code: `b2bf66c`."),
    Entry("LDOFDetector", AD, "Zhang, Hutter & Jin, PAKDD 2009", "k-NN distance detector", "removed",
          note="Inferior: median ROC AUC −0.121, worse on 27 of 37. Code: `b2bf66c`."),

    # Dropped before a benchmark ran.
    Entry("KDEOS", AD, "Schubert, Zimek & Kriegel, SDM 2014", "LOF", "dropped",
          note="The exact definition could not be checked against the paper, and the version built from "
               "secondary descriptions lost to LOF on a planted-outlier sanity check."),
    Entry("MM robust regression", REG, "Yohai, Annals of Statistics 1987", "Least squares", "dropped",
          note="Dropped at the design stage of the second wave, before any code or benchmark."),
    Entry("Non-negative garrote", REG, "Breiman, Technometrics 1995", "Lasso", "dropped",
          note="Dropped at the design stage of the second wave, before any code or benchmark."),

    # Shortlisted in the first research pass, not built yet.
    Entry("Logistic Model Trees", CLF, "Landwehr, Hall & Frank, Machine Learning 2005",
          "C4.5 / CART, logistic regression", "not built",
          note="Needs a tree builder with LogitBoost carried down the tree; only Weka has it."),
    Entry("Conditional inference forests", f"{CLF}, {REG}", "Hothorn, Hornik & Zeileis, JCGS 2006",
          "CART, random forest", "not built",
          note="Large effort (permutation-test splits); its gain is unbiased variable importance, which "
               "the accuracy-based gate cannot measure."),
    Entry("Distributional random forests", REG, "Ćevid, Michel, Näf, Bühlmann & Meinshausen, JMLR 2022",
          "Random forest, quantile regression forest", "not built",
          note="Large effort; it predicts distributions, which needs a probabilistic benchmark track first."),
    Entry("Twin SVM", CLF, "Jayadeva, Khemchandani & Chandra, IEEE TPAMI 2007", "SVM", "not built",
          note="Not reached yet."),
    Entry("Relevance vector machine", f"{CLF}, {REG}", "Tipping, JMLR 2001", "SVM, SVR", "not built",
          note="Not reached yet."),
    Entry("PIDForest", AD, "Gopalan, Sharan & Wieder, NeurIPS 2019", "Isolation Forest", "not built",
          note="Not reached yet."),
    Entry("HiCS", AD, "Keller, Müller & Böhm, ICDE 2012", "LOF, feature bagging", "not built",
          note="Not reached yet."),

    # Left out.
    Entry("NGBoost", f"{CLF}, {REG}", "Duan et al., ICML 2020", "Gradient boosting", "left out",
          note="Maintained in `ngboost`; its strength is predictive distributions, which the gate does not "
               "score yet (no NLL or CRPS metric)."),
    Entry("Sparse projection oblique forests (SPORF)", f"{CLF}, {REG}", "Tomita et al., JMLR 2020",
          "Random forest", "left out", note="Maintained in `treeple`."),
    Entry("Quantile regression forests", REG, "Meinshausen, JMLR 2006", "Random forest", "left out",
          note="Maintained in `quantile-forest`."),
    Entry("Generalized and causal random forests", REG, "Athey, Tibshirani & Wager, Annals of Statistics 2019",
          "Random forest", "left out", note="Maintained in `EconML`."),
    Entry("Hierarchical shrinkage, FIGS, GOSDT", f"{CLF}, {REG}", "ICML 2022; PNAS; ICML 2020", "CART",
          "left out", note="Maintained in `imodels`."),
    Entry("BART", REG, "Chipman, George & McCulloch, Annals of Applied Statistics 2010", "Tree ensembles",
          "left out", note="Maintained in `stochtree` and `pymc-bart`."),
    Entry("Large-margin nearest neighbours (LMNN)", CLF, "Weinberger & Saul, JMLR 2009", "k-NN", "left out",
          note="Maintained in `metric-learn` (scikit-learn-contrib)."),
    Entry("Regularized greedy forest", f"{CLF}, {REG}", "Johnson & Zhang, IEEE TPAMI 2014", "Gradient boosting",
          "left out", note="Maintained in `rgf-python`."),
    Entry("Deep Isolation Forest, iNNE, LODA, ECOD, COPOD, GLOSH", AD, "IEEE TKDE, Machine Learning, ICDM, TKDD",
          "Isolation Forest, LOF", "left out", note="Maintained in PyOD."),
    Entry("Aggregated Mondrian forests", f"{CLF}, {REG}", "Mourtada, Gaïffas & Scornet, JRSS-B 2021",
          "Random forest", "left out", note="Online learning, out of scope."),
    Entry("Robust random cut forest", AD, "Guha, Mishra, Roy & Schrijvers, ICML 2016", "Isolation Forest",
          "left out", note="Streaming, out of scope."),
    Entry("xStream", AD, "Manzoor, Lamba & Akoglu, KDD 2018", "Isolation Forest", "left out",
          note="Streaming, out of scope."),
    Entry("Online Isolation Forest", AD, "Leveni, Weigert Cassales, Pfahringer, Bifet & Boracchi, ICML 2024",
          "Isolation Forest", "left out", note="Online learning, out of scope."),
]
