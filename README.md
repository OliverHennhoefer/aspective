# aspective

Peer-reviewed variants of classic estimators that never made it into
scikit-learn, with the scikit-learn API.

Every estimator in aspective

- comes from a paper in a strong venue that compared it with its vanilla
  version in a principled way,
- cites that paper in its docstring and says when to prefer it,
- passes scikit-learn's `check_estimator` (supervised models) or follows
  PyOD's `BaseDetector` contract (anomaly detectors),
- passes the benchmark gate in [`benchmarks/`](benchmarks/): on dozens of
  public datasets it is statistically non-inferior to the vanilla model it
  improves on (see [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)).

## Installation

```bash
pip install aspective          # classifiers and regressors
pip install "aspective[pyod]"  # detectors then subclass PyOD's BaseDetector
```

The anomaly detectors work without PyOD through a small stand-in base class
with the same contract. With PyOD installed they are genuine PyOD detectors
and work inside PyOD's ensembles (LSCP, SUOD), thresholding and benchmarks.

## Estimators (first wave)

| Estimator | Improves on | Paper |
|---|---|---|
| `discriminant_analysis.RegularizedDiscriminantAnalysis` | LDA, QDA | Friedman, JASA 1989 |
| `naive_bayes.AODEClassifier` | Naive Bayes | Webb, Boughton & Wang, Machine Learning 2005 |
| `linear_model.FirthLogisticRegression` | Logistic regression | Firth, Biometrika 1993; Puhr et al., Statistics in Medicine 2017 (FLIC/FLAC) |
| `ensemble.RandomRotationClassifier`, `RandomRotationRegressor` | Random forest, any tree ensemble | Blaser & Fryzlewicz, JMLR 2016 |
| `linear_model.RelaxedLasso`, `RelaxedLassoCV` | Lasso | Meinshausen, CSDA 2007 |
| `linear_model.Bolasso` | Lasso (variable selection) | Bach, ICML 2008 |
| `anomaly.IDKDetector` | One-class SVM, Gaussian KDE | Ting, Xu, Washio & Zhou, KDD 2020 |
| `anomaly.RSHashDetector` | Isolation Forest, subspace detectors | Sathe & Aggarwal, ICDM 2016 |
| `anomaly.RobustKDEDetector` | Kernel density estimation | Kim & Scott, JMLR 2012 |

## Example

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis

X, y = load_wine(return_X_y=True)
search = GridSearchCV(
    RegularizedDiscriminantAnalysis(),
    {"lam": [0, 0.25, 0.5, 0.75, 1], "gamma": [0, 0.1, 0.5]},
).fit(X, y)
```

```python
from aspective.anomaly import IDKDetector

detector = IDKDetector(contamination=0.05, random_state=0).fit(X)
detector.decision_scores_   # higher means more anomalous
detector.predict(X)         # 0 inlier, 1 outlier, as in PyOD
```

## Development

```bash
pip install -e ".[test]"
pytest
python -m benchmarks run my-model   # see benchmarks/README.md
```
