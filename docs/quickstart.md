# Quickstart

Every aspective estimator is a scikit-learn estimator, so you use it the
way you use the model it improves on.

## Classification

Swap naive Bayes for AODE, which averages over all one-dependence models and
removes much of the bias of the independence assumption, with no structure
search and nothing to tune:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer
from aspective.naive_bayes import AODEClassifier

X, y = load_breast_cancer(return_X_y=True)

naive = make_pipeline(
    KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile"),
    CategoricalNB(),
)
aode = AODEClassifier(n_bins=5)

print(cross_val_score(naive, X, y).mean())
print(cross_val_score(aode, X, y).mean())
```

## Regression

M5 model trees fit a linear model in every leaf, so they follow a trend
instead of approximating it by a staircase:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from aspective.tree import M5ModelTreeRegressor

X, y = load_diabetes(return_X_y=True)
print(cross_val_score(M5ModelTreeRegressor(), X, y, scoring="r2").mean())
```

## Small samples and separation

Firth's logistic regression gives finite, less biased coefficients where
maximum likelihood diverges, for example under complete separation:

```python
import numpy as np
from aspective.linear_model import FirthLogisticRegression

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([0, 0, 0, 1, 1, 1])  # perfectly separated

model = FirthLogisticRegression().fit(X, y)
print(model.coef_, model.intercept_)
```

## Anomaly detection

Detectors follow PyOD's conventions: fit without labels, then read the
training scores or score new data.

```python
import numpy as np
from aspective.anomaly import ExtendedIsolationForestDetector

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(size=(500, 2)), rng.uniform(-6, 6, size=(25, 2))])

detector = ExtendedIsolationForestDetector(contamination=0.05, random_state=0).fit(X)
detector.decision_scores_  # higher means more anomalous
detector.predict(X)  # 0 inlier, 1 outlier
```

## Inside scikit-learn tooling

Because every supervised estimator passes `check_estimator`, pipelines,
`GridSearchCV`, `cross_validate`, `clone` and stacking all work:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from aspective.linear_model import AdaptiveLasso

pipe = make_pipeline(StandardScaler(), AdaptiveLasso())
search = GridSearchCV(pipe, {"adaptivelasso__alpha": [0.01, 0.1, 1.0]})
```

Next: [Choosing an estimator](user_guide/choosing.md).
