# aspective

**Peer-reviewed variants of classic estimators that never made it into
scikit-learn, with the scikit-learn API.**

Naive Bayes, LDA, k-NN, CART, random forests, AdaBoost, the lasso, logistic
regression and Isolation Forest are the workhorses of applied machine
learning. Over decades, the statistics and machine learning literature
published variants of each that fix a known weakness, and compared them
carefully with the original. Most of them never reached a mainstream Python
library. aspective collects them.

<div class="grid cards" markdown>

-   :material-file-document-check-outline: **From the literature**

    ---

    Every estimator comes from a paper in a strong venue that compared it
    with its vanilla version in a principled way, and its docstring cites
    that paper and says when to prefer it.

-   :material-scale-balance: **Benchmark-gated**

    ---

    An estimator is only included if, on dozens of public datasets, it is
    statistically non-inferior to the model it improves on. Candidates that
    fail are removed, and the [benchmarks](benchmarks.md) say which.

-   :material-puzzle-check-outline: **Drop-in**

    ---

    Supervised estimators pass scikit-learn's `check_estimator`, so they
    work with pipelines, grid search and cross-validation. Anomaly detectors
    are genuine PyOD detectors when PyOD is installed.

-   :material-compass-outline: **Easy to choose**

    ---

    Start from the model you already use, and the
    [guide](user_guide/choosing.md) lists its variants and when each helps.

</div>

## A first example

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV
from aspective.discriminant_analysis import RegularizedDiscriminantAnalysis

X, y = load_wine(return_X_y=True)
search = GridSearchCV(
    RegularizedDiscriminantAnalysis(),
    {"lam": [0, 0.25, 0.5, 0.75, 1], "gamma": [0, 0.1, 0.5]},
).fit(X, y)
print(search.best_params_, search.best_score_)
```

`lam=0, gamma=0` is QDA and `lam=1, gamma=0` is LDA, so the grid search
picks between them and everything in between.

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Choosing an estimator](user_guide/choosing.md)
- [Benchmarks](benchmarks.md)
- [API reference](api/index.md)
