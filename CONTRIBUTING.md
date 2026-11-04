# Contributing to aspective

Thanks for helping. aspective has a narrow scope and a high bar, so please
read this before opening a pull request for a new estimator.

## What belongs in aspective

An estimator belongs here when all of these hold:

1. It is a variant of a classic estimator (naive Bayes, LDA, k-NN, random
   forest, AdaBoost, CART, the lasso, logistic regression, Isolation Forest,
   ...) and not a new model family.
2. It comes from a paper in a strong venue that compares it with the vanilla
   version in a principled way.
3. It is not already in scikit-learn or PyOD.
4. It passes the benchmark gate: on the public suites in [`bench/`](bench/),
   it is statistically non-inferior to the vanilla model it improves on.

Not sure? Open an issue with the *Estimator proposal* template first.

## Development setup

aspective uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/OliverHennhoefer/aspective.git
cd aspective
uv sync --extra pyod      # library, tests, lint tools and the bench package
uv run pre-commit install # optional: lint and format on every commit
```

## Checks

```bash
uv run pytest                      # tests, including scikit-learn's check_estimator
uv run pytest --cov                # the same, with a coverage report
uv run ruff check . && uv run ruff format --check .
uv run zensical serve              # the docs site at http://localhost:8000
```

`tests/test_docs.py` runs every Python example in the README and in
`docs/`, so examples cannot silently go stale.

CI runs the tests on Python 3.10 to 3.14, with and without PyOD, and builds
and smoke-tests the wheel.

## Adding an estimator

- Put it in the module that matches scikit-learn's layout
  (`aspective/linear_model`, `aspective/naive_bayes`, ...) in a private file,
  and export it from the module's `__init__.py`.
- Supervised estimators subclass scikit-learn's mixins, declare
  `_parameter_constraints`, and must pass `check_estimator`; add them to the
  list in `tests/test_common.py`. Anomaly detectors subclass
  `aspective.anomaly._base.BaseDetector` and follow PyOD's contract.
- Write a NumPy-style docstring with *Parameters*, *Attributes* and
  *References*. Cite the paper, and say in one or two sentences when to prefer
  the estimator over the vanilla one.
- Add focused tests for the behaviour the paper describes (for example a
  limiting case that reduces to the vanilla model).
- Add a benchmark to `bench/`, run the gate, and commit the updated
  `bench/RESULTS.md`. See [`bench/README.md`](bench/README.md).
- Add a row to the estimator table in `README.md`, add the estimator to its
  page in `docs/api/` and to the table in `docs/user_guide/choosing.md`, and
  add an entry to `CHANGELOG.md` under *Unreleased*.

## Pull requests

Keep a pull request to one estimator or one change. Fill in the checklist in
the template. Every pull request needs green CI, and a new estimator also
needs a passing benchmark gate.
