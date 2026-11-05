# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 28 estimators across `discriminant_analysis`, `naive_bayes`, `neighbors`,
  `ensemble`, `tree`, `linear_model` and `anomaly`, each with the paper it
  comes from and each passing the benchmark gate in `bench/`.
- Ruff lint and format checks, test coverage reported to Codecov, CI on
  Python 3.10 to 3.14 with and without PyOD, and a wheel build check.
- Documentation site at https://oliverhennhoefer.github.io/aspective/ with a
  quickstart, a guide to choosing an estimator, the benchmark protocol and an
  API reference built from the docstrings. Every Python example in the docs
  and the README runs as a test.
- Release workflow: pushing a `v*` tag checks, builds and publishes to PyPI
  through trusted publishing. The version now lives only in
  `aspective.__version__`.
- Contributor guide, citation file, security policy, and issue and pull
  request templates.
- The regression benchmarks now run on 52 datasets: the 31 from PMLB plus
  the 21 OpenML-CTR23 datasets that meet the suite rule. `RESULTS.md` has
  per-suite leaderboards that rank every model, including standard methods
  with default settings (`uv run aspective-bench standards`).

### Fixed

- `M5ModelTreeRegressor` bounds its predictions to the training target
  range widened by 10% (new `extrapolation` parameter, as in Cubist). A leaf
  model fitted on few, nearly collinear rows had predicted R² −234 on one
  fold of superconductivity.
- `PSplineGAMRegressor` keeps each curve constant outside the training
  range instead of extrapolating it linearly, which had produced R² −46 on
  one fold of socmob.
