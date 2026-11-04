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
- Contributor guide, citation file, security policy, and issue and pull
  request templates.
