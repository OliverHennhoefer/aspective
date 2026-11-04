## Summary

<!-- What does this change and why? -->

## Checklist

- [ ] Tests pass locally (`uv run pytest`)
- [ ] Lint and format pass (`uv run ruff check . && uv run ruff format --check .`)
- [ ] `CHANGELOG.md` has an entry under *Unreleased*

### For a new estimator

- [ ] It comes from a peer-reviewed paper that compares it with its vanilla version
- [ ] The docstring cites the paper and says when to prefer the estimator
- [ ] It passes `check_estimator` (supervised) or the PyOD detector contract (anomaly)
- [ ] It passes the benchmark gate in `bench/`, and `bench/RESULTS.md` is updated
- [ ] It is listed in the README, `docs/api/` and `docs/user_guide/choosing.md`
