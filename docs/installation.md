# Installation

aspective needs Python 3.10 or newer, NumPy, SciPy and scikit-learn 1.7 or
newer.

=== "pip"

    ```bash
    pip install aspective
    ```

=== "uv"

    ```bash
    uv add aspective
    ```

## With PyOD

```bash
pip install "aspective[pyod]"
```

The anomaly detectors work without PyOD through a small stand-in base class
with the same contract. With PyOD installed they subclass PyOD's
`BaseDetector`, so they work inside PyOD's ensembles (LSCP, SUOD),
thresholding and benchmarks.

## From source

```bash
git clone https://github.com/OliverHennhoefer/aspective.git
cd aspective
uv sync --extra pyod
uv run pytest
```

This also installs the benchmark testbed in `bench/`; see
[Benchmarks](benchmarks.md).
