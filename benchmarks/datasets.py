"""Download, cache and load benchmark datasets.

Two public, peer-reviewed benchmark collections are used:

* PMLB (Olson et al., BioData Mining 2017; Romano et al., Bioinformatics 2022)
  for classification and regression. PMLB curates UCI, KEEL and OpenML
  datasets into one clean format: numeric features, no missing values.
* ADBench (Han et al., NeurIPS 2022 Datasets and Benchmarks) for anomaly
  detection. Its "Classical" collection contains the ODDS datasets and the
  other standard real-world anomaly sets.

Both are fetched from GitHub, so no API key or OpenML access is needed. Every
file is downloaded once into the cache directory, which is
``$ASPECTIVE_BENCH_CACHE`` or ``~/.cache/aspective-bench``, and checked against
its SHA-256 on later loads.
"""

import gzip
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz"
ADBENCH_URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/{name}.npz"


def cache_dir():
    root = Path(os.environ.get("ASPECTIVE_BENCH_CACHE", Path.home() / ".cache" / "aspective-bench"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_atomic(path, data):
    # Parallel workers may fetch the same file; each writes its own temporary
    # file and publishes it with an atomic rename, so readers never see a
    # partial file and concurrent writers do not collide.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _download(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "aspective-benchmarks"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if data.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{url} returned a Git LFS pointer instead of the file")
    _write_atomic(path, data)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(source, name):
    """Return the local path of a dataset file, downloading it on first use."""
    if source == "pmlb":
        url, path = PMLB_URL.format(name=name), cache_dir() / "pmlb" / f"{name}.tsv.gz"
    elif source == "adbench":
        url, path = ADBENCH_URL.format(name=name), cache_dir() / "adbench" / f"{name}.npz"
    else:
        raise ValueError(f"unknown dataset source {source!r}")
    digest_file = path.with_suffix(path.suffix + ".sha256")
    if not path.exists():
        _download(url, path)
        _write_atomic(digest_file, _sha256(path).encode())
    elif digest_file.exists() and digest_file.read_text().strip() != _sha256(path):
        raise RuntimeError(f"cached file {path} is corrupted; delete it to download it again")
    return path


def _read_pmlb(path):
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        body = f.read()
    table = np.loadtxt(io.StringIO(body), delimiter="\t", ndmin=2)
    target = header.index("target")
    X = np.delete(table, target, axis=1)
    return X, table[:, target]


def _read_adbench(path):
    with np.load(path, allow_pickle=False) as data:
        return data["X"].astype(float), data["y"].astype(int)


def _subsample(X, y, max_samples, stratify, seed=0):
    """Deterministic subsample to ``max_samples`` rows, keeping class shares."""
    if max_samples is None or len(y) <= max_samples:
        return X, y
    rng = np.random.RandomState(seed)
    if not stratify:
        idx = np.sort(rng.choice(len(y), max_samples, replace=False))
        return X[idx], y[idx]
    keep = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        k = max(2, int(round(len(members) * max_samples / len(y))))
        keep.append(rng.choice(members, min(k, len(members)), replace=False))
    idx = np.sort(np.concatenate(keep))
    return X[idx], y[idx]


def load(source, name, task, max_samples=None):
    """Load one dataset as ``(X, y)``.

    ``task`` is ``"classification"``, ``"regression"`` or ``"anomaly"``. For
    anomaly detection ``y`` is 1 for anomalies and 0 for inliers. Constant
    features are dropped, since several models cannot handle them and they
    carry no information.
    """
    path = _fetch(source, name)
    X, y = _read_pmlb(path) if source == "pmlb" else _read_adbench(path)
    if task == "classification":
        _, y = np.unique(y, return_inverse=True)
    X, y = _subsample(X, y, max_samples, stratify=task != "regression")
    X = X[:, np.ptp(X, axis=0) > 0]
    return X, y


def describe(source, name, task, max_samples=None):
    X, y = load(source, name, task, max_samples)
    info = {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}
    if task == "classification":
        info["n_classes"] = int(len(np.unique(y)))
    elif task == "anomaly":
        info["anomaly_rate"] = round(float(np.mean(y)), 4)
    return json.dumps(info)
