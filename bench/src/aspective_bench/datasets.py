"""Download, cache and load benchmark datasets.

Three public, peer-reviewed benchmark collections are used:

* PMLB (Olson et al., BioData Mining 2017; Romano et al., Bioinformatics 2022)
  for classification and regression. PMLB curates UCI, KEEL and OpenML
  datasets into one clean format: numeric features, no missing values.
* OpenML-CTR23 (Fischer, Feurer & Bischl, AutoML Conference 2023 workshop
  track) for more regression datasets. Its files come from OpenML as ARFF;
  categorical and string features are coded as integers, the way PMLB does,
  and the attributes OpenML marks to ignore (row ids, leaking targets) are
  dropped.
* ADBench (Han et al., NeurIPS 2022 Datasets and Benchmarks) for anomaly
  detection. Its "Classical" collection contains the ODDS datasets and the
  other standard real-world anomaly sets.

PMLB and ADBench are fetched from GitHub, OpenML files from openml.org; no API
key is needed. A dataset name ``openml:<data id>_<name>`` selects OpenML
whatever the suite's default source. Every file is downloaded once into the
cache directory, which is ``$ASPECTIVE_BENCH_CACHE`` or
``~/.cache/aspective-bench``, and checked against its SHA-256 on later loads;
OpenML files are also checked against the MD5 that OpenML publishes.
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
OPENML_URL = "https://api.openml.org/data/v1/download/{file_id}"

# OpenML datasets, pinned so that no API call is needed and nothing can drift:
# data id -> (file id, MD5 of the ARFF file, target, attributes to drop). The
# values are those of OpenML's dataset descriptions (default target, ignore
# and row-id attributes); OpenML dataset versions are immutable.
OPENML = {
    44956: (22111820, "81f6d8b4dce14710077ab88389254d83", "rings", ()),
    44957: (22111821, "60e512ce65710079292e99a3acc71886", "sound_pressure", ()),
    44959: (22111823, "1906eae71bd8b8142d079a4b966549ac", "strength", ()),
    44962: (22111826, "0de257b0ce71eb197e1c774a4071705b", "area", ()),
    44963: (22111827, "a7e9bb5d3d78ac0c5aad3edcb26404b0", "RMSD", ()),
    44964: (22111828, "14f13f8994cf3e2fb32a667e0818a522", "critical_temp", ()),
    44967: (22111831, "cee70f2a16e9c52694bf0bb77603b4e8", "G3", ("G1", "G2")),
    44970: (22111834, "4900e250afcfee9d523aba87895260fe", "LC50", ()),
    44971: (22111835, "06231dd5ef55e3a9d5038f4f3ad88598", "quality", ()),
    44972: (22111836, "710b6653907aba6b655fd4a59fba71eb", "quality", ()),
    44976: (22111840, "545e4913cdb8a638aa726af75381bd30", "V22", ("V23", "V24", "V25", "V26", "V27", "V28")),
    44979: (22111843, "e4c67fdd8852e51246cd03af91003046", "price", ()),
    44983: (22111847, "7a89f2ba4aa504d0df53b49b3204f426", "SALE_PRC", ("PARCELNO",)),
    44984: (22111848, "0aaf01140d0bd8624e0d186639ee9125", "wage", ()),
    44987: (22111851, "d22f9d79db5ee0bfe8a484d123829981", "counts_for_sons_current_occupation", ()),
    44989: (22111853, "3cb78791be3a56d76c7860e2812d3185", "price", ()),
    44990: (22111854, "4035c290a9c383fbc7aa3dde90b11e40", "total", ("rent_amount", "property_tax", "fire_insurance")),
    44993: (22111857, "6c3993159d394af13ed909285a3927fa", "whrswk", ("wght",)),
    44994: (22111858, "5bec24b180a78f93968b13c0f26a7e34", "Price", ()),
    45012: (22111894, "ed275765771323948ebe41edea924016", "wage_eur", ()),
    45402: (22116370, "920d059f61c6d88d34c653bdbdd04f26", "ln_votes_pop", ()),
}


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


def _download(url, path, md5=None):
    request = urllib.request.Request(url, headers={"User-Agent": "aspective-benchmarks"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if data.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{url} returned a Git LFS pointer instead of the file")
    if md5 is not None:  # an OpenML ARFF file, stored compressed
        if hashlib.md5(data).hexdigest() != md5:
            raise RuntimeError(f"{url} does not match its published MD5 {md5}")
        data = gzip.compress(data, mtime=0)
    _write_atomic(path, data)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(source, name):
    """Return the local path of a dataset file, downloading it on first use."""
    if source == "pmlb":
        url, path = PMLB_URL.format(name=name), cache_dir() / "pmlb" / f"{name}.tsv.gz"
    elif source == "adbench":
        url, path = ADBENCH_URL.format(name=name), cache_dir() / "adbench" / f"{name}.npz"
    elif source == "openml":
        file_id, md5, _, _ = OPENML[openml_id(name)]
        url, path = OPENML_URL.format(file_id=file_id), cache_dir() / "openml" / f"{name}.arff.gz"
    else:
        raise ValueError(f"unknown dataset source {source!r}")
    digest_file = path.with_suffix(path.suffix + ".sha256")
    if not path.exists():
        _download(url, path, md5 if source == "openml" else None)
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


def openml_id(name):
    """The OpenML data id of a dataset named ``<data id>_<name>``."""
    return int(name.split("_", 1)[0])


def _read_openml(path, name):
    """Numeric ``(X, y)`` from an OpenML ARFF file.

    Nominal features are coded by the position of their value in the
    declared list and string features by the sorted order of their values,
    which is how PMLB codes categories. Rows with a missing value raise, since
    the suites only admit complete datasets.
    """
    from sklearn.externals import _arff  # the liac-arff parser scikit-learn's fetch_openml uses

    with gzip.open(path, "rt", encoding="utf-8") as f:
        arff = _arff.load(f, encode_nominal=True)
    _, _, target, drop = OPENML[openml_id(name)]
    names = [attr for attr, _ in arff["attributes"]]
    columns = list(zip(*arff["data"]))
    out = []
    for attr, kind in arff["attributes"]:
        values = columns[names.index(attr)]
        if any(v is None for v in values):
            raise ValueError(f"{name}: {attr} has missing values")
        if kind == "STRING":
            levels = {v: i for i, v in enumerate(sorted(set(values)))}
            values = [levels[v] for v in values]
        out.append(np.asarray(values, dtype=float))
    table = dict(zip(names, out))
    missing = [a for a in (target, *drop) if a not in table]
    if missing:
        raise ValueError(f"{name}: attributes {missing} not in the file")
    features = [a for a in names if a != target and a not in drop]
    return np.column_stack([table[a] for a in features]), table[target]


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
    if name.startswith("openml:"):
        source, name = "openml", name.removeprefix("openml:")
    path = _fetch(source, name)
    if source == "pmlb":
        X, y = _read_pmlb(path)
    elif source == "openml":
        X, y = _read_openml(path, name)
    else:
        X, y = _read_adbench(path)
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
