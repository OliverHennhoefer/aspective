"""The OpenML reader, offline: a small ARFF file stands in for a download."""

import gzip

import numpy as np
import pytest

from aspective_bench import datasets

ARFF = """@relation 'x'
@attribute 'id' numeric
@attribute 'size' numeric
@attribute 'colour' {red,green,blue}
@attribute 'day' string
@attribute 'leak' numeric
@attribute 'y' numeric
@data
1,2.5,blue,"tue",9,10
2,3.5,red,"mon",9,20
3,4.5,green,"tue",9,30
"""


@pytest.fixture
def arff_file(tmp_path, monkeypatch):
    monkeypatch.setitem(datasets.OPENML, 99, (0, "", "y", ("id", "leak")))
    path = tmp_path / "99_toy.arff.gz"
    path.write_bytes(gzip.compress(ARFF.encode()))
    return path


def test_openml_reader_codes_categories_and_drops_ignored(arff_file):
    X, y = datasets._read_openml(arff_file, "99_toy")
    # nominal by declared order, string by sorted value; id and leak dropped
    np.testing.assert_array_equal(X, [[2.5, 2, 1], [3.5, 0, 0], [4.5, 1, 1]])
    np.testing.assert_array_equal(y, [10, 20, 30])


def test_openml_reader_rejects_missing_values(arff_file):
    arff_file.write_bytes(gzip.compress(ARFF.replace("2,3.5,red", "2,?,red").encode()))
    with pytest.raises(ValueError, match="missing"):
        datasets._read_openml(arff_file, "99_toy")


def test_every_ctr23_dataset_is_pinned():
    from aspective_bench.suites import CTR23_REGRESSION

    ids = {datasets.openml_id(name.removeprefix("openml:")) for name in CTR23_REGRESSION}
    assert ids == set(datasets.OPENML)
