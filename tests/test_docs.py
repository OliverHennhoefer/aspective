"""Run every Python example in the README and the documentation.

Each Markdown file's ``python`` code blocks run in order in one namespace, so
a later block may use names from an earlier one, as a reader would.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_examples_run(path):
    namespace = {"__name__": "__docs__"}
    for i, block in enumerate(BLOCK.findall(path.read_text(encoding="utf-8"))):
        code = compile(block, f"{path.name}[block {i}]", "exec")
        exec(code, namespace)


MODULES = [
    "discriminant_analysis",
    "naive_bayes",
    "neighbors",
    "ensemble",
    "tree",
    "linear_model",
    "anomaly",
]
PUBLIC = [
    (module, name)
    for module in MODULES
    for name in __import__(f"aspective.{module}", fromlist=["__all__"]).__all__
]


@pytest.mark.parametrize(("module", "name"), PUBLIC, ids=[n for _, n in PUBLIC])
def test_estimator_is_documented(module, name):
    """Every public estimator has an API entry, a guide row and a README row."""
    api = (ROOT / "docs" / "api" / f"{module}.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide" / "choosing.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"::: aspective.{module}.{name}\n" in api
    assert f"[aspective.{module}.{name}]" in guide
    assert name in readme
