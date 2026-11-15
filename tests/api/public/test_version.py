from __future__ import annotations

import tomllib
from pathlib import Path

import witwin.maxwell as mw

PACKAGE = Path(mw.__file__).resolve().parent


def test_version_matches_pyproject():
    pyproject = tomllib.loads((PACKAGE.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert mw.__version__ == pyproject["project"]["version"]
    assert "__version__" not in mw.__all__


def test_package_carries_py_typed_marker():
    assert (PACKAGE / "py.typed").is_file()
