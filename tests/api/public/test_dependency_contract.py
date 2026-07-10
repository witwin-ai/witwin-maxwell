from __future__ import annotations

import tomllib
from pathlib import Path


def test_runtime_and_optional_dependencies_do_not_include_slangtorch():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    groups = [pyproject["project"]["dependencies"]]
    groups.extend(pyproject["project"].get("optional-dependencies", {}).values())
    names = {
        dependency.split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].strip().lower()
        for group in groups
        for dependency in group
    }
    assert "slangtorch" not in names
    assert all(
        ".slang" not in artifact
        for artifact in pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
            "artifacts"
        ]
    )
