from __future__ import annotations

from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCENES_DIR = ROOT / "scenes"
CACHE_DIR = ROOT / "cache"
PLOTS_DIR = ROOT / "plots"
RESULTS_MD = ROOT / "RESULTS.md"


def ensure_directories() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def scenario_path_parts(scenario_name: str) -> tuple[str, ...]:
    target = f"{scenario_name}.py"
    for path in SCENES_DIR.rglob(target):
        if path.name != target:
            continue
        return path.relative_to(SCENES_DIR).with_suffix("").parts
    return (scenario_name,)


def scenario_plot_dir(scenario_name: str) -> Path:
    ensure_directories()
    return PLOTS_DIR.joinpath(*scenario_path_parts(scenario_name))


def scenario_cache_path(scenario_name: str) -> Path:
    ensure_directories()
    return CACHE_DIR.joinpath(*scenario_path_parts(scenario_name)).with_suffix(".h5")
