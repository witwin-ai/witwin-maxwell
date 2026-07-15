from __future__ import annotations

import argparse

from benchmark.runner import _run_maxwell
from benchmark.scenes import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    _, _, elapsed = _run_maxwell(
        scenario.builder(),
        frequencies=scenario.frequencies,
        run_time_factor=scenario.run_time_factor,
    )
    print(f"{scenario.name} {elapsed:.3f}")


if __name__ == "__main__":
    main()
