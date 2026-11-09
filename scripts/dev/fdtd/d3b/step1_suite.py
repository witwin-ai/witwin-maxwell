"""Run the thin-sheet gates on the fused default, recording outcomes, not timings."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import pytest
import torch

from witwin.maxwell.fdtd import thin_sheet
from witwin.maxwell.fdtd.cuda.build import STABLE_ABI_VERSION


class GateReport:
    def __init__(self):
        self.tests = []
        self.active = "session"
        self.dispatch = defaultdict(Counter)
        self.original = {}

    def pytest_sessionstart(self):
        for name in ("_gather_torch", "_scatter_torch", "_gather_fused", "_scatter_fused"):
            original = getattr(thin_sheet, name)
            self.original[name] = original

            def counted(*args, _name=name, _original=original):
                device = args[2].device.type
                self.dispatch[self.active][device + _name] += 1
                return _original(*args)

            setattr(thin_sheet, name, counted)

    def pytest_runtest_setup(self, item):
        self.active = item.nodeid

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.failed or report.skipped:
            self.tests.append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                                   expected_failure=getattr(report, "wasxfail", None),
                                   output=dict(report.sections),
                                   failure=str(report.longrepr) if report.failed else None))

    def pytest_sessionfinish(self):
        for name, original in self.original.items():
            setattr(thin_sheet, name, original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--tests", nargs="+", default=["tests/materials/thin_sheet"])
    args = parser.parse_args()
    os.environ["WITWIN_MAXWELL_FDTD_CUDA_BUILD_DIR"] = "/tmp/pcb-d3b-build"
    os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp/pcb-d3b-build"
    os.environ.pop("WITWIN_MAXWELL_THIN_SHEET_KERNEL", None)
    torch.cuda.set_per_process_memory_fraction(5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    report = GateReport()
    code = pytest.main([*args.tests, "--gpu", "-q", "--tb=short",
                        f"--basetemp=/tmp/pcb-d3b-build/{args.json.stem}-tmp"], plugins=[report])
    result = dict(abi=STABLE_ABI_VERSION, selector="default", cuda_default="fused", device=0,
                  exit_code=int(code), outcomes=dict(Counter(
                      "xfail" if test["expected_failure"] else test["outcome"] for test in report.tests)),
                  tests=report.tests, dispatch={key: dict(value) for key, value in report.dispatch.items()},
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("tests", "dispatch")}), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
