import importlib.util
from pathlib import Path

from benchmark import report as benchmark_report
from benchmark import status
from benchmark.scenes import SCENARIOS


ROOT = Path(__file__).resolve().parents[3]
LIMITS = status.SuiteLimits(
    global_limits=dict(field_l2=.1, field_linf=.1, field_corr_min=.99, flux_error=.05),
    overrides={"sentinel": dict(field_l2=.3)},
    override_notes={"sentinel": "(a): documented residual"},
    investigation={"open": "class (c): residual under investigation"},
    quarantine={"held": "reference cannot qualify"},
)


def _row(name, l2=".05", linf=".05", corr="0.999", flux=".01"):
    row = [name, *("-" for _ in range(17))]
    row[4], row[6], row[7], row[9], row[11] = l2, linf, corr, flux, "1"
    return row


def _batch_scorer():
    spec = importlib.util.spec_from_file_location("integration_batch", ROOT / "scripts/dev/integration_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_status_classes_follow_suite_tables_and_batch_scorer(tmp_path):
    rows = [
        _row("global"),
        _row("sentinel", l2=".2"),
        _row("unconverged", l2=".05 (unconverged)", flux=".01 (unconverged)"),
        _row("open", l2=".5"),
        _row("held"),
        _row("failing", flux=".2"),
        _row("tfsf_vacuum", l2="1.9"),
    ]
    classes = {row[0]: status.classify_row(row, LIMITS) for row in rows}
    assert {name: value[0] for name, value in classes.items()} == {
        "global": status.PASS_GLOBAL,
        "sentinel": status.PASS_SENTINEL,
        "unconverged": status.NOT_SCORED,
        "open": status.KNOWN_OPEN,
        "held": status.NOT_SCORED,
        "failing": status.FAIL,
        "tfsf_vacuum": status.REFERENCE_CONVENTION,
    }
    assert "field_l2 0.3 (global 0.1); (a): documented residual" == classes["sentinel"][2]
    assert "unconverged (not scored): field_l2, flux_error" in classes["unconverged"][2]
    noted = status.classify_row(_row("full_tensor_slab", l2=".2"), LIMITS)
    assert noted[0] == status.FAIL and noted[2].startswith("fails field_l2; analytic 4x4 oracle")

    results = tmp_path / "RESULTS.md"
    results.write_text("\n".join("| " + " | ".join(row) + " |" for row in rows))
    unit = dict(nodes=[f"benchmark:{row[0]}" for row in rows], metric_limits=LIMITS.global_limits,
                scenario_metric_limits=LIMITS.overrides, scenario_quarantine=LIMITS.quarantine)
    scored = {case["nodeid"].removeprefix("benchmark:"): case["status"]
              for case in _batch_scorer().benchmark_cases(results, unit)}
    assert scored == {name: value[1] for name, value in classes.items()}


def test_suite_manifest_tables_load_and_declared_rows_exist():
    limits = status.load_suite_limits()
    assert limits.global_limits == dict(field_l2=.1, field_linf=.1, field_corr_min=.99, flux_error=.05)
    assert "cone_scatter" in limits.overrides
    # full_tensor_slab passes the global limits since its regenerated 270^3 reference (bench-references); no sentinel.
    assert "full_tensor_slab" not in limits.overrides
    assert limits.override_notes["dipole_vacuum"].startswith("(a)")
    assert set(status.DECLARED_STATUS) | set(status.ROW_NOTES) <= set(SCENARIOS)
    assert not set(status.DECLARED_STATUS) & set(limits.investigation)


def test_report_keeps_row_commits_and_foreign_sections(tmp_path, monkeypatch):
    output = tmp_path / "RESULTS.md"
    monkeypatch.setattr(benchmark_report, "RESULTS_MD", output)
    foreign = "## Antenna wave-level validation\n\nantenna table\n"
    output.write_text(f"# Benchmark Results\n\n{foreign}")
    monkeypatch.setattr(benchmark_report, "_source_commit", lambda: "aaaaaaaaaaaa")
    benchmark_report.write_results_markdown([])
    rows, *_ , commits = benchmark_report._parse_existing_results()
    assert rows == {} and commits == {}
    text = output.read_text()
    assert "commit `aaaaaaaaaaaa`" in text and foreign in text

    row = _row("dipole_vacuum")
    text = text.replace("## Plot Files", "## dipole\n\n" + benchmark_report.TABLE_HEADER + "\n"
                        + benchmark_report.TABLE_SEPARATOR + "\n| " + " | ".join(row) + " |\n\n"
                        + benchmark_report.STATUS_HEADER + "\n\n" + benchmark_report.STATUS_TABLE_HEADER
                        + "\n| dipole_vacuum | x | PASS | bbbbbbbbbbbb | - |\n\n## Plot Files")
    output.write_text(text)
    benchmark_report.write_results_markdown([])
    text = output.read_text()
    assert "| dipole_vacuum | (i) pass, global limits | PASS | bbbbbbbbbbbb | - |" in text
    assert text.count(foreign) == 1
