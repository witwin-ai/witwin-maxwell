from __future__ import annotations

from datetime import datetime
import fcntl
import subprocess

from benchmark.models import ScenarioMetrics
from benchmark.paths import ROOT, RESULTS_MD, scenario_path_parts
from benchmark.status import STATUS_CLASSES, classify_row, load_suite_limits


TABLE_HEADER = (
    "| Scenario | Description | Field monitor | Component | "
    "Field L2 [smaller, <1e-1] | Shape L2 [smaller] | Field Linf [smaller, <1e-1] | "
    "Field Corr [larger, >0.99] | Field 2x-runtime change | Flux err [smaller, <5e-2] | "
    "Flux 2x-runtime change | "
    "Maxwell time (s) [smaller] | ms/step | steps/s | DFT samples | "
    "Peak GPU (MiB) | Reference cache | Updated |"
)
STATUS_HEADER = "## Row status"
STATUS_TABLE_HEADER = "| Scenario | Status | Scorer outcome | Commit | Basis |"
# Sections other writers own (RF, antenna and SAR harnesses); kept verbatim on rewrite.
FOREIGN_SECTION_HEADERS = (
    "## RF wave-level validation",
    "## Antenna wave-level validation",
    "## RF / antenna external reference generation",
    "## SAR exposure validation",
)
TABLE_SEPARATOR = (
    "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
)


def _display_path(path):
    try:
        return path.relative_to(RESULTS_MD.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_existing_results():
    if not RESULTS_MD.exists():
        return {}, {}, {}, {}, {}

    rows = {}
    plots = {}
    scalar_rows = {}
    frequency_rows = {}
    row_commits = {}
    lines = RESULTS_MD.read_text(encoding="utf-8").splitlines()
    in_table = False
    in_plots = False
    in_scalars = False
    in_frequencies = False
    in_status = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_table = in_plots = in_scalars = in_frequencies = False
            in_status = stripped == STATUS_HEADER
        if stripped.startswith("| Scenario | Description | Field monitor | Component |"):
            in_table = True
            in_plots = False
            in_scalars = False
            in_frequencies = False
            continue
        if stripped == "## Plot Files":
            in_table = False
            in_plots = True
            in_scalars = False
            in_frequencies = False
            continue
        if stripped == "## Scalar observables":
            in_table = False
            in_plots = False
            in_scalars = True
            in_frequencies = False
            continue
        if stripped == "## Per-frequency field metrics":
            in_table = False
            in_plots = False
            in_scalars = False
            in_frequencies = True
            continue
        if in_status and stripped.startswith("|") and not stripped.startswith(("| ---", "| Scenario")):
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) >= 5:
                row_commits[parts[0]] = parts[3]
            continue
        if in_table:
            if not stripped.startswith("|") or stripped.startswith("| ---"):
                continue
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) != 18:
                continue
            rows[parts[0]] = parts
            continue
        if in_plots and stripped.startswith("- `"):
            scenario_name = stripped.split("`", 2)[1]
            plots[scenario_name] = stripped
            continue
        if in_scalars and stripped.startswith("|") and not stripped.startswith("| ---"):
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) == 8 and parts[0] != "Scenario":
                scalar_rows[(parts[0], parts[1], parts[2])] = parts
            continue
        if in_frequencies and stripped.startswith("|") and not stripped.startswith("| ---"):
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) == 6 and parts[0] != "Scenario":
                parts.append("-")
            if len(parts) == 7 and parts[0] != "Scenario":
                frequency_rows[(parts[0], parts[1])] = parts
    return rows, plots, scalar_rows, frequency_rows, row_commits


def _source_commit() -> str:
    """The witwin-maxwell commit that produced a row; ``+dirty`` marks tracked edits."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        # Generated outputs (this file and the plots) do not make the producing code dirty.
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "witwin", "benchmark",
                                ":(exclude)benchmark/RESULTS.md", ":(exclude)benchmark/plots"],
                               cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def _foreign_sections(previous: str) -> list[str]:
    sections = []
    for header in FOREIGN_SECTION_HEADERS:
        start = previous.find("\n" + header + "\n")
        if start == -1:
            continue
        body = previous[start + 1:]
        ends = [index for index in (body.find("\n## ", 1), body.find("\n<!-- batch-", 1)) if index != -1]
        sections.append(body[:min(ends)] if ends else body)
    return [section.rstrip() + "\n" for section in sections]


def _status_lines(merged_rows: dict, row_commits: dict, scalar_rows: dict) -> list[str]:
    limits = load_suite_limits()
    classified = {
        name: classify_row(row, limits, [scalar for key, scalar in scalar_rows.items() if key[0] == name])
        for name, row in merged_rows.items()
    }
    counts = {status: 0 for status in STATUS_CLASSES}
    for status, _, _ in classified.values():
        counts[status] = counts.get(status, 0) + 1
    lines = [
        STATUS_HEADER,
        "",
        "Each row is classified against the benchmark suite of `scripts/dev/integration_suites.toml` "
        "(global limits, scenario sentinel limits, investigation and quarantine tables); the classes "
        "never change a limit. `Scorer outcome` is the integration-batch verdict for the aggregate row "
        "(PASS, FAIL, or SKIP when a metric is unconverged or quarantined). (i) passes every global "
        "limit; (ii) passes only through a scenario-specific sentinel limit, named with its reason; "
        "(iii) has an unconverged or quarantined metric, which is not scored; (iv) is a known-open "
        "residual under investigation; (v) differs from the reference by a documented reference "
        "convention. `fail, no waiver` rows fail a converged metric with none of these reasons. "
        "`Commit` is the witwin-maxwell commit that produced the row.",
        "",
        "| Status | Rows |",
        "| --- | ---: |",
        *(f"| {status} | {count} |" for status, count in counts.items()),
        "",
        STATUS_TABLE_HEADER,
        "| --- | --- | --- | --- | --- |",
    ]
    for name in sorted(classified):
        status, outcome, basis = classified[name]
        basis = basis.replace("|", "\\|") or "-"
        lines.append(f"| {name} | {status} | {outcome} | {row_commits.get(name, 'unknown')} | {basis} |")
    lines.append("")
    return lines


def _format_complex(value: object) -> str:
    scalar = complex(value)
    if abs(scalar.imag) <= 1.0e-14:
        return f"{scalar.real:.6e}"
    return f"{scalar.real:.6e}{scalar.imag:+.6e}j"


def _scenario_group(scenario_name: str) -> str:
    parts = scenario_path_parts(scenario_name)
    if len(parts) > 1:
        return parts[0]
    return "misc"


def _grouped_items(items: dict[str, list[str]] | dict[str, str]):
    grouped: dict[str, list[tuple[str, list[str] | str]]] = {}
    for scenario_name, item in items.items():
        grouped.setdefault(_scenario_group(scenario_name), []).append((scenario_name, item))
    for group_items in grouped.values():
        group_items.sort(key=lambda pair: scenario_path_parts(pair[0]))
    return dict(sorted(grouped.items()))


def _missing_scenario_lines(merged_rows: dict) -> list[str]:
    """Markdown for registered scenarios that carry no measured row in this file.

    Imported lazily because ``benchmark.runner`` imports this module.
    """

    from benchmark.runner import SCENARIOS

    missing = [name for name in SCENARIOS if name not in merged_rows]
    if not missing:
        return []
    lines = [
        "## Registered scenarios with no measured row",
        "",
        "These scenarios are registered in `benchmark/runner.SCENARIOS` but have no row "
        "in the tables above: the runs that produced this file never scored them. They "
        "are listed so that a family which cannot execute on the generating host is "
        "visible as unmeasured rather than silently absent. A row here is **not** a "
        "pass, a fail, or a tolerance waiver -- it is the absence of evidence.",
        "",
        "| Scenario | Solver | Description |",
        "| --- | --- | --- |",
    ]
    for name in missing:
        scenario = SCENARIOS[name]
        description = str(scenario.description).replace("|", "\\|")
        lines.append(f"| {name} | {scenario.solver} | {description} |")
    lines.append("")
    return lines


def write_results_markdown(results: list[ScenarioMetrics]) -> None:
    """Merge ``results`` into RESULTS.md; concurrent benchmark processes serialize on a lock file."""
    lock_path = RESULTS_MD.with_name(RESULTS_MD.name + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _write_results_markdown(results)


def _write_results_markdown(results: list[ScenarioMetrics]) -> None:
    import re

    previous = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    batch_sections = [match.group(0) for match in re.finditer(
        r"<!-- (batch-[\w-]+):begin -->.*?<!-- \1:end -->", previous, flags=re.DOTALL
    )]
    existing_rows, existing_plots, existing_scalar_rows, existing_frequency_rows, row_commits = (
        _parse_existing_results()
    )
    foreign_sections = _foreign_sections(previous)
    commit = _source_commit()
    merged_rows = dict(existing_rows)
    merged_plots = dict(existing_plots)
    merged_scalar_rows = dict(existing_scalar_rows)
    merged_frequency_rows = dict(existing_frequency_rows)

    for result in results:
        row_commits[result.name] = commit
        merged_scalar_rows = {
            key: row for key, row in merged_scalar_rows.items() if key[0] != result.name
        }
        merged_frequency_rows = {
            key: row for key, row in merged_frequency_rows.items() if key[0] != result.name
        }
        flux_text = "-" if result.flux_error is None else f"{result.flux_error:.4e}"
        if result.flux_converged is False:
            flux_text += " (unconverged)"
        flux_change_text = (
            "-" if result.flux_runtime_change is None else f"{result.flux_runtime_change:.4e}"
        )
        field_status = (
            " (not scored)" if not result.fields_scored
            else " (unconverged)" if result.field_converged is False else ""
        )
        field_change_text = (
            "-" if result.field_runtime_change is None else f"{result.field_runtime_change:.4e}"
        )
        ms_per_step = "-" if result.maxwell_ms_per_step is None else f"{result.maxwell_ms_per_step:.4f}"
        steps_per_second = (
            "-" if result.maxwell_steps_per_second is None else f"{result.maxwell_steps_per_second:.2f}"
        )
        dft_samples = "-" if result.maxwell_dft_samples is None else str(result.maxwell_dft_samples)
        peak_memory = (
            "-"
            if result.maxwell_peak_gpu_memory_mb is None
            else f"{result.maxwell_peak_gpu_memory_mb:.2f}"
        )
        merged_rows[result.name] = [
            result.name,
            result.description,
            result.compared_monitor,
            result.compared_component,
            f"{result.field_l2:.4e}{field_status}",
            f"{result.field_shape_l2:.4e}{field_status}",
            f"{result.field_linf:.4e}{field_status}",
            f"{result.field_corr:.4f}{field_status}",
            field_change_text,
            flux_text,
            flux_change_text,
            f"{result.maxwell_time_s:.2f}",
            ms_per_step,
            steps_per_second,
            dft_samples,
            peak_memory,
            "local" if result.tidy3d_cache_hit is None else ("hit" if result.tidy3d_cache_hit else "miss"),
            result.updated_at,
        ]
        material_rel = _display_path(result.material_source_plot)
        field_rel = _display_path(result.field_plot)
        plot_links = f"[material+source]({material_rel}), [field]({field_rel})"
        if result.diagnostic_plot is not None:
            diagnostic_rel = _display_path(result.diagnostic_plot)
            plot_links += f", [complex diagnostic]({diagnostic_rel})"
        if result.scalar_plot is not None:
            scalar_rel = _display_path(result.scalar_plot)
            plot_links += f", [scalar comparison]({scalar_rel})"
        phase_fits = [
            f"{item['frequency']:.8e} Hz: {item['field_phase_fit_rad']:+.6e}"
            for item in result.per_frequency if "field_phase_fit_rad" in item
        ]
        if phase_fits:
            plot_links += "; fitted global phase (rad, diagnostic only): " + ", ".join(phase_fits)
        merged_plots[result.name] = f"- `{result.name}`: {plot_links}"
        for item in result.scalar_metrics:
            frequency_text = f"{float(item['frequency']):.8e}"
            phase_text = (
                "-" if item["phase_error"] is None else f"{float(item['phase_error']):.4e}"
            )
            merged_scalar_rows[(result.name, frequency_text, str(item["observable"]))] = [
                result.name,
                frequency_text,
                str(item["observable"]),
                _format_complex(item["maxwell"]),
                _format_complex(item["tidy3d"]),
                f"{float(item['complex_error']):.4e}",
                f"{float(item['magnitude_error']):.4e}",
                phase_text,
            ]
        for item in result.per_frequency:
            frequency_text = f"{float(item['frequency']):.8e}"
            merged_frequency_rows[(result.name, frequency_text)] = [
                result.name,
                frequency_text,
                f"{float(item['field_l2']):.4e}{field_status}",
                f"{float(item['field_shape_l2']):.4e}{field_status}",
                f"{float(item['field_linf']):.4e}{field_status}",
                f"{float(item['field_corr']):.4f}{field_status}",
                (
                    "-"
                    if "field_energy_ratio" not in item
                    else f"{float(item['field_energy_ratio']):.4f}"
                ),
            ]

    lines = [
        "# Benchmark Results",
        "",
        "This file is auto-generated by `python -m benchmark`. Last written at witwin-maxwell "
        f"commit `{commit}`; the `Row status` section records the commit of every row.",
        "",
        "## Metric Guide",
        "",
        "- `Field L2 [smaller, <1e-1]`: absolute complex relative L2 after unit and source-spectrum conversion, without fitted phase or amplitude alignment.",
        "- `Shape L2 [smaller]`: diagnostic only, relative L2 after removing the best-fit global complex scale. The fitted global phase in Plot Files rotates the actual field toward the reference; neither diagnostic is a pass/fail gate.",
        "- `Field Linf [smaller, <1e-1]`: smaller is better; below `1e-1` usually means no large local outlier.",
        "- `Field Corr [larger, >0.99]`: larger is better; above `0.99` means field shape is strongly aligned.",
        "- `Field 2x-runtime change`: `||F_2x - F_1x||_2 / ||F_1x||_2` for the complex display monitor/component across all frequencies, without phase or amplitude alignment. Changes above `1e-2` (1%) mark field metrics `unconverged`, excluded from field pass/fail counts and summaries (including per-frequency metrics).",
        "- `Energy ratio`: weighted Maxwell/reference electric-field norm ratio for vector comparisons; unity is ideal.",
        "- `Flux err [smaller, <5e-2]`: smaller is better; below `5e-2` is a strong flux match. Flux metrics marked `unconverged` are excluded from flux pass/fail independently of field convergence.",
        "- `Flux 2x-runtime change`: maximum `abs(P_2x - P_1x) / abs(P_1x)` over flux monitors and frequencies; changes above `1e-2` (1%) are unconverged. Rows whose monitored flux is null by construction (`tfsf_vacuum`, `pml_slab_through`, `pec_box`) divide by the incident power instead, the scale of their flux error.",
        "- `(not scored)`: the display field of an undamped closed cavity never settles, so its field metrics are diagnostics only. Those rows are scored on the resonance frequency, fitted with the finite-record DFT kernel: `resonance_frequency` against the reference, `resonance_frequency_discrete_analytic` against the discrete Yee eigenfrequency (the reference column holds the analytic value) and `resonance_frequency_runtime` for the 2T against the T estimate (the reference column holds the T estimate).",
        "- Both runtime checks share one doubled-duration FDTD rerun. `-` means no applicable comparison or no convergence measurement; `WITWIN_BENCHMARK_SKIP_CONVERGENCE=1` skips both checks.",
        "- `Complex err`: symmetric complex relative error; it remains finite for near-zero S-parameters.",
        "- `Phase err`: wrapped phase difference in radians; omitted when either magnitude is numerically zero.",
        "- `Maxwell time (s) [smaller]`: original-run runtime only, excluding the convergence run, with no fixed pass/fail target.",
        "- `ms/step`, `steps/s`, and `DFT samples`: FDTD stepping diagnostics; omitted for backends that do not expose them.",
        "- `Peak GPU (MiB)`: driver-level high-water usage above the pre-run baseline, including Torch and CuPy allocations.",
        "- `Reference cache`: `hit` means a Tidy3D cloud reference was reused; `miss` means it was regenerated; `local` means another Maxwell backend supplied the reference.",
        "",
    ]

    for group_name, group_rows in _grouped_items(merged_rows).items():
        lines.extend(
            [
                f"## {group_name}",
                "",
                TABLE_HEADER,
                TABLE_SEPARATOR,
            ]
        )
        for _, row in group_rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.extend(_status_lines(merged_rows, row_commits, merged_scalar_rows))

    lines.extend(["## Plot Files", ""])
    for group_name, group_plots in _grouped_items(merged_plots).items():
        lines.extend([f"### {group_name}", ""])
        for _, plot_line in group_plots:
            lines.append(plot_line)
        lines.append("")

    if merged_scalar_rows:
        lines.extend([
            "## Scalar observables", "",
            "| Scenario | Frequency (Hz) | Observable | Maxwell | Tidy3D | Complex err | Magnitude err | Phase err (rad) |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key in sorted(merged_scalar_rows):
            lines.append("| " + " | ".join(merged_scalar_rows[key]) + " |")
        lines.append("")

    if merged_frequency_rows:
        lines.extend([
            "## Per-frequency field metrics", "",
            "| Scenario | Frequency (Hz) | Field L2 | Shape L2 | Field Linf | Field Corr | Energy ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key in sorted(merged_frequency_rows):
            lines.append("| " + " | ".join(merged_frequency_rows[key]) + " |")
        lines.append("")

    # A registered scenario that produced no row must stay visible. Without this
    # section a family that cannot run on the current host (a missing optional
    # dependency, a deliberate deferral, an aborted run) simply disappears from the
    # table, and the file silently reads as if that family had been validated.
    lines.extend(_missing_scenario_lines(merged_rows))

    # The per-medium validation-coverage table is regenerated from the coverage
    # registry so it stays in sync with media.py even across benchmark reruns.
    from benchmark.media_coverage import validation_coverage_markdown_lines

    lines.extend(validation_coverage_markdown_lines())
    lines.extend(
        [
            f"_Last regenerated: {datetime.now().astimezone().isoformat(timespec='seconds')}_",
            "",
        ]
    )
    if batch_sections:
        lines.extend(["", *batch_sections, ""])
    if foreign_sections:
        lines.extend(["", *foreign_sections])
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def record_batch_baseline(payload, artifact):
    """Update one baseline section while preserving unrelated validation tables."""
    name = payload["workload"]["name"]
    count = payload["variants"]
    control = "identical" if payload["identical"] else "varied"
    native = payload.get("include_native", False)
    graph = payload.get("native_graph", "off")
    key = f"batch-{name}-{count}-{control}" + ("-native" if native else "")
    if native and graph != "off":
        key += f"-graph-{graph}"
    begin, end = f"<!-- {key}:begin -->", f"<!-- {key}:end -->"
    title = "Execution comparison including native Batch" if native else "Independent execution baseline"
    lines = [begin, f"## {title}: {name}, N={count}, {control}", "",
             f"Device: {payload['device']}; steps: {payload['workload']['time_steps']}; "
             f"warmups/repeats: {payload['warmups']}/{payload['repeats']}; native graph policy: {graph}.", "",
             "| Mode | Warm median (s) | Range (s) |",
             "| --- | ---: | ---: |"]
    for mode, values in payload["summary"].items():
        lines.append(f"| {mode} | {values['median_wall_s']:.6f} | "
                     f"{values['min_wall_s']:.6f}–{values['max_wall_s']:.6f} |")
    if payload.get("isolated_memory"):
        lines.extend(["", "| Mode | Sampled isolated driver increment (MiB) |",
                      "| --- | ---: |"])
        for record in payload["isolated_memory"]:
            lines.append(f"| {record['mode']} | {record['driver_peak_increment_bytes'] / 1024**2:.2f} |")
    qualification = (
        "The native tier reuses a PreparedBatch under the recorded graph policy. "
        "This fixed workload does not establish business or complete scheduler acceptance."
        if native else "These independent-solver baselines do not establish native Batch or business acceptance.")
    lines.extend(["", f"[Raw measurements and memory/lifetime qualifications]({_display_path(artifact)})", "",
                  qualification, end])
    _write_batch_section(lines, begin, end)


def record_regional_material_audit(payload, artifact):
    """Retain isolated regional compiler measurements in the benchmark report."""
    key = f"batch-regional-materials-{payload['workload']}"
    begin, end = f"<!-- {key}:begin -->", f"<!-- {key}:end -->"
    lines = [begin, f"## Regional material compilation: {payload['workload']}", "",
             f"Warmups/repeats: {payload['warmups']}/{payload['repeats']}.", "",
             "| Change | Full median (s) | Local median (s) | Speedup | Maximum absolute error |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in payload["rows"]:
        lines.append(f"| {row['mode']} | {row['median_seconds']['full']:.6f} | "
                     f"{row['median_seconds']['local']:.6f} | {row['speedup']:.3f} | {row['max_absolute_error']:.3g} |")
    lines.extend(["", payload["scope"], "", f"[Raw measurements]({_display_path(artifact)})", end])
    _write_batch_section(lines, begin, end)


def record_batch_tuning_audit(payload, artifact):
    key = f"batch-tuning-{payload['workload']['name']}-n{payload['variants']}-d{len(payload['devices'])}"
    begin, end = f"<!-- {key}:begin -->", f"<!-- {key}:end -->"
    plan = payload["plan"]
    lines = [begin, f"## Batch tuning: {payload['workload']['name']}, N={payload['variants']}", "",
             "Complete streamed forward trials; one warmup and three timed repeats per candidate.", "",
             "| B | Q | Graph steps | Median s | Peak backing MiB (sum of devices) |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for row in plan["candidate_measurements"]:
        policy = row["policy"]
        lines.append(f"| {policy['batch_size']} | {policy['slots_per_device']} | {policy['graph_steps']} | "
                     f"{row['median_s']:.6f} | {sum(row['peak_bytes'].values()) / 1024**2:.2f} |")
    lines.extend(["", f"Selected policy: `{plan['selected_policy']}`. "
        f"Tuning cost: {plan['tuning']['elapsed_s']:.3f} s. "
        f"Selected/best measured candidate throughput: {plan['tuning']['selected_relative_throughput']:.4f}.",
        f"Independent hot median: {payload['holdout_median_s']:.6f} s. "
        f"Maximum probe absolute error: {payload['max_probe_absolute_error']:.3g}.",
        "Backward, all-result retention and exclusive multi-device scaling are outside this timing scope.",
        f"Raw artifact: `{artifact}`", end])
    _write_batch_section(lines, begin, end)


def record_batch_scaling_audit(payload, artifact):
    key = f"batch-scaling-{payload['workload']['name']}-n{payload['variants']}-d{len(payload['devices'])}"
    begin, end = f"<!-- {key}:begin -->", f"<!-- {key}:end -->"
    lines = [begin, f"## Batch device scaling: {payload['workload']['name']}", "", payload["scope"], "",
        "| Devices | Prepare s | Hot median s |",
        "| --- | ---: | ---: |"]
    for row in payload["rows"]:
        lines.append(f"| {', '.join(row['devices'])} | {row['prepare_s']:.6f} | {row['median_s']:.6f} |")
    lines.extend(["", f"Best single-device / parallel speedup: {payload['best_single_to_parallel_speedup']:.3f}. "
        f"Matching dual-device 1.7x gate: {payload['matching_dual_device_gate_passed']}.",
        f"Maximum probe absolute error: {payload['max_probe_absolute_error']:.3g}. "
        f"Exclusive compute observed: {payload['compute_occupancy']['exclusive_compute_observed']}.",
        payload["compute_occupancy"]["limitation"],
        f"[Raw measurements]({_display_path(artifact)})", end])
    _write_batch_section(lines, begin, end)


def _write_batch_section(lines, begin, end):
    section = "\n".join(lines) + "\n"
    current = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    if begin in current and end in current:
        before, remainder = current.split(begin, 1)
        _, after = remainder.split(end, 1)
        updated = before + section + after.lstrip("\n")
    else:
        updated = current.rstrip() + "\n\n" + section
    RESULTS_MD.write_text(updated, encoding="utf-8")
