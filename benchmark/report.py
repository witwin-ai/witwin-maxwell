from __future__ import annotations

from datetime import datetime

from benchmark.models import ScenarioMetrics
from benchmark.paths import RESULTS_MD, scenario_path_parts


TABLE_HEADER = (
    "| Scenario | Description | Field monitor | Component | "
    "Field L2 [smaller, <1e-1] | Shape L2 [smaller] | Field Linf [smaller, <1e-1] | "
    "Field Corr [larger, >0.99] | Flux err [smaller, <5e-2] | "
    "Maxwell time (s) [smaller] | ms/step | steps/s | DFT samples | "
    "Peak GPU (MiB) | Reference cache | Updated |"
)
TABLE_SEPARATOR = (
    "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | ---: | --- | --- |"
)


def _display_path(path):
    try:
        return path.relative_to(RESULTS_MD.parent).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_existing_results():
    if not RESULTS_MD.exists():
        return {}, {}, {}, {}

    rows = {}
    plots = {}
    scalar_rows = {}
    frequency_rows = {}
    lines = RESULTS_MD.read_text(encoding="utf-8").splitlines()
    in_table = False
    in_plots = False
    in_scalars = False
    in_frequencies = False
    for line in lines:
        stripped = line.strip()
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
        if in_table:
            if not stripped.startswith("|") or stripped.startswith("| ---"):
                continue
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) == 11:
                parts = [
                    *parts[:5],
                    "-",
                    *parts[5:9],
                    "-",
                    "-",
                    "-",
                    "-",
                    *parts[9:],
                ]
            if len(parts) != 16:
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
                frequency_rows[(parts[0], parts[1])] = parts
    return rows, plots, scalar_rows, frequency_rows


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


def write_results_markdown(results: list[ScenarioMetrics]) -> None:
    existing_rows, existing_plots, existing_scalar_rows, existing_frequency_rows = (
        _parse_existing_results()
    )
    merged_rows = dict(existing_rows)
    merged_plots = dict(existing_plots)
    merged_scalar_rows = dict(existing_scalar_rows)
    merged_frequency_rows = dict(existing_frequency_rows)

    for result in results:
        merged_scalar_rows = {
            key: row for key, row in merged_scalar_rows.items() if key[0] != result.name
        }
        merged_frequency_rows = {
            key: row for key, row in merged_frequency_rows.items() if key[0] != result.name
        }
        flux_text = "-" if result.flux_error is None else f"{result.flux_error:.4e}"
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
            f"{result.field_l2:.4e}",
            f"{result.field_shape_l2:.4e}",
            f"{result.field_linf:.4e}",
            f"{result.field_corr:.4f}",
            flux_text,
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
                f"{float(item['field_l2']):.4e}",
                f"{float(item['field_shape_l2']):.4e}",
                f"{float(item['field_linf']):.4e}",
                f"{float(item['field_corr']):.4f}",
            ]

    lines = [
        "# Benchmark Results",
        "",
        "This file is auto-generated by `python -m benchmark`.",
        "",
        "## Metric Guide",
        "",
        "- `Field L2 [smaller, <1e-1]`: smaller is better; below `1e-1` is a good alignment target.",
        "- `Shape L2 [smaller]`: relative L2 after removing the best-fit global complex scale; it isolates spatial-shape error from amplitude and phase conventions.",
        "- `Field Linf [smaller, <1e-1]`: smaller is better; below `1e-1` usually means no large local outlier.",
        "- `Field Corr [larger, >0.99]`: larger is better; above `0.99` means field shape is strongly aligned.",
        "- `Flux err [smaller, <5e-2]`: smaller is better; below `5e-2` is a strong flux match.",
        "- `Complex err`: symmetric complex relative error; it remains finite for near-zero S-parameters.",
        "- `Phase err`: wrapped phase difference in radians; omitted when either magnitude is numerically zero.",
        "- `Maxwell time (s) [smaller]`: runtime only, with no fixed pass/fail target.",
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
            "| Scenario | Frequency (Hz) | Field L2 | Shape L2 | Field Linf | Field Corr |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key in sorted(merged_frequency_rows):
            lines.append("| " + " | ".join(merged_frequency_rows[key]) + " |")
        lines.append("")

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
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
