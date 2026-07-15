"""Three-level FDTD grid-refinement study without a cloud reference."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json

import numpy as np
import torch

import witwin.maxwell as mw
from benchmark.metrics import align_plane_fields, best_fit_field_scale, field_l2_error
from benchmark.paths import CACHE_DIR, ROOT
from benchmark.runner import _component_plane_coords, _run_maxwell, _select_monitor_plane_field


FREQUENCY = 1.5e9
DEFAULT_RESOLUTIONS = (0.06, 0.04, 0.02666666666666667)
RESULTS_MD = ROOT / "GRID_CONVERGENCE.md"
CACHE_PATH = CACHE_DIR / "grid_convergence"


@dataclass(frozen=True)
class GridSample:
    resolution: float
    grid_shape: tuple[int, int, int]
    maxwell_time_s: float
    ms_per_step: float | None
    steps_per_second: float | None
    dft_samples: int | None
    peak_gpu_memory_mb: float


def build_scene(resolution: float) -> mw.Scene:
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(float(resolution)),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            name="sphere",
            geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=0.14),
            material=mw.Material(eps_r=4.0),
        )
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization="Ex",
            source_time=mw.GaussianPulse(frequency=FREQUENCY, fwidth=0.4e9),
        )
    )
    scene.add_monitor(
        mw.PlaneMonitor(
            "field",
            axis="y",
            position=0.0,
            fields=("Ex",),
            frequencies=(FREQUENCY,),
        )
    )
    return scene


def estimate_observed_order(
    coarse_difference: float,
    fine_difference: float,
    refinement_ratio: float,
) -> float:
    if coarse_difference <= 0.0 or fine_difference <= 0.0:
        raise ValueError("Pairwise field differences must be positive.")
    if refinement_ratio <= 1.0:
        raise ValueError("refinement_ratio must be greater than one.")
    return float(np.log(coarse_difference / fine_difference) / np.log(refinement_ratio))


def _pairwise_difference(coarse_field, coarse_coords, fine_field, fine_coords) -> float:
    aligned_coarse, aligned_fine, _ = align_plane_fields(
        coarse_field,
        fine_field,
        source_coords=coarse_coords,
        reference_coords=fine_coords,
    )
    shape_aligned, _ = best_fit_field_scale(aligned_coarse, aligned_fine)
    return field_l2_error(shape_aligned, aligned_fine)


def _run_sample(resolution: float):
    scene = build_scene(resolution)
    result, monitors, elapsed, performance = _run_maxwell(
        scene,
        frequencies=(FREQUENCY,),
        run_time_factor=10.0,
        normalize_source=True,
    )
    monitor = monitors["field"]
    field = _select_monitor_plane_field(monitor, "Ex", monitor["fields"]["Ex"])
    coords = _component_plane_coords(monitor, "Ex")
    if coords is None:
        raise RuntimeError("Grid-convergence monitor did not expose plane coordinates.")
    sample = GridSample(
        resolution=float(resolution),
        grid_shape=tuple(int(value) for value in result.prepared_scene.permittivity.shape),
        maxwell_time_s=float(elapsed),
        ms_per_step=performance["ms_per_step"],
        steps_per_second=performance["steps_per_second"],
        dft_samples=performance["dft_samples"],
        peak_gpu_memory_mb=float(performance["peak_gpu_memory_mb"]),
    )
    return sample, np.asarray(field), coords


def run_convergence(resolutions=DEFAULT_RESOLUTIONS):
    values = tuple(float(value) for value in resolutions)
    if len(values) != 3 or not values[0] > values[1] > values[2] > 0.0:
        raise ValueError("Provide exactly three positive resolutions from coarse to fine.")
    coarse_ratio = values[0] / values[1]
    fine_ratio = values[1] / values[2]
    if not np.isclose(coarse_ratio, fine_ratio, rtol=1e-6, atol=0.0):
        raise ValueError("The three resolutions must use one geometric refinement ratio.")

    runs = [_run_sample(resolution) for resolution in values]
    coarse_difference = _pairwise_difference(runs[0][1], runs[0][2], runs[1][1], runs[1][2])
    fine_difference = _pairwise_difference(runs[1][1], runs[1][2], runs[2][1], runs[2][2])
    order = estimate_observed_order(coarse_difference, fine_difference, coarse_ratio)
    return [run[0] for run in runs], coarse_difference, fine_difference, order


def _format_optional(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def render_markdown(
    samples: list[GridSample],
    coarse_difference: float,
    fine_difference: float,
    observed_order: float,
    *,
    updated_at: str,
) -> str:
    lines = [
        "# FDTD Grid Convergence",
        "",
        f"- **Updated:** {updated_at}",
        f"- **Scene:** dielectric sphere scattering at {FREQUENCY:.6e} Hz",
        "- **Method:** three geometrically refined grids; complex Ex planes are coordinate-aligned and best-fit complex-scaled before shape differencing.",
        "",
        "| Resolution (m) | Grid shape | Maxwell time (s) | ms/step | steps/s | DFT samples | Peak GPU (MiB) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample in samples:
        lines.append(
            f"| {sample.resolution:.8e} | {'x'.join(map(str, sample.grid_shape))} | "
            f"{sample.maxwell_time_s:.3f} | {_format_optional(sample.ms_per_step, '.4f')} | "
            f"{_format_optional(sample.steps_per_second, '.2f')} | "
            f"{sample.dft_samples if sample.dft_samples is not None else '-'} | "
            f"{sample.peak_gpu_memory_mb:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Best-fit-scale shape L2 |",
            "| --- | ---: |",
            f"| coarse vs medium | {coarse_difference:.6e} |",
            f"| medium vs fine | {fine_difference:.6e} |",
            "",
            f"**Observed order:** {observed_order:.4f}",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(samples, coarse_difference, fine_difference, observed_order):
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    RESULTS_MD.write_text(
        render_markdown(
            samples,
            coarse_difference,
            fine_difference,
            observed_order,
            updated_at=updated_at,
        ),
        encoding="utf-8",
    )
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    raw_path = CACHE_PATH / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    raw_path.write_text(
        json.dumps(
            {
                "samples": [asdict(sample) for sample in samples],
                "coarse_difference": coarse_difference,
                "fine_difference": fine_difference,
                "observed_order": observed_order,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the three-level FDTD grid-convergence study.")
    parser.add_argument(
        "--resolutions",
        nargs=3,
        type=float,
        default=DEFAULT_RESOLUTIONS,
        metavar=("COARSE", "MEDIUM", "FINE"),
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("Grid-convergence benchmark requires CUDA.")
    samples, coarse_difference, fine_difference, order = run_convergence(args.resolutions)
    raw_path = write_results(samples, coarse_difference, fine_difference, order)
    print(f"Wrote {RESULTS_MD} and {raw_path}")


if __name__ == "__main__":
    main()
