"""Analytic scoring of benchmark rows on a refined Maxwell grid.

The external reference of a benchmark row is computed on Maxwell's own grid and time step
(the exporter forwards the nodes and the Courant number), so the cross-code row compares two
Yee discretisations of one grid, not Maxwell against a converged answer. On the shared
24.6 mm grid the eps 4 spheres have 3.0 cells per wavelength inside the sphere at 2 GHz,
beyond the Yee band edge (sin(k dx/2) = 1.03), and the custom-grid slab has 3.6. Those rows
keep the cross-code comparison as a sentinel-limited regression row; their accuracy is scored
here, on a refined Maxwell grid, against

* the Mie series (``benchmark.analytic.mie``) for the spheres: the complex display-plane field,
  total field for the soft plane-wave rows and the TF/SF composite (total inside the TFSF box,
  scattered outside) for the TFSF row, as a shape L2 (one global complex scale removed, which
  absorbs the source normalisation and its phase gauge);
* the Fabry-Perot (Airy) slab, through the isotropic case of the Berreman oracle
  (``benchmark.analytic.slab_reference``): transmittance and reflectance from the scene's own
  flux monitors against an empty twin, and the calibrated complex line field through the slab.

``python -m benchmark analytic <row> ... [--refine R] [--out DIR]``. The refined scene keeps the
domain, geometry, sources, monitors, absorber layer count and window of the benchmark row and
divides its cell count by ``R`` per axis, so the original Yee nodes nest in the refined grid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

import witwin.maxwell as mw
from benchmark.analytic import mie

C0 = 299_792_458.0
# Base cell count per axis of the shared benchmark grid: GridSpec.uniform(0.025) over 1.28 m
# rounds up to 52 cells (24.615 mm); custom_grid_slab uses linspace(-0.64, 0.64, 54), 53 cells.
BASE_CELLS = {"uniform": 52, "custom": 53}


@dataclass(frozen=True)
class SphereRow:
    radius: float
    eps: float
    refinement: int
    tfsf_half: float | None = None  # TFSF box half-width: score the TF/SF composite


@dataclass(frozen=True)
class SlabRow:
    eps: float
    thickness: float
    centre: float
    refinement: int


ROWS = {
    "dielectric_sphere": SphereRow(radius=0.12, eps=4.0, refinement=5),
    "planewave_dielectric_sphere": SphereRow(radius=0.12, eps=4.0, refinement=5),
    "tfsf_dielectric_sphere": SphereRow(radius=0.16, eps=4.0, refinement=5, tfsf_half=0.28),
    "custom_grid_slab": SlabRow(eps=3.0, thickness=0.1, centre=0.0, refinement=4),
}

# Measured value m of each scored metric at the row's refinement on the CUDA path (A6000,
# witwin-core 7ad8bd3, 8 GiB cap, bench-respec 2026-11-12; ladders in the evidence README).
# The limits below were user-approved 2026-11-13 in the coordinator session: "Approve all (Recommended)".
MEASURED = {
    # Mie total field, Ex on y = 0, |E| >= 10% of max, z > -0.5 (downstream of the soft source).
    # dx/r: r = 1 0.731 (beyond the band edge), 4 0.0614, 5 0.0369 (15.2 cells per wavelength in
    # the sphere). The two rows are the same scene and give the same field.
    "dielectric_sphere": {"mie_shape_l2": 0.03688},
    "planewave_dielectric_sphere": {"mie_shape_l2": 0.03688},
    # Mie TF/SF composite, Ex on y = 0, |E| >= 10% of max, samples within 1.5 cells of a TFSF face
    # excluded. r = 1 0.950, 3 0.262, 5 0.110 (observed order 1.7); 96% of the residual is inside
    # the TF box, around the larger (r = 0.16 m) sphere. r = 6 exceeds the 8 GiB cap.
    "tfsf_dielectric_sphere": {"mie_shape_l2": 0.1101},
    # Airy slab (eps 3, 0.1 m) at 2 GHz: T = 0.81367, R = 0.18633. r = 2 / 3 / 4: T error 0.0020 /
    # 0.0089 / 0.0065, R error 0.0051 / 0.0121 / 0.0088 (the Fabry-Perot phase converges
    # oscillating), calibrated line field 0.199 / 0.087 / 0.0455 (second order).
    "custom_grid_slab": {"T_abs_err": 0.006542, "R_abs_err": 0.008814, "line_field_rel_l2": 0.04552},
}


def regression_budget(measured: float) -> float:
    """ceil4(max(1.25 m, m + 0.02)), the rule of the 2026-11-01 limits assessment.

    Maxwell is deterministic, so the budget only covers device, path and toolchain moves; a
    real accuracy change of the refined row shows up as a failure.
    """
    return math.ceil(max(1.25 * measured, measured + 0.02) * 1e4) / 1e4


LIMITS = {name: {key: regression_budget(value) for key, value in metrics.items()}
          for name, metrics in MEASURED.items()}


def refined_scene(name: str, refinement: int) -> mw.Scene:
    """The benchmark row with its cell count multiplied by ``refinement`` on every axis."""
    from benchmark.scenes import build_scene

    scene = build_scene(name)
    r = int(refinement)
    if r == 1:
        return scene
    if scene.grid.is_custom:
        cells = BASE_CELLS["custom"] * r
        nodes = [np.linspace(lo, hi, cells + 1) for lo, hi in scene.domain.bounds]
        grid = mw.GridSpec.custom(*nodes)
    else:
        spans = {hi - lo for lo, hi in scene.domain.bounds}
        assert len(spans) == 1, "refined_scene expects a cubic domain"
        grid = mw.GridSpec.uniform(spans.pop() / (BASE_CELLS["uniform"] * r))
    return scene.updated(grid=grid)


def tfsf_composite_mie(row: SphereRow, k: float, points: np.ndarray) -> np.ndarray:
    """Mie E: total inside the TFSF box (|x|, |y|, |z| <= half), scattered outside."""
    total, _ = mie.near_fields(row.radius, k, row.eps, points, field_kind="total")
    scattered, _ = mie.near_fields(row.radius, k, row.eps, points, field_kind="scattered")
    inside = np.all(np.abs(points) <= row.tfsf_half, axis=1)
    return np.where(inside[:, None], total, scattered)


def shape_l2(actual: np.ndarray, reference: np.ndarray) -> tuple[float, complex]:
    """Relative L2 after the best-fit global complex scale s (s * actual fitted onto reference)."""
    actual = np.asarray(actual).ravel()
    reference = np.asarray(reference).ravel()
    scale = np.vdot(actual, reference) / np.vdot(actual, actual)
    return float(np.linalg.norm(scale * actual - reference) / np.linalg.norm(reference)), complex(scale)


def sphere_mask(row: SphereRow, x: np.ndarray, z: np.ndarray, analytic: np.ndarray, *, half_span: float,
                dx: float) -> np.ndarray:
    """Scored samples on the y = 0 plane (x, z on the component's own Yee positions)."""
    X, Z = np.meshgrid(x, z, indexing="ij")
    mask = (np.abs(X) <= half_span + 1e-9) & (np.abs(Z) <= half_span + 1e-9)
    mask &= np.abs(analytic) >= 0.1 * np.abs(analytic[mask]).max()
    if row.tfsf_half is None:
        # Soft plane wave: no incident field upstream of the injection plane near z = -0.6.
        mask &= Z > -0.5
    else:
        # The TF/SF split of samples next to a box face depends on how the face snaps to the
        # grid; leave them out (1.5 cells either side of every face within the box extent).
        band = 1.5 * dx
        near_box = (np.abs(X) <= row.tfsf_half + band) & (np.abs(Z) <= row.tfsf_half + band)
        on_face = (np.abs(np.abs(X) - row.tfsf_half) < band) | (np.abs(np.abs(Z) - row.tfsf_half) < band)
        mask &= ~(near_box & on_face)
    return mask


def score_sphere(name: str, refinement: int | None = None, *, out: Path | None = None) -> dict:
    from benchmark.runner import (_PLANE_COORD_NAMES, _component_plane_coords, _resolve_source_normalization,
                                  _run_maxwell, _select_monitor_plane_field)
    from benchmark.scenes import SCENARIOS
    from witwin.maxwell.scene import place_scene, prepare_scene

    row = ROWS[name]
    r = row.refinement if refinement is None else int(refinement)
    scenario = SCENARIOS[name]
    scene = refined_scene(name, r)
    prepared = prepare_scene(place_scene(scene, "cpu"), device="cpu")
    dx = float(np.asarray(prepared.dx_primal64).min())
    normalize = _resolve_source_normalization(scene, scenario.normalize_source)
    _, monitors, elapsed, perf = _run_maxwell(
        scene, frequencies=scenario.frequencies, run_time_factor=scenario.run_time_factor,
        solver=scenario.solver, normalize_source=normalize, absorber=scenario.absorber)
    monitor, component = scenario.display_monitor, scenario.display_component
    data = monitors[monitor]
    assert data["axis"] == "y" and abs(float(data["position"])) < 1e-12 and component == "Ex"
    x, z = _component_plane_coords(data, component)
    assert _PLANE_COORD_NAMES["y"] == ("x", "z")
    X, Z = np.meshgrid(x, z, indexing="ij")
    points = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=-1)
    # A Yee sample exactly on the sphere surface has no one-sided Mie limit (normal E jumps):
    # evaluate it just outside and leave it unscored.
    radius = np.linalg.norm(points, axis=-1)
    on_surface = np.abs(radius - row.radius) <= 1e-9 * row.radius
    points[on_surface] *= (1 + 1e-6)
    frequency = float(scenario.frequencies[0])
    k = 2 * math.pi * frequency / C0
    if row.tfsf_half is None:
        analytic = mie.near_fields(row.radius, k, row.eps, points, field_kind="total")[0][:, 0]
    else:
        analytic = tfsf_composite_mie(row, k, points)[:, 0]
    analytic = analytic.reshape(X.shape)
    field = _select_monitor_plane_field(data, component, data["fields"][component], freq_index=0)
    half_span = max(abs(v) for bounds in scene.domain.bounds for v in bounds)
    mask = sphere_mask(row, x, z, analytic, half_span=half_span, dx=dx) & ~on_surface.reshape(X.shape)
    error, scale = shape_l2(field[mask], analytic[mask])
    report = {
        "row": name, "kind": "mie_tfsf_composite" if row.tfsf_half else "mie_total", "refinement": r,
        "dx_m": dx, "cells_per_wavelength_in_sphere": C0 / (frequency * math.sqrt(row.eps)) / dx,
        "frequency": frequency, "monitor": monitor, "component": component, "samples": int(mask.sum()),
        "mie_shape_l2": error, "fitted_scale_abs": abs(scale), "fitted_scale_arg": float(np.angle(scale)),
        "maxwell_seconds": elapsed, "dt": perf["dt"], "peak_gpu_memory_mb": perf["peak_gpu_memory_mb"],
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{name}_r{r}_mie.npz", x=x, z=z, maxwell=field, mie=analytic, mask=mask)
    return report


def score_slab(name: str, refinement: int | None = None, *, out: Path | None = None) -> dict:
    from benchmark.analytic.slab_reference import score_planar_scene
    from benchmark.scenes import SCENARIOS
    from witwin.maxwell.scene import place_scene, prepare_scene

    row = ROWS[name]
    r = row.refinement if refinement is None else int(refinement)
    scene = refined_scene(name, r)
    dx = float(np.asarray(prepare_scene(place_scene(scene, "cpu"), device="cpu").dz_primal64).min())
    tag = f"{name}_r{r}"
    report = score_planar_scene(tag, scene, SCENARIOS[name], row.eps * np.eye(3), row.thickness, row.centre,
                                out=out)
    report.update(row=name, kind="airy_slab", refinement=r, dx_m=dx,
                  cells_per_wavelength_in_slab=C0 / (SCENARIOS[name].frequencies[0] * math.sqrt(row.eps)) / dx)
    return report


def score(name: str, refinement: int | None = None, *, out: Path | None = None) -> dict:
    if isinstance(ROWS[name], SlabRow):
        return score_slab(name, refinement, out=out)
    return score_sphere(name, refinement, out=out)


def check(report: dict) -> list[str]:
    """Metrics of ``report`` at or above their limit (only at the row's own refinement)."""
    limits = LIMITS[report["row"]]
    return [f"{key} {report[key]:.4e} >= {limit}" for key, limit in limits.items() if not report[key] < limit]


def main(argv=None):
    from benchmark.analytic import slab_reference

    parser = argparse.ArgumentParser(description="Score benchmark rows against analytic references")
    parser.add_argument("scenes", nargs="*", default=sorted(slab_reference.SLAB_SCENES))
    parser.add_argument("--refine", type=int, default=None, help="override the row's refinement factor")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    unknown = [name for name in args.scenes if name not in ROWS and name not in slab_reference.SLAB_SCENES]
    if unknown:
        raise SystemExit(f"No analytic reference for {unknown}; available: "
                         f"{sorted(set(ROWS) | set(slab_reference.SLAB_SCENES))}")
    failed = []
    for name in args.scenes:
        if name in slab_reference.SLAB_SCENES:
            report = slab_reference.analytic_reference(name, out=args.out)
            print(json.dumps({k: v for k, v in report.items() if k != "flux"}, indent=2))
            continue
        report = score(name, args.refine, out=args.out)
        if args.refine is None or args.refine == ROWS[name].refinement:
            report["limits"] = LIMITS[name]
            report["failures"] = check(report)
            failed += [f"{name}: {item}" for item in report["failures"]]
        print(json.dumps(report, indent=2, default=float), flush=True)
        if args.out is not None:
            (args.out / f"{name}_r{report['refinement']}_analytic.json").write_text(
                json.dumps(report, indent=2, default=float))
    if failed:
        raise SystemExit("analytic limits exceeded: " + "; ".join(failed))
