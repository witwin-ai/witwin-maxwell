"""Electrical-size ladder: FDTD sphere scattering scored against the Mie series.

A sphere of diameter D sits at the origin under an x-polarized CW plane wave
along +z injected by a TFSF box. Three observables are scored against
``benchmark.analytic.mie``:

* bistatic RCS in the E-plane (phi = 0) and H-plane (phi = 90 deg), from a
  scattered-field Huygens box through the near-to-far-field transform, plus the
  complex far-field amplitude (phase referenced to the sphere centre);
* cross sections from closed-surface Poynting flux: scattering from the
  scattered-field box, absorption from a total-field box around the sphere,
  extinction = scattering + absorption;
* the total near field (Ex, Ey, Ez) on the plane z = R + g/2 inside the TFSF
  box, evaluated at each component's Yee positions with the monitor's own
  plane interpolation weights.

The ladder runs D = 1..16 wavelengths at a fixed number of cells per wavelength
in the densest medium. Every run prints its cell count and a memory estimate
before allocating and is skipped when the estimate exceeds the budget.

    python -m benchmark.analytic.sphere_ladder --material pec --diameters 1 2 4 8 16 \
        --cpl 15 20 30 --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import (
    NearFieldFarFieldTransformer,
    compute_bistatic_rcs,
    equivalent_surface_currents_from_monitor,
)

from benchmark.analytic import mie

WAVELENGTH = 1.0
FREQUENCY = mie.C0 / WAVELENGTH
PML_LAYERS = 12
# Peak torch-allocated bytes per grid cell (PML included) of a TFSF sphere run with
# the monitors below on the CUDA core path (float32 fields, dense CPML), fitted from
# the ladder's own runs on an A6000 (0.43 GiB at 1.56e6 cells, 1.44 GiB at 5.45e6
# cells); see the evidence README.
BYTES_PER_CELL = 280.0
FIXED_BYTES = 0.1 * 2**30
THETA = np.linspace(0.0, np.pi, 181)
PHI_CUTS = (0.0, 0.5 * np.pi)


@dataclass(frozen=True)
class SphereCase:
    """One sphere scattering run. Lengths in metres, frequency in Hz."""

    name: str
    radius: float
    dx: float
    material: str | complex  # mie.PEC, "vacuum" (no sphere) or eps(f)
    scene_material: object = None  # mw.Material for dispersive spheres
    frequency: float = FREQUENCY
    tf_gap: float = 0.0
    box_gap: float = 0.0
    pml_gap: float = 0.0
    transient_cycles: int = 15
    steady_cycles: int = 10
    # Lossless dielectric spheres ring. For D = 2 lambda, eps = 4 at 40 cells per
    # vacuum wavelength the closed total-field box still reports a net outflow of
    # 1.8% of the scattered power after 150/40 transient/steady cycles, 0.4% after
    # 300/80 and 0.005% after 600/160 (extinction error 2.0%, 0.66%, 0.25%);
    # arithmetic averaging settles faster (0.09% at 150/40). ladder_case() uses
    # 600/160 for dielectric rungs (evidence diag_gain).
    subpixel: str = "polarized"
    subpixel_samples: int = 3

    def __post_init__(self):
        wavelength = mie.C0 / self.frequency
        object.__setattr__(self, "tf_gap", self.tf_gap or max(0.25 * wavelength, 6 * self.dx))
        object.__setattr__(self, "box_gap", self.box_gap or max(0.15 * wavelength, 4 * self.dx))
        object.__setattr__(self, "pml_gap", self.pml_gap or max(0.25 * wavelength, 6 * self.dx))

    @property
    def k(self) -> float:
        return 2 * np.pi * self.frequency / mie.C0

    @property
    def tfsf_half(self) -> float:
        return self.radius + self.tf_gap

    @property
    def huygens_half(self) -> float:
        return self.tfsf_half + self.box_gap

    @property
    def inner_half(self) -> float:
        return self.radius + 0.5 * self.tf_gap

    @property
    def domain_half(self) -> float:
        return self.huygens_half + self.pml_gap

    def cells_per_axis(self) -> int:
        return int(math.ceil(round(2 * self.domain_half / self.dx, 9))) + 2 * PML_LAYERS

    def cells(self) -> int:
        return self.cells_per_axis() ** 3

    def memory_estimate_bytes(self) -> float:
        return self.cells() * BYTES_PER_CELL + FIXED_BYTES


def ladder_case(kind: str, diameter: float, cpl: float, *, eps: float = 4.0) -> SphereCase:
    """Ladder rung: D wavelengths, cpl cells per wavelength in the densest medium."""
    n_max = 1.0 if kind in (mie.PEC, "vacuum") else math.sqrt(eps)
    material = kind if kind in (mie.PEC, "vacuum") else complex(eps)
    label = kind if kind in (mie.PEC, "vacuum") else f"eps{eps:g}"
    horizon = {} if kind in (mie.PEC, "vacuum") else {"transient_cycles": 600, "steady_cycles": 160}
    return SphereCase(
        name=f"{label}_D{diameter:g}_cpl{cpl:g}",
        radius=0.5 * diameter * WAVELENGTH,
        dx=WAVELENGTH / (cpl * n_max),
        material=material,
        **horizon,
    )


def _closed_box(name, half, frequency):
    faces = []
    for axis_index, axis in enumerate("xyz"):
        for sign, label in ((-1, "neg"), (1, "pos")):
            position = [0.0, 0.0, 0.0]
            position[axis_index] = sign * half
            size = [2 * half] * 3
            size[axis_index] = 0.0
            faces.append(mw.monitors.FinitePlaneMonitor(
                f"{axis}_{label}", position=tuple(position), size=tuple(size),
                fields=mw.monitors.required_flux_fields(axis), frequencies=(frequency,),
                compute_flux=True, normal_direction="+" if sign > 0 else "-"))
    return mw.monitors.ClosedSurfaceMonitor(name, faces, frequencies=(frequency,))


def build_scene(case: SphereCase) -> mw.Scene:
    half = case.domain_half
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half),) * 3),
        grid=mw.GridSpec.uniform(case.dx),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        subpixel=mw.SubpixelSpec(samples=case.subpixel_samples, averaging=case.subpixel,
                                         pec="staircase"),
    )
    if case.material != "vacuum":
        if case.scene_material is not None:
            material = case.scene_material
        elif case.material == mie.PEC:
            material = mw.Material.pec()
        else:
            eps = complex(case.material)
            sigma = eps.imag * 2 * np.pi * case.frequency * 8.8541878128e-12
            material = mw.Material(eps_r=eps.real, sigma_e=sigma)
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(
                geometry=mw.Sphere(position=(0.0, 0.0, 0.0), radius=case.radius),
                material=material, name="sphere"),
        ))
    tf = case.tfsf_half
    scene = scene.updated(sources=(
        *scene.sources,
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0), polarization="Ex",
            source_time=mw.CW(frequency=case.frequency),
            injection=mw.TFSF(bounds=((-tf, tf),) * 3), name="tfsf"),
    ), monitors=(
        *scene.monitors,
        _closed_box("huygens", case.huygens_half, case.frequency),
        _closed_box("inner", case.inner_half, case.frequency),
        mw.PlaneMonitor(
            "near", axis="z", position=case.inner_half, fields=("Ex", "Ey", "Ez"),
            frequencies=(case.frequency,)),
    ))
    return scene


def _near_field_error(case: SphereCase, payload: dict) -> dict:
    limit = case.tfsf_half - 2 * case.dx
    measured, reference = [], []
    for component_index, component in enumerate(("Ex", "Ey", "Ez")):
        entry = payload["components"][component]
        data = entry["data"].detach().cpu().numpy()
        cx, cy = entry["coords"]
        mask = (np.abs(cx)[:, None] <= limit) & (np.abs(cy)[None, :] <= limit)
        xx, yy = np.meshgrid(cx, cy, indexing="ij")
        analytic = np.zeros(xx.shape, dtype=complex)
        for z, weight in zip(entry["plane_positions"], entry["plane_weights"]):
            points = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)), -1)
            e, _ = mie.near_fields(case.radius, case.k, _mie_material(case), points)
            analytic += weight * e[:, component_index].reshape(xx.shape)
        measured.append(data[mask])
        reference.append(analytic[mask])
    measured = np.concatenate(measured)
    reference = np.concatenate(reference)
    # Shape error: best-fit complex scalar removed, separating the incident wave's
    # accumulated dispersion phase (measured from the TFSF box centre) from the
    # scattered-field structure.
    scale = np.vdot(measured, reference) / np.vdot(measured, measured)
    return {
        "near_rel_l2": float(np.linalg.norm(measured - reference) / np.linalg.norm(reference)),
        "near_shape_rel_l2": float(np.linalg.norm(scale * measured - reference) / np.linalg.norm(reference)),
        "near_fit_phase_rad": float(np.angle(scale)),
        "near_max_abs": float(np.max(np.abs(measured - reference))),
        "near_points": int(measured.size),
    }


def _mie_material(case: SphereCase):
    return 1.0 + 0j if case.material == "vacuum" else case.material


def score(case: SphereCase, result) -> tuple[dict, dict]:
    """Metrics and curves for one completed run."""
    metrics: dict = {}
    material = _mie_material(case)
    reference = mie.cross_sections(case.radius, case.k, material)
    intensity = 1.0 / (2 * mie.ETA0)
    huygens = result.raw_monitor("huygens")
    inner = result.raw_monitor("inner")
    scattering = float(torch.as_tensor(huygens["power"]).real) / intensity
    absorption = -float(torch.as_tensor(inner["power"]).real) / intensity
    metrics.update(
        sigma_sca=scattering, sigma_sca_mie=reference["scattering"],
        sigma_abs=absorption, sigma_abs_mie=reference["absorption"],
        sigma_ext=scattering + absorption, sigma_ext_mie=reference["extinction"],
    )
    if case.material != "vacuum":
        metrics["ext_rel_err"] = abs(metrics["sigma_ext"] / reference["extinction"] - 1)
        metrics["sca_rel_err"] = abs(scattering / reference["scattering"] - 1)
    curves = {"theta_deg": np.degrees(THETA)}
    if case.material != "vacuum":
        currents = equivalent_surface_currents_from_monitor(result, "huygens")
        transformer = NearFieldFarFieldTransformer(currents, solver=result.solver, device="cuda")
        theta = torch.as_tensor(np.repeat(THETA[None], len(PHI_CUTS), 0), device="cuda")
        phi = torch.as_tensor(np.repeat(np.array(PHI_CUTS)[:, None], THETA.size, 1), device="cuda")
        far_radius = 1.0e4 * WAVELENGTH
        # Small direction batches: the projection workspace is batch x surface points.
        far = transformer.transform(theta, phi, radius=far_radius, batch_size=32)
        sigma = compute_bistatic_rcs(far, incident_amplitude=1.0, c=mie.C0)["rcs"].detach().cpu().numpy()
        amplitude = (far["E"].detach().cpu().numpy() * far_radius
                     * np.exp(-1j * case.k * far_radius))
        sigma_ref = np.stack([mie.bistatic_rcs(case.radius, case.k, material, THETA, p) for p in PHI_CUTS])
        directions = np.stack((np.sin(theta.cpu().numpy()) * np.cos(phi.cpu().numpy()),
                               np.sin(theta.cpu().numpy()) * np.sin(phi.cpu().numpy()),
                               np.cos(theta.cpu().numpy())), -1)
        amplitude_ref = mie.far_field(case.radius, case.k, material,
                                      directions.reshape(-1, 3)).reshape(amplitude.shape)
        floor = 1e-3 * sigma_ref.max()
        visible = sigma_ref > floor
        db = 10 * np.log10(np.maximum(sigma, floor) / np.maximum(sigma_ref, floor))
        metrics.update(
            rcs_rel_l2=float(np.linalg.norm(sigma - sigma_ref) / np.linalg.norm(sigma_ref)),
            rcs_db_mae=float(np.mean(np.abs(db[visible]))),
            far_complex_rel_l2=float(np.linalg.norm(amplitude - amplitude_ref)
                                     / np.linalg.norm(amplitude_ref)),
            back_rel_err=float(abs(sigma[0, -1] / sigma_ref[0, -1] - 1)),
            forward_rel_err=float(abs(sigma[0, 0] / sigma_ref[0, 0] - 1)),
        )
        for cut, label in enumerate(("E", "H")):
            curves[f"rcs_{label}_fdtd"] = sigma[cut]
            curves[f"rcs_{label}_mie"] = sigma_ref[cut]
    metrics.update(_near_field_error(case, result.raw_monitor("near")))
    return metrics, curves


def run_case(case: SphereCase, *, budget_gib: float = 7.5) -> dict | None:
    estimate = case.memory_estimate_bytes() / 2**30
    n = case.cells_per_axis()
    print(f"[{case.name}] dx={case.dx:.5g} N={n}^3 cells={case.cells():.3e} "
          f"est={estimate:.2f} GiB budget={budget_gib:.2f} GiB", flush=True)
    if estimate > budget_gib:
        print(f"[{case.name}] SKIP: estimate above budget", flush=True)
        return None
    scene = build_scene(case)
    torch.cuda.reset_peak_memory_stats()
    result = mw.Simulation.fdtd(scene, frequencies=[case.frequency], config=mw.FDTDConfig(
        run_time=mw.TimeConfig.auto(steady_cycles=case.steady_cycles,
                                    transient_cycles=case.transient_cycles),
        full_field_dft=False)).run()
    stats = result.stats()
    peak = torch.cuda.max_memory_allocated() / 2**30
    metrics, curves = score(case, result)
    record = {
        "name": case.name, "material": str(case.material), "diameter_wl": 2 * case.radius / (mie.C0 / case.frequency),
        "dx": case.dx, "cells_per_wl_vacuum": (mie.C0 / case.frequency) / case.dx,
        "n_axis": n, "cells": case.cells(), "est_gib": estimate, "peak_gib": peak,
        "steps": stats.get("steps_run"), "dt": stats.get("dt"), "core_runtime": getattr(result.solver, "_core_runtime", None) is not None, **metrics,
    }
    print(json.dumps({k: v for k, v in record.items()}, default=float), flush=True)
    del result, scene
    gc.collect()
    torch.cuda.empty_cache()
    return {"record": record, "curves": curves}


def observed_order(errors, densities):
    """Least-squares slope of log(error) vs log(cells per wavelength), sign flipped."""
    e = np.log(np.asarray(errors, float))
    c = np.log(np.asarray(densities, float))
    return float(-np.polyfit(c, e, 1)[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--material", default="pec", help="pec, vacuum or eps value (e.g. 4)")
    parser.add_argument("--diameters", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--cpl", type=float, nargs="+", default=[15, 20, 30])
    parser.add_argument("--budget-gib", type=float, default=7.5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="only redraw plots from CSVs in --out")
    args = parser.parse_args(argv)
    if args.plot_only:
        for path in sorted(args.out.glob("ladder_*.csv")):
            plot_ladder(path, args.out)
        return
    kind = args.material if args.material in (mie.PEC, "vacuum") else "eps"
    eps = float(args.material) if kind == "eps" else 4.0
    args.out.mkdir(parents=True, exist_ok=True)
    for cpl in args.cpl:
        for diameter in args.diameters:
            case = ladder_case(kind, diameter, cpl, eps=eps)
            csv_path = args.out / f"ladder_{case.name.split('_D')[0]}.csv"
            output = run_case(case, budget_gib=args.budget_gib)
            if output is None:
                row = {"name": case.name, "skipped": "memory", "cells": case.cells(),
                       "est_gib": case.memory_estimate_bytes() / 2**30}
            else:
                row = output["record"]
                np.savez(args.out / f"{case.name}.npz", **output["curves"])
            _update_csv(csv_path, row)
    for path in sorted(args.out.glob("ladder_*.csv")):
        plot_ladder(path, args.out)


METRICS = ("rcs_rel_l2", "far_complex_rel_l2", "ext_rel_err", "near_rel_l2")


def plot_ladder(csv_path: Path, out_dir: Path) -> None:
    """Error vs D (one line per density) and vs cells per wavelength (one line per D)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if not row.get("skipped")]
    if not rows:
        return
    label = csv_path.stem.removeprefix("ladder_")
    density = sorted({float(row["cells_per_wl_vacuum"]) for row in rows})
    diameters = sorted({float(row["diameter_wl"]) for row in rows})
    metrics = [m for m in METRICS if rows[0].get(m)]
    fig, axes = plt.subplots(2, len(metrics), figsize=(3.6 * len(metrics), 6.4), constrained_layout=True,
                             squeeze=False)
    for column, metric in enumerate(metrics):
        for cpl in density:
            pts = sorted((float(r["diameter_wl"]), float(r[metric])) for r in rows
                         if float(r["cells_per_wl_vacuum"]) == cpl)
            axes[0, column].loglog(*zip(*pts), "o-", label=f"{cpl:g} cells/lambda0")
        for diameter in diameters:
            pts = sorted((float(r["cells_per_wl_vacuum"]), float(r[metric])) for r in rows
                         if float(r["diameter_wl"]) == diameter)
            if len(pts) > 1:
                axes[1, column].loglog(*zip(*pts), "s-", label=f"D = {diameter:g} lambda")
        axes[0, column].set_xlabel("diameter [lambda0]")
        axes[1, column].set_xlabel("cells per vacuum wavelength")
        axes[0, column].set_title(metric)
        for axis in axes[:, column]:
            axis.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle(f"{label}: FDTD vs Mie")
    fig.savefig(out_dir / f"ladder_{label}.png", dpi=130)
    plt.close(fig)
    for path in sorted(out_dir.glob(f"{label}_D*.npz")):
        curves = np.load(path)
        if "rcs_E_mie" not in curves:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for axis, cut in zip(axes, ("E", "H")):
            axis.semilogy(curves["theta_deg"], curves[f"rcs_{cut}_mie"], "k-", lw=1.2, label="Mie")
            axis.semilogy(curves["theta_deg"], curves[f"rcs_{cut}_fdtd"], "C3--", lw=1.0, label="FDTD")
            axis.set_xlabel("theta [deg]")
            axis.set_ylabel("sigma [m^2]")
            axis.set_title(f"{path.stem} {cut}-plane")
        axes[0].legend(fontsize=7)
        fig.savefig(out_dir / f"rcs_{path.stem}.png", dpi=110)
        plt.close(fig)


def _update_csv(path: Path, row: dict) -> None:
    """Insert or replace one row (keyed by name) in a ladder CSV."""
    rows = []
    if path.exists():
        with path.open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["name"] != row["name"]]
    _write_csv(path, rows + [row])


def _write_csv(path: Path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
