"""Analytic (Berreman 4x4) reference option for planar benchmark scenes.

``python -m benchmark analytic full_tensor_slab [--out DIR]`` runs the benchmark
scene unchanged (plus one extra observation plane, which does not alter the
fields) and a companion run without the slab, then scores

* transmittance T = P_t(slab) / P_t(empty) and reflectance
  R = (P_r(slab) - P_r(empty)) / P_t(empty) from the scene's own flux monitors
  (the reflected-side monitor lies between the launch plane and the slab, so the
  empty run's reflected flux is minus the incident power);
* the complex Ex/Ey line field through the slab against the Berreman field,
  with the incident amplitude and phase taken from the empty run at the slab
  entrance plane.

This scores FDTD against the exact layered solution; the benchmark's own
external-reference metrics and limits are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import witwin.maxwell as mw
from benchmark.analytic import berreman

SLAB_SCENES = {
    # name: (tensor, thickness, slab z centre)
    "full_tensor_slab": (np.array([[3.0, 0.2, 0.0], [0.2, 2.5, 0.0], [0.0, 0.0, 2.0]]), 0.1, 0.0),
}


def _line(result, monitor: str, component: str, half_width: float):
    """Field averaged over |x| <= half_width on the y = 0 plane, as a function of z."""
    entry = result.raw_monitor(monitor)["components"][component]
    data = entry["data"].detach().cpu().numpy()
    x, z = entry["coords"]
    band = np.abs(x) <= half_width
    return z, data[band].mean(axis=0), data[band].std(axis=0)


def analytic_reference(name: str, *, out: Path | None = None) -> dict:
    """Score a registered benchmark scene against its Berreman reference."""
    from benchmark.scenes import SCENARIOS, build_scene

    tensor, thickness, centre = SLAB_SCENES[name]
    return score_planar_scene(name, build_scene(name), SCENARIOS[name], tensor, thickness, centre, out=out)


def score_planar_scene(name, scene, scenario, tensor, thickness, centre, *, out: Path | None = None) -> dict:
    """Run ``scene`` (a z-normal slab under a normal-incidence plane wave) and an empty twin."""
    from benchmark.runner import _resolve_source_normalization, _run_maxwell

    frequency = float(scenario.frequencies[0])
    source = next(iter(scene.resolved_sources()))
    jones = np.asarray(source.polarization, dtype=float)[:2]
    observe = mw.PlaneMonitor("analytic_line", axis="y", position=0.0, fields=("Ex", "Ey"),
                              frequencies=scenario.frequencies)
    runs = {}
    for label, structures in (("slab", list(scene.structures)), ("empty", [])):
        variant = scene.updated(structures=structures, monitors=list(scene.monitors) + [observe])
        normalize = _resolve_source_normalization(variant, scenario.normalize_source)
        result, monitors, _, _ = _run_maxwell(
            variant, frequencies=scenario.frequencies, run_time_factor=scenario.run_time_factor,
            solver=scenario.solver, normalize_source=normalize, absorber=scenario.absorber)
        runs[label] = (result, monitors)
    flux = {label: {key: float(np.real(np.asarray(monitors[key]["flux"]).ravel()[0]))
                    for key in ("reflected", "transmitted")}
            for label, (_, monitors) in runs.items()}
    incident = flux["empty"]["transmitted"]
    transmittance = flux["slab"]["transmitted"] / incident
    reflectance = (flux["slab"]["reflected"] - flux["empty"]["reflected"]) / incident

    layers = [berreman.Layer(thickness, tensor)]
    response = berreman.solve(layers, frequency)
    t_ref = float(np.sum(np.abs(response.t @ jones) ** 2))
    r_ref = float(np.sum(np.abs(response.r @ jones) ** 2))

    z0 = centre - 0.5 * thickness
    half_width = 0.3
    lines = {}
    for label, (result, _) in runs.items():
        for component in ("Ex", "Ey"):
            lines[label, component] = _line(result, "analytic_line", component, half_width)
    z_ex, empty_ex, _ = lines["empty", "Ex"]
    amplitude = np.interp(z0, z_ex, empty_ex.real) + 1j * np.interp(z0, z_ex, empty_ex.imag)
    amplitude /= jones[0]
    window = 0.3
    measured, reference, curves = [], [], {}
    for index, component in enumerate(("Ex", "Ey")):
        z, slab_line, slab_spread = lines["slab", component]
        keep = np.abs(z) <= window
        e, _ = berreman.fields(layers, frequency, z[keep], z0=z0, jones=jones)
        analytic = amplitude * e[:, index]
        measured.append(slab_line[keep])
        reference.append(analytic)
        curves[f"z_{component}"] = z[keep]
        curves[f"{component}_fdtd"] = slab_line[keep]
        curves[f"{component}_berreman"] = analytic
        curves[f"{component}_transverse_std"] = slab_spread[keep]
    measured = np.concatenate(measured)
    reference = np.concatenate(reference)
    report = {
        "scene": name,
        "frequency": frequency,
        "jones": jones.tolist(),
        "T_fdtd": transmittance, "T_berreman": t_ref, "T_abs_err": abs(transmittance - t_ref),
        "R_fdtd": reflectance, "R_berreman": r_ref, "R_abs_err": abs(reflectance - r_ref),
        "energy_fdtd": transmittance + reflectance,
        "line_field_rel_l2": float(np.linalg.norm(measured - reference) / np.linalg.norm(reference)),
        "incident_amplitude_at_entrance": [amplitude.real, amplitude.imag],
        "flux": flux,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}_berreman.json").write_text(json.dumps(report, indent=2))
        np.savez(out / f"{name}_berreman_lines.npz", **curves)
        _plot(out / f"{name}_berreman.png", curves, report, z0, thickness)
    return report


def _plot(path, curves, report, z0, thickness):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for axis, component in zip(axes, ("Ex", "Ey")):
        z = curves[f"z_{component}"]
        for part, style in ((np.real, "-"), (np.imag, "--")):
            axis.plot(z, part(curves[f"{component}_berreman"]), "k" + style, lw=1.2,
                      label=f"Berreman {part.__name__}")
            axis.plot(z, part(curves[f"{component}_fdtd"]), "C3" + style, lw=1.0,
                      label=f"FDTD {part.__name__}")
        axis.axvspan(z0, z0 + thickness, color="0.9")
        axis.set_xlabel("z [m]")
        axis.set_title(component)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{report['scene']}: T {report['T_fdtd']:.4f} vs {report['T_berreman']:.4f}, "
                 f"R {report['R_fdtd']:.4f} vs {report['R_berreman']:.4f}, "
                 f"line rel L2 {report['line_field_rel_l2']:.3g}", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score planar scenes against the Berreman oracle")
    parser.add_argument("scenes", nargs="*", default=sorted(SLAB_SCENES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in args.scenes:
        if name not in SLAB_SCENES:
            raise SystemExit(f"No analytic reference for {name!r}; available: {sorted(SLAB_SCENES)}")
        report = analytic_reference(name, out=args.out)
        print(json.dumps({k: v for k, v in report.items() if k != "flux"}, indent=2))
