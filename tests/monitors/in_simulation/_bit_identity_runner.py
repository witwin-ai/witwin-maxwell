"""Fresh-interpreter runner for the existing-monitor bit-identity check.

Usage: ``python _bit_identity_runner.py OUT.pt [with-new-monitors]``. The
``witwin.maxwell`` tree under test is selected by the caller's PYTHONPATH.
Only APIs present before the in-simulation monitors were added are used,
unless ``with-new-monitors`` is passed.
"""

import sys

import numpy as np
import torch

import witwin.maxwell as mw

# legacy subpixel pin; default coverage lives in tests/materials/compiler/test_subpixel_default_guard.py and tests/materials/compiler/test_kottke_edge.py
_LEGACY_SUBPIXEL = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase")

FREQUENCIES = (2.0e9, 2.3e9)


def _scene(with_new_monitors):
    scene = mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((-0.32, 0.32),) * 3),
        grid=mw.GridSpec.uniform(0.025),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda",
    )
    scene.add_structure(mw.Structure(
        geometry=mw.Box(position=(0, 0, 0), size=(1.28, 0.20, 0.30)),
        material=mw.Material(eps_r=4.0), name="waveguide"))
    scene.add_source(mw.ModeSource(
        position=(-0.20, 0, 0), size=(0, 0.5, 0.5), polarization="Ez",
        source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9), name="mode"))
    scene.add_source(mw.PointDipole(
        position=(0.0, 0.1, 0.2), polarization="Ey", width=0.03,
        source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9), name="dipole"))
    scene.add_monitor(mw.PointMonitor("point", (0.05, 0.02, 0.03), fields=("Ez", "Hy")))
    scene.add_monitor(mw.PlaneMonitor("plane", axis="x", position=0.1, fields=("Ez", "Ey", "Hx")))
    scene.add_monitor(mw.FluxMonitor("flux", axis="x", position=0.12))
    scene.add_monitor(mw.FinitePlaneMonitor(
        "finite", position=(0.15, 0, 0), size=(0, 0.3, 0.3), fields=("Ey", "Ez", "Hy", "Hz"), compute_flux=True))
    scene.add_monitor(mw.ModeMonitor(
        "mode_out", position=(0.15, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", direction="+"))
    scene.add_monitor(mw.ClosedSurfaceMonitor.box("box", position=(0.0, 0.1, 0.2), size=(0.15, 0.15, 0.15)))
    scene.add_monitor(mw.FieldTimeMonitor("time_point", components=("Ez",), position=(0.1, 0, 0), interval=3))
    scene.add_monitor(mw.FluxTimeMonitor("time_flux", axis="x", position=0.12, interval=2))
    if with_new_monitors:
        scene.add_monitor(mw.AngleProjectionMonitor(
            "new_projection", position=(0.0, 0.1, 0.2), size=(0.15, 0.15, 0.15), theta=(0.5, 1.5), phi=(0.0,)))
        scene.add_monitor(mw.GaussianOverlapMonitor(
            "new_overlap", position=(0.1, 0, 0), size=(0, 0.4, 0.4), beam_waist=0.1, polarization=(0, 0, 1)))
        scene.add_monitor(mw.SurfaceFieldTimeMonitor("new_surface", size=(0.2, 0.2, 0.2), interval=4))
        scene.add_monitor(mw.AuxFieldTimeMonitor("new_aux", components=("Pz",), position=(0.05, 0, 0)))
    return scene


def _flatten(prefix, value, out):
    if isinstance(value, torch.Tensor):
        out[prefix] = value.detach().cpu()
    elif isinstance(value, np.ndarray) and value.dtype != object:
        out[prefix] = torch.from_numpy(np.ascontiguousarray(value))
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}/{key}", item, out)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _flatten(f"{prefix}/{index}", item, out)
    elif isinstance(value, (bool, int, float, complex, str)) or value is None:
        out[prefix] = value


def main():
    out_path = sys.argv[1]
    with_new = len(sys.argv) > 2 and sys.argv[2] == "with-new-monitors"
    torch.manual_seed(0)
    result = mw.Simulation.fdtd(
        _scene(with_new),
        frequencies=list(FREQUENCIES),
        run_time=mw.TimeConfig(time_steps=180),
        full_field_dft=False,
    ).run()
    out = {"__file__": mw.__file__}
    for name in ("point", "plane", "flux", "finite", "mode_out", "box", "time_point", "time_flux"):
        _flatten(f"raw/{name}", result.raw_monitor(name), out)
        spectral = name not in ("time_point", "time_flux")
        for index in range(len(FREQUENCIES)) if spectral else (None,):
            _flatten(f"monitor/{name}/{index}", result.monitor(name, freq_index=index), out)
    for name in sorted(result.monitors):
        if not name.startswith("new_"):
            _flatten(f"payload/{name}", result.monitors[name], out)
    torch.save(out, out_path)


if __name__ == "__main__":
    main()
