"""Diagnose finite-wire growth: run with ``19 40 64 cu`` or ``19 40 64``.

The CSV reports maximum current and field energy in each 20-carrier-cycle block. Step numbers
are one-based. CUDA_VISIBLE_DEVICES selects the physical GPU outside this script.
"""

import math
import sys

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime import stepping

C0 = 299792458.0


def _cube(n, span, *, device):
    nodes = np.linspace(-span / 2, span / 2, n + 1)
    return mw.Scene(
        domain=mw.Domain(((float(nodes[0]), float(nodes[-1])),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device=device,
    )


def _kwargs(frequencies):
    center = (frequencies[0] + frequencies[-1]) / 2
    return dict(
        frequencies=frequencies,
        shutoff=0.0,
        full_field_dft=False,
        excitations=(mw.PortExcitation(
            "feed", source_time=mw.GaussianPulse(frequency=center, fwidth=0.8 * center)
        ),),
    )


def _surface(scene, frequencies, size):
    scene.add_monitor(mw.ClosedSurfaceMonitor.box(
        "radiation", size=size, frequencies=frequencies
    ))


def build_scene(segs, n, *, finite, port=True):
    frequency = 2.4e9
    wavelength = C0 / frequency
    span = 1.6 * wavelength
    dx = span / n
    scene = _cube(n, span, device=torch.device("cuda:0"))
    conductor = (mw.WireConductor.finite(conductivity=5.8e7)
                 if finite else mw.WireConductor.pec())
    for side, sign in (("negative", -1), ("positive", 1)):
        points = [(0.0, 0.0, sign * value)
                  for value in np.linspace(dx / 2, 0.24 * wavelength, segs + 1)]
        scene.add_thin_wire(mw.ThinWire(
            side, points, 0.12 * dx, conductor, snap="continuous"
        ))
    if port:
        scene.add_port(mw.LumpedPort(
            "feed", wire_binding=mw.WirePortBinding.nodes(
                negative=mw.WireNodeRef("negative", 0),
                positive=mw.WireNodeRef("positive", 0),
            ), reference_impedance=50.0,
        ))
    frequencies = tuple(np.linspace(0.8 * frequency, 1.2 * frequency, 11))
    _surface(scene, frequencies, (1.2 * wavelength,) * 3)
    return scene, frequencies


def main():
    segs, cycles = map(int, sys.argv[1:3])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    finite = len(sys.argv) > 4 and sys.argv[4] == "cu"
    scene, frequencies = build_scene(segs, n, finite=finite)
    kwargs = _kwargs(frequencies)
    kwargs["run_time"] = mw.TimeConfig.auto(steady_cycles=cycles, transient_cycles=8)
    prepared = mw.Simulation.fdtd(scene, **kwargs).prepare()
    solver = prepared.solver
    model = solver._wire_runtime.lossy_model
    print(f"runtime_dt={solver.dt:.17g} model_dt={None if model is None else model.dt} "
          f"certificate={None if model is None else model.spectral_radius} "
          f"orders={None if model is None else model.model_orders}", flush=True)
    if model is not None:
        print(f"fit_band_Hz={model.band} "
              f"min_companion_resistance_ohm={float(model.companion_conductance.min()):.9e} "
              f"full_band_resistance_lower_bound_ohm={model.minimum_digital_resistance:.9e}",
              flush=True)
    steps_per_cycle = 1 / (2.4e9 * solver.dt)
    original = stepping.accumulate_wire_monitors
    peak_current = 0.0
    peak_energy = 0.0
    block = 0
    onset = None
    print("block_end_step,cycle,current_peak_A,field_energy_peak_J", flush=True)

    def record(solver, step):
        nonlocal peak_current, peak_energy, block, onset
        original(solver, step)
        current = float(solver._wire_runtime.current.abs().max())
        peak_current = max(peak_current, current)
        # One ampere is far above this pulse's stable PEC response.
        if onset is None and (current > 1.0 or not math.isfinite(current)):
            onset = step + 1
            print(f"onset_current_above_1A_step={onset}", flush=True)
        electric = sum((getattr(solver, "eps_" + name).double()
                        * getattr(solver, name).double().square()).sum()
                       for name in ("Ex", "Ey", "Ez"))
        magnetic = sum((getattr(solver, "mu_" + name).double()
                        * getattr(solver, name).double().square()).sum()
                       for name in ("Hx", "Hy", "Hz"))
        volume = (1.6 * C0 / 2.4e9 / n) ** 3
        peak_energy = max(peak_energy, float(electric + magnetic) * volume / 2)
        if int((step + 1) / (20 * steps_per_cycle)) > block:
            print(f"{step + 1},{(step + 1) / steps_per_cycle:.3f},"
                  f"{peak_current:.9e},{peak_energy:.9e}", flush=True)
            peak_current = peak_energy = 0.0
            block += 1

    stepping.accumulate_wire_monitors = record
    result = prepared.run()
    stepping.accumulate_wire_monitors = original
    print(f"{result.solver_stats['steps_run']},"
          f"{result.solver_stats['steps_run'] / steps_per_cycle:.3f},"
          f"{peak_current:.9e},{peak_energy:.9e}", flush=True)
    port = result.port("feed")
    print(f"wire={'finite' if finite else 'pec'} n={n} segs={segs} cycles={cycles} "
          f"steps={result.solver_stats['steps_run']} onset={onset} "
          f"runtime_dt_after={solver.dt:.17g} "
          f"final_current_peak_A={float(solver._wire_runtime.current.abs().max()):.9e} "
          f"max_incident_power={float(torch.as_tensor(port.incident_power).abs().max()):.3e} "
          f"max|S11|={float(torch.as_tensor(port.reflection_coefficient).abs().max()):.3f}")


if __name__ == "__main__":
    main()
