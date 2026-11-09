"""Measure pulse spectra versus planned horizon without changing the runtime.

Examples (select the physical CUDA device outside this script)::

    python scripts/dev/fdtd/dft_horizon_normalisation_repro.py --scene dipole
    python scripts/dev/fdtd/dft_horizon_normalisation_repro.py --scene vacuum --normalize-source
    python scripts/dev/fdtd/dft_horizon_normalisation_repro.py --scene helix --n 48 --count 41

PEC wire geometry and drive are copied from datagen/p0-profiler f7914baf,
benchmark/datagen/scenes.py:wire. No dependency on that branch is required.
Each JSON line contains all frequencies, actual sample counts, denominators,
raw port sums and published amplitudes/powers. Fixed and decay runs share a
planned horizon. With source normalization off, multiplying an amplitude by
N/2 recovers its raw DFT sum; multiplying a power by (N/2)^2 removes the averaging factor. These
diagnostic products are not a proposed physical normalization.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import FieldProjector, compute_directivity

C0 = 299792458.0
FC = 2.4e9


def build_scene(kind, n, count):
    wavelength = C0 / FC
    span = 1.6 * wavelength
    dx = span / n
    nodes = np.linspace(-span / 2, span / 2, n + 1)
    scene = mw.Scene(
        domain=mw.Domain(((float(nodes[0]), float(nodes[-1])),) * 3),
        grid=mw.GridSpec.custom(nodes, nodes, nodes),
        boundary=mw.BoundarySpec.pml(num_layers=8),
        device="cuda:0",
    )
    pulse = mw.GaussianPulse(frequency=FC, fwidth=0.8 * FC)
    frequencies = tuple(np.linspace(0.8 * FC, 1.2 * FC, count))
    if kind == "vacuum":
        scene.add_source(mw.PointDipole(
            position=(0., 0., 0.), polarization="Ez", profile="ideal", source_time=pulse,
        ))
        excitations = ()
    else:
        for side, sign in (("negative", -1), ("positive", 1)):
            if kind == "dipole":
                segments = max(8, math.ceil((0.24 * wavelength - dx / 2) / (0.5 * dx)))
                points = [(0., 0., sign * z) for z in
                          np.linspace(dx / 2, 0.24 * wavelength, segments + 1)]
            else:
                segments = max(20, math.ceil(0.6 * wavelength / dx))
                points = [(0.07 * wavelength * (math.cos(t) - 1),
                           0.07 * wavelength * math.sin(t),
                           sign * (dx / 2 + 0.12 * wavelength * t / (4 * math.pi)))
                          for t in np.linspace(0., 4 * math.pi, segments + 1)]
            scene.add_thin_wire(mw.ThinWire(
                side, points, 0.12 * dx, mw.WireConductor.pec(), snap="continuous",
            ))
        scene.add_port(mw.LumpedPort(
            "feed", wire_binding=mw.WirePortBinding.nodes(
                negative=mw.WireNodeRef("negative", 0),
                positive=mw.WireNodeRef("positive", 0),
            ), reference_impedance=50.,
        ))
        excitations = (mw.PortExcitation("feed", source_time=pulse),)
    scene.add_monitor(mw.PointMonitor("point", (0.35 * wavelength, 0., 0.), fields=("Ez",)))
    scene.add_monitor(mw.PlaneMonitor(
        "plane", axis="x", position=0.35 * wavelength, fields=("Ez",), frequencies=frequencies,
    ))
    scene.add_monitor(mw.FluxMonitor(
        "flux", axis="x", position=0.35 * wavelength, frequencies=frequencies,
    ))
    scene.add_monitor(mw.ClosedSurfaceMonitor.box(
        "radiation", size=(1.2 * wavelength,) * 3, frequencies=frequencies,
    ))
    return scene, frequencies, excitations


def values(value):
    return torch.as_tensor(value).detach().cpu().tolist()


def run_case(args, cycles, shutoff):
    scene, frequencies, excitations = build_scene(args.scene, args.n, args.count)
    prepared = mw.Simulation.fdtd(
        scene, frequencies=frequencies, excitations=excitations,
        run_time=mw.TimeConfig.auto(steady_cycles=cycles, transient_cycles=8),
        shutoff=shutoff, full_field_dft=args.full_field_dft,
        spectral_sampler=mw.SpectralSampler(normalize_source=args.normalize_source),
    ).prepare()
    result = prepared.run()
    solver = prepared.solver
    stats = result.stats()
    entries = solver._observer_spectral_entries
    denominator = entries[0]["window_normalization"]
    point = torch.as_tensor(result.monitor("point")["Ez"])
    plane = torch.as_tensor(result.monitor("plane")["Ez"]).reshape(args.count, -1)
    row = dict(
        scene=args.scene, n=args.n, cycles=cycles, shutoff=shutoff,
        normalize_source=args.normalize_source, frequencies_Hz=frequencies,
        dt=stats["dt"], planned_steps=stats["time_steps"], steps_run=stats["steps_run"],
        shutoff_triggered=stats["shutoff_triggered"],
        time_plan=stats["time_plan"], effective_window=solver.observer_window_type,
        observer_samples=[entry["sample_count"] for entry in entries],
        observer_weight_sums=[entry["window_normalization"] for entry in entries],
        point_abs=values(point.abs()), plane_norm=values(torch.linalg.vector_norm(plane, dim=1)),
        flux=values(result.monitor("flux")["flux"]),
        closed_flux=values(result.monitor("radiation")["flux"]),
        point_abs_times_N_over_2=values(point.abs() * denominator / 2),
        raw_source_sum_abs=[abs(complex(entry["source_dft_real"], entry["source_dft_imag"]))
                            for entry in entries],
    )
    if args.full_field_dft:
        row["full_field_norm"] = values(torch.linalg.vector_norm(
            result.fields["EZ"].reshape(args.count, -1), dim=1))
        row["full_field_weight_sums"] = list(solver.dft_window_normalizations)
    projection = FieldProjector.from_monitor(result, "radiation", device=scene.device).angles(
        np.linspace(0., math.pi, 19), np.linspace(0., 2 * math.pi, 37),
    )
    angular = [compute_directivity(dict(
        theta=projection.theta, phi=projection.phi, radius=projection.radius,
        power_density=projection.power_density[index], frequency=frequency,
    )) for index, frequency in enumerate(frequencies)]
    row.update(
        far_E_norm=values(torch.linalg.vector_norm(projection.E.reshape(args.count, -1), dim=1)),
        p_rad=[float(data["P_rad"]) for data in angular],
        d_max=[float(data["D_max"]) for data in angular],
    )
    if args.scene != "vacuum":
        port = result.port("feed")
        runtime = solver._port_runtimes[0]
        accumulator = runtime.accumulator
        row.update(
            port_samples=accumulator._sample_count,
            port_weight_sums=values(accumulator._window_weight_sum),
            raw_voltage_sum_abs=values(accumulator._voltage_sum.abs()),
            raw_current_sum_abs=values(accumulator._current_sum.abs()),
            raw_drive_sum_abs=values(runtime.drive_accumulator._voltage_sum.abs()),
            voltage_abs=values(port.voltage.abs()), current_abs=values(port.current.abs()),
            a_abs=values(port.a.abs()), b_abs=values(port.b.abs()),
            p_inc=values(port.incident_power), p_ref=values(port.reflected_power),
            p_acc=values(port.accepted_power), p_delivered=values(port.delivered_power),
            p_available=values(port.available_power), s11_abs=values(port.reflection_coefficient.abs()),
            accepted_fraction=values(port.accepted_power / port.incident_power),
            p_inc_times_N_over_2_squared=values(port.incident_power * (denominator / 2) ** 2),
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("vacuum", "dipole", "helix"), default="dipole")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--cycles", type=int, nargs="+", default=(40, 80, 200))
    parser.add_argument("--mode", choices=("fixed", "decay", "both"), default="both")
    parser.add_argument("--normalize-source", action="store_true")
    parser.add_argument("--full-field-dft", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modes = {"fixed": (0.,), "decay": (1e-5,), "both": (0., 1e-5)}[args.mode]
    for cycles in args.cycles:
        for shutoff in modes:
            row = run_case(args, cycles, shutoff)
            line = json.dumps(row)
            print(line, flush=True)
            if args.output is not None:
                with args.output.open("a") as stream:
                    stream.write(line + "\n")


if __name__ == "__main__":
    main()
