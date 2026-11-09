"""Run fixed port scenes and save every port, field and monitor output.

Executed in fresh interpreters by tests/rf/lumped/test_port_step_graph_bit_identity.py
once against the frozen source tree that predates the graph-captured port step
phase and once against the working tree; the payloads must be bitwise equal:

    python port_step_graph_bit_identity_scene.py OUT.pt [fixture ...]

Every tensor reachable from the Result (fields, monitors, ports, network,
circuits) is saved, plus the final Yee fields, raw port DFT sums, lumped branch
state, drive buffers, clocks and exact per-step port V/I/drive histories. The
fixtures cover driven LumpedPort (one and two ports, conductive fill, every
source-time kind, Hann window, full-field DFT, shutoff), TerminalPort, PortSweep,
CoaxialLumpedPort, WavePort, an embedded circuit, an embedded network, wire-gap
and wire-node ports, run_until + disk resume (lumped and TerminalPort on
CopperFoil sheets), native batch execution and a port-free point-dipole scene.
Only APIs present in both trees are used.
"""
from __future__ import annotations

import dataclasses
import math
import sys
import tempfile
import warnings
from pathlib import Path

import torch

import witwin.maxwell as mw
from witwin.core import GeometryBase

# legacy subpixel pin; default coverage lives in tests/materials/compiler/test_subpixel_default_guard.py and tests/materials/compiler/test_kottke_edge.py
_LEGACY_SUBPIXEL = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase")

PULSE = mw.GaussianPulse(frequency=0.5e9, fwidth=0.3e9)
FREQS = (0.4e9, 0.6e9)

# These payloads pin the existing FDTD path against a frozen tree whose default
# is that path; an explicit False is valid in both trees. The automatic core path
# is qualified by its own gates and the default-flip census.
_SIMULATION_FDTD = mw.Simulation.fdtd


def _existing_path_fdtd(*args, **kwargs):
    kwargs.setdefault("fdtd_core_path", False)
    return _SIMULATION_FDTD(*args, **kwargs)


mw.Simulation.fdtd = staticmethod(_existing_path_fdtd)


# Exact per-step port V/I/drive histories. The step loop calls
# accumulate_wire_monitors(solver, n) from Python once per step in both trees,
# after the port/tail phase, so a wrapper can clone the live port scalars
# without touching numerics (clones only read device state in stream order).
_HISTORY = {}
_HISTORY_NAMES = ("last_voltage_midpoint", "last_branch_current", "last_voltage_after", "last_voltage_before")


def _install_history_hook():
    from witwin.maxwell.fdtd.runtime import stepping

    original = stepping.accumulate_wire_monitors

    def hooked(solver, n):
        run = _HISTORY.setdefault(id(solver), {"order": len(_HISTORY), "rows": {}})
        run["port_graph"] = bool(getattr(solver, "_port_cuda_graph_active", False))
        for index, runtime in enumerate(getattr(solver, "_port_runtimes", ())):
            rows = run["rows"]
            rows.setdefault((index, "drive_buffer"), []).append(runtime.drive_buffer.detach().clone())
            lumped = runtime.lumped
            for name in _HISTORY_NAMES:
                value = getattr(lumped, name, None) if lumped is not None else None
                if isinstance(value, torch.Tensor):
                    rows.setdefault((index, name), []).append(value.detach().clone())
        return original(solver, n)

    stepping.accumulate_wire_monitors = hooked


def _record_history(prefix, payload):
    for run in sorted(_HISTORY.values(), key=lambda item: item["order"]):
        for (index, name), values in run["rows"].items():
            payload[f"{prefix}/history/run{run['order']}/port{index}/{name}"] = torch.stack(values).cpu()
        payload[f"{prefix}/history/run{run['order']}/__port_graph"] = torch.tensor(run["port_graph"])
    _HISTORY.clear()


def _walk(prefix, value, payload, depth=0):
    if depth > 8:
        return
    if isinstance(value, torch.Tensor):
        payload[prefix] = value.detach().cpu().clone()
    elif isinstance(value, bool):
        payload[prefix] = torch.tensor(value)
    elif isinstance(value, (int, float)):
        payload[prefix] = torch.tensor(value, dtype=torch.float64)
    elif isinstance(value, complex):
        payload[prefix] = torch.tensor(value, dtype=torch.complex128)
    elif isinstance(value, dict):
        for key, item in value.items():
            _walk(f"{prefix}/{key}", item, payload, depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(f"{prefix}/{index}", item, payload, depth + 1)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _walk(f"{prefix}/{field.name}", getattr(value, field.name), payload, depth + 1)


def _record_solver(prefix, solver, payload):
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        payload[f"{prefix}/solver/{component}"] = getattr(solver, component).detach().cpu().clone()
    for index, runtime in enumerate(getattr(solver, "_port_runtimes", ())):
        key = f"{prefix}/port_runtime/{index}"
        for name in ("drive_buffer", "electric_time", "magnetic_time"):
            payload[f"{key}/{name}"] = getattr(runtime, name).detach().cpu().clone()
        payload[f"{key}/sample_index"] = torch.tensor(runtime.sample_index)
        for acc_name in ("accumulator", "drive_accumulator"):
            acc = getattr(runtime, acc_name)
            if acc is None:
                continue
            for name in ("_voltage_sum", "_current_sum", "_window_weight_sum"):
                payload[f"{key}/{acc_name}/{name}"] = getattr(acc, name).detach().cpu().clone()
            payload[f"{key}/{acc_name}/_sample_count"] = torch.tensor(acc._sample_count)
        lumped = runtime.lumped
        if lumped is not None:
            for name, value in vars(lumped).items():
                if isinstance(value, torch.Tensor) and (name.startswith("last_") or name in (
                        "inductor_current", "capacitor_voltage", "electric_history")):
                    payload[f"{key}/lumped/{name}"] = value.detach().cpu().clone()
    for index, entry in enumerate(getattr(solver, "_copper_foils", ())):
        for name in ("state", "right_e", "cumulative_loss", "spectral_real", "spectral_imag"):
            if isinstance(entry.get(name), torch.Tensor):
                payload[f"{prefix}/copper/{index}/{name}"] = entry[name].detach().cpu().clone()


def _record(prefix, result, payload):
    if prefix == "circuit":
        # The mixed drive admits time traces, coordinate metadata and port ratios.
        for name in ("t_e", "t_h"):
            _walk(f"{prefix}/monitors/{name}", result.monitor(name), payload)
        for name in ("field_index", "field_indices", "frequencies", "frequency", "position"):
            _walk(f"{prefix}/monitors/probe/{name}", result._monitors["probe"][name], payload)
        for name, port in result.ports.items():
            for field in dataclasses.fields(mw.PortData):
                if field.name not in ("voltage", "current", "available_power"):
                    _walk(f"{prefix}/ports/{name}/{field.name}", getattr(port, field.name), payload)
            for ratio in ("z_in", "reflection_coefficient", "return_loss_db", "vswr"):
                _walk(f"{prefix}/ports/{name}/{ratio}", getattr(port, ratio), payload)
        names = ("circuits", "network")
    else:
        # Inspect stored arrays even when public multi-frequency volume access is disabled.
        _walk(f"{prefix}/fields", result._fields, payload)
        names = ("monitors", "ports", "circuits", "network", "embedded_networks")
    for name in names:
        _walk(f"{prefix}/{name}", getattr(result, name, None), payload)
    solver = getattr(result, "solver", None)
    if solver is not None:
        _record_solver(prefix, solver, payload)
        stats = result.solver_stats or {}
        payload[f"{prefix}/steps_run"] = torch.tensor(int(stats.get("steps_run", -1)))
        if prefix in ("circuit", "circuit_ongoing", "circuit_transient"):
            payload[f"{prefix}/__field_graph"] = torch.tensor(solver._cuda_graph_active)
            payload[f"{prefix}/__circuit_graph"] = torch.tensor(solver._circuit_graph_active)


def _port_time_monitors(position, extent):
    return [mw.FieldTimeMonitor("t_e", components=("Ex", "Ey", "Ez"), position=position, size=extent),
            mw.FieldTimeMonitor("t_h", components=("Hx", "Hy", "Hz"), position=position, size=extent)]


def _lumped_scene(*, second_port=False, conductive=True, monitors=True):
    structures = [
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.75), size=(0.5, 0.5, 0.1)), mw.Material.pec(), name="signal"),
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.25), size=(0.5, 0.5, 0.1)), mw.Material.pec(), name="ground"),
    ]
    if conductive:
        structures.append(mw.Structure(mw.Box(position=(0.5, 0.5, 0.5), size=(0.5, 0.5, 0.4)),
                                       mw.Material(eps_r=2.2, sigma_e=0.02), name="lossy_fill"))
    ports = [mw.LumpedPort("p1", positive=(0.5, 0.5, 0.7), negative=(0.5, 0.5, 0.3), voltage_path=mw.AxisPath("z"),
                           current_surface=mw.Box(position=(0.5, 0.5, 0.45), size=(0.1, 0.1, 0.0)),
                           reference_impedance=50.0)]
    if second_port:
        ports.append(mw.LumpedPort("p2", positive=(1.2, 0.5, 0.6), negative=(1.2, 0.5, 0.4),
                                   voltage_path=mw.AxisPath("z"),
                                   current_surface=mw.Box(position=(1.2, 0.5, 0.45), size=(0.1, 0.1, 0.0)),
                                   reference_impedance=75.0))
    scene = mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((0.0, 1.6), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        structures=structures,
        ports=ports,
        lumped_elements=(mw.Resistor("shunt", positive=(1.4, 0.5, 0.6), negative=(1.4, 0.5, 0.4), resistance=100.0),)
        if second_port else (),
        device="cuda",
    )
    if monitors:
        for monitor in _port_time_monitors((0.5, 0.5, 0.5), (0.2, 0.2, 0.4)):
            scene.add_monitor(monitor)
        scene.add_monitor(mw.PointMonitor("probe", (1.0, 0.5, 0.5), fields=("Ez",)))
    return scene


def _sim(scene, *, steps, excitations, source_time=PULSE, **kwargs):
    if excitations == "p1":
        excitations = mw.PortExcitation("p1", source_time=source_time)
    options = dict(frequencies=FREQS, run_time=mw.TimeConfig(time_steps=steps),
                   spectral_sampler=mw.SpectralSampler(window="none"))
    options.update(kwargs)
    return mw.Simulation.fdtd(scene, excitations=excitations, **options)


def _terminal_lumped_scene():
    structures = (
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.8), size=(0.5, 0.5, 0.2)), mw.Material.pec(), name="signal"),
        mw.Structure(mw.Box(position=(0.5, 0.5, 0.2), size=(0.5, 0.5, 0.2)), mw.Material.pec(), name="ground"),
    )
    ports = (
        mw.TerminalPort("terminal", positive_terminal=mw.TerminalRef("signal"),
                        negative_terminal=mw.TerminalRef("ground"), integration_path=mw.AxisPath("z"),
                        reference_plane=0.45, reference_impedance=50.0),
        mw.LumpedPort("lumped", positive=(1.2, 0.5, 0.6), negative=(1.2, 0.5, 0.4), voltage_path=mw.AxisPath("z"),
                      current_surface=mw.Box(position=(1.2, 0.5, 0.45), size=(0.1, 0.1, 0.0)),
                      reference_impedance=75.0),
    )
    return mw.Scene(
        subpixel_samples=_LEGACY_SUBPIXEL,
        domain=mw.Domain(bounds=((0.0, 1.6), (0.0, 1.0), (0.0, 1.0))),
        grid=mw.GridSpec.uniform(0.1),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        structures=structures,
        ports=ports,
        lumped_elements=(mw.Resistor("shunt", positive=(1.4, 0.5, 0.6), negative=(1.4, 0.5, 0.4), resistance=100.0),),
        device="cuda",
    )


class _Tube(GeometryBase):
    kind = "bit_identity_tube"

    def __init__(self, *, inner_radius, outer_radius, length, device=None):
        super().__init__(position=(0.0, 0.0, 0.0), device=device)
        self.inner_radius, self.outer_radius, self.length = inner_radius, outer_radius, length

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        radial = torch.sqrt(dy.square() + dz.square())
        axial = torch.abs(dx) - 0.5 * self.length
        return torch.maximum(torch.maximum(radial - self.outer_radius, axial),
                             -torch.maximum(radial - self.inner_radius, axial))


def _wave_scene(dx=0.02):
    def port(name, x, direction):
        contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
        mode = mw.WaveModeSpec("tem", polarization="Ey", voltage_path=((x, 0.04, 0.0), (x, 0.16, 0.0)),
                               current_contour=mw.Box(position=(contour_x, 0.0, 0.0), size=(0.0, 0.22, 0.22)))
        return mw.WavePort(name, position=(x, 0.0, 0.0), size=(0.0, 0.40, 0.40), direction=direction,
                           reference_plane=x, modes=(mode,))

    length = 2.0 * (0.12 + 6 * dx) + 8.0 * dx
    scene = mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((-0.12, 0.12), (-0.22, 0.22), (-0.22, 0.22))),
                     grid=mw.GridSpec.uniform(dx), boundary=mw.BoundarySpec.pml(num_layers=6),
                     ports=(port("left", -0.04, "+"), port("right", 0.04, "-")), device="cuda")
    scene.add_structure(mw.Structure(mw.Cylinder(position=(0.0, 0.0, 0.0), radius=0.04, height=length, axis="x"),
                                     mw.Material.pec(), name="inner_conductor"))
    scene.add_structure(mw.Structure(_Tube(inner_radius=0.16, outer_radius=0.20, length=length),
                                     mw.Material.pec(), name="outer_conductor"))
    return scene


def _coax_scene():
    h, a, b, z0, z1, half, length = 0.25e-3, 1.0e-3, 2.3e-3, 1.0e-3, 6.0e-3, 3.0e-3, 7.0e-3
    scene = mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((-half, half), (-half, half), (0.0, length))),
                     grid=mw.GridSpec.uniform(h), boundary=mw.BoundarySpec.pec(), device="cuda")
    circle = [(b * math.cos(2 * math.pi * k / 96), b * math.sin(2 * math.pi * k / 96)) for k in range(96)]
    square = [(-half - h, -half - h), (half + h, -half - h), (half + h, half + h), (-half - h, half + h)]
    scene.add_structure(mw.Structure(mw.ComplexPolySlab([square, circle], bounds=(0.0, length), axis="z"),
                                     mw.Material.pec(), name="shield"))
    scene.add_structure(mw.Structure(mw.Cylinder(position=(0, 0, 0.5 * (z0 + z1)), radius=a, height=z1 - z0, axis="z"),
                                     mw.Material.pec(), name="inner"))
    scene.add_port(mw.CoaxialLumpedPort("feed", center=(0.0, 0.0, z0), inner_diameter=2 * a, outer_diameter=2 * b,
                                        normal_axis="z", reference_impedance=50.0))
    scene.add_port(mw.CoaxialLumpedPort("probe", center=(0.0, 0.0, 3.5e-3), inner_diameter=2 * a,
                                        outer_diameter=2 * b, normal_axis="z"))
    scene.add_lumped_element(mw.CoaxialLumpedResistor("load", center=(0, 0, z1), inner_diameter=2 * a,
                                                      outer_diameter=2 * b, normal_axis="z", resistance=75.0))
    return scene


def _circuit_scene(*, waveform=mw.SineWaveform(0.0, 0.01, 0.5e9)):
    circuit = mw.Circuit("driven")
    node = circuit.node("input")
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.0))
    circuit.add(mw.CurrentSource("I1", circuit.ground, node, waveform=waveform))
    circuit.bind_port("p2", positive=node, negative=circuit.ground)
    return _lumped_scene(second_port=True).clone(circuits=(circuit,))


def _box(position, size):
    return mw.Box(position=torch.tensor(position, dtype=torch.float64), size=torch.tensor(size, dtype=torch.float64))


def _copper_terminal_scene():
    # Mirrors tests/rf/terminal/test_planar_terminal_targets.py (_scene axis 2,
    # copper): a named TerminalPort between two CopperFoil sheets with a finite
    # voltage aperture, the PCB printed-line port/foil configuration in miniature.
    def sheet(name, coordinate):
        return mw.Structure(_box((0.0, 0.0, coordinate), (0.006, 0.006, 0.0)),
                            mw.CopperFoil(thickness=35e-6, frequency_range=(0.1e9, 3.0e9)), name=name)

    port = mw.TerminalPort("feed", mw.TerminalRef("signal"), mw.TerminalRef("ground"), mw.AxisPath("z"), 0.0005,
                           current_surface=_box((0.0, 0.0, 0.0005), (0.005, 0.005, 0.0)),
                           voltage_aperture=_box((0.0, 0.0, 0.0), (0.002, 0.002, 0.004)))
    return mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((-0.008, 0.008),) * 3), grid=mw.GridSpec.uniform(0.001),
                    boundary=mw.BoundarySpec.pml(num_layers=4), device="cuda",
                    structures=(sheet("signal", 0.002), sheet("ground", -0.002)), ports=(port,))


def _copper_sim(steps):
    return mw.Simulation.fdtd(_copper_terminal_scene(), frequencies=(0.8e9, 1.2e9),
                              run_time=mw.TimeConfig(time_steps=steps), full_field_dft=False,
                              spectral_sampler=mw.SpectralSampler(window="none", normalize_source=False),
                              excitations=mw.PortExcitation("feed", source_impedance=50.0,
                                                            source_time=mw.GaussianPulse(1e9, fwidth=0.8e9)))


def _wire_gap_scene():
    wire = mw.ThinWire(name="dipole", points=((-0.08, -0.08, 0.0), (-0.04, -0.04, 0.0), (0.04, 0.04, 0.0),
                                              (0.08, 0.08, 0.0)),
                       radius=2.0e-3, conductor=mw.WireConductor.pec(),
                       endpoints=(mw.WireEnd.open(), mw.WireEnd.open()), snap="strict")
    port = mw.LumpedPort("feed", wire_binding=mw.WirePortBinding.gap(negative=mw.WireNodeRef("dipole", 1),
                                                                      positive=mw.WireNodeRef("dipole", 2)),
                         reference_impedance=50.0)
    return mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((-0.12, 0.12),) * 3), grid=mw.GridSpec.uniform(0.04),
                    boundary=mw.BoundarySpec.none(), thin_wires=(wire,), ports=(port,), device="cuda")


def _wire_node_scene():
    negative = mw.ThinWire("negative_arm", ((-0.08, -0.08, 0.0), (-0.04, -0.04, 0.0)), 2.0e-3,
                           mw.WireConductor.pec(), endpoints=(mw.WireEnd.open(), mw.WireEnd.open()))
    positive = mw.ThinWire("positive_arm", ((0.04, 0.04, 0.0), (0.08, 0.08, 0.0)), 2.0e-3,
                           mw.WireConductor.pec(), endpoints=(mw.WireEnd.open(), mw.WireEnd.open()))
    port = mw.TerminalPort("feed", wire_binding=mw.WirePortBinding.nodes(
        negative=mw.WireNodeRef("negative_arm", 1), positive=mw.WireNodeRef("positive_arm", 0)))
    return mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((-0.12, 0.12),) * 3), grid=mw.GridSpec.uniform(0.04),
                    boundary=mw.BoundarySpec.none(), thin_wires=(negative, positive), ports=(port,), device="cuda")


def _resume(make_sim, stop, tmp):
    checkpoint = make_sim().prepare().run_until(stop)
    path = Path(tmp) / f"checkpoint_{stop}.pt"
    checkpoint.save(path)
    del checkpoint
    loaded = mw.FDTDResumeCheckpoint.load(path, map_location="cpu")
    return make_sim().prepare().run(resume_from=loaded)


def _custom_time():
    times = torch.linspace(0.0, 12e-9, 241, dtype=torch.float64)
    values = torch.exp(-((times - 4e-9) / 1.2e-9) ** 2) * torch.sin(2 * math.pi * 0.5e9 * times)
    return mw.CustomSourceTime(times, values, interpolation="cubic")


def fixtures():
    out = {}
    out["lumped1"] = lambda: _sim(_lumped_scene(), steps=333, excitations="p1").run()
    out["lumped1_nograph"] = lambda: _sim(_lumped_scene(), steps=333, excitations="p1", cuda_graph=False).run()
    out["lumped1_full_dft"] = lambda: _sim(_lumped_scene(), steps=333, excitations="p1", full_field_dft=True).run()
    out["lumped1_hanning"] = lambda: _sim(_lumped_scene(), steps=333, excitations="p1",
                                          spectral_sampler=mw.SpectralSampler(window="hanning")).run()
    out["lumped1_shutoff"] = lambda: _sim(_lumped_scene(), steps=900, excitations="p1", shutoff=1e-3,
                                          shutoff_check_interval=50).run()
    out["lumped1_cw"] = lambda: _sim(_lumped_scene(conductive=False), steps=257, excitations="p1",
                                     source_time=mw.CW(frequency=0.5e9), frequencies=(0.5e9,)).run()
    out["lumped1_ricker"] = lambda: _sim(_lumped_scene(conductive=False), steps=257, excitations="p1",
                                         source_time=mw.RickerWavelet(frequency=0.5e9), frequencies=(0.5e9,)).run()
    out["lumped1_custom"] = lambda: _sim(_lumped_scene(conductive=False), steps=257, excitations="p1",
                                         source_time=_custom_time(), frequencies=(0.5e9,)).run()
    out["lumped1_broadband"] = lambda: _sim(_lumped_scene(conductive=False), steps=257, excitations="p1",
                                            source_time=mw.BroadbandPulse(freq_range=(0.3e9, 0.7e9))).run()
    out["lumped2_one_driven"] = lambda: _sim(_lumped_scene(second_port=True), steps=301, excitations="p1").run()
    out["lumped2_sweep"] = lambda: _sim(_lumped_scene(second_port=True, monitors=False), steps=301,
                                        excitations=mw.PortSweep(source_time=PULSE)).run()
    out["terminal_direct"] = lambda: _sim(_terminal_lumped_scene(), steps=400,
                                          excitations=mw.PortExcitation("terminal", source_time=PULSE)).run()
    out["terminal_sweep"] = lambda: _sim(_terminal_lumped_scene(), steps=400,
                                         excitations=mw.PortSweep(source_time=PULSE)).run()
    out["coax"] = lambda: mw.Simulation.fdtd(_coax_scene(), frequencies=(1.0e9, 2.0e9, 3.0e9),
                                             excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(
                                                 frequency=2.0e9, fwidth=1.5e9)),
                                             run_time=mw.TimeConfig(time_steps=700),
                                             spectral_sampler=mw.SpectralSampler(window="none")).run()
    out["wave"] = lambda: mw.Simulation.fdtd(_wave_scene(), frequencies=(1.0e9,), excitations=mw.PortExcitation("left"),
                                             run_time=mw.TimeConfig(time_steps=700),
                                             spectral_sampler=mw.SpectralSampler(window="hanning"),
                                             full_field_dft=False).run()
    out["circuit"] = lambda: _sim(_circuit_scene(), steps=301, excitations="p1").run()
    out["circuit_ongoing"] = lambda: _sim(_circuit_scene(waveform=None), steps=301, excitations="p1",
                                         source_time=mw.CW(frequency=0.5e9)).run()
    out["circuit_transient"] = lambda: _sim(_circuit_scene(waveform=mw.SineWaveform(
        0.0, 0.01, 0.5e9, damping=1e9)), steps=301, excitations="p1").run()
    out["wire_gap"] = lambda: mw.Simulation.fdtd(_wire_gap_scene(), frequencies=(1.0e9, 1.25e9),
                                                 excitations=mw.PortExcitation("feed", source_impedance="matched",
                                                     source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.5e9)),
                                                 run_time=mw.TimeConfig(time_steps=161),
                                                 spectral_sampler=mw.SpectralSampler(window="none"),
                                                 full_field_dft=False).run()
    out["wire_node"] = lambda: mw.Simulation.fdtd(_wire_node_scene(), frequencies=(1.0e9,),
                                                  excitations=mw.PortExcitation("feed", source_time=mw.GaussianPulse(
                                                      frequency=1.0e9, fwidth=0.5e9)),
                                                  run_time=mw.TimeConfig(time_steps=161),
                                                  spectral_sampler=mw.SpectralSampler(window="none"),
                                                  full_field_dft=False).run()
    tmp = tempfile.mkdtemp(prefix="port_graph_resume_")
    lumped_sim = lambda: _sim(_lumped_scene(), steps=333, excitations="p1")
    out["resume_lumped_100"] = lambda: _resume(lumped_sim, 100, tmp)
    out["resume_lumped_64"] = lambda: _resume(lumped_sim, 64, tmp)
    out["resume_lumped_1"] = lambda: _resume(lumped_sim, 1, tmp)
    out["copper_terminal"] = lambda: _copper_sim(301).run()
    out["resume_copper_terminal_150"] = lambda: _resume(lambda: _copper_sim(301), 150, tmp)
    out["resume_copper_terminal_96"] = lambda: _resume(lambda: _copper_sim(301), 96, tmp)
    out["dipole"] = lambda: mw.Simulation.fdtd(
        mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((0.0, 1.6), (0.0, 1.0), (0.0, 1.0))), grid=mw.GridSpec.uniform(0.1),
                 boundary=mw.BoundarySpec.pml(num_layers=6),
                 structures=[mw.Structure(mw.Box(position=(0.8, 0.5, 0.5), size=(0.3, 0.3, 0.3)),
                                          mw.Material(eps_r=3.0, sigma_e=0.05), name="lossy")],
                 sources=[mw.PointDipole(position=(0.4, 0.5, 0.5), polarization="Ez", source_time=PULSE)],
                 monitors=[mw.PointMonitor("probe", (1.2, 0.5, 0.5), fields=("Ez",)),
                           *_port_time_monitors((0.6, 0.5, 0.5), (0.2, 0.2, 0.2))],
                 device="cuda"),
        frequencies=FREQS, run_time=mw.TimeConfig(time_steps=301), full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none")).run()
    def network():
        def port(index):
            x = -0.03 + index * 0.06
            return mw.LumpedPort(name=f"field_{index}", positive=(x, 0.0, 0.005), negative=(x, 0.0, -0.005),
                                 voltage_path=mw.AxisPath("z"), reference_impedance=50.0,
                                 current_surface=mw.Box(position=(x, 0.0, -0.0025), size=(0.005, 0.005, 0.0)))

        band = torch.linspace(1.0e9, 5.0e9, 17, dtype=torch.float64)
        incidence = torch.tensor((1.0, -1.0), dtype=torch.float64)
        model = mw.StateSpaceNetwork(
            A=torch.tensor(((-1.0 / (50.0 * 5.0e-12),),), dtype=torch.float64),
            B=(incidence / (50.0 * 5.0e-12)).reshape(1, 2), C=(-incidence / 50.0).reshape(2, 1),
            D=torch.outer(incidence, incidence) / 50.0 + 2.0e-3 * torch.eye(2, dtype=torch.float64),
            representation="Y", port_order=("network_0", "network_1"))
        block = mw.NetworkBlock(name="two_port_load", fit=False, model=model,
                                network=mw.NetworkData.from_y(frequencies=band, y=model.evaluate(band), z0=50.0,
                                                              port_names=model.port_order),
                                connections={"network_0": "field_0", "network_1": "field_1"})
        scene = mw.Scene(subpixel_samples=_LEGACY_SUBPIXEL, domain=mw.Domain(bounds=((-0.04, 0.04), (-0.02, 0.02), (-0.02, 0.02))),
                         grid=mw.GridSpec.uniform(0.005), boundary=mw.BoundarySpec.none(),
                         sources=(mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez",
                                                 source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.5e9)),),
                         ports=(port(0), port(1)), networks=(block,), device="cuda")
        return mw.Simulation.fdtd(scene, frequencies=(2.5e9, 3.0e9), run_time=mw.TimeConfig(time_steps=161),
                                  spectral_sampler=mw.SpectralSampler(window="none")).run()

    out["network_embedded"] = network

    def batch(batch_size):
        def run():
            sims = [_sim(_lumped_scene(monitors=False).clone(
                        monitors=[mw.PointMonitor("probe", (1.0, 0.5, 0.5), fields=("Ez",))]),
                        steps=333, excitations="p1") for _ in range(2)]
            execution = mw.BatchExecution(devices=("cuda:0",), batch_size=batch_size)
            with mw.Simulation.batch(sims, execution=execution).prepare() as prepared:
                results = prepared.run()
                return [results[i] for i in range(len(sims))]
        return run

    out["batch_b1"] = batch(1)
    out["batch_b2"] = batch(2)
    return out


def main(output, selected):
    warnings.filterwarnings("ignore", message=".*records a volume time series.*")
    payload = {}
    _install_history_hook()
    table = fixtures()
    names = selected or list(table)
    for name in names:
        result = table[name]()
        if isinstance(result, list):
            for index, item in enumerate(result):
                _record(f"{name}/lane{index}", item, payload)
            result = result[0]
        else:
            _record(name, result, payload)
        _record_history(name, payload)
        stats = result.solver_stats or {}
        payload[f"{name}/__port_graph"] = torch.tensor(bool(stats.get("port_cuda_graph_active", False)))
        print(f"{name}: steps_run={stats.get('steps_run')} field_graph={stats.get('cuda_graph_active')} "
              f"port_graph={stats.get('port_cuda_graph_active')} tensors={sum(k.startswith(name + '/') for k in payload)}",
              flush=True)
        del result
    payload["__module"] = mw.__file__
    torch.save(payload, output)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
