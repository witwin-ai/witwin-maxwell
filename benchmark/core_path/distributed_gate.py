"""Distributed gate: two-GPU joint solves against frozen payloads and single-device runs.

Every ``DIST_SCENES`` case is a two-shard ``FDTDConfig(parallel=...)`` simulation on
``cuda:0``/``cuda:1`` (in-process CUDA P2P transport). A worker captures, in one fresh
process, the distributed arm (result phasors and monitors, port/circuit/network/loss
outputs, the gathered owned Yee fields, discrete DFT/observer sample counts, the loss and
density gradient of the trainable cases, and the resumed and uninterrupted runs of the
resume cases) and the same scene run on one device through the default path. ``compare``
checks the distributed arm against the frozen payload of the existing distributed path
(``<store>/gates/distributed``) and reports the distributed-versus-single-device
deviation of both arms.

Examples::

    python -m benchmark.core_path.distributed_gate worker --scene dist_vacuum_none --output <dir>
    python -m benchmark.core_path.distributed_gate compare --output <dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import witwin.maxwell as mw

from .gates import discrete_leaf, result_tensors, tensors

DIST_SCENES = (
    "dist_vacuum_none", "dist_pec_dielectric", "dist_cpml_dense", "dist_cpml_slab_dispersive",
    "dist_dispersive_none", "dist_mixed_pec_pmc_mur", "dist_uniform_current", "dist_lumped_circuit",
    "dist_network", "dist_thin_wire", "dist_copper_foil", "dist_shutoff", "dist_gradient_standard",
    "dist_gradient_cpml", "dist_resume",
)
DEVICES = ("cuda:0", "cuda:1")
FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


@dataclass(frozen=True)
class DistCase:
    name: str
    simulation: object                 # callable(parallel: bool, steps: int | None) -> Simulation
    steps: int
    gradient: bool = False             # record loss and density gradient (trainable density)
    resume_step: int | None = None     # also run run_until(step) + run(resume_from=...)


def _parallel(*, overlap=True, gather_fields=True):
    return mw.fdtd.FDTDParallelConfig(devices=DEVICES, transport="cuda_p2p", overlap=overlap,
                                      gather_fields=gather_fields, result_device=DEVICES[0])


def _config(parallel, steps, *, dt=None, full_field_dft=True, **options):
    run_time = mw.TimeConfig(time_steps=steps) if dt is None else mw.TimeConfig(time_steps=steps, dt=dt)
    return mw.FDTDConfig(parallel=parallel, run_time=run_time, full_field_dft=full_field_dft,
                         spectral_sampler=mw.fdtd.SpectralSampler(window="none"), device="cuda:0", **options)


def _scene(x, y, z, boundary, **kwargs):
    x, y, z = (np.asarray(axis, dtype=np.float64) for axis in (x, y, z))
    return mw.Scene(domain=mw.Domain(bounds=tuple((float(axis[0]), float(axis[-1])) for axis in (x, y, z))),
                    grid=mw.GridSpec.custom(x, y, z), boundary=boundary, **kwargs)


def _dipole(position, *, polarization="Ez", name="drive", source_time=None):
    return mw.PointDipole(name=name, position=position, polarization=polarization, profile="ideal",
                          source_time=source_time or mw.GaussianPulse(frequency=1e9, fwidth=0.5e9))


def _vacuum_none(parallel, steps):
    # 10 cells split 5/5: the dipole sits on the split node; the probe sits inside rank 0's
    # last owned cell (every component interpolates across the split).
    x = np.linspace(-0.5, 0.5, 11)
    scene = _scene(x, np.linspace(-0.2, 0.2, 9), np.linspace(-0.2, 0.2, 9), mw.BoundarySpec.none(),
                   sources=[_dipole((0.0, 0.0, 0.0))],
                   monitors=[mw.PointMonitor("seam_probe", (-0.04, 0.01, 0.02), fields=FIELDS),
                             mw.PointMonitor("probe", (0.23, 0.0, 0.0), fields=FIELDS),
                             mw.PlaneMonitor(name="y_plane", axis="y", position=0.05, fields=FIELDS,
                                             frequencies=(0.8e9, 1.0e9))])
    return mw.Simulation.fdtd(scene, frequencies=(0.8e9, 1.0e9),
                              config=_config(_parallel() if parallel else None, steps))


def _pec_dielectric(parallel, steps):
    # Uneven graded x grid (11 cells, 6/5), PEC faces, a conductive dielectric across the split.
    x = (-0.55, -0.44, -0.33, -0.21, -0.10, 0.00, 0.09, 0.19, 0.30, 0.42, 0.53, 0.65)
    scene = _scene(x, np.linspace(-0.2, 0.2, 9), np.linspace(-0.2, 0.2, 9), mw.BoundarySpec.pec(),
                   structures=[mw.Structure(name="slab", geometry=mw.Box(position=(0.0, 0.0, 0.0),
                                                                        size=(0.4, 0.25, 0.3)),
                                            material=mw.Material(eps_r=3.0, sigma_e=0.01))],
                   sources=[_dipole((-0.3, 0.0, 0.05), polarization="Ey")],
                   monitors=[mw.monitors.FieldTimeMonitor("trace", components=FIELDS, position=(0.12, 0.0, 0.0),
                                                          start=2, stop=steps - 2, interval=3),
                             mw.FluxMonitor(name="flux", axis="x", position=0.25, frequencies=(1.0e9,))])
    return mw.Simulation.fdtd(scene, frequencies=1.0e9,
                              config=_config(_parallel(overlap=False) if parallel else None, steps))


def _cpml_scene(memory_mode, material, monitors):
    x = np.linspace(-0.6, 0.6, 13)
    scene = _scene(x, np.linspace(-0.4, 0.4, 9), np.linspace(-0.4, 0.4, 9),
                   mw.BoundarySpec.pml(num_layers=2, strength=1.0e6),
                   structures=[mw.Structure(name="body", geometry=mw.Box(position=(0.0, 0.0, 0.0),
                                                                        size=(0.5, 0.4, 0.4)), material=material)],
                   sources=[_dipole((-0.3, 0.0, 0.0))], monitors=monitors)
    return scene, mw.fdtd.CPMLConfig(memory_mode=memory_mode)


def _cpml_dense(parallel, steps):
    scene, cpml = _cpml_scene("dense", mw.Material(eps_r=3.0), [
        mw.monitors.ClosedSurfaceMonitor.box("surface", position=(0.0, 0.0, 0.0), size=(0.7, 0.5, 0.5),
                                             frequencies=(1.0e9, 1.2e9)),
        mw.PointMonitor("probe", (0.2, 0.05, 0.0), fields=FIELDS)])
    return mw.Simulation.fdtd(scene, frequencies=(1.0e9, 1.2e9),
                              config=_config(_parallel() if parallel else None, steps, cpml=cpml))


def _cpml_slab_dispersive(parallel, steps):
    material = mw.Material(eps_r=2.0, sigma_e=0.02, lorentz_poles=(
        mw.LorentzPole(delta_eps=0.5, resonance_frequency=2.0e9, gamma=2.0e8),))
    scene, cpml = _cpml_scene("slab", material, [
        mw.PlaneMonitor(name="z_plane", axis="z", position=0.05, fields=("Ex", "Ez", "Hy"), frequencies=(1.0e9,)),
        mw.FluxMonitor(name="flux", axis="x", position=0.3, frequencies=(1.0e9,))])
    return mw.Simulation.fdtd(scene, frequencies=1.0e9,
                              config=_config(_parallel() if parallel else None, steps, cpml=cpml))


def _dispersive_none(parallel, steps):
    x = np.linspace(-0.4, 0.4, 13)
    axis = np.linspace(-0.2, 0.2, 9)
    structures = [
        mw.Structure(name="debye", geometry=mw.Box(position=(-0.1, 0.0, 0.0), size=(0.2, 0.3, 0.3)),
                     material=mw.Material(eps_r=2.0, debye_poles=(mw.DebyePole(delta_eps=1.5, tau=2e-10),))),
        mw.Structure(name="drude", geometry=mw.Box(position=(0.1, 0.0, 0.0), size=(0.15, 0.3, 0.3)),
                     material=mw.Material(eps_r=1.5, drude_poles=(mw.DrudePole(plasma_frequency=3e9, gamma=4e8),))),
        mw.Structure(name="magnetic", geometry=mw.Box(position=(0.0, 0.1, 0.0), size=(0.3, 0.1, 0.3)),
                     material=mw.Material(eps_r=2.0, mu_lorentz_poles=(
                         mw.LorentzPole(delta_eps=0.5, resonance_frequency=2.5e9, gamma=0.3e9),))),
    ]
    scene = _scene(x, axis, axis, mw.BoundarySpec.none(), structures=structures,
                   sources=[_dipole((-0.25, 0.0, 0.0))],
                   monitors=[mw.PointMonitor("probe", (0.2, 0.0, 0.0), fields=FIELDS)])
    return mw.Simulation.fdtd(scene, frequencies=1.0e9,
                              config=_config(_parallel() if parallel else None, steps))


def _mixed_faces(parallel, steps):
    boundary = mw.BoundarySpec.faces(default="none", x=("pec", "pmc"), y=("mur", "mur"), z=("pec", "mur"))
    scene = _scene(np.linspace(-0.6, 0.6, 13), np.linspace(-0.4, 0.4, 9), np.linspace(-0.4, 0.4, 9), boundary,
                   sources=[_dipole((-0.3, 0.0, 0.0))],
                   monitors=[mw.PointMonitor("probe", (0.25, 0.1, -0.1), fields=FIELDS)])
    return mw.Simulation.fdtd(scene, frequencies=1.0e9,
                              config=_config(_parallel() if parallel else None, steps))


def _uniform_current(parallel, steps):
    scene = _scene(np.linspace(-0.6, 0.6, 13), np.linspace(-0.4, 0.4, 9), np.linspace(-0.4, 0.4, 9),
                   mw.BoundarySpec.none(),
                   sources=[mw.UniformCurrentSource(size=(0.5, 0.3, 0.3), center=(0.0, 0.0, 0.0), polarization="Ez",
                                                    source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
                                                    name="cross_interface_current")],
                   monitors=[mw.PointMonitor("probe", (0.3, 0.0, 0.0), fields=FIELDS)])
    return mw.Simulation.fdtd(scene, frequencies=1.0e9,
                              config=_config(_parallel() if parallel else None, steps))


def _port(name, x):
    return mw.LumpedPort(name=name, positive=(x, 0.0, 0.004), negative=(x, 0.0, -0.004),
                         voltage_path=mw.AxisPath("z"),
                         current_surface=mw.Box(position=(x, 0.0, -0.002), size=(0.012, 0.012, 0.0)),
                         reference_impedance=50.0)


def _lumped_circuit(parallel, steps):
    circuit = mw.circuits.Circuit("two_port")
    left, right = circuit.node("left"), circuit.node("right")
    circuit.add(mw.circuits.CurrentSource("Iexcite", left, circuit.ground, 0.0,
                                          waveform=mw.circuits.SineWaveform(0.0, 1.0e-3, 3.0e9)))
    circuit.add(mw.rf.Resistor("Rlink", left, right, 75.0))
    circuit.add(mw.rf.Resistor("Rload", right, circuit.ground, 50.0))
    circuit.bind_port("left_port", positive=left, negative=circuit.ground)
    circuit.bind_port("right_port", positive=right, negative=circuit.ground)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.024, 0.024),) * 3), grid=mw.GridSpec.uniform(0.004),
                     boundary=mw.BoundarySpec.none(), ports=(_port("left_port", -0.008), _port("right_port", 0.008)),
                     circuits=(circuit,))
    return mw.Simulation.fdtd(scene, frequencies=3.0e9,
                              config=_config(_parallel() if parallel else None, steps, full_field_dft=False))


def _network(parallel, steps):
    resistance, capacitance, shunt = 50.0, 5.0e-12, 2.0e-3
    incidence = torch.tensor((1.0, -1.0), dtype=torch.float64)
    model = mw.rf.rational.StateSpaceNetwork(
        A=torch.tensor(((-1.0 / (resistance * capacitance),),), dtype=torch.float64),
        B=(incidence / (resistance * capacitance)).reshape(1, 2), C=(-incidence / resistance).reshape(2, 1),
        D=torch.outer(incidence, incidence) / resistance + shunt * torch.eye(2, dtype=torch.float64),
        representation="Y", port_order=("net_0", "net_1"))
    frequencies = torch.linspace(0.2e9, 6.0e9, 30, dtype=torch.float64)
    data = mw.rf.NetworkData.from_y(frequencies=frequencies, y=model.evaluate(frequencies), z0=50.0,
                                    port_names=model.port_order)
    block = mw.rf.NetworkBlock(name="split_net", network=data,
                               connections={"net_0": "left_port", "net_1": "right_port"}, fit=False, model=model)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.024, 0.024),) * 3), grid=mw.GridSpec.uniform(0.004),
                     boundary=mw.BoundarySpec.none(),
                     sources=(_dipole((0.0, 0.0, 0.0), source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.8e9)),),
                     ports=(_port("left_port", -0.008), _port("right_port", 0.008)), networks=(block,))
    return mw.Simulation.fdtd(scene, frequencies=3.0e9,
                              config=_config(_parallel() if parallel else None, steps, full_field_dft=False))


def _thin_wire(parallel, steps):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.12, 0.12),) * 3), grid=mw.GridSpec.uniform(0.04),
                     boundary=mw.BoundarySpec.none(),
                     thin_wires=(
                         mw.rf.ThinWire(name="wire", points=((-0.08, 0.0, 0.0), (0.08, 0.0, 0.0)), radius=2.0e-3,
                                        conductor=mw.rf.WireConductor.pec(), snap="strict"),
                     ),
                     sources=(
                         _dipole((-0.04, 0.0, 0.0), polarization="Ex",
                                 source_time=mw.CW(frequency=2.0e9, amplitude=10.0)),
                     ),
                     monitors=(mw.rf.WireMonitor(name="wire_state", wire="wire", frequencies=(2.0e9,)),))
    return mw.Simulation.fdtd(scene, frequencies=(2.0e9,),
                              config=_config(_parallel() if parallel else None, steps))


def _copper_scene(*, parallel, steps, sampled=None, full_field_dft=True):
    structures = [
        mw.Structure(name="foil", geometry=mw.Box(position=torch.tensor((0.0, 0.0, 0.0), dtype=torch.float64),
                                                  size=torch.tensor((0.02, 0.02, 0.0), dtype=torch.float64)),
                     material=mw.media.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 2e9))),
        mw.Structure(name="substrate", geometry=mw.Box(position=(0.0, 0.0, -0.005), size=(0.02, 0.02, 0.01)),
                     material=mw.Material(eps_r=2.2, sigma_e=0.02, debye_poles=(mw.DebyePole(delta_eps=0.8, tau=1e-10),))),
    ]
    surface = mw.monitors.ClosedSurfaceMonitor.box("surface", position=(0.0, 0.0, 0.0), size=(0.04,) * 3,
                                                   frequencies=(1e9, 1.2e9), sample_spacing=sampled)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.03, 0.03),) * 3), grid=mw.GridSpec.uniform(0.005),
                     boundary=mw.BoundarySpec.pml(num_layers=3), structures=structures,
                     sources=[_dipole((-0.005, 0.0, -0.005), source_time=mw.CW(frequency=1e9))],
                     monitors=(surface,))
    parallel_config = _parallel(gather_fields=full_field_dft) if parallel else None
    return mw.Simulation.fdtd(scene, frequencies=(1e9, 1.2e9),
                              config=_config(parallel_config, steps, dt=5e-12, full_field_dft=full_field_dft))


def _copper_foil(parallel, steps):
    return _copper_scene(parallel=parallel, steps=steps, sampled=0.006)


def _shutoff(parallel, steps):
    # A zero-DC pulse (phase pi/2) lets the field energy decay below the shutoff fraction
    # (the run stops at step 271 of 640 on one device); normalize_source exercises the
    # source DFT and the DFT sample counts of a stopped run.
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(0.025),
                     boundary=mw.BoundarySpec.pml(num_layers=3),
                     structures=[mw.Structure(name="design", geometry=mw.Box(position=(0.05, 0.0, 0.0),
                                                                            size=(0.07, 0.08, 0.09)),
                                              material=mw.Material(eps_r=2.0))],
                     sources=[_dipole((-0.05, 0.0, 0.0), source_time=mw.GaussianPulse(
                         frequency=1e9, fwidth=1e9, phase=math.pi / 2))],
                     monitors=[mw.PointMonitor("probe", (0.1, 0.0, 0.0), fields=FIELDS)])
    config = _config(_parallel() if parallel else None, steps, dt=2e-11, shutoff=1e-4, shutoff_check_interval=8)
    config = replace(config, spectral_sampler=mw.fdtd.SpectralSampler(window="none", normalize_source=True))
    return mw.Simulation.fdtd(scene, frequencies=1.0e9, config=config)


def _density_scene(density, *, pml):
    if pml:
        x, boundary = np.linspace(-0.6, 0.6, 21), mw.BoundarySpec.pml(num_layers=4)
    else:
        x, boundary = np.linspace(-0.5, 0.5, 11), mw.BoundarySpec.none()
    scene = mw.Scene(subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
                     domain=mw.Domain(bounds=((float(x[0]), float(x[-1])), (-0.2, 0.2), (-0.2, 0.2))),
                     grid=mw.GridSpec.custom(x, np.linspace(-0.2, 0.2, 9), np.linspace(-0.2, 0.2, 9)),
                     boundary=boundary,
                     material_regions=(
                         mw.MaterialRegion(name="design", geometry=mw.Box(position=(0.0, 0.0, 0.0),
                                                                           size=(0.6, 0.3, 0.3)),
                                           density=density, eps_bounds=(1.0, 6.0)),
                     ),
                     sources=(
                         _dipole((-0.3, 0.0, 0.0), source_time=mw.GaussianPulse(frequency=1e9, fwidth=5.0e8)),
                     ),
                     monitors=(mw.PointMonitor("probe", (0.1, 0.0, 0.0), fields=("Ez",)),))
    return scene


def _density_values():
    generator = torch.Generator().manual_seed(1)
    return (0.4 + 0.2 * torch.rand((5, 4, 4), generator=generator)).double()


def _gradient(pml):
    def build(parallel, steps):
        density = _density_values().to("cuda:0").requires_grad_(True)
        config = _config(_parallel(overlap=False, gather_fields=False) if parallel else None, steps,
                         full_field_dft=False, adjoint_checkpoint_stride=7)
        return mw.Simulation.fdtd(_density_scene(density, pml=pml), frequencies=1.0e9, config=config), density
    return build


def _resume(parallel, steps):
    return _copper_scene(parallel=parallel, steps=steps)


def build_case(name: str) -> DistCase:
    cases = {
        "dist_vacuum_none": DistCase(name, _vacuum_none, 48),
        "dist_pec_dielectric": DistCase(name, _pec_dielectric, 48),
        "dist_cpml_dense": DistCase(name, _cpml_dense, 64),
        "dist_cpml_slab_dispersive": DistCase(name, _cpml_slab_dispersive, 64),
        "dist_dispersive_none": DistCase(name, _dispersive_none, 48),
        "dist_mixed_pec_pmc_mur": DistCase(name, _mixed_faces, 48),
        "dist_uniform_current": DistCase(name, _uniform_current, 40),
        "dist_lumped_circuit": DistCase(name, _lumped_circuit, 120),
        "dist_network": DistCase(name, _network, 120),
        "dist_thin_wire": DistCase(name, _thin_wire, 24),
        "dist_copper_foil": DistCase(name, _copper_foil, 96),
        "dist_shutoff": DistCase(name, _shutoff, 640),
        "dist_gradient_standard": DistCase(name, _gradient(False), 60, gradient=True),
        "dist_gradient_cpml": DistCase(name, _gradient(True), 60, gradient=True),
        "dist_resume": DistCase(name, _resume, 96, resume_step=47),
    }
    return cases[name]


def _outputs(result, prefix):
    payload = tensors(result_tensors(result), prefix)
    outputs = {"ports": result.ports, "conductor_losses": mw.rf.conductor_losses(result),
               "circuits": mw.circuits.circuits(result), "embedded_networks": mw.rf.embedded_networks(result)}
    payload.update(tensors(outputs, f"{prefix}/outputs"))
    stats = result.stats()
    payload.update(tensors({key: tuple(stats.get(key) or ()) for key in ("dft_sample_counts", "observer_sample_counts")},
                           f"{prefix}/counts"))
    return payload


def _fields(solver, prefix):
    if hasattr(solver, "shards"):
        values = {name: solver._gather_component(name, tuple(getattr(shard.solver, name) for shard in solver.shards))
                  for name in FIELDS}
    else:
        values = {name: getattr(solver, name) for name in FIELDS}
    return tensors(values, f"{prefix}/fields")


def _shard_state(solver, prefix):
    """Full shard storage (owned and ghost planes) and ADE/CPML state, for diagnosis only."""
    from witwin.maxwell.fdtd.checkpoint import graph_state_tensors

    payload = {}
    for shard in solver.shards:
        payload.update(tensors(dict(graph_state_tensors(shard.solver)), f"{prefix}/{shard.rank}"))
    return payload


def _meta(result):
    solver = result.solver
    stats = dict(getattr(solver, "parallel_stats", {}) or {})
    return {"execution_path": "core" if all(getattr(shard.solver, "_core_runtime", None) is not None
                                            for shard in getattr(solver, "shards", ())) else None,
            "core_shards": [getattr(shard.solver, "_core_runtime", None) is not None
                            for shard in getattr(solver, "shards", ())],
            "dft_tables": [getattr(shard.solver, "_dft_weighted_cos_table", None) is not None
                           and bool(getattr(shard.solver, "dft_enabled", False))
                           for shard in getattr(solver, "shards", ())],
            "parallel_stats": {key: stats.get(key) for key in (
                "overlap_requested", "overlap_active", "transport", "halo_bytes_per_step", "executed_steps",
                "resumed_from_step", "partitions")},
            "shutoff_step": getattr(solver, "_shutoff_step", None)}


def _run(case, *, parallel, steps):
    built = case.simulation(parallel, steps)
    simulation, density = built if case.gradient else (built, None)
    with torch.set_grad_enabled(case.gradient):
        result = simulation.run()
        payload = _outputs(result, "run")
        if case.gradient:
            spectrum = result.monitor("probe").sel(component="Ez").values
            loss = (spectrum.real ** 2 + spectrum.imag ** 2).sum()
            loss.backward()
            payload["gradient/loss"] = loss.detach().cpu().clone()
            payload["gradient/density"] = density.grad.detach().cpu().clone()
    if result.solver is not None and not case.gradient:
        payload.update(_fields(result.solver, "run"))
    meta = _meta(result) if result.solver is not None else {"execution_path": None}
    if parallel and result.solver is not None and hasattr(result.solver, "shards") and not case.gradient:
        payload.update(_shard_state(result.solver, "shard_state"))
    if case.resume_step is not None:
        checkpoint = case.simulation(parallel, steps).prepare().run_until(case.resume_step)
        resumed = case.simulation(parallel, steps).prepare().run(resume_from=checkpoint)
        payload.update(_outputs(resumed, "resumed"))
        payload.update(_fields(resumed.solver, "resumed"))
        meta["resumed"] = _meta(resumed)
    return payload, meta


def _memory_cap():
    cap = os.environ.get("DIST_GATE_MEMCAP_GIB")
    if cap:
        for index in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(
                float(cap) * 1024 ** 3 / torch.cuda.get_device_properties(index).total_memory, index)


def capture(args) -> None:
    _memory_cap()
    case = build_case(args.scene)
    destination = args.output / "candidate" / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    payload, info = {}, {}
    for arm, parallel in (("dist", True), ("single", False)):
        for horizon, steps in (("full", case.steps), ("half", case.steps // 2)):
            if arm == "single" and case.resume_step is not None and horizon == "half":
                continue
            values, meta = _run(case, parallel=parallel, steps=steps)
            payload.update({f"{arm}/{horizon}/{key}": value for key, value in values.items()})
            info[f"{arm}/{horizon}"] = meta
    torch.save(payload, destination / "tensors.pt")
    (destination / "capture.json").write_text(json.dumps({
        "gate": "distributed", "scene": args.scene, "process_id": os.getpid(), "steps": case.steps,
        "arms": info, "tensor_names": sorted(payload)}, indent=2, default=str) + "\n")


def _rows(reference, candidate):
    rows = {}
    for name, expected in reference.items():
        actual = candidate[name]
        same = actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)
        row = {"equal": bool(same), "discrete": bool(discrete_leaf(name, expected))}
        if not same and actual.shape == expected.shape and expected.numel() and not row["discrete"]:
            wide = torch.complex128 if expected.is_complex() else torch.float64
            row.update(max_abs=float((actual.to(wide) - expected.to(wide)).abs().max()),
                       max_ref=float(expected.abs().max()))
        rows[name] = row
    return rows


def _versus_single(payload, arm_prefix):
    """Distributed against single-device values for the shared output/field keys of one horizon."""
    rows = {}
    for key, value in payload.items():
        if not key.startswith(f"dist/{arm_prefix}/") or "/shard_state/" in key or "/counts/" in key:
            continue
        other = "single/" + key[len("dist/"):]
        if other not in payload or payload[other].shape != value.shape or not value.is_floating_point() and not value.is_complex():
            continue
        wide = torch.complex128 if value.is_complex() else torch.float64
        difference = float((value.to(wide) - payload[other].to(wide)).abs().max()) if value.numel() else 0.0
        rows[key] = {"max_abs": difference, "max_ref": float(payload[other].abs().max()) if value.numel() else 0.0,
                     "equal": bool(torch.equal(value, payload[other]))}
    return rows


def compare(args) -> int:
    from tests.support import frozen_oracles

    summary, failed = {}, False
    for scene in args.scenes or DIST_SCENES:
        # The frozen store (with any superseding re-pin) unless a baseline directory is given.
        directory = (Path(args.baseline) / scene if args.baseline is not None
                     else frozen_oracles.scene_directory("distributed", scene))
        candidate = torch.load(args.output / "candidate" / scene / "tensors.pt", weights_only=True)
        reference = torch.load(directory / "tensors.pt", weights_only=True)
        assert reference.keys() == candidate.keys(), (scene, sorted(reference.keys() ^ candidate.keys())[:10])
        gated = {key: value for key, value in reference.items() if "/shard_state/" not in key}
        rows = _rows(gated, candidate)
        ghosts = _rows({key: value for key, value in reference.items() if "/shard_state/" in key}, candidate)
        info = json.loads((args.output / "candidate" / scene / "capture.json").read_text())
        single = {horizon: _versus_single(candidate, horizon) for horizon in ("full", "half")}
        reference_single = {horizon: _versus_single(reference, horizon) for horizon in ("full", "half")}
        row = {
            "frozen": {"items": len(rows), "equal": sum(r["equal"] for r in rows.values())},
            "shard_state": {"items": len(ghosts), "equal": sum(r["equal"] for r in ghosts.values()),
                            "unequal": sorted(key for key, value in ghosts.items() if not value["equal"])},
            "paths": sorted({str(meta.get("execution_path")) for meta in info["arms"].values()}),
            "core_shards": info["arms"]["dist/full"].get("core_shards"),
            "dft_tables": info["arms"]["dist/full"].get("dft_tables"),
            "versus_single": {horizon: {
                "items": len(values), "equal": sum(value["equal"] for value in values.values()),
                "max_rel": max((value["max_abs"] / value["max_ref"] for value in values.values() if value["max_ref"]),
                               default=0.0),
                "same_as_reference": all(values[key]["max_abs"] == reference_single[horizon][key]["max_abs"]
                                         for key in values if key in reference_single[horizon])}
                for horizon, values in single.items()},
            "unequal": {key: value for key, value in rows.items() if not value["equal"]},
        }
        failed |= bool(row["unequal"])
        summary[scene] = row
        print(scene, row["frozen"], row["paths"], "shard_state", row["shard_state"]["equal"], "/",
              row["shard_state"]["items"], len(row["unequal"]), flush=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(failed)


def determinism(args) -> int:
    """Fresh-process captures of one tree must be bitwise equal (Rule 1)."""
    first, *others = args.dirs
    unequal, items = [], 0
    for scene in args.scenes or DIST_SCENES:
        reference = torch.load(first / "candidate" / scene / "tensors.pt", weights_only=True)
        for other in others:
            candidate = torch.load(other / "candidate" / scene / "tensors.pt", weights_only=True)
            assert reference.keys() == candidate.keys(), scene
            for key, value in reference.items():
                items += 1
                if not torch.equal(value, candidate[key]):
                    unequal.append(f"{other.name}:{scene}:{key}")
    print(f"determinism: {items} comparisons, {len(unequal)} unequal", *unequal[:20], sep="\n")
    return int(bool(unequal))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--scene", required=True, choices=DIST_SCENES)
    worker.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("compare")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--baseline", type=Path)
    check.add_argument("--scenes", nargs="*")
    repeat = sub.add_parser("determinism")
    repeat.add_argument("--dirs", nargs="+", type=Path, required=True)
    repeat.add_argument("--scenes", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "worker":
        capture(args)
        return 0
    if args.command == "determinism":
        return determinism(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
