"""Forward-tape and gradient gates for differentiable FDTD runs (W7 row 14).

A differentiable run records a forward tape: the adjoint checkpoint states at the
checkpoint stride and the bridge's forward output tensors. The reverse sweep replays
segments from those checkpoints, so the tape and the gradients together pin a
gradient run. Each capture runs one scene in a fresh process and writes
``<output>/<arm>/<scene>/{full,half}.pt`` (the half horizon for Rule 2 run-level
comparisons) plus ``meta.json`` (execution path, graph chunks, checkpoint steps).
The baseline arm is the frozen existing-path capture (store key
``gates/grad/<scene>/...``); ``--baseline`` points a comparison at another directory.

Examples::

    python -m benchmark.core_path.grad_gates capture --output <evidence-root>/grad --arm candidate
    python -m benchmark.core_path.grad_gates compare --output <evidence-root>/grad
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import witwin.maxwell as mw


def _dipole(position=(0.0, 0.0, -0.18)):
    return mw.PointDipole(position=position, polarization="Ez", width=0.04,
                          source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0))


def _probe(position=(0.0, 0.0, 0.18)):
    return mw.PointMonitor("probe", position, fields=("Ez",))


def _design(density, *, position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24), eps_bounds=(1.0, 6.0)):
    return mw.MaterialRegion(name="design", geometry=mw.Box(position=position, size=size),
                             density=density, eps_bounds=eps_bounds, mu_bounds=(1.0, 1.0))


def _density(shape=(2, 2, 2), value=0.4):
    return torch.full(shape, value, device="cuda", requires_grad=True)


# Density gradients need arithmetic sampling (the default kottke_edge refuses them).
ARITHMETIC = {"subpixel": mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")}


def _cube(boundary, *, half=0.6, spacing=0.12, **collections):
    return mw.Scene(domain=mw.Domain(bounds=((-half, half),) * 3), grid=mw.GridSpec.uniform(spacing),
                    boundary=boundary, **ARITHMETIC, **collections)


def _config(steps, *, stride=None, full_field_dft=False, cuda_graph=True, dt=None, window="none", **options):
    return mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=steps, dt=dt), full_field_dft=full_field_dft,
                         spectral_sampler=mw.fdtd.SpectralSampler(window=window), cuda_graph=cuda_graph,
                         adjoint_checkpoint_stride=stride, **options)


class _EpsModule(mw.SceneModule):
    """SceneModule trainable (materialized without a graph in the bridge forward)."""

    def __init__(self, time_monitor=False):
        super().__init__()
        self.eps_r = torch.nn.Parameter(torch.tensor(3.5, device="cuda"))
        self.time_monitor = time_monitor

    def to_scene(self):
        monitors = [_probe()]
        if self.time_monitor:
            monitors.append(mw.monitors.FieldTimeMonitor("trace", components=("Ez",),
                                                         position=(0.0, 0.0, 0.06), interval=1))
        return _cube(mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                     structures=[mw.Structure(geometry=mw.Box(position=(0.06, 0.06, 0.18), size=(0.24, 0.24, 0.24)),
                                              material=mw.Material(eps_r=self.eps_r))],
                     sources=[_dipole()], monitors=monitors)


def _module_case(steps, *, time_monitor=False, **config):
    module = _EpsModule(time_monitor=time_monitor)
    return mw.Simulation.fdtd(module, frequencies=[1e9], config=_config(steps, **config)), list(module.parameters())


def _pec_scalar(steps, **config):
    eps = torch.tensor(3.0, device="cuda", requires_grad=True)
    scene = _cube(mw.BoundarySpec.pec(),
                  structures=[mw.Structure(geometry=mw.Box(position=(0.06, 0.0, 0.12), size=(0.36, 0.36, 0.24)),
                                           material=mw.Material(eps_r=eps))],
                  sources=[_dipole()], monitors=[_probe()])
    return mw.Simulation.fdtd(scene, frequencies=[1e9], config=_config(steps, **config)), [eps]


def _density_fields(steps):
    density = _density()
    scene = _cube(mw.BoundarySpec.pml(num_layers=2, strength=1.0), material_regions=[_design(density)],
                  sources=[_dipole()],
                  monitors=[_probe(), mw.PlaneMonitor("plane", axis="z", position=0.3, fields=("Ex", "Ez"))])
    return mw.Simulation.fdtd(scene, frequencies=[0.9e9, 1.1e9],
                              config=_config(steps, full_field_dft=True)), [density]


def _tfsf(steps):
    density = _density((2, 1, 1), 0.5)
    scene = mw.Scene(
        **ARITHMETIC,
        domain=mw.Domain(bounds=((-1.2, 1.2),) * 3), grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        material_regions=[_design(density, position=(0.06, 0.0, 0.0), size=(0.24, 0.12, 0.12))],
        sources=[mw.PlaneWave(name="tfsf_pw", direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0),
                              source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.25e9, amplitude=40.0),
                              injection=mw.TFSF(bounds=((-0.36, 0.36),) * 3))],
        monitors=[mw.PointMonitor("probe", (0.18, 0.0, 0.0), fields=("Ez",))])
    return mw.Simulation.fdtd(scene, frequencies=[1e9], config=_config(steps)), [density]


def _static_slab(material, size=(0.60, 0.60, 0.12)):
    return mw.Structure(name="slab", geometry=mw.Box(position=(0.0, 0.0, 0.0), size=size), material=material)


def _slab_case(steps, material, size=(0.60, 0.60, 0.12)):
    density = _density()
    scene = _cube(mw.BoundarySpec.pml(num_layers=2, strength=1.0), structures=[_static_slab(material, size)],
                  material_regions=[_design(density)], sources=[_dipole()], monitors=[_probe()])
    return mw.Simulation.fdtd(scene, frequencies=[1e9], config=_config(steps)), [density]


def _lorentz_pole(steps):
    scale = torch.tensor(1.0, device="cuda", requires_grad=True)
    pole = mw.LorentzPole(delta_eps=2.0 * scale, resonance_frequency=1.8e9, gamma=2e8)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, 0.3),) * 3), grid=mw.GridSpec.uniform(0.06),
                     boundary=mw.BoundarySpec.pec(),
                     structures=[mw.Structure(geometry=mw.Sphere(radius=0.143),
                                              material=mw.Material(eps_r=2.0, lorentz_poles=(pole,)))],
                     sources=[mw.PointDipole(position=(-0.12, 0.0, 0.0), polarization="Ez", width=0.06,
                                             source_time=mw.CW(frequency=1e9))])
    return mw.Simulation.fdtd(scene, frequencies=(1e9,), config=_config(steps, dt=3e-12, full_field_dft=True)), [scale]


def _port(termination=None):
    return mw.LumpedPort(name="feed", positive=(0.0, 0.0, 0.005), negative=(0.0, 0.0, -0.005),
                         voltage_path=mw.AxisPath("z"),
                         current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
                         reference_impedance=50.0, termination=termination)


def _port_scene(port, **collections):
    return mw.Scene(domain=mw.Domain(bounds=((-0.02, 0.02),) * 3), grid=mw.GridSpec.uniform(0.005),
                    boundary=mw.BoundarySpec.none(), ports=(port,), **collections)


def _lumped_port(steps):
    resistance = torch.tensor(40.0, device="cuda", requires_grad=True)
    scene = _port_scene(_port(mw.rf.SeriesRLC(r=resistance)), sources=[mw.PointDipole(
        position=(0.005, 0.0, 0.0), polarization="Ez", width=0.002, source_time=mw.CW(frequency=3.0e9, amplitude=1000.0))])
    return mw.Simulation.fdtd(scene, frequencies=3.0e9, config=_config(steps)), [resistance]


def _circuit_port(steps):
    resistance = torch.tensor(25.0, device="cuda", requires_grad=True)
    circuit = mw.circuits.Circuit("gradient_network")
    source, middle, output = (circuit.node(name) for name in ("input", "middle", "output"))
    circuit.add(mw.rf.Resistor("R1", source, middle, resistance))
    circuit.add(mw.rf.Inductor("L1", middle, output, 0.5e-9))
    circuit.add(mw.rf.Capacitor("C1", output, circuit.ground, 1.0e-12))
    circuit.add(mw.circuits.CurrentSource("I1", circuit.ground, source,
                                          waveform=mw.circuits.SineWaveform(0.0, 0.01, 3.0e9)))
    circuit.bind_port("feed", positive=source, negative=circuit.ground)
    scene = _port_scene(_port(), circuits=(circuit,))
    return mw.Simulation.fdtd(scene, frequencies=3.0e9, config=_config(steps)), [resistance]


def _mur(steps):
    density = _density()
    scene = _cube(mw.BoundarySpec.mur(), material_regions=[_design(density)], sources=[_dipole()],
                  monitors=[_probe()])
    return mw.Simulation.fdtd(scene, frequencies=[1e9], config=_config(steps, stride=13)), [density]


def _thin_wire(steps):
    density = _density((1, 1, 1), 0.5)
    scene = mw.Scene(
        **ARITHMETIC,
        domain=mw.Domain(bounds=((-0.12, 0.12),) * 3), grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.none(),
        material_regions=[mw.MaterialRegion(name="host", geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.48,) * 3),
                                            density=density, eps_bounds=(1.0, 3.0), mu_bounds=(1.0, 1.0))],
        thin_wires=[mw.rf.ThinWire(name="wire", points=((0.0, 0.0, -0.08), (0.0, 0.0, 0.08)), radius=0.004,
                                   conductor=mw.rf.WireConductor.pec())],
        sources=[mw.PointDipole(name="drive", position=(0.04, 0.0, 0.0), polarization="Ez", width=0.04,
                                source_time=mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9, amplitude=10.0))],
        monitors=[mw.PointMonitor("probe", (0.0, 0.04, 0.0), fields=("Ez",))])
    return mw.Simulation.fdtd(scene, frequencies=[2.0e9], config=_config(steps, stride=4)), [density]


def _guarded_cpml(steps):
    # PEC-closed transverse faces with an explicit CPML absorber: the passivity guard samples.
    density = _density()
    scene = _cube(mw.BoundarySpec.faces(default="pec", num_layers=4, z="pml"), material_regions=[_design(density)],
                  sources=[_dipole()], monitors=[_probe()])
    return mw.Simulation.fdtd(scene, frequencies=[1e9], config=_config(steps, absorber="cpml")), [density]


def _bloch(steps):
    import math

    density = _density((2, 1, 1), 0.5)
    scene = mw.Scene(
        **ARITHMETIC,
        domain=mw.Domain(bounds=((-0.6, 0.6),) * 3), grid=mw.GridSpec.uniform(0.15),
        boundary=mw.BoundarySpec.bloch((math.pi / 1.2, 0.0, 0.0)),
        material_regions=[_design(density, position=(0.075, 0.0, 0.0), size=(0.30, 0.15, 0.15))],
        sources=[mw.PointDipole(position=(-0.3, 0.0, 0.0), polarization="Ez", width=0.06,
                                source_time=mw.GaussianPulse(frequency=1e9, fwidth=0.25e9, amplitude=50.0))],
        monitors=[mw.PointMonitor("probe", (0.3, 0.0, 0.0), fields=("Ez",))])
    return mw.Simulation.fdtd(scene, frequencies=[1e9], config=_config(steps)), [density]


def _bloch_copper_foil(steps):
    # W7 row 16b: trainable foil thickness and conductivity on a Bloch-periodic copper foil with a PML x axis
    # (complex fields beside separate physics; the copper-foil adjoint fixture's foil and drive).
    thickness = torch.tensor(1e-6, device="cuda", dtype=torch.float64, requires_grad=True)
    conductivity = torch.tensor(1e6, device="cuda", dtype=torch.float64, requires_grad=True)
    foil = mw.media.CopperFoil(thickness=thickness, conductivity=conductivity, frequency_range=(2e9, 4e9), mode_count=16)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.03, 0.03),) * 3), grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.faces(default="bloch", x="pml", num_layers=2, bloch_wavevector=(0.0, 7.0, 11.0)),
        structures=[mw.Structure(mw.Box(position=(0.0, 0.0, 0.0), size=(0.0, 0.06, 0.06)), foil, name="foil")],
        sources=[mw.UniformCurrentSource(name="drive", center=(-0.015, 0.0, -0.0025), size=(0.005, 0.005, 0.005),
                                         polarization="Ez", source_time=mw.CW(3e9, amplitude=1e4))],
        monitors=[mw.PointMonitor("probe", (0.01, 0.0, 0.0), fields=("Ey", "Ez"))])
    return mw.Simulation.fdtd(scene, frequencies=3e9, config=_config(steps, stride=7)), [thickness, conductivity]


# name -> (build(time_steps) -> (simulation, trainable leaves), full-horizon steps)
GRADIENT_SCENES = {
    "pec_scalar_eps": (_pec_scalar, 64),
    "pml_module_eps": (_module_case, 64),
    "pml_density_fields": (_density_fields, 64),
    "tfsf_density": (_tfsf, 64),
    "lorentz_pole": (_lorentz_pole, 96),
    "conductive_slab": (lambda steps: _slab_case(steps, mw.Material(eps_r=1.0, sigma_e=0.05), (1.2, 1.2, 0.12)), 64),
    "kerr_slab": (lambda steps: _slab_case(steps, mw.Material(eps_r=2.0, kerr_chi3=1e-10)), 64),
    "full_aniso_slab": (lambda steps: _slab_case(steps, mw.Material(epsilon_tensor=mw.media.Tensor3x3(
        ((4.0, 0.0, 1.2), (0.0, 4.0, 1.2), (1.2, 1.2, 5.0)))), (0.36, 0.36, 0.12)), 64),
    "lumped_port_rlc": (_lumped_port, 32),
    "circuit_port": (_circuit_port, 16),
    "time_monitor": (lambda steps: _module_case(steps, time_monitor=True), 64),
    "mur_density": (_mur, 96),
    "stride_odd": (lambda steps: _pec_scalar(steps, stride=5), 37),
    "eager_graphs_off": (lambda steps: _module_case(steps, cuda_graph=False), 64),
    "thin_wire": (_thin_wire, 48),
    "guarded_cpml": (_guarded_cpml, 96),
    "bloch_density": (_bloch, 48),
    "bloch_copper_foil": (_bloch_copper_foil, 48),
}


def capture_scene(name: str, steps: int) -> tuple[dict[str, torch.Tensor], dict]:
    """Run one differentiable scene and return its tape, loss and gradients."""
    from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge

    build, _ = GRADIENT_SCENES[name]
    simulation, leaves = build(steps)
    bridges, outputs = [], []
    accept, rebuild = _FDTDGradientBridge.accept_forward, _FDTDGradientBridge.rebuild_forward_outputs

    def accept_spy(bridge, *args, **kwargs):
        bridges.append(bridge)
        return accept(bridge, *args, **kwargs)

    def rebuild_spy(bridge, output_tensors):
        outputs.append(tuple(output_tensors))
        return rebuild(bridge, output_tensors)

    with patch.object(_FDTDGradientBridge, "accept_forward", accept_spy), \
            patch.object(_FDTDGradientBridge, "rebuild_forward_outputs", rebuild_spy):
        result = simulation.run()
    assert len(bridges) == 1 and len(outputs) == 1, (len(bridges), len(outputs))
    bridge, tensors = bridges[0], outputs[0]
    terms = [tensor.abs().square().sum() for tensor in tensors
             if (tensor.is_floating_point() or tensor.is_complex()) and tensor.requires_grad]
    loss = sum(term / term.detach().clamp_min(1e-30) * (index + 1) for index, term in enumerate(terms))
    gradients = torch.autograd.grad(loss, leaves, allow_unused=True)
    payload = {"loss": loss.detach().cpu()}
    for index, tensor in enumerate(tensors):
        payload[f"outputs/{index}"] = tensor.detach().cpu()
    for index, checkpoint in enumerate(bridge._last_checkpoints):
        payload[f"checkpoints/{index}/step"] = torch.tensor(checkpoint.step)
        for key, tensor in checkpoint.tensors.items():
            payload[f"checkpoints/{index}/{key}"] = tensor.detach().cpu()
    for field in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        payload[f"final/{field}"] = getattr(result.solver, field).detach().cpu()
    for index, gradient in enumerate(gradients):
        assert gradient is not None, f"{name}: gradient {index} dropped"
        payload[f"gradients/{index}"] = gradient.detach().cpu()
    solver = result.solver
    meta = {
        "scene": name, "steps": steps,
        "passivity_guard_status": getattr(solver, "passivity_guard_status", None),
        "execution_path": "core" if getattr(solver, "_core_runtime", None) is not None else None,
        "core_step_graph_active": bool(getattr(solver, "_core_step_graph_active", False)),
        "core_step_graph_chunks": list(getattr(solver, "_core_step_graph_chunks", [])),
        "checkpoint_steps": [checkpoint.step for checkpoint in bridge._last_checkpoints],
        "outputs": len(tensors), "gradients": len(gradients),
        "loss": float(loss),
    }
    return payload, meta


def worker(args) -> None:
    if args.cap_gib is not None:
        torch.cuda.set_per_process_memory_fraction(
            args.cap_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
    _, steps = GRADIENT_SCENES[args.scene]
    destination = args.output / args.arm / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    metas = {}
    for horizon, count in (("full", steps), ("half", steps // 2)):
        payload, meta = capture_scene(args.scene, count)
        torch.save(payload, destination / f"{horizon}.pt")
        metas[horizon] = meta
    (destination / "meta.json").write_text(json.dumps(metas, indent=1) + "\n")
    print(json.dumps(metas["full"]))


def capture(args) -> None:
    failed = []
    for name in args.scenes or GRADIENT_SCENES:
        command = [sys.executable, "-m", "benchmark.core_path.grad_gates", "worker", "--scene", name,
                   "--output", str(args.output), "--arm", args.arm]
        if args.cap_gib is not None:
            command += ["--cap-gib", str(args.cap_gib)]
        rc = subprocess.run(command, env=os.environ).returncode
        print(f"{name} rc={rc}", flush=True)
        if rc:
            failed.append(name)
    sys.exit(1 if failed else 0)


def frozen_payload(scene: str, horizon: str) -> dict[str, torch.Tensor]:
    from tests.support import frozen_oracles

    return frozen_oracles.load(f"gates/grad/{scene}/{horizon}.pt")


def compare_payloads(reference, candidate, *, ref_half=None, new_half=None):
    """Bitwise first; Rule 2 run-level receipts for continuous movers."""
    assert reference.keys() == candidate.keys(), sorted(reference.keys() ^ candidate.keys())
    unequal = [key for key in reference if reference[key].dtype != candidate[key].dtype
               or reference[key].shape != candidate[key].shape or not torch.equal(reference[key], candidate[key])]
    report = {"tensors": len(reference), "unequal": unequal, "tier": "T0" if not unequal else None}
    if unequal and ref_half is not None:
        from benchmark.core_path.gates import rule2_rows, tier_summary

        rows = rule2_rows({key: reference[key] for key in unequal}, {key: candidate[key] for key in unequal},
                          level="run", ref_half={key: ref_half[key] for key in unequal},
                          new_half={key: new_half[key] for key in unequal})
        report["rule2"] = rows
        summary = tier_summary(rows)
        report["tier"] = "T2" if summary["T2"] else "T1"
    return report


def compare(args) -> None:
    results, failed = {}, []
    for name in args.scenes or GRADIENT_SCENES:
        root = args.output / args.arm / name
        load = (lambda horizon: torch.load(args.baseline / name / f"{horizon}.pt", weights_only=True)) \
            if args.baseline is not None else (lambda horizon: frozen_payload(name, horizon))
        reference, ref_half = load("full"), load("half")
        candidate = torch.load(root / "full.pt", weights_only=True)
        new_half = torch.load(root / "half.pt", weights_only=True)
        report = compare_payloads(reference, candidate, ref_half=ref_half, new_half=new_half)
        report["meta"] = json.loads((root / "meta.json").read_text())["full"]
        results[name] = report
        print(f"{name}: {report['tensors']} tensors, {len(report['unequal'])} unequal, {report['tier']}, "
              f"path={report['meta']['execution_path']} graph={report['meta']['core_step_graph_active']}", flush=True)
        if report["tier"] != "T0" and report["tier"] != "T1":
            failed.append(name)
    (args.output / args.arm / "comparison.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    sys.exit(1 if failed else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("capture", "worker", "compare"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--arm", default="candidate")
        if command == "worker":
            child.add_argument("--scene", required=True, choices=tuple(GRADIENT_SCENES))
        else:
            child.add_argument("--scenes", nargs="*", choices=tuple(GRADIENT_SCENES))
        if command != "compare":
            child.add_argument("--cap-gib", type=float, help="per-process CUDA memory cap")
        if command == "compare":
            child.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    {"capture": capture, "worker": worker, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
