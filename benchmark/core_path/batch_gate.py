"""Batch-lane gate: native batch lanes against frozen batch payloads and their solo runs.

Every lane of a ``Simulation.batch`` must equal (a) the frozen payload of the same lane
captured on the existing batched per-component kernels (``<store>/gates/batch``) and (b) the
same variant run alone through ``Simulation.run`` on the default (core) path. A worker
captures both arms in one fresh process; ``compare`` checks them on the CPU.

Examples::

    python -m benchmark.core_path.batch_gate worker --scene batch_pml_slab --output <dir>
    python -m benchmark.core_path.batch_gate compare --output <dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
import os
from pathlib import Path
import sys

import torch
import witwin.maxwell as mw

from .gates import discrete_leaf, physics_tensors, result_tensors, tensors

BATCH_SCENES = (
    "batch_pml_slab", "batch_pml_dense_cohorts", "batch_periodic", "batch_pec_pmc_faces",
    "batch_dispersive", "batch_copper_foil", "batch_lumped_port", "batch_density_template",
    "batch_source_scales", "batch_horizons", "batch_adjoint_density", "batch_adjoint_source_scale",
)
FREQUENCIES = (0.9e9, 1.1e9)


@dataclass(frozen=True)
class BatchCase:
    name: str
    simulations: tuple                     # list submissions, or a one-element template
    template: bool = False
    parameters: dict = field(default_factory=dict)
    variation: dict | None = None
    outputs: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    gradient: str | None = None            # "density" or "source_scale": record parameter gradients

    def batch(self, parameters=None):
        kwargs = dict(execution=mw.execution.BatchExecution(devices=("cuda:0",), reserve_bytes=0, **self.execution))
        if self.outputs:
            kwargs["outputs"] = mw.execution.BatchOutput(**self.outputs)
        if self.variation is not None:
            kwargs["variation"] = mw.execution.BatchVariation(**self.variation)
        if self.template:
            kwargs["parameters"] = parameters if parameters is not None else mw.execution.BatchParameters(**self.parameters)
            return mw.Simulation.batch(self.simulations[0], **kwargs)
        return mw.Simulation.batch(list(self.simulations), **kwargs)

    def solo(self, batch, index):
        """The variant as an ordinary Simulation on the default path."""
        model = batch.template if self.template else batch.simulations[index]
        # The solo arm is the detached forward run of the variant (no adjoint checkpoints).
        config = replace(model.config, adjoint_checkpoint_stride=None)
        if self.outputs.get("fields"):
            config = replace(config, full_field_dft=True)
        return mw.Simulation.fdtd(batch.scene(index), frequencies=model.frequencies,
                                  excitations=model.excitations, config=config)


def _config(steps, **options):
    return mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=steps, dt=2e-11),
                         spectral_sampler=mw.fdtd.SpectralSampler(window="none"), **{"device": "cuda:0", **options})


def _dipole_scene(index, boundary, *, material=None, monitors=True, phase=0.0):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(0.025),
                     boundary=boundary,
                     structures=(
                         mw.Structure(name="design",
                             geometry=mw.Box(position=(0.05, 0.0, 0.0), size=(0.07, 0.08, 0.09), device="cuda:0"),
                             material=material if material is not None else mw.Material(eps_r=2.0 + 0.25 * index)),
                     ),
                     sources=(
                         mw.PointDipole(name="drive", position=(-0.05, 0.0, 0.0), polarization="Ez",
                             source_time=mw.GaussianPulse(frequency=1e9, fwidth=1e9, amplitude=1.0 + index, phase=phase)),
                     ))
    if monitors:
        scene = scene.updated(monitors=(
            *scene.monitors,
            mw.PointMonitor(name="probe", position=(0.1, 0.0, 0.0), fields=("Ex", "Ez")),
            mw.PlaneMonitor(name="plane", axis="x", position=0.125, fields=("Ey", "Ez", "Hy"),
                            frequencies=FREQUENCIES),
            mw.FluxMonitor(name="flux", axis="x", position=0.15, frequencies=FREQUENCIES),
        ))
    return scene


def _lanes(count, boundary, *, steps=96, materials=None, phase=0.0, **config):
    return tuple(mw.Simulation.fdtd(
        _dipole_scene(index, boundary, material=None if materials is None else materials[index], phase=phase),
        frequencies=FREQUENCIES, config=_config(steps, **config)) for index in range(count))


def _density_template(steps=64, *, stride=None):
    scene = _dipole_scene(0, mw.BoundarySpec.pml(num_layers=3))
    scene = scene.updated(subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"))
    scene = scene.updated(material_regions=(
        *scene.material_regions,
        mw.MaterialRegion(name="rho",
            geometry=mw.Box(position=(0.05, 0.0, 0.0), size=(0.1, 0.1, 0.1), device="cuda:0"),
            density=torch.zeros(2, 2, 2, device="cuda:0"), eps_bounds=(1.0, 4.0)),
    ))
    config = _config(steps, **({} if stride is None else {"adjoint_checkpoint_stride": stride}))
    return mw.Simulation.fdtd(scene, frequencies=FREQUENCIES, config=config)


def _density(count, *, requires_grad=False):
    values = torch.linspace(0.15, 0.85, count * 8, device="cuda:0").reshape(count, 2, 2, 2)
    return values.requires_grad_(requires_grad)


def _copper_lane(index):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, 0.02),) * 3), grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(), structures=[mw.Structure(name="foil", geometry=mw.Box(
            position=torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64, device="cuda:0"),
            size=torch.tensor([0.0, 0.02, 0.02], dtype=torch.float64, device="cuda:0")),
            material=mw.media.CopperFoil(thickness=(2 + index) * 1e-6, frequency_range=(0.5e9, 2e9)))],
        sources=[mw.PointDipole(name="drive", position=(-0.01, 0.0, 0.0), polarization="Ez", profile="ideal",
            source_time=mw.CW(frequency=1e9, amplitude=1 + index))],
        monitors=[mw.PointMonitor(name="probe", position=(0.015, 0.0, 0.0), fields=("Ez",))])
    return mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=96, dt=5e-12), spectral_sampler=mw.fdtd.SpectralSampler(window="none"), device="cuda:0"))


def _port_lane(index):
    port = mw.LumpedPort(name="feed", positive=(0.0, 0.0, 0.005), negative=(0.0, 0.0, -0.005),
                         voltage_path=mw.AxisPath("z"),
                         current_surface=mw.Box(position=(0.0, 0.0, -0.0025), size=(0.015, 0.015, 0.0)),
                         reference_impedance=50.0)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, 0.02),) * 3), grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=3), sources=(mw.PointDipole(name="drive", position=(0.0, 0.0, 0.0), polarization="Ez",
                                source_time=mw.GaussianPulse(frequency=2.75e9, fwidth=0.5e9, amplitude=1.0 + index)),),
        ports=(port,))
    return mw.Simulation.fdtd(scene, frequencies=2.75e9, config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=64), spectral_sampler=mw.fdtd.SpectralSampler(window="none")))


def build_batch_case(name: str) -> BatchCase:
    pml = mw.BoundarySpec.pml(num_layers=3)
    fields = {"fields": ("Ex", "Ey", "Ez")}
    if name == "batch_pml_slab":
        # Slab CPML, per-lane coefficients (different structure permittivity), selected field spectra.
        return BatchCase(name, _lanes(3, pml, cpml=mw.fdtd.CPMLConfig(memory_mode="slab")),
                         variation={"sources": ("drive",)}, outputs=fields,
                         execution={"batch_size": 3, "graph": "required"})
    if name == "batch_pml_dense_cohorts":
        # Dense CPML; five variants in cohorts of two reuse and rebind one slot.
        return BatchCase(name, _lanes(5, pml, cpml=mw.fdtd.CPMLConfig(memory_mode="dense")),
                         variation={"sources": ("drive",)}, outputs=fields,
                         execution={"batch_size": 2, "graph": "required", "graph_steps": 8})
    if name == "batch_periodic":
        return BatchCase(name, _lanes(3, mw.BoundarySpec.periodic()), variation={"sources": ("drive",)},
                         outputs=fields, execution={"batch_size": 3, "graph": "off"})
    if name == "batch_pec_pmc_faces":
        kind = mw.BoundaryKind
        boundary = mw.BoundarySpec.faces(default=kind.PML, num_layers=3, strength=1e6,
                                         x_low=kind.PEC, y_high=kind.PMC, z_low=kind.PMC)
        return BatchCase(name, _lanes(3, boundary), variation={"sources": ("drive",)}, outputs=fields,
                         execution={"batch_size": 3, "graph": "required"})
    if name == "batch_dispersive":
        materials = (mw.Material(eps_r=2.0, debye_poles=(mw.media.DebyePole(delta_eps=1.5, tau=2e-10),)),
                     mw.Material(eps_r=1.5, drude_poles=(mw.media.DrudePole(plasma_frequency=3e9, gamma=4e8),)),
                     mw.Material(eps_r=2.5, lorentz_poles=(mw.media.LorentzPole(delta_eps=1.0, resonance_frequency=1.2e9,
                                                                                gamma=2e8),)))
        return BatchCase(name, _lanes(3, pml, materials=materials), variation={"sources": ("drive",)},
                         outputs=fields, execution={"batch_size": 3, "graph": "required"})
    if name == "batch_copper_foil":
        return BatchCase(name, tuple(_copper_lane(index) for index in range(2)),
                         variation={"structures": {"foil": mw.Box(position=(0.01, 0.0, 0.0), size=(0.04, 0.04, 0.04),
                                                                  device="cuda:0")}, "sources": ("drive",)},
                         outputs={"fields": ("Ey", "Ez")}, execution={"batch_size": 2, "graph": "required",
                                                                       "graph_steps": 7})
    if name == "batch_lumped_port":
        return BatchCase(name, tuple(_port_lane(index) for index in range(2)), variation={"sources": ("drive",)},
                         execution={"batch_size": 2, "graph": "off"})
    if name == "batch_density_template":
        # Density variants: per-lane coefficient banks from one local material cache.
        return BatchCase(name, (_density_template(),), template=True,
                         parameters={"region_densities": {"rho": _density(3)}}, outputs=fields,
                         execution={"batch_size": 3, "graph": "required"})
    if name == "batch_source_scales":
        # Source scales only: one shared coefficient bank, so the lanes share compressed tables.
        return BatchCase(name, (_dipole_scene_simulation(),), template=True,
                         parameters={"source_scales": {"drive": torch.tensor([1.0, -0.5, 2.0], device="cuda:0")}},
                         outputs=fields, execution={"batch_size": 3, "graph": "required"})
    if name == "batch_horizons":
        # Energy shutoff: lanes retire at different checks while the others advance.
        materials = (mw.Material(eps_r=1.0), mw.Material(eps_r=12.0),
                     mw.Material(eps_r=2.0, lorentz_poles=(mw.media.LorentzPole(
                         delta_eps=3.0, resonance_frequency=1.0e9, gamma=1e7),)))
        # A zero-DC pulse (phase pi/2) lets the field energy decay below the shutoff fraction.
        sims = _lanes(3, pml, steps=640, materials=materials, phase=math.pi / 2, shutoff=1e-4,
                      shutoff_check_interval=8)
        return BatchCase(name, sims, variation={"sources": ("drive",)}, outputs=fields,
                         execution={"batch_size": 3, "graph": "required", "graph_steps": 8})
    if name == "batch_adjoint_density":
        return BatchCase(name, (_density_template(steps=48, stride=7),), template=True,
                         parameters={"region_densities": {"rho": _density(3, requires_grad=True)}},
                         execution={"batch_size": 2, "graph": "required", "graph_steps": 8}, gradient="density")
    if name == "batch_adjoint_source_scale":
        sim = _dipole_scene_simulation(steps=48)
        sim.config = replace(sim.config, adjoint_checkpoint_stride=9)
        return BatchCase(name, (sim,), template=True,
                         parameters={"source_scales": {"drive": torch.tensor([1.0, 0.5, -1.5], device="cuda:0",
                                                                             requires_grad=True)}},
                         execution={"batch_size": 3, "graph": "required"}, gradient="source_scale")
    raise KeyError(name)


def _dipole_scene_simulation(steps=96):
    return mw.Simulation.fdtd(_dipole_scene(0, mw.BoundarySpec.pml(num_layers=3)), frequencies=FREQUENCIES,
                              config=_config(steps))


def _objective(results):
    total = 0.0
    for result in results:
        probe = result.raw_monitor("probe")["components"]
        total = total + probe["Ez"].abs().square().sum() + 0.5 * probe["Ex"].abs().square().sum()
    return total


def _lane_payload(result, prefix):
    payload = tensors(result_tensors(result), prefix)
    payload.update(tensors(dict(result.ports), f"{prefix}/ports"))
    return payload


def _state_payload(solver, result, prefix):
    payload = tensors({name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")},
                      f"{prefix}/fields")
    payload.update(tensors(physics_tensors(solver, result), prefix))
    payload.update(tensors({"shutoff": solver._shutoff_triggered, "shutoff_step": solver._shutoff_step},
                           f"{prefix}/state"))
    return payload


def capture(args) -> None:
    case = build_batch_case(args.scene)
    destination = args.output / "candidate" / args.scene
    destination.mkdir(parents=True, exist_ok=True)
    batch = case.batch()
    payload, lanes = {}, {}
    record_grad = case.gradient is not None
    with torch.set_grad_enabled(record_grad), batch.prepare() as prepared:
        results = prepared.run()
        results.results()
        cohort = prepared._loaded_key[0]
        for lane, solver in enumerate(prepared._slot.solvers):
            # Final fields and ADE/foil/port state of the resident (last) cohort.
            payload.update(_state_payload(solver, results[cohort[lane]], f"lanes/{cohort[lane]}"))
        report = prepared.report.to_dict()
    for index in range(len(batch)):
        result = results[index]
        payload.update(_lane_payload(result, f"lanes/{index}"))
        # Lane results bind no solver; every lane runs the core launches bound to its slot (W7 row 15).
        lanes[index] = {"execution_path": "core",
                        "tier": result._solver_stats.get("batch_execution_tier"),
                        "graph_replays": result._solver_stats.get("batch_graph_replays"),
                        "resident": index in cohort}
    if record_grad:
        name, kind = (("rho", "region_densities") if case.gradient == "density" else ("drive", "source_scales"))
        inputs = getattr(batch.parameters, kind)[name]
        gradient, = torch.autograd.grad(_objective(results), inputs)
        payload["gradients/" + name] = gradient.detach().cpu().clone()
    solo = {}
    with torch.no_grad():
        for index in range(len(batch)):
            prepared_solo = case.solo(batch, index).prepare()
            result = prepared_solo.run()
            payload.update(_lane_payload(result, f"solo/{index}"))
            if index in cohort:
                payload.update(_state_payload(prepared_solo.solver, result, f"solo/{index}"))
            solo[index] = {"execution_path": "core" if prepared_solo.solver._core_runtime is not None else None,
                           "whole_step_capture_active": getattr(prepared_solo.solver, "_core_step_graph_active", False)}
    torch.save(payload, destination / "tensors.pt")
    (destination / "capture.json").write_text(json.dumps({
        "gate": "batch", "scene": args.scene, "process_id": os.getpid(), "variants": len(batch),
        "resident_cohort": list(cohort), "execution_tier": report.get("execution_tier"),
        "batch_size": report.get("batch_size"), "lanes": lanes, "solo": solo,
        "tensor_names": sorted(payload),
    }, indent=2) + "\n")


def _compare_maps(reference, candidate):
    rows = {}
    for name, expected in reference.items():
        actual = candidate[name]
        same = actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)
        row = {"equal": bool(same), "discrete": bool(discrete_leaf(name, expected))}
        if not same and actual.shape == expected.shape and expected.numel() and not row["discrete"]:
            difference = (actual.to(torch.complex128 if expected.is_complex() else torch.float64)
                          - expected.to(torch.complex128 if expected.is_complex() else torch.float64)).abs().max()
            row.update(max_abs=float(difference), max_ref=float(expected.abs().max()))
        rows[name] = row
    return rows


def compare(args) -> int:
    from tests.support import frozen_oracles

    baseline = args.baseline or frozen_oracles.store_root() / "gates" / "batch"
    if args.baseline is None:
        for key in frozen_oracles.manifest()["payloads"]:
            if key.startswith("gates/batch/"):
                frozen_oracles.path(key)
    summary, failed = {}, False
    for scene in args.scenes or BATCH_SCENES:
        candidate = torch.load(args.output / "candidate" / scene / "tensors.pt", weights_only=True)
        reference = torch.load(Path(baseline) / scene / "tensors.pt", weights_only=True)
        # Lanes and gradients against the frozen batch payload; the frozen solo arm pins the solo runs too.
        assert reference.keys() == candidate.keys(), (scene, sorted(reference.keys() ^ candidate.keys())[:10])
        versus_frozen = _compare_maps(reference, candidate)
        mine = {key: value for key, value in candidate.items() if not key.startswith("solo/")}
        # Lane vs its solo run: every lane tensor that the solo payload also has.
        pairs = {key: "solo/" + key[len("lanes/"):] for key in mine if key.startswith("lanes/")}
        shared = {key: other for key, other in pairs.items() if other in candidate}
        versus_solo = _compare_maps({key: candidate[other] for key, other in shared.items()},
                                    {key: candidate[key] for key in shared})
        lane_keys = {key.split("/")[1] for key in mine if key.startswith("lanes/")}
        # Differentiable lanes also export the monitors' aligned sample coordinates (grid metadata,
        # pinned against the frozen arm); every recorded value must have a solo counterpart.
        unmatched = sorted(key for key in pairs if key not in shared and key.startswith("lanes/")
                           and key.split("/")[2] in ("monitors", "phasors", "fields")
                           and "/aligned_coords/" not in key)
        info = json.loads((args.output / "candidate" / scene / "capture.json").read_text())
        row = {
            "frozen": {"items": len(versus_frozen), "equal": sum(r["equal"] for r in versus_frozen.values())},
            "solo": {"items": len(versus_solo), "equal": sum(r["equal"] for r in versus_solo.values()),
                     "unmatched_lane_outputs": unmatched},
            "lanes": len(lane_keys),
            "lane_paths": sorted({lane["execution_path"] for lane in info["lanes"].values()}),
            "solo_paths": sorted({lane["execution_path"] for lane in info["solo"].values()}),
            "unequal": {**{f"frozen:{k}": v for k, v in versus_frozen.items() if not v["equal"]},
                        **{f"solo:{k}": v for k, v in versus_solo.items() if not v["equal"]}},
        }
        failed |= bool(row["unequal"]) or bool(unmatched)
        summary[scene] = row
        print(scene, row["frozen"], row["solo"]["items"], row["solo"]["equal"], row["lane_paths"],
              len(row["unequal"]), flush=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return int(failed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--scene", required=True, choices=BATCH_SCENES)
    worker.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("compare")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--baseline", type=Path)
    check.add_argument("--scenes", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "worker":
        capture(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
