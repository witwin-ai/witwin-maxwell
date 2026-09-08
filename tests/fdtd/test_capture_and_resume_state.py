"""Nonzero-state capture and schema-complete persistence regressions."""
from dataclasses import fields

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import FDTDCheckpointSchema, FDTDCheckpointState


@pytest.mark.parametrize("family", ["network_state_names", "network_delay_state_names", "wire_state_names"])
def test_resume_disk_round_trip_keeps_every_physics_state_family(tmp_path, family):
    values = {field.name: () for field in fields(FDTDCheckpointSchema) if field.name != "version"}
    values["field_names"] = ("Ex",)
    values[family] = ("auxiliary",)
    schema = FDTDCheckpointSchema(version=2, **values)
    checkpoint = mw.FDTDResumeCheckpoint(step=3, total_steps=8, dt=1e-12,
        physics=FDTDCheckpointState(3, schema, {"Ex": torch.tensor([1.]), "auxiliary": torch.tensor([2.])}),
        signature={}, auxiliary={})
    path = tmp_path / "state.pt"
    checkpoint.save(path)
    restored = mw.FDTDResumeCheckpoint.load(path)
    assert restored.physics.schema == schema
    for name, value in checkpoint.physics.tensors.items():
        assert torch.equal(restored.physics.tensors[name], value)


def _solver(kind, *, mur=False):
    materials = {
        "debye": lambda: mw.Material.debye(eps_inf=2., delta_eps=3., tau=2e-10),
        "drude": lambda: mw.Material.drude(eps_inf=2., plasma_frequency=2e9, gamma=1e8),
        "lorentz": lambda: mw.Material.lorentz(eps_inf=2., delta_eps=3., resonance_frequency=2e9, gamma=1e8),
        "magnetic": lambda: mw.Material(mu_lorentz_poles=(mw.LorentzPole(delta_eps=3., resonance_frequency=2e9, gamma=1e8),)),
    }
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1, 0.1),) * 3), grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.faces(default="mur") if mur else mw.BoundarySpec.pec(), device="cuda")
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0., 0., 0.), size=(0.2,) * 3), material=materials[kind]()))
    scene.add_source(mw.PointDipole(position=(0., 0., 0.), polarization="Ez", source_time=mw.CW(frequency=1e9)))
    return mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=32)).prepare().solver


def _physical_tensors(solver):
    tensors = {name: getattr(solver, name) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    for family in ("_dispersive_templates", "_magnetic_dispersive_templates"):
        for component, templates in getattr(solver, family).items():
            for kind in ("debye", "drude", "lorentz"):
                for index, entry in enumerate(templates[kind]):
                    for key in ("polarization", "current"):
                        if key in entry:
                            tensors[f"{component}_{kind}_{index}_{key}"] = entry[key]
    for index, entry in enumerate(solver._mur_state):
        for key in ("prev_boundary", "prev_adjacent"):
            tensors[f"mur_{index}_{key}"] = entry[key]
    return tensors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["debye", "drude", "lorentz", "magnetic"])
@pytest.mark.parametrize("fail_capture", [False, True])
def test_nonzero_material_memory_survives_capture_and_failure(kind, fail_capture, monkeypatch):
    from witwin.maxwell.fdtd.cuda.runtime.graph import CudaGraphRunner
    from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner

    solver = _solver(kind)
    solver.Ex.fill_(1.)
    solver.Hx.fill_(0.01)
    tensors = _physical_tensors(solver)
    saved = {name: tensor.clone() for name, tensor in tensors.items()}
    if fail_capture:
        def fail_after_warmup(self, fn):
            fn()
            fn()
            raise RuntimeError("injected capture failure")
        monkeypatch.setattr(CudaGraphRunner, "capture", fail_after_warmup)
    _make_field_update_runner(solver, True)
    assert solver._cuda_graph_active is not fail_capture
    for name, tensor in tensors.items():
        assert torch.equal(tensor, saved[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_tail_capture_preserves_nonzero_mur_history():
    from witwin.maxwell.fdtd.runtime.stepping import _make_tail_runner

    solver = _solver("debye", mur=True)
    solver.Ex.fill_(1.)
    for entry in solver._mur_state:
        entry["prev_boundary"].fill_(0.2)
        entry["prev_adjacent"].fill_(0.3)
    solver.enable_dft(1e9, window_type="none", end_step=32)
    assert solver.build_dft_step_tables(32)
    solver._cuda_graph_active = True
    tensors = _physical_tensors(solver)
    saved = {name: tensor.clone() for name, tensor in tensors.items()}
    assert _make_tail_runner(solver, True) is not None
    for name, tensor in tensors.items():
        assert torch.equal(tensor, saved[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dispersive_electrostatic_initial_condition_graph_matches_eager():
    from tests.electrostatic.test_initial_condition import _uniform_dc_result, BOUNDS, H

    initial = mw.ElectrostaticInitialCondition.from_result(_uniform_dc_result(device="cuda"))
    results = []
    for graph in (False, True):
        scene = mw.Scene(domain=mw.Domain(bounds=BOUNDS), grid=mw.GridSpec.uniform(H),
            boundary=mw.BoundarySpec.periodic(), device="cuda")
        scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.25,) * 3, size=(1.,) * 3),
            material=mw.Material.debye(eps_inf=2., delta_eps=3., tau=2e-10)))
        result = mw.Simulation.fdtd(scene, frequency=1e9, initial_condition=initial,
            run_time=mw.TimeConfig(time_steps=1200), full_field_dft=True, cuda_graph=graph).run()
        assert result.stats()["cuda_graph_active"] is graph
        results.append(result)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(results[0].solver, name), getattr(results[1].solver, name)), name
    assert torch.equal(results[0].field("Ez"), results[1].field("Ez"))
