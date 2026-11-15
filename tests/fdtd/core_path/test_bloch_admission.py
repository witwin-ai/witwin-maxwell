"""Complex (Bloch split) fields on the core path: admission, closure order and device tables."""
from types import SimpleNamespace

import pytest
import torch

from benchmark.core_path.scenes import BLOCH_SCENES, BLOCH_SEPARATE_SCENES, GATE_SCENES, build_core_case
from witwin.maxwell.fdtd import cut_surface
from witwin.maxwell.fdtd.runtime import core_path, core_tables, stepping
from witwin.maxwell.scene import prepare_scene


@pytest.fixture(autouse=True)
def forbid_cuda_init(monkeypatch, request):
    if request.node.get_closest_marker("host_validation"):
        monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **kw: pytest.fail("Host check initialized CUDA."))


@pytest.mark.host_validation
@pytest.mark.parametrize("name", BLOCH_SCENES)
def test_bloch_gate_scenes_are_admitted(name):
    assert name in GATE_SCENES
    case = build_core_case(name)
    assert case.scene.boundary.uses_kind("bloch")
    grid = prepare_scene(case.scene, device="cpu", lazy_meshgrid=True)
    assert grid.Nx > 0 and grid.Ny > 0 and grid.Nz > 0
    solver = SimpleNamespace(scene=SimpleNamespace(scene=case.scene), complex_fields_enabled=True)
    assert core_path.core_path_step_loop_blockers(solver) == []


@pytest.mark.host_validation
@pytest.mark.parametrize("name", BLOCH_SEPARATE_SCENES)
def test_bloch_with_separate_physics_is_admitted(name):
    # W7 row 16b: the separate-physics pieces that admit complex fields run on core with them.
    assert name in GATE_SCENES
    case = build_core_case(name)
    assert case.scene.boundary.uses_kind("bloch")
    solver = SimpleNamespace(scene=SimpleNamespace(scene=case.scene), complex_fields_enabled=True,
                             _port_runtimes=[object()])
    assert core_path.core_path_has_separate_physics(solver)


@pytest.mark.host_validation
@pytest.mark.parametrize("cpml", [False, True])
def test_complex_closure_order(monkeypatch, cpml):
    events = []
    runtime = core_path.CorePathRuntime.__new__(core_path.CorePathRuntime)
    runtime.step = torch.zeros(1, dtype=torch.int64)
    runtime.magnetic = lambda imag=False: events.append("yee_h_imag" if imag else "yee_h")
    runtime.electric = lambda: pytest.fail("Complex fields never launch the real fused E.")
    runtime.extension = SimpleNamespace(CoreStepLaunch=lambda *args:
        SimpleNamespace(launch=lambda: events.append("commit")))
    line = SimpleNamespace(launch=lambda: events.append("line_e"))
    runtime.sources = SimpleNamespace(auxiliary=line, rows={}, groups={
        name: [SimpleNamespace(launch=lambda n=name: events.append(n))] for name in ("magnetic", "electric")})
    solver = SimpleNamespace(scene=SimpleNamespace(scene=SimpleNamespace()), tfsf_enabled=True, observers_enabled=False,
        complex_fields_enabled=True, uses_cpml=cpml, has_pec_faces=False, _tfsf_state={},
        _advance_magnetic_dispersive_state=lambda: events.append("ade_h"),
        _apply_magnetic_dispersive_corrections=lambda: events.append("correct_h"),
        _advance_dispersive_state=lambda: events.append("ade_e"),
        _apply_dispersive_corrections=lambda: events.append("correct_e"))
    monkeypatch.setattr(core_tables, "compile_spectral_tables", lambda *a:
        tuple(SimpleNamespace(launch=lambda p=part: events.append(p)) for part in ("dft", "dft_imag")))
    monkeypatch.setattr(core_tables, "compile_monitor_routes", lambda *a: None)
    monkeypatch.setattr(core_tables, "compile_auxiliary_fusion", lambda *a: None)
    monkeypatch.setattr(stepping, "enforce_pec_boundaries", lambda s: events.append("pec"))
    monkeypatch.setattr(stepping, "apply_mur_boundaries", lambda s: events.append("mur"))
    monkeypatch.setattr(stepping, "update_electric_fields_bloch", lambda s: events.append("bloch_e"))
    monkeypatch.setattr(stepping, "update_electric_fields_bloch_cpml", lambda s: events.append("bloch_e_cpml"))
    monkeypatch.setattr(cut_surface, "apply_cut_surface_constraints", lambda s: events.append("cut"))
    from witwin.maxwell.fdtd.excitation import tfsf_apply
    monkeypatch.setattr(tfsf_apply, "apply_tfsf_h_correction", lambda s, t: events.append("tfsf_h"))
    monkeypatch.setattr(tfsf_apply, "apply_tfsf_e_correction", lambda s, t: events.append("tfsf_e"))
    monkeypatch.setattr(tfsf_apply, "advance_tfsf_auxiliary_magnetic", lambda s: events.append("line_h"))
    step, _ = runtime.prepare_step(solver, use_gpu_dft=True, time_steps=32)
    step()
    # _field_update_block's complex branch, then the eager TFSF/source/tail order of solve_steps.
    assert events == ["ade_h", "yee_h", "yee_h_imag", "tfsf_h", "line_h", "magnetic", "correct_h", "ade_e",
                      "bloch_e_cpml" if cpml else "bloch_e", "tfsf_e", "line_e", "electric", "correct_e",
                      "mur", "cut", "dft", "dft_imag", "commit"]


def _simulation(name, steps):
    from benchmark.core_path.gates import make_simulation

    return make_simulation(build_core_case(name), steps=steps, full_field_dft=True)


def _payload(result):
    from benchmark.core_path.gates import export_tensors, physics_tensors, result_tensors, tensors

    solver = result.solver
    payload = tensors({name + suffix: getattr(solver, name + suffix)
                       for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") for suffix in ("", "_imag")}, "fields")
    # Separate physics: lumped and port state, port spectra, copper losses and the export-time loss channels.
    return payload | result_tensors(result) | physics_tensors(solver, result) | export_tensors(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Core execution requires CUDA.")
@pytest.mark.parametrize("name", ["bloch_dipole_slab", "bloch_cw_tfsf", *BLOCH_SEPARATE_SCENES])
def test_bloch_whole_step_core_equals_existing_path(name):
    """The K=32 and K=1 core graphs equal the existing-runner run bitwise.

    The reference is the existing-runner payload frozen in addendum w7-row16e (W7 row 16e).
    """
    from tests.support import frozen_oracles
    runs = {}
    # The fixed-k TFSF pulse reaches the cell after about 100 steps on this grid.
    steps = 160 if name == "bloch_tfsf_resistor" else 40
    for label, k in (("k32", 32), ("k1", 1)):
        prepared = _simulation(name, steps).prepare()
        prepared.solver._core_runtime.chunk_size = k
        result = prepared.run()
        solver = result.solver
        assert solver.complex_fields_enabled
        # TFSF beside separate physics keeps the per-step core loop.
        graph = name != "bloch_tfsf_resistor"
        assert solver._core_step_graph_active is graph
        assert solver._core_step_graph_chunks == ([] if not graph else [32, 8] if k == 32 else [1] * 40)
        runs[label] = _payload(result)
    reference = frozen_oracles.load(f"tests/fdtd/core_path/test_bloch_admission.py::{name}")
    assert any(value.abs().max() > 0 for key, value in reference.items() if key.startswith("fields/E"))
    for label in ("k32", "k1"):
        assert runs[label].keys() == reference.keys()
        for key, value in reference.items():
            assert torch.equal(value, runs[label][key]), (label, key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fixed-k line tables are device tensors.")
def test_fixed_k_line_tables_match_the_host_formulas():
    from witwin.maxwell.sources import evaluate_source_time

    prepared = _simulation("bloch_oblique", 24).prepare()
    solver = prepared.solver
    line = solver._tfsf_state["auxiliary_grid"]
    assert solver._tfsf_state["provider"] == "plane_wave_fixed_k" and line.clock.is_cuda
    line.prepare_tables(24)
    for step in (0, 7, 23):
        value = float(evaluate_source_time(line.source_time, step * line.dt)) * 2 * line.courant
        assert line.drive_rows[step].item() == complex(torch.tensor(value, dtype=torch.float32).item(), 0.0)
        phase = torch.exp(1j * line.omega * (step * line.dt))
        assert torch.equal(line.phase_rows[step], phase)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DFT tables are device tensors.")
def test_complex_dft_uses_device_tables():
    solver = _simulation("bloch_dipole_slab", 16).prepare().solver
    solver.enable_dft((2.0e9,), window_type="none", end_step=16)
    assert solver.build_dft_step_tables(16)
    assert all(buffers["aux_real"] is not None for buffers in solver._dft_batched_fields.values())


_BLOCH_REFUSALS = {
    # Each W1 separate-physics scene under Bloch x/y faces refuses complex fields on every path (W7 row 16b).
    ("w1_lumped_port", True): "Driven lumped port 'feed' under Bloch boundaries",
    ("w1_coax_port", False): "only passive LTI R/L/C and LumpedPort terminations",
    ("w1_terminal_port", False): "has a terminal on CopperFoil 'signal'; complex-field Bloch coupling requires",
    ("w1_circuit", True): "Circuits/MNA and embedded networks under Bloch boundaries",
    ("w1_network", True): "Circuits/MNA and embedded networks under Bloch boundaries",
    ("w1_thin_wire", True): "Thin-wire FDTD does not yet support Bloch-periodic phase topology",
    ("w1_cut_surface", True): "PEC cut-surface projection currently requires real isotropic linear fields",
    ("w1_sheet_feed", True): "Conformal thin sheets do not yet support periodic or Bloch boundaries",
    ("w1_sibc", True): "Bloch-periodic run carries complex phase-shifted fields",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD preparation requires CUDA.")
@pytest.mark.parametrize("name,driven", list(_BLOCH_REFUSALS), ids=[f"{name}-{'driven' if driven else 'undriven'}"
                                                                    for name, driven in _BLOCH_REFUSALS])
def test_unsupported_bloch_separate_physics_refuses_on_every_path(name, driven):
    import witwin.maxwell as mw
    from dataclasses import replace

    case = build_core_case(name)
    boundary = mw.BoundarySpec.faces(default="pml", num_layers=2, x="bloch", y="bloch", bloch_wavevector=(1., 2., 0.))
    case = replace(case, scene=case.scene.updated(boundary=boundary), excitations=case.excitations if driven else ())
    with pytest.raises(NotImplementedError, match=_BLOCH_REFUSALS[name, driven]):
        _simulation_of(case).prepare()


def _simulation_of(case):
    from benchmark.core_path.gates import make_simulation

    return make_simulation(case, steps=8, full_field_dft=True)
