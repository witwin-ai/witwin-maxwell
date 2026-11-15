"""Conformal thin sheets on the native CUDA backend.

The patch runs as device-resident kernels next to the native Yee kernels
(inside the captured CUDA graph by default). The reference arm is the frozen
Torch thin-sheet oracle (tests/oracles/thin_sheet_torch.py) on the same device.
Tolerances fixed before the runs:

* Fused patch vs the Torch oracle over 400 cavity steps: monitor series and
  final fields within 1e-5 of their peak, copper heat within 1e-5 relative.
  (One patch step from identical inputs is gated bitwise or to 1e-13 in
  test_thin_sheet_fused_copies.py and test_thin_sheet_fused_copper.py.)
* The Chebyshev chord solve has forward error <= 1e-9 against a dense solve,
  for the fused kernels and the oracle alike.
* Graph replay and eager execution are bit-identical.
* Physics gates: cavity energy drift <= 1e-3 over 20000 steps, copper
  passivity, exact PEC watertightness, copper leakage below the two-port
  floor, axis-aligned reproduction of the native CopperFoil and PlanarPEC
  paths within 2e-5, oblique T/R within 0.03 of the analytic sheet.
"""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

from tests.oracles import thin_sheet_torch

from . import harness as h
from . import test_thin_sheet_runtime as runtime_tests
from .harness import pec_material, rectangle_mesh

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native thin sheets require CUDA.")

NORMAL = runtime_tests.NORMAL
CENTER = runtime_tests.CENTER


def _structure(kind):
    if kind == "none":
        return None
    if kind == "pec":
        return mw.Structure(geometry=rectangle_mesh(CENTER, NORMAL, 0.03, 0.025, sheet_model="conformal_two_sided"),
                            material=pec_material())
    return mw.Structure(geometry=rectangle_mesh(CENTER, NORMAL, 0.03, 0.025),
                        material=mw.media.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 4e9)))


def _cavity(kind, device):
    structure = _structure(kind)
    scene = runtime_tests._cavity(structure if structure is not None else _structure("pec"))
    scene = scene.updated(monitors=(mw.monitors.FieldTimeMonitor("p", components=("Ex", "Ey", "Ez"), position=(0.03, -0.02, 0.01)),))
    if structure is None:
        scene = scene.updated(structures=())
    return scene


def _run(kind, steps=400, **kwargs):
    scene = _cavity(kind, "cuda")
    return mw.Simulation.fdtd(scene, frequencies=[2e9], config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=steps), **kwargs)).run()


def _observables(result):
    solver = result.solver
    series = torch.cat([torch.as_tensor(result.monitor("p").sel(component=c).values).double().cpu().reshape(-1)
                        for c in ("Ex", "Ey", "Ez")])
    fields = torch.cat([getattr(solver, name).double().cpu().reshape(-1) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")])
    return series, fields


def _relative(a, b):
    return float((a - b).abs().max() / a.abs().max())


@pytest.mark.parametrize("kind", ["pec", "copper"])
def test_fused_patch_matches_torch_oracle_over_400_steps(kind, monkeypatch):
    fused = _run(kind)
    with monkeypatch.context() as patch:
        thin_sheet_torch.install(patch)
        reference = _run(kind, cuda_graph=False)
    assert fused.solver._cuda_graph_active and fused.solver._thin_sheets
    (series_r, fields_r), (series_f, fields_f) = _observables(reference), _observables(fused)
    errors = (_relative(series_r, series_f), _relative(fields_r, fields_f))
    heat = (reference.solver._thin_sheets[0].cumulative_loss, fused.solver._thin_sheets[0].cumulative_loss)
    print(dict(kind=kind, series=errors[0], fields=errors[1], heat=heat))
    assert errors[0] <= 1e-5 and errors[1] <= 1e-5
    if kind == "copper":
        assert heat[0] > 0 and abs(heat[1] - heat[0]) <= 1e-5 * heat[0]


def test_copper_crank_nicolson_solve_forward_error():
    """rhs = A x_true for a rough (random) and a smooth x_true; forward error <= 1e-9, fused and oracle."""
    scene = _cavity("copper", "cuda")
    sheet = mw.Simulation.fdtd(scene, frequencies=[2e9],
                               config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=1))).prepare().solver._thin_sheets[0]
    errors = {}
    for name, truth in copper_test_vectors(sheet).items():
        truth = truth.to(sheet.device)
        rhs = torch.mv(sheet.copper["S"], truth)
        solutions = {"fused": sheet.chord_solve(rhs), "oracle": thin_sheet_torch.chord_solve(sheet, rhs)}
        for arm, x in solutions.items():
            errors[(name, arm)] = float(torch.linalg.vector_norm(x - truth) / torch.linalg.vector_norm(truth))
        errors[(name, "fused-oracle")] = float(torch.linalg.vector_norm(solutions["fused"] - solutions["oracle"])
                                               / torch.linalg.vector_norm(truth))
    print(dict(errors=errors))
    assert max(errors.values()) <= 1e-9


def copper_test_vectors(sheet):
    """A rough (independent Gaussian) and a smooth (low-order trigonometric) copper stencil field."""
    position = torch.as_tensor(np.repeat(np.array([path.mean(0) for path in sheet.complex.chord_path]), 2, axis=0))
    span = (position.max(0).values - position.min(0).values).clamp_min(1e-12)
    phase = (position - position.min(0).values) / span
    smooth = (torch.sin(2.1 * phase[:, 0] + 0.3) * torch.cos(1.7 * phase[:, 1] - 0.2)
              + 0.5 * torch.sin(1.3 * phase[:, 2] + 0.9)).to(torch.float64)
    rough = torch.randn(len(position), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    return {"rough": rough, "smooth": smooth}


def test_graph_replay_is_bit_identical_to_eager():
    graph = _run("copper", steps=300)
    eager = _run("copper", steps=300, cuda_graph=False)
    assert graph.solver._cuda_graph_active and not eager.solver._cuda_graph_active
    for a, b in zip(_observables(graph), _observables(eager)):
        assert torch.equal(a, b)
    assert graph.solver._thin_sheets[0].cumulative_loss == eager.solver._thin_sheets[0].cumulative_loss
    for a, b in zip(graph.solver._thin_sheets[0].state_tensors(), eager.solver._thin_sheets[0].state_tensors()):
        assert torch.equal(a, b)
    print(dict(graph_eager_max_difference=0, steps=300))


def test_cuda_tilted_pec_cavity_energy_for_twenty_thousand_steps():
    runtime_tests.check_pec_cavity_energy()


def test_cuda_tilted_copper_cavity_is_passive_for_twenty_thousand_steps():
    runtime_tests.check_copper_cavity_passivity()


def test_cuda_cavity_spanning_pec_sheet_is_watertight():
    runtime_tests.check_pec_watertight("cuda")


def test_cuda_cavity_spanning_copper_leaks_only_through_the_copper_model():
    runtime_tests.check_copper_leak("cuda")


@pytest.mark.parametrize("material", ["copper", "pec"])
def test_cuda_axis_aligned_mesh_reproduces_native_planar_paths(material):
    runtime_tests.check_axis_aligned(material, "cuda")


@pytest.mark.parametrize("g,polarization", [(0.5 / h.ETA0, "te"), (4.0 / h.ETA0, "tm"), (0.0, "te")])
def test_cuda_oblique_scattering_matches_thin_sheet_formulas(g, polarization):
    material = pec_material() if g == 0 else h.resistive_foil(g)
    transmission, reflection, info = h.measure_transmission(material, 30.0, polarization, 0.015, absorber="cpml")
    T, R = complex(transmission[1]), complex(reflection[1])
    if g == 0:
        expected_t, expected_r = 0.0, -1.0
    else:
        expected_t = h.thin_sheet_transmission(g, 30.0, polarization)
        expected_r = (h.thin_sheet_reflection_te(g, 30.0) if polarization == "te"
                      else h.thin_sheet_reflection_tm_ey(g, 30.0))
    print(dict(g=g * h.ETA0, pol=polarization, T=T, R=R, dT=abs(T - expected_t), dR=abs(R - expected_r), **info))
    assert abs(T - expected_t) <= (1e-6 if g == 0 else 0.03)
    assert abs(R - expected_r) <= 0.03


def test_periodic_boundaries_are_rejected_on_cuda():
    scene = _cavity("copper", "cuda")
    scene = scene.updated(boundary=mw.BoundarySpec.faces(default="pec", z="periodic"))
    with pytest.raises(NotImplementedError, match="periodic or Bloch"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=2))).prepare()


def test_native_batch_rejects_thin_sheets():
    from witwin.maxwell.fdtd.runtime.batch import validate_native_solver
    solver = mw.Simulation.fdtd(_cavity("pec", "cuda"), frequencies=[2e9],
                                config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=2))).prepare().solver
    with pytest.raises(NotImplementedError, match="_thin_sheets"):
        validate_native_solver(solver)


def test_trainable_copper_is_rejected_on_cuda():
    thickness = torch.tensor(2e-6, dtype=torch.float64, device="cuda", requires_grad=True)
    foil = mw.media.CopperFoil(thickness=thickness, frequency_range=(0.5e9, 4e9), mode_count=64)
    scene = _cavity("copper", "cuda")
    scene = scene.updated(structures=(mw.Structure(geometry=rectangle_mesh(CENTER, NORMAL, 0.03, 0.025), material=foil),))
    with pytest.raises(NotImplementedError, match="forward-only; trainable copper parameters are not supported"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=2))).run()
