"""Conformal thin sheets on the native CUDA backend.

The patch runs as device-resident Torch operators next to the native Yee
kernels (inside the captured CUDA graph by default). Tolerances fixed before
the runs:

* CUDA vs the CPU reference backend over 400 cavity steps: monitor series and
  final fields within 1e-5 of their peak, copper heat within 1e-5 relative.
  Both backends advance float32 Yee fields whose kernels round differently
  (operation order, fused multiply-add), so agreement is bounded by float32
  accumulation, not by the patch: the same scene without a sheet is measured
  as the baseline and the sheet may not more than triple it.
* One patch step from identical inputs: all private float64 unknowns
  within 1e-12 relative, float32 field rows within one float32 rounding
  (2.4e-7 relative). This includes the locally implicit chord fields, their
  history and forcing, and the copper diffusion state.
* Graph replay and eager execution are bit-identical.
* Physics gates are those of the CPU suite: cavity energy drift <= 1e-3 over
  20000 steps, copper passivity, exact PEC watertightness, copper leakage below
  the two-port floor, axis-aligned reproduction of the native CopperFoil and
  PlanarPEC paths within 2e-5, oblique T/R within 0.03 of the analytic sheet.
"""
import math

import numpy as np
import pytest
import torch

import witwin.maxwell as mw

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
                        material=mw.CopperFoil(thickness=2e-6, frequency_range=(0.5e9, 4e9)))


def _cavity(kind, device):
    structure = _structure(kind)
    scene = runtime_tests._cavity(structure if structure is not None else _structure("pec"), device=device)
    scene.monitors = (mw.FieldTimeMonitor("p", components=("Ex", "Ey", "Ez"), position=(0.03, -0.02, 0.01)),)
    if structure is None:
        scene.structures = ()
    return scene


def _run(kind, device, steps=400, **kwargs):
    scene = _cavity(kind, device)
    return mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=steps),
                              backend=h.backend_for(scene), **kwargs).run()


def _observables(result):
    solver = result.solver
    series = torch.cat([torch.as_tensor(result.monitors["p"]["components"][c]).double().cpu().reshape(-1)
                        for c in ("Ex", "Ey", "Ez")])
    fields = torch.cat([getattr(solver, name).double().cpu().reshape(-1) for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")])
    return series, fields


def _relative(a, b):
    return float((a - b).abs().max() / a.abs().max())


@pytest.fixture(scope="module")
def float32_baseline():
    cpu, cuda = _observables(_run("none", "cpu")), _observables(_run("none", "cuda"))
    return tuple(_relative(a, b) for a, b in zip(cpu, cuda))


@pytest.mark.parametrize("kind", ["pec", "copper"])
def test_cuda_matches_cpu_reference(kind, float32_baseline):
    cpu, cuda = _run(kind, "cpu"), _run(kind, "cuda")
    assert cuda.solver._cuda_graph_active and cuda.solver._thin_sheets
    (series_c, fields_c), (series_g, fields_g) = _observables(cpu), _observables(cuda)
    errors = (_relative(series_c, series_g), _relative(fields_c, fields_g))
    heat = (cpu.solver._thin_sheets[0].cumulative_loss, cuda.solver._thin_sheets[0].cumulative_loss)
    print(dict(kind=kind, series=errors[0], fields=errors[1], baseline=float32_baseline, heat=heat))
    assert errors[0] <= 1e-5 and errors[1] <= 1e-5
    assert errors[0] <= 3 * max(float32_baseline[0], 1e-7) and errors[1] <= 3 * max(float32_baseline[1], 1e-7)
    if kind == "copper":
        assert heat[0] > 0 and abs(heat[1] - heat[0]) <= 1e-5 * heat[0]


@pytest.mark.parametrize("kind", ["pec", "copper"])
def test_one_patch_step_matches_cpu_in_float64(kind):
    """Identical inputs through the patch alone (no native kernels) on both devices."""
    generator = torch.Generator().manual_seed(7)
    prepared = {}
    for device in ("cpu", "cuda"):
        scene = _cavity(kind, device)
        prepared[device] = mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=1),
                                              backend=h.backend_for(scene)).prepare().solver
    names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    values = {name: torch.randn(getattr(prepared["cpu"], name).shape, generator=generator) for name in names}
    sheets = {device: solver._thin_sheets[0] for device, solver in prepared.items()}
    extra_e = torch.randn(sheets["cpu"].e_extra.shape, generator=generator, dtype=torch.float64)
    extra_h = torch.randn(sheets["cpu"].h_extra.shape, generator=generator, dtype=torch.float64)
    state = None if sheets["cpu"].copper is None else torch.randn(
        sheets["cpu"].copper["state"].shape, generator=generator, dtype=torch.float64) * 1e-3
    chord_inputs = {} if state is None else {
        name: torch.randn(getattr(sheets["cpu"], name).shape, generator=generator, dtype=torch.float64)
        for name in ("chord_e", "chord_e_prev", "chord_f", "x_prev")}
    for device, solver in prepared.items():
        for name in names:
            getattr(solver, name).copy_(values[name])
        sheet = sheets[device]
        sheet.e_extra.copy_(extra_e)
        sheet.h_extra.copy_(extra_h)
        if state is not None:
            sheet.copper["state"].copy_(state)
            for name, value in chord_inputs.items():
                getattr(sheet, name).copy_(value)
        sheet.begin(solver)
        sheet.magnetic(solver)
        sheet.electric(solver)
    cpu, cuda = sheets["cpu"], sheets["cuda"]
    private = [(getattr(cpu, name), getattr(cuda, name))
               for name in ("h_extra", "e_extra", "e_new", "h_new", *chord_inputs)]
    if state is not None:
        private.extend(((cpu.copper["state"], cuda.copper["state"]), (cpu.loss, cuda.loss)))
    worst_private = max(_relative(a, b.cpu()) for a, b in private)
    worst_field = max(_relative(getattr(prepared["cpu"], n).double(), getattr(prepared["cuda"], n).double().cpu())
                      for n in names)
    print(dict(kind=kind, private=worst_private, field=worst_field))
    assert worst_private <= 1e-12
    assert worst_field <= 2.4e-7


def test_copper_crank_nicolson_solve_forward_error_on_both_devices():
    """rhs = A x_true for a rough (random) and a smooth x_true; forward error <= 1e-9 on CPU and CUDA."""
    sheets = {}
    for device in ("cpu", "cuda"):
        scene = _cavity("copper", device)
        sheets[device] = mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=1),
                                            backend=h.backend_for(scene)).prepare().solver._thin_sheets[0]
    errors = {}
    for name, truth in copper_test_vectors(sheets["cpu"]).items():
        rhs = torch.mv(sheets["cpu"].copper["S"], truth)
        solutions = {device: sheet.chord_solve(rhs.to(sheet.device)).cpu() for device, sheet in sheets.items()}
        for device, x in solutions.items():
            errors[(name, device)] = float(torch.linalg.vector_norm(x - truth) / torch.linalg.vector_norm(truth))
        errors[(name, "cpu-cuda")] = float(torch.linalg.vector_norm(solutions["cpu"] - solutions["cuda"])
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
    graph = _run("copper", "cuda", steps=300)
    eager = _run("copper", "cuda", steps=300, cuda_graph=False)
    assert graph.solver._cuda_graph_active and not eager.solver._cuda_graph_active
    for a, b in zip(_observables(graph), _observables(eager)):
        assert torch.equal(a, b)
    assert graph.solver._thin_sheets[0].cumulative_loss == eager.solver._thin_sheets[0].cumulative_loss
    for a, b in zip(graph.solver._thin_sheets[0].state_tensors(), eager.solver._thin_sheets[0].state_tensors()):
        assert torch.equal(a, b)
    print(dict(graph_eager_max_difference=0, steps=300))


def test_cuda_tilted_pec_cavity_energy_for_twenty_thousand_steps():
    runtime_tests.check_pec_cavity_energy("cuda")


def test_cuda_tilted_copper_cavity_is_passive_for_twenty_thousand_steps():
    runtime_tests.check_copper_cavity_passivity("cuda")


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
    transmission, reflection, info = h.measure_transmission(material, 30.0, polarization, 0.015, device="cuda", absorber="cpml")
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
    scene.boundary = mw.BoundarySpec.faces(default="pec", z="periodic")
    with pytest.raises(NotImplementedError, match="periodic or Bloch"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2)).prepare()


def test_native_batch_rejects_thin_sheets():
    from witwin.maxwell.fdtd.runtime.batch import validate_native_solver
    solver = mw.Simulation.fdtd(_cavity("pec", "cuda"), frequencies=[2e9],
                                run_time=mw.TimeConfig(time_steps=2)).prepare().solver
    with pytest.raises(NotImplementedError, match="_thin_sheets"):
        validate_native_solver(solver)


def test_trainable_copper_is_rejected_on_cuda():
    thickness = torch.tensor(2e-6, dtype=torch.float64, device="cuda", requires_grad=True)
    foil = mw.CopperFoil(thickness=thickness, frequency_range=(0.5e9, 4e9), mode_count=64)
    scene = _cavity("copper", "cuda")
    scene.structures = (mw.Structure(geometry=rectangle_mesh(CENTER, NORMAL, 0.03, 0.025), material=foil),)
    with pytest.raises(NotImplementedError, match="forward-only; trainable copper parameters are not supported"):
        mw.Simulation.fdtd(scene, frequencies=[2e9], run_time=mw.TimeConfig(time_steps=2)).run()
