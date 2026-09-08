"""Independent adjoint identity and native face/corner ordering checks."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary.mur import mur_state_name, pullback_mur_boundaries, replay_mur_boundaries
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from witwin.maxwell.fdtd.runtime.stepping import apply_mur_boundaries

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _solver(complex_fields):
    boundary = (mw.BoundarySpec.faces(default="mur", x="bloch", bloch_wavevector=(0.7, 0., 0.))
                if complex_fields else mw.BoundarySpec.mur())
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3),
        grid=mw.GridSpec.uniform(0.1), boundary=boundary, device="cuda")
    return mw.Simulation.fdtd(scene, frequency=1e9).prepare().solver


@pytest.mark.parametrize("complex_fields", [False, True])
def test_mur_replay_matches_native_faces_and_transpose_inner_product(complex_fields):
    solver = _solver(complex_fields)
    torch.manual_seed(742)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).normal_()
        if complex_fields:
            getattr(solver, name + "_imag").normal_()
    for entry in solver._mur_state:
        entry["prev_boundary"].normal_()
        entry["prev_adjacent"].normal_()
    initial = capture_checkpoint_state(solver, 0).tensors
    apply_mur_boundaries(solver)
    actual = capture_checkpoint_state(solver, 1).tensors
    replayed = replay_mur_boundaries(solver, initial, initial)
    for name in initial:
        torch.testing.assert_close(replayed[name], actual[name], rtol=5e-6, atol=1e-6)

    # Pure algebra in double precision: independent random primal/cotangent,
    # including all histories and shared corner degrees of freedom.
    x = {name: value.cpu().double() for name, value in initial.items()}
    y = {name: torch.randn_like(value) for name, value in x.items()}
    advanced = replay_mur_boundaries(solver, x, x)
    transposed = pullback_mur_boundaries(solver, y)
    left = sum((advanced[name] * y[name]).sum() for name in x)
    right = sum((x[name] * transposed[name]).sum() for name in x)
    torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def test_complex_mur_is_covariant_under_global_phase_rotation():
    solver = _solver(True)
    state = capture_checkpoint_state(solver, 0).tensors
    state = {name: torch.randn_like(value, dtype=torch.float64) for name, value in state.items()}
    pairs = [(name, name + "_imag") for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")]
    entries = solver._mur_state
    for index, entry in enumerate(entries):
        if entry["field"].endswith("_imag"):
            continue
        imaginary_index = next(i for i, other in enumerate(entries)
            if other["field"] == entry["field"] + "_imag" and other["axis"] == entry["axis"]
            and other["boundary_index"] == entry["boundary_index"])
        pairs.extend((mur_state_name(index, history), mur_state_name(imaginary_index, history))
                     for history in ("prev_boundary", "prev_adjacent"))

    def rotate(values):
        output = dict(values)
        for real, imag in pairs:
            output[real] = math.cos(0.37) * values[real] - math.sin(0.37) * values[imag]
            output[imag] = math.sin(0.37) * values[real] + math.cos(0.37) * values[imag]
        return output

    rotated = rotate(state)
    actual = replay_mur_boundaries(solver, rotated, rotated)
    expected = rotate(replay_mur_boundaries(solver, state, state))
    for name in state:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)
