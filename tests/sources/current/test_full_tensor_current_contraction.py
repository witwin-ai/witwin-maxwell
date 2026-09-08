"""Impressed-current sources use the complete inverse permittivity operator."""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.injection import inject_source_terms

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_TENSOR = np.array(((2., .5, .2), (.5, 3., -.3), (.2, -.3, 2.5)))


def _scene(kind, axis, boundary):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.3, 0.3),) * 3), grid=mw.GridSpec.uniform(.1),
        boundary=boundary, device="cuda")
    scene.add_structure(mw.Structure(geometry=mw.Box(position=(0.,) * 3, size=(1.,) * 3),
        material=mw.Material(epsilon_tensor=mw.Tensor3x3(_TENSOR.tolist()))))
    waveform = mw.CW(frequency=1e9)
    if kind == "point":
        position = [-.3, -.3, -.3] if boundary.uses_kind("periodic") else [0., 0., 0.]
        position[axis] = -.25 if boundary.uses_kind("periodic") else .05
        source = mw.PointDipole(position=tuple(position), polarization="E" + "xyz"[axis],
            profile="ideal", source_time=waveform)
    elif kind == "uniform":
        source = mw.UniformCurrentSource(size=(.6,) * 3, polarization="E" + "xyz"[axis], source_time=waveform)
    else:
        coords = tuple(np.linspace(-.3, .3, 7) for _ in range(3))
        values = np.ones((7, 7, 7))
        source = mw.CustomCurrentSource(mw.CurrentDataset(coords, {"J" + "xyz"[axis]: values}), source_time=waveform)
    scene.add_source(source)
    return scene


def _independent_current_response(solver, source_axis, periodic):
    # Small CPU oracle: reconstruct the declared impressed current and apply
    # the analytic homogeneous inverse tensor on the four neighboring Yee edges.
    shapes = [tuple(getattr(solver, name).shape) for name in ("Ex", "Ey", "Ez")]
    current = np.zeros(shapes[source_axis])
    for term in solver._source_terms:
        if term.get("periodic_compiled", False):
            continue
        patch = term["patch"].detach().cpu().double().numpy()
        region = tuple(slice(start, start + size) for start, size in zip(term["offsets"], patch.shape))
        eps = getattr(solver, "eps_" + term["field_name"])[region].cpu().double().numpy()
        current[region] += -patch * eps / solver.dt
    if periodic:
        for axis in range(3):
            if axis == source_axis:
                continue
            lo, hi = [slice(None)] * 3, [slice(None)] * 3
            lo[axis], hi[axis] = 0, -1
            total = current[tuple(lo)] + current[tuple(hi)]
            current[tuple(lo)] = total
            current[tuple(hi)] = total
    inverse = np.linalg.inv(_TENSOR)
    output = []
    for target_axis, shape in enumerate(shapes):
        field = np.zeros(shape)
        if target_axis == source_axis:
            field = -solver.dt / solver.eps0 * inverse[target_axis, source_axis] * current
        else:
            for index in np.ndindex(shape):
                for target_side in range(2):
                    for source_side in range(2):
                        sample = list(index)
                        sample[target_axis] += target_side
                        sample[source_axis] += source_side - 1
                        if periodic:
                            sample[source_axis] %= current.shape[source_axis]
                            for axis in range(3):
                                if axis != source_axis:
                                    sample[axis] %= current.shape[axis] - 1
                        if any(value < 0 or value >= limit for value, limit in zip(sample, current.shape)):
                            continue
                        field[index] -= .25 * solver.dt / solver.eps0 * inverse[target_axis, source_axis] * current[tuple(sample)]
        output.append(torch.as_tensor(field, device=solver.device, dtype=solver.Ex.dtype))
    return output


@pytest.mark.parametrize("kind", ["point", "uniform", "custom"])
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("periodic", [False, True])
def test_current_source_matches_analytic_inverse_tensor(kind, axis, periodic):
    boundary = mw.BoundarySpec.periodic() if periodic else mw.BoundarySpec.none()
    solver = mw.Simulation.fdtd(_scene(kind, axis, boundary), frequency=1e9).prepare().solver
    expected = _independent_current_response(solver, axis, periodic)
    inject_source_terms(solver, time_value=0.)
    for name, value in zip(("Ex", "Ey", "Ez"), expected):
        torch.testing.assert_close(getattr(solver, name), value, rtol=2e-6, atol=2e-6)
    assert all(bool(getattr(solver, name).abs().max() > 0) for name in ("Ex", "Ey", "Ez"))


def test_current_transverse_response_obeys_target_pec_faces():
    solver = mw.Simulation.fdtd(_scene("uniform", 0, mw.BoundarySpec.pec()), frequency=1e9).prepare().solver
    inject_source_terms(solver, time_value=0.)
    assert bool(solver.Ey[1:-1, :, 1:-1].abs().max() > 0)
    assert torch.count_nonzero(solver.Ey[0]) == 0
    assert torch.count_nonzero(solver.Ey[-1]) == 0
    assert torch.count_nonzero(solver.Ez[0]) == 0
    assert torch.count_nonzero(solver.Ez[-1]) == 0
