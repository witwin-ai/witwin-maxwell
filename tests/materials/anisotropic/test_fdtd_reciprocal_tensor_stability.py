"""Independent spectral and transpose gates for spatially varying full tensors."""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.runtime.stepping import _field_update_block, update_magnetic_fields

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _tensor_scene(*, nonuniform=False, conductive=False):
    generator = np.random.default_rng(6193)
    for _ in range(3):
        angle = generator.uniform(-np.pi, np.pi, (4, 4))
        strength = 10 ** generator.uniform(0, 2, (4, 4))
    x = np.array([0., .07, .16, .30, .4]) if nonuniform else np.linspace(0., .4, 5)
    y = np.array([0., .04, .14, .25, .4]) if nonuniform else np.linspace(0., .4, 5)
    scene = mw.Scene(domain=mw.Domain(bounds=((0., .4), (0., .4), (0., .2))),
        grid=mw.GridSpec.custom(x_coords=x, y_coords=y, z_coords=(0., .1, .2)),
        boundary=mw.BoundarySpec.periodic(), device="cuda")
    for i in range(5):
        for j in range(5):
            a, s = angle[i % 4, j % 4], strength[i % 4, j % 4]
            c, t = np.cos(a), np.sin(a)
            tensor = ((c*c*s+t*t, c*t*(s-1), 0.),
                      (c*t*(s-1), t*t*s+c*c, 0.), (0., 0., 1.))
            sigma = .01 * s if conductive else 0.
            scene.add_structure(mw.Structure(
                geometry=mw.Box(position=(float(x[i]), float(y[j]), .1), size=(.01, .01, .6)),
                material=mw.Material(epsilon_tensor=mw.Tensor3x3(tensor), sigma_e=sigma)))
    return scene


@pytest.mark.parametrize("nonuniform", [False, True])
def test_variable_tensor_native_curl_operator_is_self_adjoint_and_positive(nonuniform):
    solver = mw.Simulation.fdtd(_tensor_scene(nonuniform=nonuniform), frequency=.7e9,
        run_time=mw.TimeConfig(time_steps=600), cuda_graph=False).prepare().solver
    columns = []
    for position in range(16):
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            getattr(solver, name).zero_()
        i, j = divmod(position, 4)
        solver.Hz[i, j, :].fill_(1.)
        _field_update_block(solver, 0.)
        update_magnetic_fields(solver, solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
        columns.append(solver.Hz[:, :, 0].cpu().numpy().reshape(-1).copy())
    operator = (np.eye(16) - np.stack(columns, axis=1)) / (solver.c * solver.dt)**2
    volume = np.outer(solver.scene.dx_primal64, solver.scene.dy_primal64).reshape(-1)
    symmetric = np.sqrt(volume)[:, None] * operator / np.sqrt(volume)[None, :]
    np.testing.assert_allclose(symmetric, symmetric.T, rtol=2e-5, atol=2e-4)
    eigenvalues = np.linalg.eigvalsh((symmetric + symmetric.T) / 2)
    assert eigenvalues.min() >= -2e-4
    assert eigenvalues.max() * (solver.c * solver.dt)**2 < 4

    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).zero_()
    solver.Hz.copy_(torch.as_tensor(np.random.default_rng(21).normal(size=(4, 4, 1)),
                                   device="cuda", dtype=torch.float32).expand_as(solver.Hz))
    initial = float(solver.Hz.norm())
    for step in range(600):
        _field_update_block(solver, step * solver.dt)
    assert float(solver.Hz.norm()) < 2 * initial


def test_variable_conductive_tensor_native_joint_step_is_passive():
    solver = mw.Simulation.fdtd(_tensor_scene(nonuniform=True, conductive=True), frequency=.7e9,
        run_time=mw.TimeConfig(time_steps=100), cuda_graph=False).prepare().solver
    assert solver._time_step_metadata["material_rate_bound"] > 0
    impedance = np.sqrt(solver.mu0 / solver.eps0)
    columns = []
    for position in range(48):
        state = torch.zeros((3, 4, 4), device="cuda")
        state.flatten()[position] = 1.
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            getattr(solver, name).zero_()
        solver.Ex[:, :4, :] = (impedance * state[0])[:, :, None]
        solver.Ex[:, 4, :] = solver.Ex[:, 0, :]
        solver.Ey[:4, :, :] = (impedance * state[1])[:, :, None]
        solver.Ey[4, :, :] = solver.Ey[0, :, :]
        solver.Hz.copy_(state[2, :, :, None].expand_as(solver.Hz))
        _field_update_block(solver, 0.)
        columns.append(torch.cat((solver.Ex[:, :4, 0].flatten() / impedance,
                                  solver.Ey[:4, :, 0].flatten() / impedance,
                                  solver.Hz[:, :, 0].flatten())).cpu().numpy())
    assert np.abs(np.linalg.eigvals(np.stack(columns, axis=1))).max() <= 1 + 2e-6
    # Continue the last nonzero basis state with no forcing. A time-step rate
    # bound must control the actual carried fields, not just a local pole.
    for step in range(1000):
        _field_update_block(solver, step * solver.dt)
    assert torch.isfinite(solver.Hz).all()
    assert float(solver.Hz.norm()) < 2
    assert float(solver.Ex.norm() / impedance) < 2
    assert float(solver.Ey.norm() / impedance) < 2


def test_native_anisotropic_transpose_matches_independent_autograd():
    from witwin.maxwell.fdtd.adjoint.core import _collocate_curl
    from witwin.maxwell.fdtd.cuda.backend import _full_aniso_curl_adjoint

    shapes = ((3, 5, 4), (4, 4, 4), (4, 5, 3))
    generator = torch.Generator(device="cuda").manual_seed(773)
    fields = [torch.randn(shape, generator=generator, device="cuda", requires_grad=True) for shape in shapes]
    coefficients = {(i, j): torch.randn((*shapes[i], 4), generator=generator, device="cuda")
                    for i in range(3) for j in range(3) if i != j}
    seeds = [torch.randn(shape, generator=generator, device="cuda") for shape in shapes]
    objective = sum((_collocate_curl(fields[j], coefficients[i, j], edge_axis=i, curl_axis=j) * seeds[i]).sum()
                    for i, j in coefficients)
    expected = torch.autograd.grad(objective, fields)
    actual = [torch.empty_like(field) for field in fields]
    _full_aniso_curl_adjoint(AdjCurlX=actual[0], AdjCurlY=actual[1], AdjCurlZ=actual[2],
        AdjEx=seeds[0], AdjEy=seeds[1], AdjEz=seeds[2],
        CoeffExY=coefficients[0, 1], CoeffExZ=coefficients[0, 2],
        CoeffEyX=coefficients[1, 0], CoeffEyZ=coefficients[1, 2],
        CoeffEzX=coefficients[2, 0], CoeffEzY=coefficients[2, 1])
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-6)
