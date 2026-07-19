"""Phase 4 slice: finite-difference gradient gates for point and averaged SAR.

Small float64 CPU problems. The E-field amplitude and a 3D mass-density grid are
autograd leaves, so those gates compare autograd against central differences. The
scalar ``Material.sigma_e`` is floated at compile time (not an autograd leaf in
this build), so the conductivity gate is a central-difference-vs-analytic check
of the reduction's sensitivity instead of an autograd comparison.
"""

import numpy as np
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.scene import prepare_scene

_DX = 0.25


def _build(*, amp, sigma=0.5, rho=1000.0, freq=1.0e9):
    """Uniform lossy cube with a constant complex field of amplitude ``amp``.

    ``amp`` and ``rho`` may be tensors (autograd leaves); ``rho`` as a tensor is a
    3D grid over the box.
    """
    material = mw.Material(sigma_e=sigma, mass_density=rho, name="tissue")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 2.0),) * 3),
        grid=mw.GridSpec.uniform(_DX),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(position=(1.0, 1.0, 1.0), size=(8.0, 8.0, 8.0)),
                material=material,
            ),
        ),
        device="cpu",
    )
    monitor = mw.PowerLossMonitor(
        "loss", position=(1.0, 1.0, 1.0), size=(1.0, 1.0, 1.0),
        frequencies=(freq,), channels=("conduction",),
    )
    scene.add_monitor(monitor)
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    amp_t = amp if torch.is_tensor(amp) else torch.tensor(amp, dtype=torch.float64)
    fields = {}
    for component in ("Ex", "Ey", "Ez"):
        shape = compiled.full_component_shapes[component]
        base = torch.ones(shape, dtype=torch.complex128)
        fields[component.upper()] = amp_t.to(torch.complex128) * base
    result = mw.Result(
        method="fdtd", scene=scene, prepared_scene=prepared,
        frequencies=(freq,), fields=fields,
    )
    return result


def _grid_shape():
    result = _build(amp=1.0)
    prepared = result.prepared_scene
    return (prepared.Nx, prepared.Ny, prepared.Nz)


def _point_objective(sar):
    total = sar.point_sar("total")
    return torch.nansum(total)


def _averaged_objective(sar, m0):
    field = sar.averaged_sar(m0)
    return torch.nansum(field)


def test_point_sar_field_gradient_matches_central_difference():
    amp = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    sar = _build(amp=amp).sar("loss")
    obj = _point_objective(sar)
    (grad,) = torch.autograd.grad(obj, amp)

    # The GPU-first reducer runs in float32, so autograd/FD agree to float32
    # precision (~1e-3), not float64. That still pins the gradient's value.
    eps = 1e-4
    plus = _point_objective(_build(amp=1.7 + eps).sar("loss")).item()
    minus = _point_objective(_build(amp=1.7 - eps).sar("loss")).item()
    fd = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(float(grad), fd, rtol=2e-3, atol=1e-8)


def test_averaged_sar_field_gradient_matches_central_difference():
    m0 = 1000.0 * (3 * _DX) ** 3
    averaging = mw.SARAveraging(mass=(m0,))
    amp = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    sar = _build(amp=amp).sar("loss", averaging=averaging)
    obj = _averaged_objective(sar, m0)
    (grad,) = torch.autograd.grad(obj, amp)

    eps = 1e-4
    plus = _averaged_objective(_build(amp=1.3 + eps).sar("loss", averaging=averaging), m0).item()
    minus = _averaged_objective(_build(amp=1.3 - eps).sar("loss", averaging=averaging), m0).item()
    fd = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(float(grad), fd, rtol=2e-3, atol=1e-8)


def test_point_sar_density_grid_gradient_matches_central_difference():
    shape = _grid_shape()
    rho0 = 1000.0
    rho_grid = torch.full(shape, rho0, dtype=torch.float32, requires_grad=True)
    sar = _build(amp=2.0, rho=rho_grid).sar("loss")
    obj = _point_objective(sar)
    (grad,) = torch.autograd.grad(obj, rho_grid)

    # Perturb a single interior voxel and compare to the analytic autograd grad.
    idx = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    eps = 1.0
    grid_plus = torch.full(shape, rho0, dtype=torch.float32)
    grid_plus[idx] += eps
    grid_minus = torch.full(shape, rho0, dtype=torch.float32)
    grid_minus[idx] -= eps
    plus = _point_objective(_build(amp=2.0, rho=grid_plus).sar("loss")).item()
    minus = _point_objective(_build(amp=2.0, rho=grid_minus).sar("loss")).item()
    fd = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(float(grad[idx]), fd, rtol=2e-3, atol=1e-9)


def test_point_sar_conductivity_central_difference_matches_analytic():
    """SAR is linear in conductivity: d(sum SAR)/d(sigma) == sum(SAR)/sigma."""
    sigma0 = 0.6
    sar = _build(amp=2.0, sigma=sigma0).sar("loss")
    base = float(_point_objective(sar))

    # A wide step beats float32 catastrophic cancellation; SAR is exactly linear
    # in sigma, so truncation error is zero regardless of step.
    eps = 0.05
    plus = float(_point_objective(_build(amp=2.0, sigma=sigma0 + eps).sar("loss")))
    minus = float(_point_objective(_build(amp=2.0, sigma=sigma0 - eps).sar("loss")))
    fd = (plus - minus) / (2 * eps)
    analytic = base / sigma0
    np.testing.assert_allclose(fd, analytic, rtol=1e-4, atol=1e-9)


def test_point_sar_density_central_difference_matches_analytic():
    """Point SAR ~ 1/rho, so d(sum SAR)/d(rho) == -sum(SAR)/rho for a uniform density."""
    rho0 = 950.0
    sar = _build(amp=2.0, rho=rho0).sar("loss")
    base = float(_point_objective(sar))

    eps = 1.0
    plus = float(_point_objective(_build(amp=2.0, rho=rho0 + eps).sar("loss")))
    minus = float(_point_objective(_build(amp=2.0, rho=rho0 - eps).sar("loss")))
    fd = (plus - minus) / (2 * eps)
    analytic = -base / rho0
    np.testing.assert_allclose(fd, analytic, rtol=1e-3, atol=1e-9)
