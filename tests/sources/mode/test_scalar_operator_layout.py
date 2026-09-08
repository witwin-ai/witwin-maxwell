"""Independent stencil oracles for rectangular scalar mode apertures."""
import numpy as np
import pytest
import torch

from witwin.maxwell.fdtd.excitation.modes.eigensolvers import (
    _build_scalar_operator,
    _build_scalar_operator_torch_dense,
    _build_scalar_operator_torch_sparse,
    _lobpcg_initial_guess,
)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (1, 4), (4, 1)])
@pytest.mark.parametrize("complex_potential", [False, True])
def test_scalar_operator_matches_independent_rectangular_stencil(shape, complex_potential):
    nu, nv = shape
    du, dv = 0.7, 1.3
    potential = np.arange((nu + 2) * (nv + 2)).reshape(nu + 2, nv + 2) * 0.31
    if complex_potential:
        potential = potential + 0.03j * potential**2
    vector = np.sin(np.arange(nu * nv) + 0.2).reshape(nu, nv)
    padded = np.pad(vector, 1)
    expected = ((padded[:-2, 1:-1] - 2 * vector + padded[2:, 1:-1]) / du**2
                + (padded[1:-1, :-2] - 2 * vector + padded[1:-1, 2:]) / dv**2
                + potential[1:-1, 1:-1] * vector).ravel()
    operator, _, _ = _build_scalar_operator(potential, du, dv)
    np.testing.assert_allclose(operator @ vector.ravel(), expected, rtol=1e-12, atol=1e-12)
    tensor = torch.as_tensor(potential).requires_grad_()
    for builder in (_build_scalar_operator_torch_dense, _build_scalar_operator_torch_sparse):
        operator, _, _ = builder(tensor, du, dv)
        got = operator @ torch.as_tensor(vector.ravel(), dtype=tensor.dtype)
        torch.testing.assert_close(got, torch.as_tensor(expected), rtol=1e-12, atol=1e-12)
        grad, = torch.autograd.grad(got.real.sum(), tensor)
        torch.testing.assert_close(grad[1:-1, 1:-1].real, torch.as_tensor(vector), rtol=1e-12, atol=1e-12)


def test_sparse_starting_mode_uses_aperture_layout():
    nu, nv = 3, 4
    initial = _lobpcg_initial_guess(nu, nv, k=1, device="cpu", dtype=torch.float64)[:, 0]
    u = torch.arange(1, nu + 1, dtype=torch.float64)
    v = torch.arange(1, nv + 1, dtype=torch.float64)
    expected = torch.sin(torch.pi * u / (nu + 1))[:, None] * torch.sin(torch.pi * v / (nv + 1))[None, :]
    torch.testing.assert_close(initial, expected.flatten(), rtol=1e-12, atol=1e-12)
