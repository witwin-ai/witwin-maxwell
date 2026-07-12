from __future__ import annotations

import pytest
import torch

from witwin.maxwell.fdtd.boundary import BOUNDARY_BLOCH, BOUNDARY_NONE
from witwin.maxwell.fdtd.cuda.backend import get_native_fdtd_module


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    ("normal_axis", "tangent_axis"),
    [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)],
)
def test_native_cpml_correction_is_exact_discrete_transpose(normal_axis, tangent_axis):
    """Check the mixed Bloch/CPML local operator without a Torch adjoint copy."""
    torch.manual_seed(120 + 3 * normal_axis + tangent_axis)
    shape = (5, 6, 7)
    device = "cuda"
    psi = torch.randn(shape, device=device)
    derivative = torch.randn(shape, device=device)
    adj_field = torch.randn(shape, device=device)
    adj_psi = torch.randn(shape, device=device)
    curl = torch.randn(shape, device=device)
    axis_size = shape[normal_axis]
    b = torch.rand(axis_size, device=device)
    c = torch.randn(axis_size, device=device)
    inv_kappa = 0.5 + torch.rand(axis_size, device=device)
    sign = -1.0 if (normal_axis + tangent_axis) % 2 else 1.0

    view_shape = [1, 1, 1]
    view_shape[normal_axis] = axis_size
    b3 = b.view(view_shape)
    c3 = c.view(view_shape)
    k3 = inv_kappa.view(view_shape)
    active = torch.zeros(shape, dtype=torch.bool, device=device)
    interior = [slice(None)] * 3
    interior[normal_axis] = slice(1, -1)
    active[tuple(interior)] = True
    psi_post = torch.where(active, b3 * psi + c3 * derivative, psi)
    field_correction = torch.where(
        active,
        sign * curl * ((k3 - 1.0) * derivative + psi_post),
        torch.zeros_like(psi),
    )

    adj_psi_prev = torch.empty_like(psi)
    adj_derivative = torch.empty_like(derivative)
    get_native_fdtd_module().reverseCpmlCorrection3D(
        AdjPsiPrev=adj_psi_prev,
        AdjDerivative=adj_derivative,
        AdjField=adj_field,
        AdjPsiPost=adj_psi,
        Curl=curl,
        B=b,
        C=c,
        InvKappa=inv_kappa,
        NormalAxis=normal_axis,
        TangentAxis=tangent_axis,
        TangentLowMode=BOUNDARY_BLOCH,
        TangentHighMode=BOUNDARY_BLOCH,
        Sign=sign,
    ).launchRaw()

    lhs = torch.sum(field_correction * adj_field) + torch.sum(psi_post * adj_psi)
    rhs = torch.sum(psi * adj_psi_prev) + torch.sum(derivative * adj_derivative)
    torch.testing.assert_close(lhs, rhs, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("parameter", "gradient_output"),
    [("chi2", "GradChi2"), ("chi3", "GradChi3"), ("tpa", "GradTpa")],
)
def test_native_nonlinear_coefficient_pullback_matches_central_difference(
    parameter, gradient_output
):
    shape = (3, 3, 3)
    device = "cuda"
    values = {
        "eps": 2.0,
        "chi2": 0.2,
        "chi3": 0.3,
        "tpa": 0.1,
        "sigma": 0.05,
        "fsq": 0.4,
        "own": 0.25,
        "external": 0.9,
    }
    dt = 0.05
    eps0 = 1.0
    tensors = {
        name: torch.full(shape, value, device=device)
        for name, value in values.items()
    }
    torch.manual_seed(303)
    h_pos = torch.randn(shape, device=device)
    h_neg = torch.randn(shape, device=device)
    psi_pos = torch.randn(shape, device=device)
    psi_neg = torch.randn(shape, device=device)
    adj_post = torch.randn(shape, device=device)
    adj_psi_pos_post = torch.randn(shape, device=device)
    adj_psi_neg_post = torch.randn(shape, device=device)
    b_pos = torch.tensor([0.7, 0.8, 0.75], device=device)
    c_pos = torch.tensor([0.1, -0.2, 0.15], device=device)
    k_pos = torch.tensor([0.9, 0.85, 0.8], device=device)
    b_neg = torch.tensor([0.65, 0.78, 0.72], device=device)
    c_neg = torch.tensor([-0.1, 0.12, 0.18], device=device)
    k_neg = torch.tensor([0.82, 0.88, 0.86], device=device)
    inv = torch.ones(3, device=device)
    outputs = {name: torch.empty(shape, device=device) for name in (
        "AdjPrev", "GradEps", "GradChi2", "GradChi3", "GradTpa", "GFsq",
        "AdjPsiPosPrev", "AdjPsiNegPrev", "AdjDPos", "AdjDNeg",
    )}

    get_native_fdtd_module().reverseElectricComponentCpmlNonlinear3D(
        component=0,
        **outputs,
        AdjPost=adj_post,
        AdjPsiPosPost=adj_psi_pos_post,
        AdjPsiNegPost=adj_psi_neg_post,
        EPrev=tensors["own"],
        ExternalDecay=tensors["external"],
        Eps=tensors["eps"],
        Chi2=tensors["chi2"],
        Chi3=tensors["chi3"],
        Tpa=tensors["tpa"],
        SigmaStatic=tensors["sigma"],
        Fsq=tensors["fsq"],
        Dt=dt,
        Eps0=eps0,
        PsiPos=psi_pos,
        PsiNeg=psi_neg,
        BPos=b_pos,
        CPos=c_pos,
        InvKappaPos=k_pos,
        BNeg=b_neg,
        CNeg=c_neg,
        InvKappaNeg=k_neg,
        HPosMid=h_pos,
        HNegMid=h_neg,
        InvPos=inv,
        InvNeg=inv,
        LowModeA=BOUNDARY_NONE,
        HighModeA=BOUNDARY_NONE,
        LowModeB=BOUNDARY_NONE,
        HighModeB=BOUNDARY_NONE,
    ).launchRaw()

    i = j = k = 1
    d_pos = h_pos[i, j, k] - h_pos[i, j - 1, k]
    d_neg = h_neg[i, j, k] - h_neg[i, j, k - 1]
    psi_p = b_pos[j] * psi_pos[i, j, k] + c_pos[j] * d_pos
    psi_n = b_neg[k] * psi_neg[i, j, k] + c_neg[k] * d_neg
    curl_term = d_pos * k_pos[j] + psi_p - d_neg * k_neg[k] - psi_n

    def objective(value):
        local = dict(values)
        local[parameter] = value
        eff = local["eps"] + eps0 * (
            local["chi2"] * local["own"] + local["chi3"] * local["fsq"]
        )
        sigma = local["sigma"] + local["tpa"] * local["fsq"]
        q = 0.5 * sigma * dt
        decay = local["external"] * (eff - q) / (eff + q)
        curl_coeff = local["external"] * dt / (eff + q)
        return adj_post[i, j, k].item() * (
            decay * local["own"] + curl_coeff * curl_term.item()
        )

    h = 1e-3
    expected = (objective(values[parameter] + h) - objective(values[parameter] - h)) / (2 * h)
    assert outputs[gradient_output][i, j, k].item() == pytest.approx(expected, rel=3e-4, abs=2e-5)
