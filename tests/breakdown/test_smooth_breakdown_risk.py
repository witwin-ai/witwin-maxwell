"""Gates for the differentiable smooth breakdown-risk surrogate (Phase-5 slice).

Capability level: differentiable-surrogate (non-physical, non-regulatory). These
tests validate the surrogate as an OPTIMIZATION objective, not a physical model:

* gradient flows from upstream source/material parameters through a recorded
  ``|E|(t)`` field to the risk scalar, matching central differences in float64
  on a small synthetic scene;
* the risk is monotone increasing in the source amplitude;
* the risk collapses to (numerically) zero far below the critical field;
* the colocation reused from the physical stress accumulator reproduces the
  analytic magnitude on a uniform field;
* the result is typed and tagged unmistakably non-physical / non-regulatory.

Each headline gate carries a recorded falsification (see the acceptance doc).
The surrogate is pure ``torch`` (no CUDA kernels), so these run on CPU float64.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.breakdown_risk import (
    SMOOTH_BREAKDOWN_CAPABILITY_LEVEL,
    SmoothBreakdownRisk,
    SmoothBreakdownRiskData,
)
from witwin.maxwell.breakdown_stress import colocate_electric_magnitude


# --------------------------------------------------------------------------- #
# Synthetic differentiable |E|(t, cell) field parameterized by source/material  #
# --------------------------------------------------------------------------- #


def _field_series(source_amplitude, material_scale, *, T=40, region=(3, 2, 2), dtype=torch.float64):
    """A small differentiable ``|E|(t, cell)`` driven by a source and a material param.

    ``source_amplitude`` scales the field linearly; ``material_scale`` divides it
    (a stand-in for a permittivity-like screening of the local field). A fixed
    time envelope and a fixed per-cell spatial pattern make the peak land in one
    corner cell, so the risk has non-trivial per-cell structure.
    """

    steps = torch.arange(T, dtype=dtype)
    envelope = torch.exp(-((steps - T / 3.0) ** 2) / (2.0 * (T / 6.0) ** 2))  # (T,)
    nx, ny, nz = region
    ramp = torch.linspace(0.4, 1.0, nx, dtype=dtype)
    pattern = ramp[:, None, None] * torch.ones((nx, ny, nz), dtype=dtype)  # (region)
    amp = source_amplitude / material_scale
    field = amp * envelope[:, None, None, None] * pattern[None, ...]
    return field  # (T, nx, ny, nz)


def _risk_scalar(source_amplitude, material_scale, risk: SmoothBreakdownRisk, dt=1e-10):
    field = _field_series(source_amplitude, material_scale)
    data = risk.evaluate(field, dt)
    return data.risk


# --------------------------------------------------------------------------- #
# Gradient gate (FD check, small scene, float64)                               #
# --------------------------------------------------------------------------- #


def _central_difference(fn, value, delta):
    plus = fn(value + delta)
    minus = fn(value - delta)
    return (plus - minus) / (2.0 * delta)


def test_gradient_flows_to_source_and_material_matches_fd():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    dt = 1.0e-10

    amp0 = torch.tensor(1.5e6, dtype=torch.float64, requires_grad=True)
    mat0 = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)

    risk_value = _risk_scalar(amp0, mat0, risk, dt=dt)
    risk_value.backward()
    grad_amp = float(amp0.grad)
    grad_mat = float(mat0.grad)

    # Central differences in float64, holding the other parameter fixed.
    amp_c = float(amp0.detach())
    mat_c = float(mat0.detach())

    def f_amp(a):
        t = torch.tensor(a, dtype=torch.float64)
        return float(_risk_scalar(t, torch.tensor(mat_c, dtype=torch.float64), risk, dt=dt))

    def f_mat(m):
        t = torch.tensor(m, dtype=torch.float64)
        return float(_risk_scalar(torch.tensor(amp_c, dtype=torch.float64), t, risk, dt=dt))

    fd_amp = _central_difference(f_amp, amp_c, amp_c * 1e-5)
    fd_mat = _central_difference(f_mat, mat_c, mat_c * 1e-5)

    assert grad_amp != 0.0 and grad_mat != 0.0, "surrogate produced a zero gradient"
    assert math.copysign(1.0, grad_amp) == math.copysign(1.0, fd_amp)
    assert math.copysign(1.0, grad_mat) == math.copysign(1.0, fd_mat)
    assert abs(grad_amp - fd_amp) / abs(fd_amp) < 1e-4, (grad_amp, fd_amp)
    assert abs(grad_mat - fd_mat) / abs(fd_mat) < 1e-4, (grad_mat, fd_mat)
    # Material screens the field, source drives it: opposite-sign sensitivities.
    assert grad_amp > 0.0 and grad_mat < 0.0


def test_gradient_reaches_scene_module_parameters():
    """Autograd threads a trainable torch.nn.Parameter through to the risk scalar."""

    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    amplitude = torch.nn.Parameter(torch.tensor(1.4e6, dtype=torch.float64))
    material = torch.nn.Parameter(torch.tensor(1.2, dtype=torch.float64))
    field = _field_series(amplitude, material)
    data = risk.evaluate(field, 1.0e-10)
    data.risk.backward()
    assert amplitude.grad is not None and float(amplitude.grad) > 0.0
    assert material.grad is not None and float(material.grad) < 0.0


# --------------------------------------------------------------------------- #
# Monotonicity gate                                                            #
# --------------------------------------------------------------------------- #


def test_risk_monotone_in_source_amplitude():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    dt = 1.0e-10
    amplitudes = [0.8e6, 1.0e6, 1.2e6, 1.5e6, 2.0e6]
    values = [
        float(_risk_scalar(torch.tensor(a, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64), risk, dt=dt))
        for a in amplitudes
    ]
    for lower, upper in zip(values, values[1:]):
        assert upper > lower, values


# --------------------------------------------------------------------------- #
# Zero-risk far below threshold                                                #
# --------------------------------------------------------------------------- #


def test_risk_vanishes_far_below_threshold():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e4)
    # Peak field is ~1e-2 of the critical field: margin ~ -50 widths -> sigmoid underflows.
    field = _field_series(
        torch.tensor(1.0e4, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    assert float(field.max()) < 0.02 * risk.critical_field
    data = risk.evaluate(field, 1.0e-10)
    # Margin is ~ -50 widths, so both the dose and the instantaneous surrogate are
    # negligible against an at-threshold dose (order 1e-9 s, see the test below).
    assert float(data.risk) < 1e-18
    assert float(data.peak_instant_risk) < 1e-15


def test_risk_grows_from_negligible_to_finite_across_threshold():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=5.0e4)
    dt = 1.0e-10
    far = float(_risk_scalar(torch.tensor(2.0e5, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64), risk, dt=dt))
    at = float(_risk_scalar(torch.tensor(2.0e6, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64), risk, dt=dt))
    assert far < 1e-11
    assert at > 1e-10
    assert at > 1e6 * far


# --------------------------------------------------------------------------- #
# Colocation reuse (same |E| as the physical stress accumulator)               #
# --------------------------------------------------------------------------- #


def test_from_components_matches_manual_colocation_and_uniform_field():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    T, nx, ny, nz = 5, 3, 2, 2
    dtype = torch.float64
    # Node-overhang Yee blocks: each component one node larger on its two node axes.
    ex = torch.rand((T, nx, ny + 1, nz + 1), dtype=dtype) * 1.0e6
    ey = torch.rand((T, nx + 1, ny, nz + 1), dtype=dtype) * 1.0e6
    ez = torch.rand((T, nx + 1, ny + 1, nz), dtype=dtype) * 1.0e6

    manual = colocate_electric_magnitude(ex, ey, ez)
    assert tuple(manual.shape) == (T, nx, ny, nz)
    via_components = risk.evaluate_from_components(ex, ey, ez, 1.0e-10)
    via_manual = risk.evaluate(manual, 1.0e-10)
    torch.testing.assert_close(via_components.risk, via_manual.risk)

    # Uniform field: colocation reproduces the analytic magnitude exactly. Each
    # Yee component keeps its own staggered overhang shape.
    val = 3.0e5
    uni_ex = torch.full((T, nx, ny + 1, nz + 1), val, dtype=dtype)
    uni_ey = torch.full((T, nx + 1, ny, nz + 1), val, dtype=dtype)
    uni_ez = torch.full((T, nx + 1, ny + 1, nz), val, dtype=dtype)
    mag = colocate_electric_magnitude(uni_ex, uni_ey, uni_ez)
    torch.testing.assert_close(mag, torch.full((T, nx, ny, nz), math.sqrt(3) * val, dtype=dtype))


def test_batched_colocation_matches_per_step_loop():
    dtype = torch.float64
    T, nx, ny, nz = 4, 2, 3, 2
    ex = torch.rand((T, nx, ny + 1, nz + 1), dtype=dtype)
    ey = torch.rand((T, nx + 1, ny, nz + 1), dtype=dtype)
    ez = torch.rand((T, nx + 1, ny + 1, nz), dtype=dtype)
    batched = colocate_electric_magnitude(ex, ey, ez)
    for t in range(T):
        per_step = colocate_electric_magnitude(ex[t], ey[t], ez[t])
        torch.testing.assert_close(batched[t], per_step)


# --------------------------------------------------------------------------- #
# Reductions and diagnostics                                                   #
# --------------------------------------------------------------------------- #


def test_softmax_reduction_between_mean_and_peak():
    dt = 1.0e-10
    field = _field_series(
        torch.tensor(1.8e6, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    sum_risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5, reduction="sum")
    mean_risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5, reduction="mean")
    soft_risk = SmoothBreakdownRisk(
        critical_field=1.0e6, width=2.0e5, reduction="softmax", temperature=1.0e-11
    )

    per_cell = sum_risk.evaluate(field, dt).soft_duration_map
    mean_val = float(mean_risk.evaluate(field, dt).risk)
    soft_val = float(soft_risk.evaluate(field, dt).risk)
    hard_peak = float(per_cell.max())
    plain_mean = float(per_cell.mean())

    assert plain_mean == pytest.approx(mean_val, rel=1e-9)
    assert mean_val <= soft_val <= hard_peak + 1e-30
    assert soft_val > mean_val


def test_occupancy_weighting_zeroes_unoccupied_cells():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    field = _field_series(
        torch.tensor(1.8e6, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    region = field.shape[1:]
    full = risk.evaluate(field, 1.0e-10).risk
    occ = torch.ones(region, dtype=torch.float64)
    occ[0] = 0.0  # drop the weakest-field plane
    masked = risk.evaluate(field, 1.0e-10, occupancy=occ).risk
    assert float(masked) < float(full)
    zero = risk.evaluate(field, 1.0e-10, occupancy=torch.zeros(region, dtype=torch.float64)).risk
    assert float(zero) == 0.0


def test_damage_exponent_produces_extra_map_only():
    base = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    dmg = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5, damage_exponent=2.0)
    field = _field_series(
        torch.tensor(1.8e6, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    base_data = base.evaluate(field, 1.0e-10)
    dmg_data = dmg.evaluate(field, 1.0e-10)
    assert base_data.soft_damage_map is None
    assert dmg_data.soft_damage_map is not None
    # The primary risk scalar is unchanged by enabling the auxiliary damage map.
    torch.testing.assert_close(base_data.risk, dmg_data.risk)


# --------------------------------------------------------------------------- #
# Typing / non-physical tagging                                               #
# --------------------------------------------------------------------------- #


def test_result_is_typed_and_tagged_non_physical():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    data = risk.evaluate(
        _field_series(torch.tensor(1.5e6, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)),
        1.0e-10,
        name="stress_zone",
    )
    assert isinstance(data, SmoothBreakdownRiskData)
    assert data.capability_level == SMOOTH_BREAKDOWN_CAPABILITY_LEVEL
    assert "non-physical" in data.capability_level and "non-regulatory" in data.capability_level
    assert data.provenance["non_physical"] is True
    assert data.provenance["non_regulatory"] is True
    assert data.provenance["model_version"] == "smooth-breakdown-risk-1"
    # Public API surface.
    assert mw.SmoothBreakdownRisk is SmoothBreakdownRisk
    assert mw.SmoothBreakdownRiskData is SmoothBreakdownRiskData


# --------------------------------------------------------------------------- #
# Fail-closed validation                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(critical_field=0.0, width=1.0), "critical_field"),
        (dict(critical_field=1.0, width=0.0), "width"),
        (dict(critical_field=1.0, width=1.0, reduction="bogus"), "reduction"),
        (dict(critical_field=1.0, width=1.0, reduction="softmax"), "temperature"),
        (dict(critical_field=1.0, width=1.0, damage_exponent=-1.0), "damage_exponent"),
    ],
)
def test_config_fails_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SmoothBreakdownRisk(**kwargs)


def test_evaluate_rejects_bad_inputs():
    risk = SmoothBreakdownRisk(critical_field=1.0e6, width=2.0e5)
    with pytest.raises(TypeError):
        risk.evaluate([1.0, 2.0], 1.0e-10)
    with pytest.raises(ValueError, match="time axis"):
        risk.evaluate(torch.ones(5, dtype=torch.float64), 1.0e-10)
    with pytest.raises(ValueError, match="dt"):
        risk.evaluate(torch.ones((3, 2), dtype=torch.float64), 0.0)
    field = _field_series(torch.tensor(1.5e6, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="occupancy"):
        risk.evaluate(field, 1.0e-10, occupancy=torch.ones((2, 2), dtype=torch.float64))
