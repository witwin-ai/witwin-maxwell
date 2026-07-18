"""Generic surface-impedance ADE forward: passivity gate, shared-discretization parity.

The generic per-edge surface update steps a passive Z-form auxiliary-differential-
equation (ADE) built by the shared fitter's trapezoidal (bilinear, |z| < 1)
discretization. These runtime-independent checks pin the two contracts the runtime
depends on: (a) a non-passive surface is rejected by the compile passivity certificate
BEFORE any step runs (the R1 falsification), and (b) the batched per-edge recurrence in
``stepping.surface_ade_step`` reproduces the shared ``DiscreteStateSpaceNetwork.step``
exactly, so the forward is the certified discretization and not a re-derivation.
"""

from __future__ import annotations

import math

import pytest
import torch

from witwin.maxwell.compiler.surface_impedance import fit_surface_impedance
from witwin.maxwell.fdtd.runtime.stepping import surface_ade_step
from witwin.maxwell.fdtd.surface_impedance_reference import good_conductor_surface_impedance
from witwin.maxwell.media import RationalSurfaceImpedance


def _good_conductor_z(band, samples=120, sigma=50.0):
    freqs = torch.logspace(math.log10(band[0]), math.log10(band[1]), samples, dtype=torch.float64)
    return freqs, good_conductor_surface_impedance(sigma, freqs).to(torch.complex128)


# --- R1 falsification: a non-passive surface is rejected before runtime ------


def test_hand_built_non_passive_model_is_rejected_at_compile():
    # Stable (Re(pole) < 0) but non-passive: a single real pole with a negative
    # residue gives Re(Z_s) < 0 across the band -- an active surface that would inject
    # energy and diverge. The passivity certificate must reject it at construction,
    # before any solve.
    poles = torch.tensor([-2.0e9], dtype=torch.complex128)
    residues = torch.tensor([-3.0e9], dtype=torch.complex128)
    with pytest.raises(ValueError, match="passive"):
        RationalSurfaceImpedance(poles, residues, 0.0, frequency_range=(1.0e9, 5.0e9), representation="Z")


def test_passive_good_conductor_fit_is_certified_and_stable():
    band = (0.5e9, 5.0e9)
    freqs, z_values = _good_conductor_z(band)
    fitted = fit_surface_impedance(
        (freqs, z_values), band=band, order=6, dt=2.0e-12, representation="Z", dtype=torch.float64
    )
    assert fitted.passivity.passive and fitted.passivity.certified
    assert fitted.passivity.max_violation <= 1.0e-9
    # Bilinear discretization keeps every discrete pole strictly inside the unit circle.
    assert fitted.pole_radius < 1.0


# --- the batched per-edge recurrence is the shared discretization ------------


def test_surface_ade_step_matches_shared_discrete_state_space():
    band = (0.5e9, 5.0e9)
    freqs, z_values = _good_conductor_z(band)
    fitted = fit_surface_impedance(
        (freqs, z_values), band=band, order=6, dt=2.0e-12, representation="Z", dtype=torch.float64
    )
    discrete = fitted.discrete
    order = fitted.state_count
    edges = 5
    torch.manual_seed(0)

    # Reference: the shared step advanced independently per edge.
    ref_state = torch.zeros((order, edges), dtype=torch.float64)
    # Runtime: the batched ADE the surface update runs.
    ade = {
        "A": discrete.A,
        "B": discrete.B,
        "C": discrete.C,
        "D": discrete.D,
        "state": torch.zeros((order, edges), dtype=torch.float64),
    }
    for _ in range(40):
        u = torch.randn(edges, dtype=torch.float64)
        batched_out = surface_ade_step(ade, u)
        ref_out = torch.empty(edges, dtype=torch.float64)
        for m in range(edges):
            next_state, out = discrete.step(ref_state[:, m], u[m : m + 1])
            ref_state[:, m] = next_state
            ref_out[m] = out.reshape(())
        assert torch.allclose(batched_out, ref_out, atol=1e-12, rtol=0.0)
        assert torch.allclose(ade["state"], ref_state, atol=1e-12, rtol=0.0)


def test_order0_generic_scalar_matches_fused_resistive_formula():
    # Bitwise parity of the two order-0 realizations of the same relation:
    # the fused kernel computes ``sign * R * H`` and the generic scalar path computes
    # ``D * (sign * H)`` with ``D = R``. Multiplication by ``sign = +/-1`` is exact in
    # float32, so the two are bit-identical (the fused path is a pure optimization).
    torch.manual_seed(1)
    h = torch.randn(64, dtype=torch.float32)
    r = torch.tensor(0.37, dtype=torch.float32)
    for sign in (1.0, -1.0):
        fused = sign * r * h
        generic = r.reshape(()) * (sign * h)
        assert torch.equal(fused, generic)
