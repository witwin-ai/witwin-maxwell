"""Slice 2a: general (non-axis-aligned) bias gyromagnetic forward.

Gates (track G3a):
  * Rotation equivalence (headline) -- the general-bias code path reduces to the
    axis-aligned fast path BIT-FOR-BIT for an axis-aligned bias (b = z, x, y), on
    both the pure-recurrence primitive and the coupled implicit-midpoint forward
    step, and on the real CUDA FDTD field update.
  * Oblique bias vs oracle (headline) -- b = (1,1,1)/sqrt(3) CW response matches the
    Phase-0 torch reference oracle to ``reference_polder_rtol``; falsified by
    flipping the gyrotropy sign.
  * Oblique passivity -- discrete magnetic energy does not grow at zero damping.
  * The general path builds and runs (the former general-bias fail-closed guard is
    lifted); mixed-bias and Bloch still fail closed (covered in the sibling suite).

The mock-solver gates test the exact runtime code path at float64 without the CUDA
field update; the CUDA gates build the real FDTD solver.
"""

from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.core import Box
from witwin.maxwell.fdtd import ferrite_reference as oracle
from witwin.maxwell.fdtd.runtime.gyromagnetic import (
    advance_gyromagnetic_state,
    build_gyromagnetic,
    initialize_gyromagnetic_state,
    snapshot_gyromagnetic_drive,
    step_gyromagnetic_coupled,
)
from witwin.maxwell.scene import prepare_scene

MU_0 = 4.0e-7 * math.pi
OBLIQUE = (1.0, 1.0, 1.0)  # b_hat = (1,1,1)/sqrt(3)


def _ferrite(bias=(0.0, 0.0, 1.75e5), **kwargs):
    base = dict(eps_r=14.5, saturation_magnetization=1.40e5, bias_field=bias, gilbert_damping=2.0e-3)
    base.update(kwargs)
    return mw.GyromagneticFerrite(**base)


def _uniform_ferrite_scene(material, *, spacing=0.02, half=0.06):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(10.0, 10.0, 10.0)), material=material))
    return prepare_scene(scene)


class _MockSolver:
    """Minimal solver surface the gyromagnetic runtime touches (no field update)."""

    def __init__(self, scene, dt, *, dtype=torch.float64):
        self.scene = scene
        self.dt = float(dt)
        self.device = torch.device("cpu")
        self.Nx, self.Ny, self.Nz = int(scene.Nx), int(scene.Ny), int(scene.Nz)
        self.Hx = torch.zeros((self.Nx, self.Ny - 1, self.Nz - 1), dtype=dtype)
        self.Hy = torch.zeros((self.Nx - 1, self.Ny, self.Nz - 1), dtype=dtype)
        self.Hz = torch.zeros((self.Nx - 1, self.Ny - 1, self.Nz), dtype=dtype)


# --- rotation equivalence: general path reduces to the fast path bit-for-bit ---


@pytest.mark.parametrize("axis", [2, 0, 1])
def test_general_reduces_to_fast_bitwise(axis):
    """A general-bias solver forced onto an axis-aligned bias must reproduce the
    axis-aligned fast path BIT-FOR-BIT.

    This is the rotation-equivalence headline: the general path is a pure
    coordinate rotation of the same discretized update, so for b = +e_axis the
    local frame ``[u|v|w]`` is a signed permutation and every general-path
    projection collapses (``1*H + 0*H + 0*H``) to exactly the single-component
    fast-path arithmetic. Both b = z and b = x (the 90-degree-rotated axis) are
    exercised.

    Falsification: perturbing any general-path projection field (e.g.
    ``state['ux'] += eps``) makes the coupled step diverge from the fast path and
    this bitwise assertion fails -- see the acceptance doc.
    """
    bias = [0.0, 0.0, 0.0]
    bias[axis] = 1.75e5
    scene = _uniform_ferrite_scene(_ferrite(bias=tuple(bias), gilbert_damping=5e-3))
    fast = _MockSolver(scene, 1e-12)
    build_gyromagnetic(fast, scene)
    gen = _MockSolver(scene, 1e-12)
    build_gyromagnetic(gen, scene, force_general=True)
    assert fast._gyromagnetic_state["general"] is False
    assert gen._gyromagnetic_state["general"] is True

    torch.manual_seed(0)
    for _ in range(40):
        pre = [torch.randn_like(fast.Hx), torch.randn_like(fast.Hy), torch.randn_like(fast.Hz)]
        for solver in (fast, gen):
            solver.Hx.copy_(pre[0]); solver.Hy.copy_(pre[1]); solver.Hz.copy_(pre[2])
            snapshot_gyromagnetic_drive(solver)
        tmp = [torch.randn_like(fast.Hx), torch.randn_like(fast.Hy), torch.randn_like(fast.Hz)]
        for solver in (fast, gen):
            solver.Hx.copy_(tmp[0]); solver.Hy.copy_(tmp[1]); solver.Hz.copy_(tmp[2])
            step_gyromagnetic_coupled(solver)
        for name in ("Hx", "Hy", "Hz"):
            assert torch.equal(getattr(fast, name), getattr(gen, name)), f"{name} not bitwise-equal"
        for name in ("m_u", "m_v"):
            assert torch.equal(fast._gyromagnetic_state[name], gen._gyromagnetic_state[name])


def test_general_path_perturbation_breaks_bitwise_reduction():
    """Falsification harness for the reduction gate: corrupting a general-path
    projection field makes the general step diverge from the fast path, so the
    bitwise reduction has teeth (it is not vacuously satisfied)."""
    scene = _uniform_ferrite_scene(_ferrite(bias=(0.0, 0.0, 1.75e5)))
    fast = _MockSolver(scene, 1e-12)
    build_gyromagnetic(fast, scene)
    gen = _MockSolver(scene, 1e-12)
    build_gyromagnetic(gen, scene, force_general=True)
    gen._gyromagnetic_state["ux"].add_(1.0e-6)  # corrupt the u.H gather
    torch.manual_seed(1)
    fast.Hx.copy_(torch.randn_like(fast.Hx)); gen.Hx.copy_(fast.Hx)
    snapshot_gyromagnetic_drive(fast); snapshot_gyromagnetic_drive(gen)
    step_gyromagnetic_coupled(fast); step_gyromagnetic_coupled(gen)
    assert not torch.equal(fast.Hx, gen.Hx)


# --- oblique bias vs the Phase-0 torch reference oracle -----------------------


def _extract_chi_general(solver, omega, *, settle_periods=300, window_periods=300):
    """Drive a uniform lab field H = u * cos(omega t) and DFT-extract local chi.

    Driving purely along the local ``u`` axis gives h_u = u.H = cos, h_v = v.H = 0
    (u,v orthonormal), so the extracted ``(m_u, m_v)`` are directly
    ``(chi_uu, chi_vu)`` at the drive frequency -- the general-path analogue of the
    axis-aligned CW extractor. With a spatially-uniform field the identity
    collocation is exact, isolating the arbitrary-axis ADE precession.
    """
    state = solver._gyromagnetic_state
    ux = float(state["ux"].flatten()[0])
    uy = float(state["uy"].flatten()[0])
    uz = float(state["uz"].flatten()[0])
    hxs, hys, hzs = state["hx_slice"], state["hy_slice"], state["hz_slice"]
    dt = solver.dt
    steps_per_period = int(round(2.0 * math.pi / omega / dt))
    n_settle = settle_periods * steps_per_period
    n_window = window_periods * steps_per_period
    initialize_gyromagnetic_state(solver)
    acc = torch.zeros(2, dtype=torch.complex128)
    for n in range(n_settle + n_window):
        c = math.cos(omega * (n + 0.5) * dt)
        solver.Hx.zero_(); solver.Hy.zero_(); solver.Hz.zero_()
        solver.Hx[hxs] = ux * c
        solver.Hy[hys] = uy * c
        solver.Hz[hzs] = uz * c
        advance_gyromagnetic_state(solver)
        if n >= n_settle:
            t_state = (n + 1) * dt
            phase = complex(math.cos(omega * t_state), math.sin(omega * t_state))
            m_u = complex(float(state["m_u"].flatten()[0]))
            m_v = complex(float(state["m_v"].flatten()[0]))
            acc = acc + torch.tensor([m_u, m_v], dtype=torch.complex128) * phase
    chi = 2.0 * acc / n_window
    return chi[0], chi[1]


def _oracle_params(material):
    return oracle.FerriteReferenceParameters(
        saturation_magnetization=material.saturation_magnetization,
        bias_magnitude=material.bias_magnitude,
        bias_unit_vector=material.bias_unit_vector,
        gilbert_damping=material.gilbert_damping,
        gyromagnetic_ratio=material.gyromagnetic_ratio,
        mu_infinity=material.mu_infinity,
    )


def test_oblique_cw_matches_oracle():
    """b = (1,1,1)/sqrt(3): CW susceptibility matches the frozen discrete Polder
    oracle to ``reference_polder_rtol`` (a pure coordinate rotation preserves the
    scalar response)."""
    material = _ferrite(bias=OBLIQUE, gilbert_damping=1.0e-2)
    omega = 0.8 * material.omega_0
    dt = (2.0 * math.pi / omega) / 64
    scene = _uniform_ferrite_scene(material)
    solver = _MockSolver(scene, dt)
    build_gyromagnetic(solver, scene)
    assert solver._gyromagnetic_state["general"] is True

    chi_uu, chi_vu = _extract_chi_general(solver, omega)
    chi_ref = oracle.discrete_susceptibility(_oracle_params(material), omega, dt)
    rtol = oracle.ACCEPTANCE_BUDGET.reference_polder_rtol
    assert chi_uu == pytest.approx(complex(chi_ref[0, 0]), rel=rtol)
    assert chi_vu == pytest.approx(complex(chi_ref[1, 0]), rel=rtol)


def test_oblique_cw_gyrotropy_sign_falsification():
    """The oblique-vs-oracle gate has teeth: flipping the gyrotropy (precession)
    sign in the runtime makes the extracted off-diagonal response mismatch the
    oracle's kappa sign.

    Falsification: negating the skew off-diagonals of the compiled Phi/Gamma
    reverses the precession sense (kappa -> -kappa), so chi_vu flips sign while the
    oracle keeps the original sign -- the parity gate then fails, proving it is not
    passing on magnitude alone.
    """
    material = _ferrite(bias=OBLIQUE, gilbert_damping=1.0e-2)
    omega = 0.8 * material.omega_0
    dt = (2.0 * math.pi / omega) / 64
    scene = _uniform_ferrite_scene(material)
    solver = _MockSolver(scene, dt)
    build_gyromagnetic(solver, scene)
    state = solver._gyromagnetic_state
    for name in ("phi01", "phi10", "gamma01", "gamma10"):
        state[name].mul_(-1.0)  # reverse precession handedness -> flip kappa sign

    chi_uu, chi_vu = _extract_chi_general(solver, omega)
    chi_ref = oracle.discrete_susceptibility(_oracle_params(material), omega, dt)
    rtol = oracle.ACCEPTANCE_BUDGET.reference_polder_rtol
    # Diagonal (mu) unaffected, but the gyrotropic off-diagonal now has the wrong sign.
    assert chi_uu == pytest.approx(complex(chi_ref[0, 0]), rel=1e-3)
    assert chi_vu != pytest.approx(complex(chi_ref[1, 0]), rel=rtol)
    # It matches the sign-flipped oracle instead (kappa -> -kappa).
    assert chi_vu == pytest.approx(-complex(chi_ref[1, 0]), rel=1e-3)


# --- oblique passivity (discrete energy non-growth at zero damping) -----------


def test_oblique_energy_non_growth_lossless():
    """alpha = 0 oblique bias: undriven precession conserves the ADE energy exactly.

    The Cayley propagator is orthogonal in the local (u,v) plane independent of the
    lab orientation of that plane, so passivity is orientation-invariant.
    """
    material = _ferrite(bias=OBLIQUE, gilbert_damping=0.0)
    scene = _uniform_ferrite_scene(material)
    solver = _MockSolver(scene, 1.0e-12)
    build_gyromagnetic(solver, scene)
    state = solver._gyromagnetic_state
    state["m_u"].fill_(1.0)
    state["m_v"].fill_(-0.3)
    e0 = float((state["m_u"] ** 2 + state["m_v"] ** 2).sum())
    solver.Hx.zero_(); solver.Hy.zero_(); solver.Hz.zero_()
    for _ in range(200000):
        advance_gyromagnetic_state(solver)
    e1 = float((state["m_u"] ** 2 + state["m_v"] ** 2).sum())
    assert abs(e1 / e0 - 1.0) <= oracle.ACCEPTANCE_BUDGET.passive_energy_residual
    assert abs(e1 / e0 - 1.0) < 1e-9  # zero-growth (not merely bounded)


# --- the general path now builds (former fail-closed guard lifted) ------------


def test_oblique_bias_builds_and_engages_general_path():
    """A general (non-axis-aligned) bias now builds and engages the general path
    (the slice-1c axis-aligned fail-closed guard is lifted in slice 2a)."""
    scene = _uniform_ferrite_scene(_ferrite(bias=OBLIQUE))
    solver = _MockSolver(scene, 1e-12)
    build_gyromagnetic(solver, scene)
    assert solver.gyromagnetic_enabled
    state = solver._gyromagnetic_state
    assert state["general"] is True
    # u, v, w = b form a right-handed orthonormal frame; w is the bias unit vector.
    u = torch.tensor([state[k].flatten()[0] for k in ("ux", "uy", "uz")], dtype=torch.float64)
    v = torch.tensor([state[k].flatten()[0] for k in ("vx", "vy", "vz")], dtype=torch.float64)
    assert float(torch.dot(u, v)) == pytest.approx(0.0, abs=1e-12)
    assert float(u.norm()) == pytest.approx(1.0, abs=1e-12)
    assert float(v.norm()) == pytest.approx(1.0, abs=1e-12)


# --- CUDA integration: real FDTD field update --------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _cuda_scene(material, *, boundary=None, half_z=0.12):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-half_z, half_z))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=boundary or mw.BoundarySpec.pml(num_layers=6),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.08)), material=material)
    )
    return scene


@cuda
def test_cuda_general_matches_fast_bitwise():
    """On the real FDTD solver, forcing the general path on an axis-aligned (b=z)
    ferrite reproduces the fast-path propagation BIT-FOR-BIT.

    This binds the rotation-equivalence reduction to the production field update
    (not just the mock ADE), proving the general gather/scatter is correct in the
    runtime that writes the staggered Yee H.
    """
    from witwin.maxwell.fdtd.solver import FDTD

    def _run(force_general):
        solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
        solver.init_field()
        if force_general:
            build_gyromagnetic(solver, solver.scene, force_general=True)
        state = solver._gyromagnetic_state
        gen = torch.Generator(device=solver.Hx.device).manual_seed(4321)
        slc = state["hx_slice"] if state["general"] else state["u_slice"]
        seed = torch.rand(solver.Hx[slc].shape, generator=gen, device=solver.Hx.device, dtype=solver.Hx.dtype)
        solver.Hx[slc] = seed
        solver.solve(time_steps=200)
        return {n: getattr(solver, n).clone() for n in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    fast_fields = _run(False)
    gen_fields = _run(True)
    for name, fast in fast_fields.items():
        assert torch.equal(fast, gen_fields[name]), f"{name} fast vs forced-general mismatch"


@cuda
def test_cuda_oblique_forward_stable_and_precesses():
    """A real FDTD oblique-bias (1,1,1) ferrite runs stably and populates the
    magnetization; the general path is engaged."""
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite(bias=OBLIQUE, gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    assert state["general"] is True
    solver.Hx[state["hx_slice"]] = 1.0
    solver.solve(time_steps=200)
    assert torch.isfinite(solver.Hx).all() and torch.isfinite(solver.Hy).all()
    assert torch.isfinite(solver.Hz).all()
    assert float(state["m_u"].abs().max()) > 0.0
    assert float(state["m_v"].abs().max()) > 0.0
