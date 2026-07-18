"""Slice 1c: gyromagnetic ferrite forward runtime.

Gates (design brief slice 1c):
  * CUDA/runtime-vs-torch-reference parity -- the Phase-0 LLG oracle is the judge;
  * circular eigen-permeabilities / bias-reversal non-reciprocity;
  * discrete-energy non-growth at zero damping (falsifiable zero-growth form);
  * composes with PML / conductivity;
  * ferrite-free scene adds zero gyromagnetic operations (gated flag);
  * general (non-axis-aligned) bias and Bloch fail closed.

The runtime-vs-oracle gates run on a lightweight float64 mock solver so they test
the exact stepping code path without the CUDA field update; the integration gates
build the real FDTD solver on CUDA.
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
    apply_gyromagnetic_correction,
    build_gyromagnetic,
    initialize_gyromagnetic_state,
)
from witwin.maxwell.scene import prepare_scene

MU_0 = 4.0e-7 * math.pi


def _ferrite(bias=(0.0, 0.0, 1.75e5), **kwargs):
    base = dict(eps_r=14.5, saturation_magnetization=1.40e5, bias_field=bias, gilbert_damping=2.0e-3)
    base.update(kwargs)
    return mw.GyromagneticFerrite(**base)


def _uniform_ferrite_scene(material, *, spacing=0.02, half=0.06):
    """Small scene fully filled by the ferrite (all overlap cells active/uniform)."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(10.0, 10.0, 10.0)), material=material))
    return prepare_scene(scene)


class _MockSolver:
    """Minimal solver surface build_gyromagnetic / advance / apply touch."""

    def __init__(self, scene, dt, *, dtype=torch.float64):
        self.scene = scene
        self.dt = float(dt)
        self.device = torch.device("cpu")
        self.Nx, self.Ny, self.Nz = int(scene.Nx), int(scene.Ny), int(scene.Nz)
        self.Hx = torch.zeros((self.Nx, self.Ny - 1, self.Nz - 1), dtype=dtype)
        self.Hy = torch.zeros((self.Nx - 1, self.Ny, self.Nz - 1), dtype=dtype)
        self.Hz = torch.zeros((self.Nx - 1, self.Ny - 1, self.Nz), dtype=dtype)


def _build_mock(material, dt):
    scene = _uniform_ferrite_scene(material)
    solver = _MockSolver(scene, dt)
    build_gyromagnetic(solver, scene)
    return solver


# --- exact wiring parity: advance == Phi m + Gamma h -------------------------


def test_advance_matches_cayley_recurrence_exactly():
    """The runtime advance is exactly m <- Phi m + Gamma h at every active cell."""
    material = _ferrite(gilbert_damping=5e-3)
    dt = 1.0e-12
    solver = _build_mock(material, dt)
    state = solver._gyromagnetic_state
    phi = torch.tensor([[state["phi00"].flatten()[0], state["phi01"].flatten()[0]],
                        [state["phi10"].flatten()[0], state["phi11"].flatten()[0]]], dtype=torch.float64)
    gamma = torch.tensor([[state["gamma00"].flatten()[0], state["gamma01"].flatten()[0]],
                          [state["gamma10"].flatten()[0], state["gamma11"].flatten()[0]]], dtype=torch.float64)
    m_ref = torch.zeros(2, dtype=torch.float64)
    torch.manual_seed(0)
    for _ in range(50):
        hu, hv = float(torch.randn(())), float(torch.randn(()))
        solver.Hx[state["u_slice"]] = hu / state["sign_u"]
        solver.Hy[state["v_slice"]] = hv / state["sign_v"]
        advance_gyromagnetic_state(solver)
        m_ref = phi @ m_ref + gamma @ torch.tensor([hu, hv], dtype=torch.float64)
        assert float(solver._gyromagnetic_state["m_u"].flatten()[0]) == pytest.approx(float(m_ref[0]), abs=1e-14)
        assert float(solver._gyromagnetic_state["m_v"].flatten()[0]) == pytest.approx(float(m_ref[1]), abs=1e-14)


# --- CW parity vs the frozen discrete Polder response ------------------------


def _extract_chi(solver, omega, *, settle_periods=300, window_periods=300):
    """Drive a lab-frame transverse H = (cos(omega t), 0) and DFT-extract lab chi.

    Returns ``(chi_uu, chi_vu)`` in the LAB frame: the drive is applied on the
    ``u``-axis lab H component and the response is read back through the local
    basis signs (``m_lab = sign * m_local``). For a ``+`` axis-aligned bias the lab
    and local frames coincide; for a reversed bias the lab off-diagonal flips sign.
    """
    dt = solver.dt
    state = solver._gyromagnetic_state
    su, sv = state["sign_u"], state["sign_v"]
    hu_field = getattr(solver, state["u_attr"])
    hv_field = getattr(solver, state["v_attr"])
    steps_per_period = int(round(2.0 * math.pi / omega / dt))
    n_settle = settle_periods * steps_per_period
    n_window = window_periods * steps_per_period
    initialize_gyromagnetic_state(solver)
    acc = torch.zeros(2, dtype=torch.complex128)
    for n in range(n_settle + n_window):
        t_half = (n + 0.5) * dt
        hu_field.zero_()
        hv_field.zero_()
        hu_field[state["u_slice"]] = math.cos(omega * t_half)  # lab-frame drive
        advance_gyromagnetic_state(solver)
        if n >= n_settle:
            t_state = (n + 1) * dt
            phase = complex(math.cos(omega * t_state), math.sin(omega * t_state))
            m_u = complex(su * float(state["m_u"].flatten()[0]))
            m_v = complex(sv * float(state["m_v"].flatten()[0]))
            acc = acc + torch.tensor([m_u, m_v], dtype=torch.complex128) * phase
    chi = 2.0 * acc / n_window
    return chi[0], chi[1]


def test_cw_response_matches_frozen_discrete_polder():
    material = _ferrite(gilbert_damping=1.0e-2)
    omega_0 = material.omega_0
    omega = 0.8 * omega_0
    steps_per_period = 64
    dt = (2.0 * math.pi / omega) / steps_per_period
    solver = _build_mock(material, dt)
    chi_uu, chi_vu = _extract_chi(solver, omega)

    params = oracle.FerriteReferenceParameters(
        saturation_magnetization=material.saturation_magnetization,
        bias_magnitude=material.bias_magnitude,
        bias_unit_vector=material.bias_unit_vector,
        gilbert_damping=material.gilbert_damping,
        gyromagnetic_ratio=material.gyromagnetic_ratio,
        mu_infinity=material.mu_infinity,
    )
    chi_ref = oracle.discrete_susceptibility(params, omega, dt)
    # chi_uu = mu - mu_inf ; chi_vu = +i*kappa (contract 2.3/2.4).
    assert chi_uu == pytest.approx(complex(chi_ref[0, 0]), rel=2e-3)
    assert chi_vu == pytest.approx(complex(chi_ref[1, 0]), rel=2e-3)


def test_bias_reversal_flips_kappa():
    """Bias reversal (b -> -b) flips the gyrotropic off-diagonal sign (non-reciprocity)."""
    omega_0 = _ferrite().omega_0
    omega = 0.8 * omega_0
    steps_per_period = 64
    dt = (2.0 * math.pi / omega) / steps_per_period
    up = _build_mock(_ferrite(bias=(0.0, 0.0, 1.75e5), gilbert_damping=1e-2), dt)
    down = _build_mock(_ferrite(bias=(0.0, 0.0, -1.75e5), gilbert_damping=1e-2), dt)
    _, kappa_up = _extract_chi(up, omega)
    _, kappa_down = _extract_chi(down, omega)
    # chi_vu = +i*kappa; the diagonal is unchanged, the off-diagonal flips sign.
    assert complex(kappa_down) == pytest.approx(-complex(kappa_up), rel=1e-6, abs=1e-9)


# --- discrete energy non-growth at zero damping ------------------------------


def test_energy_non_growth_lossless():
    """alpha = 0: free precession conserves the ADE energy exactly (no growth)."""
    material = _ferrite(gilbert_damping=0.0)
    dt = 1.0e-12
    solver = _build_mock(material, dt)
    state = solver._gyromagnetic_state
    state["m_u"].fill_(1.0)
    state["m_v"].fill_(-0.3)
    e0 = float((state["m_u"] ** 2 + state["m_v"] ** 2).sum())
    # Undriven precession: zero H drive, advance many steps.
    solver.Hx.zero_()
    solver.Hy.zero_()
    for _ in range(200000):
        advance_gyromagnetic_state(solver)
    e1 = float((state["m_u"] ** 2 + state["m_v"] ** 2).sum())
    assert abs(e1 / e0 - 1.0) <= oracle.ACCEPTANCE_BUDGET.passive_energy_residual
    assert abs(e1 / e0 - 1.0) < 1e-9  # zero-growth (not merely bounded)


def test_energy_decays_when_lossy():
    material = _ferrite(gilbert_damping=1e-2)
    dt = 1.0e-12
    solver = _build_mock(material, dt)
    state = solver._gyromagnetic_state
    state["m_u"].fill_(1.0)
    solver.Hx.zero_()
    solver.Hy.zero_()
    e0 = float((state["m_u"] ** 2 + state["m_v"] ** 2).sum())
    e_prev = e0
    for _ in range(500):
        advance_gyromagnetic_state(solver)
        e = float((state["m_u"] ** 2 + state["m_v"] ** 2).sum())
        assert e <= e_prev * (1.0 + 1e-12)  # monotone decay
        e_prev = e
    assert e_prev < 0.9 * e0  # strictly decayed (contraction, not conservation)


# --- fail-closed boundaries --------------------------------------------------


def test_general_bias_fails_closed():
    material = _ferrite(bias=(1.0e5, 1.0e5, 1.0e5))
    with pytest.raises(NotImplementedError, match="axis-aligned"):
        _build_mock(material, 1e-12)


def test_bloch_boundary_fails_closed():
    material = _ferrite()
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.06, 0.06))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.bloch(wavevector=(0.0, 0.0, 0.0)),
        device="cpu",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(10.0, 10.0, 10.0)), material=material))
    solver = _MockSolver(prepare_scene(scene), 1e-12)
    with pytest.raises(NotImplementedError, match="Bloch"):
        build_gyromagnetic(solver, solver.scene)


def test_ferrite_free_scene_disables_runtime():
    """A ferrite-free scene allocates no state and every hook is a no-op."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.06, 0.06))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(0.05, 0.05, 0.05)), material=mw.Material(eps_r=4.0)))
    solver = _MockSolver(prepare_scene(scene), 1e-12)
    build_gyromagnetic(solver, solver.scene)
    assert solver.gyromagnetic_enabled is False
    assert not hasattr(solver, "_gyromagnetic_state")
    before = solver.Hx.clone()
    advance_gyromagnetic_state(solver)  # no-op
    apply_gyromagnetic_correction(solver)  # no-op
    initialize_gyromagnetic_state(solver)  # no-op
    assert torch.equal(solver.Hx, before)


# --- axis fast paths ---------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_axis_fast_paths_build(axis):
    bias = [0.0, 0.0, 0.0]
    bias[axis] = 1.75e5
    solver = _build_mock(_ferrite(bias=tuple(bias)), 1e-12)
    assert solver.gyromagnetic_enabled
    assert solver._gyromagnetic_state["axis"] == axis


# --- CUDA integration (real FDTD solver) -------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _cuda_scene(material, *, boundary=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.12, 0.12))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=boundary or mw.BoundarySpec.pml(num_layers=6),
        device="cuda",
    )
    scene.add_structure(
        mw.Structure(geometry=Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.08)), material=material)
    )
    return scene


@cuda
def test_cuda_solver_enables_and_allocates_state():
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite()), frequency=8e9)
    solver.init_field()
    assert solver.gyromagnetic_enabled
    state = solver._gyromagnetic_state
    overlap = (solver.Nx - 1, solver.Ny - 1, solver.Nz - 1)
    assert state["m_u"].shape == overlap
    assert state["m_u"].device.type == "cuda"


@cuda
def test_cuda_ferrite_free_scene_disables():
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(mw.Material(eps_r=14.5)), frequency=8e9)
    solver.init_field()
    assert solver.gyromagnetic_enabled is False


@cuda
def test_cuda_forward_stable_and_precesses():
    """Seeding a transverse H excites the coupled precession without blowing up."""
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    solver.Hx[state["u_slice"]] = 1.0  # seed a uniform transverse field in the slab
    solver.solve(time_steps=200)
    assert torch.isfinite(solver.Hx).all() and torch.isfinite(solver.Hy).all()
    # The gyromagnetic coupling drives the magnetization state (m != 0).
    assert float(state["m_u"].abs().max()) > 0.0
    assert float(state["m_v"].abs().max()) > 0.0  # off-diagonal precession populated m_v


@cuda
def test_cuda_composes_with_conductivity():
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite(sigma_e=0.5)), frequency=8e9)
    solver.init_field()
    assert solver.gyromagnetic_enabled and solver.conductive_enabled
    solver.solve(time_steps=50)
    assert torch.isfinite(solver.Hx).all()


@cuda
def test_cuda_graph_capturable():
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    # No per-step allocation / host sync in the gyromagnetic block -> capturable.
    solver.solve(time_steps=50, use_cuda_graph=True)
    assert torch.isfinite(solver.Hx).all()
