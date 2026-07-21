"""Slice 1c: gyromagnetic ferrite forward runtime.

Gates (design brief slice 1c):
  * CUDA/runtime-vs-torch-reference parity -- the Phase-0 LLG oracle is the judge;
  * circular eigen-permeabilities / bias-reversal non-reciprocity;
  * discrete-energy non-growth at zero damping (falsifiable zero-growth form);
  * composes with PML / conductivity;
  * ferrite-free scene adds zero gyromagnetic operations (gated flag);
  * mixed-bias-direction and Bloch fail closed.

General (non-axis-aligned) bias is no longer fail-closed: slice 2a lifts that
guard and routes an arbitrary bias through the general-bias path. Its gates live
in ``test_gyromagnetic_general_bias.py``.

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
    snapshot_gyromagnetic_drive,
    step_gyromagnetic_coupled,
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


def test_general_bias_builds_via_general_path():
    """A general (non-axis-aligned) bias no longer fails closed: slice 2a routes it
    through the general-bias path (dense u/v projection fields, identity
    collocation). Full physics gates live in test_gyromagnetic_general_bias.py."""
    material = _ferrite(bias=(1.0e5, 1.0e5, 1.0e5))
    solver = _build_mock(material, 1e-12)
    assert solver.gyromagnetic_enabled
    assert solver._gyromagnetic_state["general"] is True


def _two_region_bias_scene(bias_low, bias_high, *, spacing=0.02, half=0.06):
    """Ferrite in z<0 with ``bias_low`` and z>0 with ``bias_high`` (both axis-aligned)."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, -half / 2.0), size=(10.0, 10.0, half)),
            material=_ferrite(bias=bias_low),
        )
    )
    scene.add_structure(
        mw.Structure(
            geometry=Box(position=(0.0, 0.0, half / 2.0), size=(10.0, 10.0, half)),
            material=_ferrite(bias=bias_high),
        )
    )
    return prepare_scene(scene)


def test_mixed_sign_bias_fails_closed():
    """A scene mixing +z and -z ferrites must fail closed, not silently invert the
    non-reciprocity of the -bias region.

    Falsification: the old guard used ``torch.unique(fast_axis)``, which collapses
    +z and -z to the same axis code, so this scene built successfully and the single
    global transverse sign (taken from cell 0) applied a +z response to the -z-owned
    cells -- inverting kappa exactly where a latching (opposed-bias) circulator needs
    it opposed. Guarding on bias-vector uniformity closes the hole.
    """
    scene = _two_region_bias_scene((0.0, 0.0, 1.75e5), (0.0, 0.0, -1.75e5))
    solver = _MockSolver(scene, 1e-12)
    with pytest.raises(NotImplementedError, match="single uniform bias direction"):
        build_gyromagnetic(solver, scene)


def test_mixed_axis_bias_fails_closed():
    """A scene mixing +z and +x axis-aligned biases must also fail closed."""
    scene = _two_region_bias_scene((0.0, 0.0, 1.75e5), (1.75e5, 0.0, 0.0))
    solver = _MockSolver(scene, 1e-12)
    with pytest.raises(NotImplementedError, match="single uniform bias direction"):
        build_gyromagnetic(solver, scene)


def test_uniform_sign_two_region_builds():
    """Two co-aligned (+z / +z) ferrite regions still build: the guard rejects only
    non-uniform bias directions, not multiple same-bias structures."""
    scene = _two_region_bias_scene((0.0, 0.0, 1.75e5), (0.0, 0.0, 1.75e5))
    solver = _MockSolver(scene, 1e-12)
    build_gyromagnetic(solver, scene)
    assert solver.gyromagnetic_enabled
    assert solver._gyromagnetic_state["axis"] == 2


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


def _run_seeded_forward(use_graph, *, steps=200, seed=1234):
    """Run a ferrite forward with a deterministic seeded transverse field.

    Returns the final fields, the magnetization state, and whether the
    field-update block was actually captured into a CUDA graph.
    """
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    gen = torch.Generator(device=solver.Hx.device).manual_seed(seed)
    seed_field = torch.rand(
        solver.Hx[state["u_slice"]].shape,
        generator=gen,
        device=solver.Hx.device,
        dtype=solver.Hx.dtype,
    )
    solver.Hx[state["u_slice"]] = seed_field
    solver.solve(time_steps=steps, use_cuda_graph=use_graph)
    fields = {name: getattr(solver, name).clone() for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    return (
        fields,
        state["m_u"].clone(),
        state["m_v"].clone(),
        bool(getattr(solver, "_cuda_graph_active", False)),
    )


@cuda
def test_cuda_graph_eager_field_parity():
    """CUDA-graph replay must advance the magnetization recurrence bit-identically
    to eager, not freeze it at capture-time pointers.

    Falsification: with the previous state-rebinding advance (``state[...] = ...``
    to fresh allocations), the captured graph replayed kernels against frozen
    capture-time buffers, so the recurrence stalled and eager/graph diverged at
    rel ~ 1. In-place buffer updates make the two paths bitwise-identical.
    """
    eager_fields, mu_eager, mv_eager, _ = _run_seeded_forward(False)
    graph_fields, mu_graph, mv_graph, captured = _run_seeded_forward(True)

    assert captured, (
        "the field-update block (including the gyromagnetic advance/correction) was "
        "not captured into a CUDA graph, so this test did not exercise graph replay"
    )
    for name, eager in eager_fields.items():
        assert torch.equal(eager, graph_fields[name]), f"{name} eager/graph mismatch"
    assert torch.equal(mu_eager, mu_graph)
    assert torch.equal(mv_eager, mv_graph)
    # The recurrence actually advanced under replay (the frozen-pointer bug left the
    # graph magnetization near zero while eager grew).
    assert float(mu_graph.abs().max()) > 0.0
    assert float(mv_graph.abs().max()) > 0.0


@cuda
def test_cuda_gyromagnetic_block_zero_per_step_alloc():
    """The real coupled forward hot path (snapshot + coupled step) performs no
    per-step allocation.

    Uses the caching allocator's cumulative allocation counter: after a warmup
    step, the count must not increase across repeated snapshot/coupled-step calls.
    """
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    solver.Hx[state["u_slice"]] = 0.5
    device = solver.Hx.device
    # Warmup: absorb any one-time lazy allocations before counting.
    snapshot_gyromagnetic_drive(solver)
    step_gyromagnetic_coupled(solver)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    for _ in range(25):
        snapshot_gyromagnetic_drive(solver)
        step_gyromagnetic_coupled(solver)
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    assert after == before, (
        f"gyromagnetic snapshot/coupled step allocated {after - before} blocks over 25 steps"
    )


# --- driven closed-cavity energy passivity (real coupled solver) -------------


def _cavity_scene(material, *, spacing=0.02, half=0.06):
    """PEC-walled lossless cavity fully filled by the ferrite (closed system)."""
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-half, half), (-half, half), (-half, half))),
        grid=mw.GridSpec.uniform(spacing),
        boundary=mw.BoundarySpec.pec(),
        device="cuda",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(10.0, 10.0, 10.0)), material=material))
    return scene


def _cavity_total_energy(solver):
    """Positive-definite total-energy proxy of the coupled field+magnetization system.

    ``eps0 |E|^2 + mu0 |H|^2`` is the field proxy (both kinds, so the E<->H sloshing
    of the leapfrog does not read as growth), plus the magnetization energy ``mu0
    |m|^2`` -- the ADE degrees of freedom the gyromagnetic coupling drives. A passive
    coupling keeps this bounded; a coupling that injects energy each step grows it
    secularly.
    """
    e_energy = sum(float((getattr(solver, n).double() ** 2).sum()) for n in ("Ex", "Ey", "Ez"))
    h_energy = sum(float((getattr(solver, n).double() ** 2).sum()) for n in ("Hx", "Hy", "Hz"))
    state = solver._gyromagnetic_state
    m_energy = float((state["m_u"].double() ** 2 + state["m_v"].double() ** 2).sum())
    return solver.eps0 * e_energy + solver.mu0 * (h_energy + m_energy)


@cuda
def test_cuda_driven_cavity_energy_non_growth_lossless():
    """alpha = 0 closed cavity: the coupled field+magnetization discrete energy of the
    REAL solver (field update + gyromagnetic correction) does not grow.

    This is the coupled-system passivity gate the contract's Risk-2 flags -- distinct
    from the undriven ADE-subsystem test, which forces H = 0 and only exercises the
    Cayley propagator. Here a seeded transverse H drives the magnetization while the
    magnetization corrects H back through the Yee update, in a lossless PEC cavity.

    Falsification: see ``test_cuda_driven_cavity_energy_growth_detected`` -- corrupting
    the back-reaction (energy injection) makes this same non-growth bound fail.
    """
    from witwin.maxwell.fdtd.solver import FDTD

    solver = FDTD(_cavity_scene(_ferrite(gilbert_damping=0.0)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    solver.Hx[state["u_slice"]] = 0.5  # smooth uniform transverse seed in the slab

    e0 = _cavity_total_energy(solver)
    assert e0 > 0.0
    peak = e0
    for _ in range(60):
        solver.solve(time_steps=100)
        peak = max(peak, _cavity_total_energy(solver))
    # 6000 lossless steps: a per-step energy injection of even 1e-4 would compound to
    # ~+80%. The leapfrog+Cayley coupling stays bounded well under a few percent.
    assert peak <= e0 * 1.02, f"energy grew: peak/e0 = {peak / e0:.4f}"
    assert torch.isfinite(solver.Hx).all()


@cuda
def test_cuda_driven_cavity_energy_growth_detected():
    """The non-growth gate has teeth: a corrupted back-reaction (energy injection)
    makes the coupled-energy bound fail.

    Falsification harness for ``test_cuda_driven_cavity_energy_non_growth_lossless``:
    replacing the correction with one that adds a small positive multiple of H each
    step turns the lossless cavity into an active one, and the same energy proxy grows
    past the 2% bound.
    """
    from witwin.maxwell.fdtd.solver import FDTD
    from witwin.maxwell.fdtd.runtime import gyromagnetic as gyro

    solver = FDTD(_cavity_scene(_ferrite(gilbert_damping=0.0)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    solver.Hx[state["u_slice"]] = 0.5

    def _corrupt_step(s):
        gyro.step_gyromagnetic_coupled(s)  # the real (passive) coupled step
        st = s._gyromagnetic_state
        hu = getattr(s, st["u_attr"])
        hu[st["u_slice"]].add_(hu[st["u_slice"]], alpha=1.0e-3)  # inject energy each step

    solver._step_gyromagnetic_coupled = lambda: _corrupt_step(solver)

    e0 = _cavity_total_energy(solver)
    peak = e0
    for _ in range(60):
        solver.solve(time_steps=100)
        peak = max(peak, _cavity_total_energy(solver))
        if peak > e0 * 1.02:
            break
    assert peak > e0 * 1.02, "corrupted (energy-injecting) coupling was not detected by the gate"


# --- fail-closed consumer guards (adjoint / FDFD) ----------------------------


def test_adjoint_rejects_gyromagnetic_medium():
    """The differentiable path must reject a ferrite scene: the non-reciprocal
    magnetization-ADE correction has no reverse core, so gradients would be for a
    reciprocal medium.

    Falsification: before the guard, ``_unsupported_adjoint_medium`` returned None for
    a ferrite scene (the material reports is_anisotropic=False, mu_tensor=None,
    is_magnetic_dispersive=False), so the adjoint accepted it and silently mis-simulated.
    """
    from witwin.maxwell.fdtd.adjoint.bridge import _unsupported_adjoint_medium

    scene = _uniform_ferrite_scene(_ferrite())
    message = _unsupported_adjoint_medium(scene)
    assert message is not None
    assert "GyromagneticFerrite" in message

    # A ferrite-free scene is still accepted (guard is specific, not blanket).
    plain = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.06, 0.06))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.none(),
        device="cpu",
    )
    plain.add_structure(mw.Structure(geometry=Box(size=(0.05, 0.05, 0.05)), material=mw.Material(eps_r=4.0)))
    assert _unsupported_adjoint_medium(prepare_scene(plain)) is None


@cuda
def test_fdfd_rejects_gyromagnetic_medium():
    """FDFD must fail closed on a ferrite: the static compile lowers it to the
    diagonal background only, so the frequency-domain solve would silently drop the
    off-diagonal Polder gyrotropy and simulate a reciprocal medium.
    """
    from witwin.maxwell.fdfd.solver import FDFD

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.06, 0.06), (-0.06, 0.06), (-0.06, 0.06))),
        grid=mw.GridSpec.uniform(0.02),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )
    scene.add_structure(mw.Structure(geometry=Box(size=(10.0, 10.0, 10.0)), material=_ferrite()))
    solver = FDFD(scene, frequency=8e9)
    with pytest.raises(NotImplementedError, match="FDFD does not support GyromagneticFerrite"):
        solver._ensure_material_components()


# --- checkpoint / resume round-trip ------------------------------------------


@cuda
def test_checkpoint_roundtrip_preserves_gyromagnetic_state():
    """A ferrite checkpoint must carry the magnetization state and restore it exactly;
    dropping a gyromagnetic name must trip the schema-drift guard.

    Falsification: before the schema carried m_u/m_v/dm_u/dm_v, a mid-run save/resume
    silently re-zeroed the magnetization (init_field on the resumed solver), and
    validate_checkpoint_state could not notice a state family the schema never listed.
    """
    from witwin.maxwell.fdtd.solver import FDTD
    from witwin.maxwell.fdtd.checkpoint import (
        FDTDCheckpointState,
        capture_checkpoint_state,
        gyromagnetic_state_name,
        restore_checkpoint_state,
        validate_checkpoint_state,
    )

    solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    solver.Hx[state["u_slice"]] = 0.7
    solver.solve(time_steps=40)

    checkpoint = capture_checkpoint_state(solver, step=40)
    for tensor_name in ("m_u", "m_v", "dm_u", "dm_v"):
        schema_name = gyromagnetic_state_name(tensor_name)
        assert schema_name in checkpoint.schema.gyromagnetic_state_names
        assert schema_name in checkpoint.schema.state_names
        assert schema_name in checkpoint.tensors
    m_u_saved = solver._gyromagnetic_state["m_u"].clone()
    m_v_saved = solver._gyromagnetic_state["m_v"].clone()
    assert float(m_u_saved.abs().max()) > 0.0  # non-trivial magnetization was captured

    fresh = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    fresh.init_field()
    assert float(fresh._gyromagnetic_state["m_u"].abs().max()) == 0.0  # starts re-zeroed
    restore_checkpoint_state(fresh, checkpoint)
    assert torch.equal(fresh._gyromagnetic_state["m_u"], m_u_saved)
    assert torch.equal(fresh._gyromagnetic_state["m_v"], m_v_saved)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert torch.equal(getattr(fresh, name), getattr(solver, name))

    # Falsification: drop one gyromagnetic name -> schema drift raise.
    dropped = dict(checkpoint.tensors)
    del dropped[gyromagnetic_state_name("m_u")]
    with pytest.raises(RuntimeError, match="layout drifted"):
        validate_checkpoint_state(
            FDTDCheckpointState(step=checkpoint.step, schema=checkpoint.schema, tensors=dropped)
        )


@cuda
def test_resume_checkpoint_serializes_gyromagnetic_state(tmp_path):
    """The resume checkpoint serializes the magnetization state and its schema names
    through save/load (the physics schema snapshot must carry the gyromagnetic
    names, or the reloaded schema would drift from the tensor dict).

    Falsification: before ``gyromagnetic_state_names`` was added to the resume schema
    snapshot fields, the reloaded schema dropped the gyromagnetic names while the
    serialized tensors kept them, so ``validate_checkpoint_state`` raised on load.
    """
    from witwin.maxwell.fdtd.solver import FDTD
    from witwin.maxwell.fdtd.checkpoint import gyromagnetic_state_name
    from witwin.maxwell.fdtd.resume import FDTDResumeCheckpoint, capture_resume_checkpoint

    solver = FDTD(_cuda_scene(_ferrite(gilbert_damping=1e-2)), frequency=8e9)
    solver.init_field()
    state = solver._gyromagnetic_state
    solver.Hx[state["u_slice"]] = 0.7
    solver.solve(time_steps=40)
    m_u_saved = solver._gyromagnetic_state["m_u"].clone()
    assert float(m_u_saved.abs().max()) > 0.0

    checkpoint = capture_resume_checkpoint(solver, step=40, total_steps=200)
    path = tmp_path / "ferrite_resume.pt"
    checkpoint.save(path)
    loaded = FDTDResumeCheckpoint.load(path, map_location="cuda")  # validates on load

    schema_name = gyromagnetic_state_name("m_u")
    assert schema_name in loaded.physics.schema.gyromagnetic_state_names
    assert torch.equal(loaded.physics.tensors[schema_name].to(m_u_saved.device), m_u_saved)
