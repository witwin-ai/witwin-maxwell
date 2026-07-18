"""Gyromagnetic (Polder) ferrite forward runtime (plan 08, slice 1c).

Advances the linearized-LLG magnetization ADE locally at each ferrite cell and
applies the resulting non-reciprocal correction to the magnetic field, as the
magnetic mirror of the full-anisotropy correction on the electric side. The
gyrotropy is carried entirely by this local magnetization state; the background
permeability ``mu_infinity`` (and eps/sigma) is compiled through the ordinary
diagonal material path, so the plain Yee magnetic update already produces the
background-only field ``H_tmp`` and this module adds only ``H -= dM/mu_inf``.

Frozen physics contract: ``docs/reference/ferrite-physics-contract.md``. The
per-cell implicit-midpoint (Cayley) propagator ``Phi``/``Gamma`` comes from the
compiled layout (``CompiledGyromagneticLayout.with_timestep``), whose matrices
are the twins of the Phase-0 verification oracle. ``Phi``/``Gamma`` are the
*material* ADE model (validated against the oracle by the CW-parity gates); the
field <-> magnetization *coupling* is a separate concern handled below.

Coupling passivity (contract Risk-2). An explicit coupling -- advancing ``m``
from the pre-update ``H`` and then correcting ``H`` with the resulting ``dM`` --
is NOT passive: the one-step feedback delay makes the closed field+magnetization
loop weakly anti-damped, so a lossless closed cavity grows without bound at any
timestep. The delivered coupling is instead the implicit midpoint (Crank-Nicolson)
of the coupled system: ``m`` is driven by the time-centred ``(H_pre + H_tmp)/2``
and the correction feeds ``dM`` back into ``H`` simultaneously, solved per cell by
a precomputed 2x2 inverse. That closed form is discretely non-growing at zero
damping (see the driven-cavity energy gate). It runs in two hooks straddling the
plain Yee magnetic update: ``snapshot_gyromagnetic_drive`` (pre) records the
pre-update transverse ``H``, and ``step_gyromagnetic_coupled`` (post) does the
coupled solve and the correction.

Slice 1c scope: axis-aligned uniform bias (z/x/y fast path). The two transverse
magnetic components co-locate on the shared Yee overlap
``(Nx-1, Ny-1, Nz-1)`` (identity collocation ``C = I``, the contract-sanctioned
regime for transversely-uniform fields, contract section 5). A general
(non-axis-aligned) bias needs the local-frame rotation plus the 4-point
collocation of the arbitrary-bias kernel and fails closed here (Phase 2).

Every entry point is gated by ``solver.gyromagnetic_enabled``; a ferrite-free
scene allocates no state and issues no operations. The persistent state
(magnetization, dense coefficient fields, per-step scratch) is preallocated once
at build time and the hot path performs no host synchronization (no ``.item()`` /
device-to-host transfer) and no per-step allocation (all updates are in-place
``copy_``/``mul(out=)``/``addcmul_``/``add_``/``sub_``), so the magnetic block
stays CUDA-graph capturable and its captured kernels advance the same persistent
buffers on every replay.
"""

from __future__ import annotations

import torch

# Overlap slice for each transverse magnetic component: the sub-block of the Yee
# H tensor that co-locates with the shared (Nx-1, Ny-1, Nz-1) cell overlap.
#   Hx: (Nx, Ny-1, Nz-1) -> [:Nx-1, :, :]
#   Hy: (Nx-1, Ny, Nz-1) -> [:, :Ny-1, :]
#   Hz: (Nx-1, Ny-1, Nz) -> [:, :, :Nz-1]
_COMPONENT_ATTR = {0: "Hx", 1: "Hy", 2: "Hz"}


def _overlap_slice(axis: int, nx: int, ny: int, nz: int):
    if axis == 0:
        return (slice(0, nx - 1), slice(None), slice(None))
    if axis == 1:
        return (slice(None), slice(0, ny - 1), slice(None))
    return (slice(None), slice(None), slice(0, nz - 1))


def build_gyromagnetic(solver, scene):
    """Compile the ferrite layout and precompute the dense forward-runtime state.

    Sets ``solver.gyromagnetic_enabled`` and, when enabled, the transverse
    component slices, the dense per-cell Cayley (``Phi``/``Gamma``) and coupled
    implicit-midpoint (``B``/``C``) fields, the active mask, and the preallocated
    magnetization / drive / scratch buffers. Fails closed on a non-axis-aligned
    (general) bias, on a scene mixing bias directions, and on a Bloch boundary.
    """
    layout = scene.compile_gyromagnetic_materials(dt=solver.dt, device=solver.device)
    solver._gyromagnetic_layout = layout
    solver.gyromagnetic_enabled = bool(layout.enabled)
    if not layout.enabled:
        return

    if scene.boundary.uses_kind("bloch"):
        raise NotImplementedError(
            "FDTD gyromagnetic ferrite requires the real-valued magnetic update: a "
            "Bloch-periodic run carries complex phase-shifted fields, and the real "
            "magnetization-ADE correction would break the Bloch phase relation. Use a "
            "real-field boundary (PML/PEC/PMC/periodic) with the ferrite."
        )

    fast_axis = layout.fast_axis
    if int(fast_axis.min()) < 0:
        raise NotImplementedError(
            "FDTD gyromagnetic ferrite forward advances an axis-aligned uniform bias "
            "(z/x/y fast path): a general (non-axis-aligned) bias needs the local-frame "
            "rotation and 4-point Yee collocation of the arbitrary-bias kernel, which "
            "co-locates the three staggered H components. Align every ferrite bias with "
            "a single Cartesian axis."
        )

    # The fast path applies a single global (axis, transverse signs) taken from the
    # local basis of cell 0, so every active cell must share one bias direction --
    # sign included. Guarding on torch.unique(fast_axis) is not enough: the axis
    # code collapses +e_k and -e_k to the same value, so a scene mixing +z and -z
    # ferrites would pass an axis-only guard while their v-column signs are opposed,
    # and the global sign would silently invert the non-reciprocity of the -bias
    # region (e.g. a latching circulator). Guard on the bias unit vectors themselves
    # so mixed axes OR opposed signs on one axis both fail closed.
    bias = layout.bias_unit
    if not torch.allclose(bias, bias[:1], atol=1.0e-9):
        raise NotImplementedError(
            "FDTD gyromagnetic ferrite forward requires a single uniform bias direction "
            "(sign included) across the scene for the z/x/y fast path: a scene mixing bias "
            "axes, or mixing opposed signs on one axis (e.g. +z and -z ferrites), needs the "
            "per-cell local-frame signs of the arbitrary-bias kernel. Align every ferrite bias "
            "with one signed Cartesian direction."
        )

    axis = int(fast_axis[0])
    basis = layout.local_basis[0]  # columns [u | v | w], w == bias
    u_axis = int(torch.argmax(torch.abs(basis[:, 0])))
    v_axis = int(torch.argmax(torch.abs(basis[:, 1])))
    sign_u = float(torch.sign(basis[u_axis, 0]))
    sign_v = float(torch.sign(basis[v_axis, 1]))

    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    field_dtype = solver.Hx.dtype
    device = solver.device

    # Dense node fields (Nx, Ny, Nz) scattered from the sparse layout, then sliced
    # to the (Nx-1, Ny-1, Nz-1) cell overlap. For a single uniform ferrite these
    # are constant over the active block; the dense form keeps the update a plain
    # masked elementwise op (CUDA-graph friendly, no gather).
    def _dense_from(values, fill=0.0):
        flat = torch.full((nx * ny * nz,), float(fill), dtype=field_dtype, device=device)
        flat[layout.active_index] = values.to(dtype=field_dtype, device=device)
        return flat.reshape(nx, ny, nz)

    phi = layout.phi  # (N, 2, 2)
    gamma = layout.gamma
    overlap = (slice(0, nx - 1), slice(0, ny - 1), slice(0, nz - 1))

    # Coupled implicit-midpoint (Crank-Nicolson) coefficients. The coupled system
    # per active cell, with the time-centred drive ``H_mid = (H_pre + H_new)/2`` and
    # the simultaneous back-reaction ``H_new = H_tmp - (m_new - m_old)/mu_inf``, is
    #
    #     (I + Gamma/(2 mu_inf)) m_new
    #         = (Phi + Gamma/(2 mu_inf)) m_old + (Gamma/2) (H_pre + H_tmp),
    #
    # so ``m_new = B m_old + C (H_pre + H_tmp)`` with the precomputed 2x2 blocks
    # ``B = A^-1 (Phi + Gamma/(2 mu_inf))`` and ``C = A^-1 (Gamma/2)``. Unlike the
    # explicit ``m = Phi m + Gamma H_pre`` recurrence, this closed form is discretely
    # non-growing at zero damping (the midpoint rule preserves the coupled invariant).
    layout_dtype = phi.dtype
    num_active = phi.shape[0]
    mu_col = layout.mu_infinity.to(dtype=layout_dtype).view(num_active, 1, 1)
    identity = torch.eye(2, dtype=layout_dtype, device=phi.device).expand(num_active, 2, 2)
    half_gamma_over_mu = gamma / (2.0 * mu_col)
    a_matrix = identity + half_gamma_over_mu
    a_inv = torch.linalg.inv(a_matrix)
    b_matrix = a_inv @ (phi + half_gamma_over_mu)
    c_matrix = a_inv @ (gamma / 2.0)

    mask_flat = torch.zeros(nx * ny * nz, dtype=torch.bool, device=device)
    mask_flat[layout.active_index] = True
    active = mask_flat.reshape(nx, ny, nz)[overlap].contiguous()

    state = {
        "axis": axis,
        "u_attr": _COMPONENT_ATTR[u_axis],
        "v_attr": _COMPONENT_ATTR[v_axis],
        "u_slice": _overlap_slice(u_axis, nx, ny, nz),
        "v_slice": _overlap_slice(v_axis, nx, ny, nz),
        "sign_u": sign_u,
        "sign_v": sign_v,
        "active": active,
        # Material ADE propagator (used by the pure-recurrence primitives / oracle
        # parity tests).
        "phi00": _dense_from(phi[:, 0, 0])[overlap].contiguous(),
        "phi01": _dense_from(phi[:, 0, 1])[overlap].contiguous(),
        "phi10": _dense_from(phi[:, 1, 0])[overlap].contiguous(),
        "phi11": _dense_from(phi[:, 1, 1])[overlap].contiguous(),
        "gamma00": _dense_from(gamma[:, 0, 0])[overlap].contiguous(),
        "gamma01": _dense_from(gamma[:, 0, 1])[overlap].contiguous(),
        "gamma10": _dense_from(gamma[:, 1, 0])[overlap].contiguous(),
        "gamma11": _dense_from(gamma[:, 1, 1])[overlap].contiguous(),
        # Coupled implicit-midpoint blocks (the real field-coupled forward path).
        "b00": _dense_from(b_matrix[:, 0, 0])[overlap].contiguous(),
        "b01": _dense_from(b_matrix[:, 0, 1])[overlap].contiguous(),
        "b10": _dense_from(b_matrix[:, 1, 0])[overlap].contiguous(),
        "b11": _dense_from(b_matrix[:, 1, 1])[overlap].contiguous(),
        "c00": _dense_from(c_matrix[:, 0, 0])[overlap].contiguous(),
        "c01": _dense_from(c_matrix[:, 0, 1])[overlap].contiguous(),
        "c10": _dense_from(c_matrix[:, 1, 0])[overlap].contiguous(),
        "c11": _dense_from(c_matrix[:, 1, 1])[overlap].contiguous(),
        # 1/mu_inf where active, 0 elsewhere: keeps the correction masked.
        "inv_mu_inf": _dense_from(1.0 / layout.mu_infinity)[overlap].contiguous(),
    }
    overlap_shape = state["active"].shape
    # Persistent magnetization state (m) plus per-step scratch (drive gathers hu/hv,
    # pre-update snapshot h_pre_*, staged propagator output new_u/new_v, staged
    # increment dm_*, staged correction corr_*). All preallocated once here; the hot
    # path only writes into these buffers in place, so it performs zero per-step
    # allocation and the pointers stay fixed under CUDA-graph replay.
    for name in (
        "m_u", "m_v", "dm_u", "dm_v",
        "hu", "hv", "new_u", "new_v", "corr_u", "corr_v",
        "h_pre_u", "h_pre_v",
    ):
        state[name] = torch.zeros(overlap_shape, dtype=field_dtype, device=device)
    # Float mask for the masked increment (dm zeroed on inactive cells).
    state["active_f"] = state["active"].to(dtype=field_dtype)
    solver._gyromagnetic_state = state


def initialize_gyromagnetic_state(solver):
    """Reset the magnetization / increment buffers to zero (no reallocation)."""
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    for name in ("m_u", "m_v", "dm_u", "dm_v", "h_pre_u", "h_pre_v"):
        state[name].zero_()


# --- material ADE primitives (pure recurrence, oracle-parity path) -----------


def advance_gyromagnetic_state(solver):
    """Advance the magnetization ADE one step by the pure Cayley recurrence.

    ``m <- Phi m + Gamma h`` at every active cell, with ``h`` the current (signed)
    transverse ``H`` drive, then stages ``dm = m_new - m_old``. This is the
    *material* ADE model, validated against the Phase-0 oracle by the CW-parity
    gates; the field-coupled forward path uses the passive coupling in
    :func:`step_gyromagnetic_coupled`, not this explicit recurrence. No host
    synchronization and no per-step allocation.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    hu = state["hu"]
    hv = state["hv"]
    torch.mul(getattr(solver, state["u_attr"])[state["u_slice"]], state["sign_u"], out=hu)
    torch.mul(getattr(solver, state["v_attr"])[state["v_slice"]], state["sign_v"], out=hv)

    m_u = state["m_u"]
    m_v = state["m_v"]
    new_u = state["new_u"]
    new_v = state["new_v"]
    torch.mul(state["phi00"], m_u, out=new_u)
    new_u.addcmul_(state["phi01"], m_v)
    new_u.addcmul_(state["gamma00"], hu)
    new_u.addcmul_(state["gamma01"], hv)
    torch.mul(state["phi10"], m_u, out=new_v)
    new_v.addcmul_(state["phi11"], m_v)
    new_v.addcmul_(state["gamma10"], hu)
    new_v.addcmul_(state["gamma11"], hv)
    dm_u = state["dm_u"]
    dm_v = state["dm_v"]
    active_f = state["active_f"]
    torch.sub(new_u, m_u, out=dm_u)
    torch.sub(new_v, m_v, out=dm_v)
    dm_u.mul_(active_f)
    dm_v.mul_(active_f)
    m_u.add_(dm_u)
    m_v.add_(dm_v)


def apply_gyromagnetic_correction(solver):
    """Subtract the staged magnetization increment from the field (pure form).

    ``H^{n+1} = H_tmp - dM/mu_inf`` using the ``dm`` staged by
    :func:`advance_gyromagnetic_state`. Kept as the material-primitive counterpart
    of the pure recurrence; the real forward path uses
    :func:`step_gyromagnetic_coupled`. No per-step allocation.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    inv_mu = state["inv_mu_inf"]
    hu = getattr(solver, state["u_attr"])
    hv = getattr(solver, state["v_attr"])
    corr_u = state["corr_u"]
    corr_v = state["corr_v"]
    torch.mul(state["dm_u"], inv_mu, out=corr_u)
    torch.mul(state["dm_v"], inv_mu, out=corr_v)
    corr_u.mul_(state["sign_u"])
    corr_v.mul_(state["sign_v"])
    hu[state["u_slice"]].sub_(corr_u)
    hv[state["v_slice"]].sub_(corr_v)


# --- coupled implicit-midpoint forward path (passive) ------------------------


def snapshot_gyromagnetic_drive(solver):
    """Record the pre-update transverse H drive (one half of the midpoint average).

    Called before the plain Yee magnetic update overwrites ``H``. Stores the signed
    ``H^{n-1/2}`` into persistent buffers so the post-update coupled step can form
    the time-centred drive ``(H_pre + H_tmp)/2``. No per-step allocation.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    h_pre_u = state["h_pre_u"]
    h_pre_v = state["h_pre_v"]
    torch.mul(getattr(solver, state["u_attr"])[state["u_slice"]], state["sign_u"], out=h_pre_u)
    torch.mul(getattr(solver, state["v_attr"])[state["v_slice"]], state["sign_v"], out=h_pre_v)


def step_gyromagnetic_coupled(solver):
    """Coupled implicit-midpoint magnetization advance + non-reciprocal correction.

    Called after the plain Yee magnetic update (so ``H`` now holds the
    background-only ``H_tmp``). Forms the time-centred drive ``H_pre + H_tmp``,
    solves the coupled system in closed form ``m_new = B m_old + C (H_pre + H_tmp)``,
    then applies ``H^{n+1} = H_tmp - dM/mu_inf``. Discretely non-growing at zero
    damping. All updates are in place; no host synchronization, no per-step
    allocation.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    # Time-centred drive sum: h_pre + h_tmp (post-update H), gathered into scratch.
    hu = state["hu"]
    hv = state["hv"]
    torch.mul(getattr(solver, state["u_attr"])[state["u_slice"]], state["sign_u"], out=hu)
    torch.mul(getattr(solver, state["v_attr"])[state["v_slice"]], state["sign_v"], out=hv)
    hu.add_(state["h_pre_u"])
    hv.add_(state["h_pre_v"])

    m_u = state["m_u"]
    m_v = state["m_v"]
    new_u = state["new_u"]
    new_v = state["new_v"]
    # m_new = B @ m_old + C @ (h_pre + h_tmp), staged into scratch in place.
    torch.mul(state["b00"], m_u, out=new_u)
    new_u.addcmul_(state["b01"], m_v)
    new_u.addcmul_(state["c00"], hu)
    new_u.addcmul_(state["c01"], hv)
    torch.mul(state["b10"], m_u, out=new_v)
    new_v.addcmul_(state["b11"], m_v)
    new_v.addcmul_(state["c10"], hu)
    new_v.addcmul_(state["c11"], hv)

    dm_u = state["dm_u"]
    dm_v = state["dm_v"]
    active_f = state["active_f"]
    torch.sub(new_u, m_u, out=dm_u)
    torch.sub(new_v, m_v, out=dm_v)
    dm_u.mul_(active_f)
    dm_v.mul_(active_f)
    m_u.add_(dm_u)
    m_v.add_(dm_v)

    # Non-reciprocal correction: H^{n+1} = H_tmp - dM/mu_inf on the same edges.
    inv_mu = state["inv_mu_inf"]
    corr_u = state["corr_u"]
    corr_v = state["corr_v"]
    torch.mul(dm_u, inv_mu, out=corr_u)
    torch.mul(dm_v, inv_mu, out=corr_v)
    corr_u.mul_(state["sign_u"])
    corr_v.mul_(state["sign_v"])
    getattr(solver, state["u_attr"])[state["u_slice"]].sub_(corr_u)
    getattr(solver, state["v_attr"])[state["v_slice"]].sub_(corr_v)
