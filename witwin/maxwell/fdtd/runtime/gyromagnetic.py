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
are the twins of the Phase-0 verification oracle.

Slice 1c scope: axis-aligned uniform bias (z/x/y fast path). The two transverse
magnetic components co-locate on the shared Yee overlap
``(Nx-1, Ny-1, Nz-1)`` (identity collocation ``C = I``, the contract-sanctioned
regime for transversely-uniform fields, contract section 5). A general
(non-axis-aligned) bias needs the local-frame rotation plus the 4-point
collocation of the arbitrary-bias kernel and fails closed here (Phase 2).

Every entry point is gated by ``solver.gyromagnetic_enabled``; a ferrite-free
scene allocates no state and issues no operations. The persistent state
(magnetization, staged increment, dense coefficient fields) is preallocated once
at build time and the hot path performs no host synchronization (no ``.item()`` /
device-to-host transfer), so the magnetic block stays CUDA-graph capturable with a
capture-stable allocation pattern.
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
    component slices, the dense per-cell Cayley fields, the active mask, and the
    preallocated magnetization / delta buffers. Fails closed on a non-axis-aligned
    (general) bias or a scene mixing bias axes -- both need the arbitrary-bias
    kernel (Phase 2).
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

    axes = torch.unique(layout.fast_axis)
    if int(axes.min()) < 0 or int(axes.numel()) > 1:
        raise NotImplementedError(
            "FDTD gyromagnetic ferrite forward advances an axis-aligned uniform bias "
            "(z/x/y fast path): a general (non-axis-aligned) bias, or a scene mixing "
            "bias axes, needs the local-frame rotation and 4-point Yee collocation of "
            "the arbitrary-bias kernel, which co-locates the three staggered H "
            "components. Align every ferrite bias with a single Cartesian axis."
        )

    axis = int(axes[0])
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
        "phi00": _dense_from(phi[:, 0, 0])[overlap].contiguous(),
        "phi01": _dense_from(phi[:, 0, 1])[overlap].contiguous(),
        "phi10": _dense_from(phi[:, 1, 0])[overlap].contiguous(),
        "phi11": _dense_from(phi[:, 1, 1])[overlap].contiguous(),
        "gamma00": _dense_from(gamma[:, 0, 0])[overlap].contiguous(),
        "gamma01": _dense_from(gamma[:, 0, 1])[overlap].contiguous(),
        "gamma10": _dense_from(gamma[:, 1, 0])[overlap].contiguous(),
        "gamma11": _dense_from(gamma[:, 1, 1])[overlap].contiguous(),
        # 1/mu_inf where active, 0 elsewhere: keeps the correction masked.
        "inv_mu_inf": _dense_from(1.0 / layout.mu_infinity)[overlap].contiguous(),
    }
    overlap_shape = state["active"].shape
    for name in ("m_u", "m_v", "dm_u", "dm_v"):
        state[name] = torch.zeros(overlap_shape, dtype=field_dtype, device=device)
    solver._gyromagnetic_state = state


def initialize_gyromagnetic_state(solver):
    """Reset the magnetization / delta buffers to zero (no reallocation)."""
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    for name in ("m_u", "m_v", "dm_u", "dm_v"):
        state[name].zero_()


def advance_gyromagnetic_state(solver):
    """Advance the magnetization ADE one implicit-midpoint step and stage ``dM``.

    Uses the pre-update transverse H (the leapfrog drive ``h^{n+1/2}``) to advance
    ``m`` with the Cayley propagator, then stages ``dm = m^{n+1} - m^n`` for the
    post-update correction. No host synchronization.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    hu = state["sign_u"] * getattr(solver, state["u_attr"])[state["u_slice"]]
    hv = state["sign_v"] * getattr(solver, state["v_attr"])[state["v_slice"]]

    m_u = state["m_u"]
    m_v = state["m_v"]
    new_u = state["phi00"] * m_u + state["phi01"] * m_v + state["gamma00"] * hu + state["gamma01"] * hv
    new_v = state["phi10"] * m_u + state["phi11"] * m_v + state["gamma10"] * hu + state["gamma11"] * hv
    active = state["active"]
    new_u = torch.where(active, new_u, m_u)
    new_v = torch.where(active, new_v, m_v)
    state["dm_u"] = new_u - m_u
    state["dm_v"] = new_v - m_v
    state["m_u"] = new_u
    state["m_v"] = new_v


def apply_gyromagnetic_correction(solver):
    """Subtract the magnetization increment from the background-updated field.

    ``B = mu_0 (mu_inf H + M)`` with ``dB/dt = -curl E`` gives
    ``H^{n+1} = H_tmp - dM/mu_inf``, where ``H_tmp`` is the plain (background-only)
    Yee update. Deposits back onto the same transverse edges the drive gathered
    from (identity collocation ``C = I``), so at zero damping the field<->ADE
    power exchange injects no net energy.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    inv_mu = state["inv_mu_inf"]
    hu = getattr(solver, state["u_attr"])
    hv = getattr(solver, state["v_attr"])
    hu[state["u_slice"]] -= state["sign_u"] * state["dm_u"] * inv_mu
    hv[state["v_slice"]] -= state["sign_v"] * state["dm_v"] * inv_mu
