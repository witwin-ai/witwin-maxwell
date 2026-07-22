"""Gyromagnetic (Polder) ferrite forward runtime (plan 08, slices 1c / 2a).

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

Two bias regimes share one 2x2 magnetization solve and diverge only in how the
transverse RF field is gathered from / scattered back to the staggered Yee ``H``:

* **Axis-aligned fast path** (``fast_axis >= 0``). The bias ``b`` is ``+/-`` a
  Cartesian axis, so the local frame ``[u|v|w]`` is a signed permutation: ``m_u``
  is driven by a single lab ``H`` component (``sign_u * H_{u_axis}``) and ``m_v``
  by another. Identity collocation ``C = I`` (each transverse component truncated
  to the shared ``(Nx-1, Ny-1, Nz-1)`` cell overlap), the contract-sanctioned
  regime for transversely-uniform fields (contract section 5).

* **General-bias path** (``fast_axis < 0``, slice 2a). The bias ``b`` is an
  arbitrary unit vector. The local transverse axes ``u``, ``v`` (columns of the
  right-handed ``R = [u|v|w]``, ``w = b``) have all three Cartesian components, so
  the RF drive is the projection ``h_u = u . H``, ``h_v = v . H`` gathered from all
  three lab ``H`` components, and the back-reaction ``dM = dm_u u + dm_v v`` is
  scattered onto all three. This is a pure coordinate rotation of the SAME
  discretized update: because ``R`` is orthonormal (``[u|v]^T [u|v] = I``), the
  local-frame reduction ``h^new = h_tmp - dm/mu_inf`` -- and therefore the coupled
  ``m_new = B m_old + C (h_pre + h_tmp)`` solve -- is algebraically identical to
  the fast path, so the general path reproduces it bit-for-bit when ``b`` happens
  to be axis-aligned. The same identity collocation ``C = I`` (per-component
  truncation to the cell overlap) is reused (contract section 5); no 4-point
  gather is introduced.

A mixed-bias scene (different bias axes, or opposed signs on one axis such as a
``+z`` / ``-z`` latching circulator, or differing magnitudes/materials) is
supported through the general path: the compiled layout already stores a per-cell
bias, local frame ``[u|v|w]`` and per-cell ``Phi``/``Gamma``/``B``/``C``, and the
magnetization ADE is purely local (no spatial coupling in the magnetization
update -- fields couple only through the ordinary reciprocal Yee curl). A
mixed-bias scene is therefore the direct sum of independent per-cell passive
blocks: each cell precesses around its own ``b`` with the correct handedness
(the right-handed local frame flips the lab-frame off-diagonal for an opposed
bias), and per-cell passivity gives global passivity. Only a Bloch-periodic
ferrite still fails closed (the real magnetization-ADE correction cannot carry
the complex Bloch phase).

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

# Overlap slice for each magnetic component: the sub-block of the Yee H tensor
# that co-locates with the shared (Nx-1, Ny-1, Nz-1) cell overlap.
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


def build_gyromagnetic(solver, scene, *, force_general=False):
    """Compile the ferrite layout and precompute the dense forward-runtime state.

    Sets ``solver.gyromagnetic_enabled`` and, when enabled, the transverse
    gather/scatter description (a single-component signed pair for the axis-aligned
    fast path, or dense ``u``/``v`` projection fields for the general-bias path),
    the dense per-cell Cayley (``Phi``/``Gamma``) and coupled implicit-midpoint
    (``B``/``C``) fields, the active mask, and the preallocated magnetization /
    drive / scratch buffers. A uniform axis-aligned bias uses the fast path; a
    uniform oblique bias or any mixed-bias scene uses the per-cell general path.
    Fails closed only on a Bloch boundary.

    ``force_general`` selects the general-bias path even when the bias is
    axis-aligned. It changes no physics (the general path reduces to the fast path
    bit-for-bit for an axis-aligned bias); it exists so the rotation-equivalence
    gate can drive both code paths on one physical scene.
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

    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    field_dtype = solver.Hx.dtype
    device = solver.device

    overlap = (slice(0, nx - 1), slice(0, ny - 1), slice(0, nz - 1))

    def _dense_from(values, fill=0.0):
        flat = torch.full((nx * ny * nz,), float(fill), dtype=field_dtype, device=device)
        flat[layout.active_index] = values.to(dtype=field_dtype, device=device)
        return flat.reshape(nx, ny, nz)

    phi = layout.phi  # (N, 2, 2)
    gamma = layout.gamma

    # Coupled implicit-midpoint (Crank-Nicolson) coefficients. The coupled system
    # per active cell, with the time-centred drive ``h_mid = (h_pre + h_new)/2`` and
    # the simultaneous back-reaction ``h_new = h_tmp - (m_new - m_old)/mu_inf`` (the
    # ``[u|v]^T`` projection of the lab correction; orthonormal so it is exactly the
    # local ``dm/mu_inf``), is
    #
    #     (I + Gamma/(2 mu_inf)) m_new
    #         = (Phi + Gamma/(2 mu_inf)) m_old + (Gamma/2) (h_pre + h_tmp),
    #
    # so ``m_new = B m_old + C (h_pre + h_tmp)`` with the precomputed 2x2 blocks
    # ``B = A^-1 (Phi + Gamma/(2 mu_inf))`` and ``C = A^-1 (Gamma/2)``. This closed
    # form is frame-independent (identical for the fast and general paths) and is
    # discretely non-growing at zero damping.
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
    state["active_f"] = state["active"].to(dtype=field_dtype)

    # Persistent magnetization state (m) plus per-step scratch (drive gather hu/hv,
    # pre-update snapshot h_pre_*, staged propagator output new_u/new_v, staged
    # increment dm_*). Preallocated once; the hot path only writes into these
    # buffers in place, so it performs zero per-step allocation and the pointers
    # stay fixed under CUDA-graph replay.
    for name in (
        "m_u", "m_v", "dm_u", "dm_v",
        "hu", "hv", "new_u", "new_v",
        "h_pre_u", "h_pre_v",
    ):
        state[name] = torch.zeros(overlap_shape, dtype=field_dtype, device=device)

    # The axis-aligned fast path assumes ONE signed Cartesian bias for the whole
    # scene (a single global transverse frame). It is used only when the bias is
    # uniform (sign included) and axis-aligned. Every other case -- a uniform
    # oblique bias, or a mixed-bias scene (different axes, or opposed signs on one
    # axis, e.g. a latching circulator) -- routes to the per-cell general path,
    # whose dense u/v projection fields carry each cell's own local frame. Because
    # the ADE is purely local (no spatial coupling in the magnetization update) and
    # the compiled layout already stores per-cell bias_unit / local_basis /
    # phi / gamma, a mixed-bias scene is the direct sum of independent per-cell
    # passive blocks: correct and passive without a dedicated mixed-bias kernel.
    bias = layout.bias_unit
    uniform_bias = bool(torch.allclose(bias, bias[:1], atol=1.0e-9))
    fast_axis = int(layout.fast_axis[0])
    if uniform_bias and fast_axis >= 0 and not force_general:
        _build_fast_axis_state(solver, layout, state, overlap_shape, field_dtype)
    else:
        _build_general_state(solver, layout, state, overlap_shape, field_dtype, _dense_from)

    solver._gyromagnetic_state = state


def _build_fast_axis_state(solver, layout, state, overlap_shape, field_dtype):
    """Axis-aligned fast path: a single signed lab component drives each of m_u/m_v."""
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    basis = layout.local_basis[0]  # columns [u | v | w], w == bias
    u_axis = int(torch.argmax(torch.abs(basis[:, 0])))
    v_axis = int(torch.argmax(torch.abs(basis[:, 1])))
    state.update(
        general=False,
        axis=int(layout.fast_axis[0]),
        u_attr=_COMPONENT_ATTR[u_axis],
        v_attr=_COMPONENT_ATTR[v_axis],
        u_slice=_overlap_slice(u_axis, nx, ny, nz),
        v_slice=_overlap_slice(v_axis, nx, ny, nz),
        sign_u=float(torch.sign(basis[u_axis, 0])),
        sign_v=float(torch.sign(basis[v_axis, 1])),
    )
    for name in ("corr_u", "corr_v"):
        state[name] = torch.zeros(overlap_shape, dtype=field_dtype, device=solver.device)


def _build_general_state(solver, layout, state, overlap_shape, field_dtype, dense_from):
    """General-bias path: dense u/v projection fields gather from / scatter to all H.

    ``u``/``v`` are the first two columns of the per-cell local basis (lab frame),
    each broadcast into a dense ``(Nx-1, Ny-1, Nz-1)`` field masked to zero on
    inactive cells. The drive is ``h_u = u . H`` (gathered from Hx/Hy/Hz truncated
    to the cell overlap) and the back-reaction ``dM = dm_u u + dm_v v`` scatters
    onto all three. Per-cell ``u``/``v`` fields make a future mixed-bias relaxation
    a data change only, not a code change.
    """
    nx, ny, nz = solver.Nx, solver.Ny, solver.Nz
    basis = layout.local_basis  # (N, 3, 3), columns [u | v | w]
    state["general"] = True
    # Fixed overlap slices for each lab H component (identity collocation, C = I).
    state["hx_slice"] = _overlap_slice(0, nx, ny, nz)
    state["hy_slice"] = _overlap_slice(1, nx, ny, nz)
    state["hz_slice"] = _overlap_slice(2, nx, ny, nz)
    # Dense per-cell projection fields u = basis[:, :, 0], v = basis[:, :, 1].
    for comp, attr in ((0, "ux"), (1, "uy"), (2, "uz")):
        state[attr] = dense_from(basis[:, comp, 0])[
            (slice(0, nx - 1), slice(0, ny - 1), slice(0, nz - 1))
        ].contiguous()
    for comp, attr in ((0, "vx"), (1, "vy"), (2, "vz")):
        state[attr] = dense_from(basis[:, comp, 1])[
            (slice(0, nx - 1), slice(0, ny - 1), slice(0, nz - 1))
        ].contiguous()
    for name in ("corr_x", "corr_y", "corr_z"):
        state[name] = torch.zeros(overlap_shape, dtype=field_dtype, device=solver.device)


def initialize_gyromagnetic_state(solver):
    """Reset the magnetization / increment buffers to zero (no reallocation)."""
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    for name in ("m_u", "m_v", "dm_u", "dm_v", "h_pre_u", "h_pre_v"):
        state[name].zero_()


# --- transverse RF drive gather ----------------------------------------------


def _gather_drive(solver, state, hu, hv):
    """Fill ``hu``/``hv`` with the transverse RF drive from the current lab ``H``.

    Fast path: a single signed component per transverse axis. General path: the
    projection ``h_u = u . H`` / ``h_v = v . H`` over all three lab components,
    each truncated to the cell overlap (identity collocation).
    """
    if not state["general"]:
        torch.mul(getattr(solver, state["u_attr"])[state["u_slice"]], state["sign_u"], out=hu)
        torch.mul(getattr(solver, state["v_attr"])[state["v_slice"]], state["sign_v"], out=hv)
        return
    hx = solver.Hx[state["hx_slice"]]
    hy = solver.Hy[state["hy_slice"]]
    hz = solver.Hz[state["hz_slice"]]
    torch.mul(state["ux"], hx, out=hu)
    hu.addcmul_(state["uy"], hy)
    hu.addcmul_(state["uz"], hz)
    torch.mul(state["vx"], hx, out=hv)
    hv.addcmul_(state["vy"], hy)
    hv.addcmul_(state["vz"], hz)


def _scatter_correction(solver, state):
    """Apply ``H -= dM/mu_inf`` with the staged ``dm_u``/``dm_v``.

    Fast path: subtract the signed increment from the two driven lab components.
    General path: scatter ``dM = dm_u u + dm_v v`` onto all three lab components.
    """
    inv_mu = state["inv_mu_inf"]
    dm_u = state["dm_u"]
    dm_v = state["dm_v"]
    if not state["general"]:
        corr_u = state["corr_u"]
        corr_v = state["corr_v"]
        torch.mul(dm_u, inv_mu, out=corr_u)
        torch.mul(dm_v, inv_mu, out=corr_v)
        corr_u.mul_(state["sign_u"])
        corr_v.mul_(state["sign_v"])
        getattr(solver, state["u_attr"])[state["u_slice"]].sub_(corr_u)
        getattr(solver, state["v_attr"])[state["v_slice"]].sub_(corr_v)
        return
    corr_x = state["corr_x"]
    corr_y = state["corr_y"]
    corr_z = state["corr_z"]
    torch.mul(dm_u, state["ux"], out=corr_x)
    corr_x.addcmul_(dm_v, state["vx"])
    corr_x.mul_(inv_mu)
    torch.mul(dm_u, state["uy"], out=corr_y)
    corr_y.addcmul_(dm_v, state["vy"])
    corr_y.mul_(inv_mu)
    torch.mul(dm_u, state["uz"], out=corr_z)
    corr_z.addcmul_(dm_v, state["vz"])
    corr_z.mul_(inv_mu)
    solver.Hx[state["hx_slice"]].sub_(corr_x)
    solver.Hy[state["hy_slice"]].sub_(corr_y)
    solver.Hz[state["hz_slice"]].sub_(corr_z)


# --- material ADE primitives (pure recurrence, oracle-parity path) -----------


def advance_gyromagnetic_state(solver):
    """Advance the magnetization ADE one step by the pure Cayley recurrence.

    ``m <- Phi m + Gamma h`` at every active cell, with ``h`` the current transverse
    RF drive (single signed component per axis on the fast path, ``u . H`` / ``v . H``
    on the general path), then stages ``dm = m_new - m_old``. This is the *material*
    ADE model, validated against the Phase-0 oracle by the CW-parity gates; the
    field-coupled forward path uses the passive coupling in
    :func:`step_gyromagnetic_coupled`, not this explicit recurrence. No host
    synchronization and no per-step allocation.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    hu = state["hu"]
    hv = state["hv"]
    _gather_drive(solver, state, hu, hv)

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
    _scatter_correction(solver, solver._gyromagnetic_state)


# --- coupled implicit-midpoint forward path (passive) ------------------------


def snapshot_gyromagnetic_drive(solver):
    """Record the pre-update transverse RF drive (one half of the midpoint average).

    Called before the plain Yee magnetic update overwrites ``H``. Stores the
    ``H^{n-1/2}`` transverse drive into persistent buffers so the post-update
    coupled step can form the time-centred drive ``(H_pre + H_tmp)/2``. No per-step
    allocation.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    _gather_drive(solver, state, state["h_pre_u"], state["h_pre_v"])


def step_gyromagnetic_coupled(solver):
    """Coupled implicit-midpoint magnetization advance + non-reciprocal correction.

    Called after the plain Yee magnetic update (so ``H`` now holds the
    background-only ``H_tmp``). Forms the time-centred drive ``h_pre + h_tmp``,
    solves the coupled system in closed form ``m_new = B m_old + C (h_pre + h_tmp)``,
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
    _gather_drive(solver, state, hu, hv)
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

    # Non-reciprocal correction: H^{n+1} = H_tmp - dM/mu_inf.
    _scatter_correction(solver, state)
