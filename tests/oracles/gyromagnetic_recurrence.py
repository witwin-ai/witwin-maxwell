"""Torch forms of the gyromagnetic magnetization step (frozen test oracles).

Moved out of ``witwin.maxwell.fdtd.runtime.gyromagnetic``, whose drive snapshot and
coupled implicit-midpoint step are CUDA kernels (``fdtd/cuda/kernels/gyromagnetic.cu``):

* :func:`snapshot_gyromagnetic_drive` / :func:`step_gyromagnetic_coupled` are the
  former Torch runtime hooks, the reference for the kernels and the float64 form the
  mock-solver gates step;
* :func:`advance_gyromagnetic_state` / :func:`apply_gyromagnetic_correction` are the
  pure Cayley recurrence of the material ADE, which only serves the CW-parity gates
  against :mod:`tests.oracles.ferrite_reference`.

They read the runtime's prepared state (``build_gyromagnetic``) and keep their own
scratch buffers in it, allocated on first use.
"""
from __future__ import annotations

import torch


def _scratch(state, *names):
    """Per-state oracle scratch, allocated once (dense overlap shape, field dtype)."""
    for name in names:
        if f"oracle_{name}" not in state:
            state[f"oracle_{name}"] = torch.zeros_like(state["m_u"])
    return [state[f"oracle_{name}"] for name in names]


def _active(state):
    if "oracle_active_f" not in state:
        state["oracle_active_f"] = state["active"].to(dtype=state["m_u"].dtype)
    return state["oracle_active_f"]


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
        corr_u, corr_v = _scratch(state, "corr_u", "corr_v")
        torch.mul(dm_u, inv_mu, out=corr_u)
        torch.mul(dm_v, inv_mu, out=corr_v)
        corr_u.mul_(state["sign_u"])
        corr_v.mul_(state["sign_v"])
        getattr(solver, state["u_attr"])[state["u_slice"]].sub_(corr_u)
        getattr(solver, state["v_attr"])[state["v_slice"]].sub_(corr_v)
        return
    corr_x, corr_y, corr_z = _scratch(state, "corr_x", "corr_y", "corr_z")
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


def _stage_increment(state, new_u, new_v):
    """``dm = (m_new - m_old)`` masked to the ferrite, then ``m += dm``."""
    m_u, m_v = state["m_u"], state["m_v"]
    dm_u, dm_v = state["dm_u"], state["dm_v"]
    active_f = _active(state)
    torch.sub(new_u, m_u, out=dm_u)
    torch.sub(new_v, m_v, out=dm_v)
    dm_u.mul_(active_f)
    dm_v.mul_(active_f)
    m_u.add_(dm_u)
    m_v.add_(dm_v)


def snapshot_gyromagnetic_drive(solver):
    """Torch reference of the runtime drive snapshot: ``h_pre = drive(H^{n-1/2})``."""
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    _gather_drive(solver, state, state["h_pre_u"], state["h_pre_v"])


def step_gyromagnetic_coupled(solver):
    """Torch reference of the runtime coupled step.

    ``m_new = B m_old + C (h_pre + h_tmp)``, ``dm = m_new - m_old`` and
    ``H^{n+1} = H_tmp - dM/mu_inf``, op for op as the former runtime hook.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    hu, hv, new_u, new_v = _scratch(state, "hu", "hv", "new_u", "new_v")
    _gather_drive(solver, state, hu, hv)
    hu.add_(state["h_pre_u"])
    hv.add_(state["h_pre_v"])
    m_u, m_v = state["m_u"], state["m_v"]
    torch.mul(state["b00"], m_u, out=new_u)
    new_u.addcmul_(state["b01"], m_v)
    new_u.addcmul_(state["c00"], hu)
    new_u.addcmul_(state["c01"], hv)
    torch.mul(state["b10"], m_u, out=new_v)
    new_v.addcmul_(state["b11"], m_v)
    new_v.addcmul_(state["c10"], hu)
    new_v.addcmul_(state["c11"], hv)
    _stage_increment(state, new_u, new_v)
    _scatter_correction(solver, state)


def advance_gyromagnetic_state(solver):
    """Advance the magnetization ADE one step by the pure Cayley recurrence.

    ``m <- Phi m + Gamma h`` at every active cell, with ``h`` the current transverse
    RF drive (single signed component per axis on the fast path, ``u . H`` / ``v . H``
    on the general path), then stages ``dm = m_new - m_old``. This is the *material*
    ADE model, validated against the Phase-0 oracle (``ferrite_reference.py``) by the
    CW-parity gates; the field-coupled forward path uses the passive coupling of
    :func:`step_gyromagnetic_coupled`, not this explicit recurrence.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    state = solver._gyromagnetic_state
    hu, hv, new_u, new_v = _scratch(state, "hu", "hv", "new_u", "new_v")
    _gather_drive(solver, state, hu, hv)
    m_u, m_v = state["m_u"], state["m_v"]
    torch.mul(state["phi00"], m_u, out=new_u)
    new_u.addcmul_(state["phi01"], m_v)
    new_u.addcmul_(state["gamma00"], hu)
    new_u.addcmul_(state["gamma01"], hv)
    torch.mul(state["phi10"], m_u, out=new_v)
    new_v.addcmul_(state["phi11"], m_v)
    new_v.addcmul_(state["gamma10"], hu)
    new_v.addcmul_(state["gamma11"], hv)
    _stage_increment(state, new_u, new_v)


def apply_gyromagnetic_correction(solver):
    """Subtract the staged magnetization increment from the field (pure form).

    ``H^{n+1} = H_tmp - dM/mu_inf`` using the ``dm`` staged by
    :func:`advance_gyromagnetic_state`.
    """
    if not getattr(solver, "gyromagnetic_enabled", False):
        return
    _scatter_correction(solver, solver._gyromagnetic_state)
