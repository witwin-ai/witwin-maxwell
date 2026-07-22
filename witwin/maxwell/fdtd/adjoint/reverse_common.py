from __future__ import annotations

from dataclasses import dataclass

import torch


def _runtime():
    from . import core as _adjoint

    return _adjoint


@dataclass
class _CpmlReverseContext:
    pre_step_adjoint: dict[str, torch.Tensor]
    grad_eps_ex: torch.Tensor
    grad_eps_ey: torch.Tensor
    grad_eps_ez: torch.Tensor
    magnetic_output_adjoint: dict[str, torch.Tensor]
    adj_d_hz_dy: torch.Tensor
    adj_d_hy_dz: torch.Tensor
    adj_d_hx_dz: torch.Tensor
    adj_d_hz_dx: torch.Tensor
    adj_d_hy_dx: torch.Tensor
    adj_d_hx_dy: torch.Tensor
    adj_d_ez_dy: torch.Tensor
    adj_d_ey_dz: torch.Tensor
    adj_d_ex_dz: torch.Tensor
    adj_d_ez_dx: torch.Tensor
    adj_d_ey_dx: torch.Tensor
    adj_d_ex_dy: torch.Tensor
    ex_curl: torch.Tensor
    ey_curl: torch.Tensor
    ez_curl: torch.Tensor


class _ReverseScratch:
    """Per-backward-call reuse pool for the single-GPU adjoint reverse sweep.

    The bridge attaches one instance to the solver for the duration of one
    backward pass. Every reverse buffer whose shape is fixed across the whole
    sweep (the post-step adjoint snapshot, the CPML reverse context, the eps-cast
    dynamic electric curls) is allocated once here and refreshed in place each
    step instead of being freshly ``torch.zeros_like``-d, eliminating per-step
    allocator churn and lowering peak memory.

    The pool is intentionally single-GPU only: the distributed reverse drives
    per-shard solver objects that never carry a scratch attribute, so
    :func:`allocate_cpml_reverse_context` falls back to fresh allocation there and
    the multi-GPU path stays byte-for-byte unchanged.
    """

    __slots__ = ("_snapshot", "_cpml_ctx", "_cpml_signature", "curls", "curls_key")

    def __init__(self):
        self._snapshot = None
        self._cpml_ctx = None
        self._cpml_signature = None
        self.curls = None
        self.curls_key = None

    def snapshot(self, adjoint_state):
        """Return a reusable value-copy of ``adjoint_state`` (the post-step seed).

        Replaces the per-step ``{name: value.clone()}`` snapshot the backward loop
        used to allocate. The buffers are fully overwritten by ``copy_`` every
        step and are only read within the step (the reverse kernels consume them
        and the magnetic-output seed clones the H entries), so reuse is safe.
        """
        buf = self._snapshot
        if buf is None or not _layout_matches(buf, adjoint_state):
            buf = {name: torch.empty_like(tensor) for name, tensor in adjoint_state.items()}
            self._snapshot = buf
        for name, tensor in adjoint_state.items():
            buf[name].copy_(tensor)
        return buf


# The fused CPML electric/magnetic reverse kernels assign these pre-step adjoint
# entries unconditionally at every grid cell (adjoint.cu
# reverse_electric/magnetic_component_cpml write with `=`), so a reused context
# never needs to re-zero them. Every *other* key a composed runner leaves in the
# pre-step dict (electric/magnetic dispersive currents, added-in psi on a
# standard grid, etc.) must start each step at zero -- the dispersive/nonlinear
# runners fold onto or conditionally seed those entries and rely on the fresh
# baseline zero. Reusing the buffers means re-zeroing exactly that complement.
_CPML_PRE_WRITTEN_NAMES = frozenset(
    {
        "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
        "psi_ex_y", "psi_ex_z", "psi_ey_x", "psi_ey_z", "psi_ez_x", "psi_ez_y",
        "psi_hx_y", "psi_hx_z", "psi_hy_x", "psi_hy_z", "psi_hz_x", "psi_hz_y",
    }
)


def _layout_matches(buffers, reference) -> bool:
    if buffers.keys() != reference.keys():
        return False
    for name, tensor in reference.items():
        cached = buffers[name]
        if cached.shape != tensor.shape or cached.dtype != tensor.dtype:
            return False
    return True


def allocate_reverse_buffers(forward_state, *, eps_ex, eps_ey, eps_ez):
    grad_eps_ex = torch.zeros_like(eps_ex)
    grad_eps_ey = torch.zeros_like(eps_ey)
    grad_eps_ez = torch.zeros_like(eps_ez)
    pre_step_adjoint = {
        name: torch.zeros_like(tensor)
        for name, tensor in forward_state.items()
    }
    return pre_step_adjoint, grad_eps_ex, grad_eps_ey, grad_eps_ez


def dynamic_electric_curls(solver, *, eps_ex, eps_ey, eps_ez):
    """Cast the frozen base electric curls by the eps leaves.

    The result is a pure function of ``solver.cex_curl`` and the frozen eps
    leaves, both constant across the whole backward sweep. When the solver carries
    a reverse scratch (single-GPU backward), the curls are memoized on it keyed by
    the eps-leaf identity so every reverse step of the sweep shares one triple
    instead of recomputing the mul/div/contiguous each step.
    """
    scratch = getattr(solver, "_reverse_ctx_scratch", None)
    if scratch is not None:
        key = (id(eps_ex), id(eps_ey), id(eps_ez))
        if scratch.curls is not None and scratch.curls_key == key:
            return scratch.curls
    runtime = _runtime()
    curls = (
        runtime._dynamic_electric_curl(solver.cex_curl, solver.eps_Ex, eps_ex),
        runtime._dynamic_electric_curl(solver.cey_curl, solver.eps_Ey, eps_ey),
        runtime._dynamic_electric_curl(solver.cez_curl, solver.eps_Ez, eps_ez),
    )
    if scratch is not None:
        # The native reverse kernels read these curls as coefficient values only;
        # the eps gradient is produced analytically inside the kernels, never by
        # autograd through the curls. Detach before caching so the sweep does not
        # pin one dynamic-curl graph in memory for its whole duration.
        scratch.curls = tuple(curl.detach() for curl in curls)
        scratch.curls_key = key
        return scratch.curls
    return curls


def allocate_cpml_reverse_context(
    solver,
    forward_state,
    adjoint_state,
    *,
    eps_ex,
    eps_ey,
    eps_ez,
) -> _CpmlReverseContext:
    """Allocate the per-step reverse buffers the native CPML runner writes into.

    The mid-step magnetic field the electric update consumed is reconstructed by
    the shared Torch replay helper (``_forward_magnetic_fields``) at call sites,
    matching the analytic Torch reference exactly, so this context only owns the
    reverse-math outputs: the pre-step adjoint (electric/magnetic + psi), the eps
    gradient accumulators, the mid-step magnetic-output adjoint the electric
    reverse folds its curl(H) contribution into, the 12 curl-derivative adjoint
    scratch buffers, and the eps-cast dynamic electric curls.
    """
    ex_curl, ey_curl, ez_curl = dynamic_electric_curls(
        solver,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )

    scratch = getattr(solver, "_reverse_ctx_scratch", None)
    if scratch is not None:
        ctx = scratch._cpml_ctx
        signature = tuple(forward_state.keys())
        if ctx is None or scratch._cpml_signature != signature:
            ctx = _new_cpml_context(
                forward_state,
                adjoint_state,
                eps_ex=eps_ex,
                eps_ey=eps_ey,
                eps_ez=eps_ez,
            )
            scratch._cpml_ctx = ctx
            scratch._cpml_signature = signature
        else:
            # Re-seed only the mid-step magnetic adjoint (the electric folds add
            # into it, so it must start at the incoming post-step H adjoint). The
            # pre-step adjoint, eps-gradient, and curl-derivative scratch buffers
            # are fully assigned by the fused reverse kernels at every cell before
            # any read (adjoint.cu reverse_electric/magnetic_component_cpml write
            # unconditionally over the whole grid), so they need no re-zeroing.
            ctx.magnetic_output_adjoint["Hx"].copy_(adjoint_state["Hx"].detach())
            ctx.magnetic_output_adjoint["Hy"].copy_(adjoint_state["Hy"].detach())
            ctx.magnetic_output_adjoint["Hz"].copy_(adjoint_state["Hz"].detach())
            # Re-zero only the pre-step entries the fused kernels do not assign
            # (dispersive currents, standard-grid synthetic psi, ...); composed
            # runners fold onto these and expect the baseline fresh zero.
            for name, buffer in ctx.pre_step_adjoint.items():
                if name not in _CPML_PRE_WRITTEN_NAMES:
                    buffer.zero_()
        ctx.ex_curl = ex_curl
        ctx.ey_curl = ey_curl
        ctx.ez_curl = ez_curl
        return ctx

    ctx = _new_cpml_context(
        forward_state,
        adjoint_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    ctx.ex_curl = ex_curl
    ctx.ey_curl = ey_curl
    ctx.ez_curl = ez_curl
    return ctx


def _new_cpml_context(
    forward_state,
    adjoint_state,
    *,
    eps_ex,
    eps_ey,
    eps_ez,
) -> _CpmlReverseContext:
    pre_step_adjoint, grad_eps_ex, grad_eps_ey, grad_eps_ez = allocate_reverse_buffers(
        forward_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    return _CpmlReverseContext(
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        magnetic_output_adjoint={
            "Hx": adjoint_state["Hx"].detach().clone(),
            "Hy": adjoint_state["Hy"].detach().clone(),
            "Hz": adjoint_state["Hz"].detach().clone(),
        },
        adj_d_hz_dy=torch.zeros_like(forward_state["Ex"]),
        adj_d_hy_dz=torch.zeros_like(forward_state["Ex"]),
        adj_d_hx_dz=torch.zeros_like(forward_state["Ey"]),
        adj_d_hz_dx=torch.zeros_like(forward_state["Ey"]),
        adj_d_hy_dx=torch.zeros_like(forward_state["Ez"]),
        adj_d_hx_dy=torch.zeros_like(forward_state["Ez"]),
        adj_d_ez_dy=torch.zeros_like(forward_state["Hx"]),
        adj_d_ey_dz=torch.zeros_like(forward_state["Hx"]),
        adj_d_ex_dz=torch.zeros_like(forward_state["Hy"]),
        adj_d_ez_dx=torch.zeros_like(forward_state["Hy"]),
        adj_d_ey_dx=torch.zeros_like(forward_state["Hz"]),
        adj_d_ex_dy=torch.zeros_like(forward_state["Hz"]),
        ex_curl=None,
        ey_curl=None,
        ez_curl=None,
    )


def replay_standard_magnetic_step(
    solver,
    forward_state,
    *,
    forward_module,
    block_size,
    time_value,
    resolved_source_terms,
    include_imag: bool = False,
):
    runtime = _runtime()
    hx_mid = forward_state["Hx"].detach().clone()
    hy_mid = forward_state["Hy"].detach().clone()
    hz_mid = forward_state["Hz"].detach().clone()

    forward_module.updateMagneticFieldHxStandard3D(
        Hx=hx_mid,
        Ey=forward_state["Ey"],
        Ez=forward_state["Ez"],
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        invDy=solver.inv_dy_h,
        invDz=solver.inv_dz_h,
    ).launchRaw()
    forward_module.updateMagneticFieldHyStandard3D(
        Hy=hy_mid,
        Ex=forward_state["Ex"],
        Ez=forward_state["Ez"],
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
    ).launchRaw()
    forward_module.updateMagneticFieldHzStandard3D(
        Hz=hz_mid,
        Ex=forward_state["Ex"],
        Ey=forward_state["Ey"],
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
    ).launchRaw()

    magnetic_fields = {
        "Hx": hx_mid,
        "Hy": hy_mid,
        "Hz": hz_mid,
    }
    if include_imag:
        hx_mid_imag = forward_state["Hx_imag"].detach().clone()
        hy_mid_imag = forward_state["Hy_imag"].detach().clone()
        hz_mid_imag = forward_state["Hz_imag"].detach().clone()
        forward_module.updateMagneticFieldHxStandard3D(
            Hx=hx_mid_imag,
            Ey=forward_state["Ey_imag"],
            Ez=forward_state["Ez_imag"],
            HxDecay=solver.chx_decay,
            HxCurl=solver.chx_curl,
            invDy=solver.inv_dy_h,
            invDz=solver.inv_dz_h,
        ).launchRaw()
        forward_module.updateMagneticFieldHyStandard3D(
            Hy=hy_mid_imag,
            Ex=forward_state["Ex_imag"],
            Ez=forward_state["Ez_imag"],
            HyDecay=solver.chy_decay,
            HyCurl=solver.chy_curl,
            invDx=solver.inv_dx_h,
            invDz=solver.inv_dz_h,
        ).launchRaw()
        forward_module.updateMagneticFieldHzStandard3D(
            Hz=hz_mid_imag,
            Ex=forward_state["Ex_imag"],
            Ey=forward_state["Ey_imag"],
            HzDecay=solver.chz_decay,
            HzCurl=solver.chz_curl,
            invDx=solver.inv_dx_h,
            invDy=solver.inv_dy_h,
        ).launchRaw()
        magnetic_fields.update(
            Hx_imag=hx_mid_imag,
            Hy_imag=hy_mid_imag,
            Hz_imag=hz_mid_imag,
        )

    return runtime._apply_resolved_magnetic_source_terms(
        magnetic_fields,
        solver=solver,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
