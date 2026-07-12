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
    runtime = _runtime()
    return (
        runtime._dynamic_electric_curl(solver.cex_curl, solver.eps_Ex, eps_ex),
        runtime._dynamic_electric_curl(solver.cey_curl, solver.eps_Ey, eps_ey),
        runtime._dynamic_electric_curl(solver.cez_curl, solver.eps_Ez, eps_ez),
    )


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
    pre_step_adjoint, grad_eps_ex, grad_eps_ey, grad_eps_ez = allocate_reverse_buffers(
        forward_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    ex_curl, ey_curl, ez_curl = dynamic_electric_curls(
        solver,
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
        ex_curl=ex_curl,
        ey_curl=ey_curl,
        ez_curl=ez_curl,
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
    ).launchRaw(
        blockSize=block_size,
        gridSize=runtime._adjoint_launch_shape(solver, "Hx", hx_mid),
    )
    forward_module.updateMagneticFieldHyStandard3D(
        Hy=hy_mid,
        Ex=forward_state["Ex"],
        Ez=forward_state["Ez"],
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
    ).launchRaw(
        blockSize=block_size,
        gridSize=runtime._adjoint_launch_shape(solver, "Hy", hy_mid),
    )
    forward_module.updateMagneticFieldHzStandard3D(
        Hz=hz_mid,
        Ex=forward_state["Ex"],
        Ey=forward_state["Ey"],
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
    ).launchRaw(
        blockSize=block_size,
        gridSize=runtime._adjoint_launch_shape(solver, "Hz", hz_mid),
    )

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
        ).launchRaw(
            blockSize=block_size,
            gridSize=runtime._adjoint_launch_shape(solver, "Hx", hx_mid_imag),
        )
        forward_module.updateMagneticFieldHyStandard3D(
            Hy=hy_mid_imag,
            Ex=forward_state["Ex_imag"],
            Ez=forward_state["Ez_imag"],
            HyDecay=solver.chy_decay,
            HyCurl=solver.chy_curl,
            invDx=solver.inv_dx_h,
            invDz=solver.inv_dz_h,
        ).launchRaw(
            blockSize=block_size,
            gridSize=runtime._adjoint_launch_shape(solver, "Hy", hy_mid_imag),
        )
        forward_module.updateMagneticFieldHzStandard3D(
            Hz=hz_mid_imag,
            Ex=forward_state["Ex_imag"],
            Ey=forward_state["Ey_imag"],
            HzDecay=solver.chz_decay,
            HzCurl=solver.chz_curl,
            invDx=solver.inv_dx_h,
            invDy=solver.inv_dy_h,
        ).launchRaw(
            blockSize=block_size,
            gridSize=runtime._adjoint_launch_shape(solver, "Hz", hz_mid_imag),
        )
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
