from __future__ import annotations

from dataclasses import dataclass

import torch

from .profiler import _BackwardProfiler, _ReverseStepResult


def _runtime():
    from . import core as _adjoint

    return _adjoint


@dataclass
class _CpmlReverseContext:
    hx_mid: torch.Tensor
    hy_mid: torch.Tensor
    hz_mid: torch.Tensor
    psi_hx_y_mid: torch.Tensor
    psi_hx_z_mid: torch.Tensor
    psi_hy_x_mid: torch.Tensor
    psi_hy_z_mid: torch.Tensor
    psi_hz_x_mid: torch.Tensor
    psi_hz_y_mid: torch.Tensor
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


@dataclass
class _DispersiveReverseContext:
    runtime: object
    active_profiler: _BackwardProfiler
    adjoint_module: object
    block_size: tuple[int, int, int]
    dispersive_output_adjoint: dict[str, torch.Tensor]
    correction_grad_eps: dict[str, torch.Tensor]
    source_adjoint_state: dict[str, torch.Tensor]
    adjusted_adjoint_state: dict[str, torch.Tensor]


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
        hx_mid=forward_state["Hx"].detach().clone(),
        hy_mid=forward_state["Hy"].detach().clone(),
        hz_mid=forward_state["Hz"].detach().clone(),
        psi_hx_y_mid=forward_state["psi_hx_y"].detach().clone(),
        psi_hx_z_mid=forward_state["psi_hx_z"].detach().clone(),
        psi_hy_x_mid=forward_state["psi_hy_x"].detach().clone(),
        psi_hy_z_mid=forward_state["psi_hy_z"].detach().clone(),
        psi_hz_x_mid=forward_state["psi_hz_x"].detach().clone(),
        psi_hz_y_mid=forward_state["psi_hz_y"].detach().clone(),
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


def replay_cpml_magnetic_step(
    solver,
    forward_state,
    context: _CpmlReverseContext,
    *,
    forward_module,
    block_size,
    time_value,
    resolved_source_terms,
):
    runtime = _runtime()
    forward_module.updateMagneticFieldHx3D(
        Hx=context.hx_mid,
        Ey=forward_state["Ey"],
        Ez=forward_state["Ez"],
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        PsiHxY=context.psi_hx_y_mid,
        PsiHxZ=context.psi_hx_z_mid,
        InvKappaHxY=solver.cpml_inv_kappa_h_y,
        ByHxY=solver.cpml_b_h_y,
        CyHxY=solver.cpml_c_h_y,
        InvKappaHxZ=solver.cpml_inv_kappa_h_z,
        ByHxZ=solver.cpml_b_h_z,
        CyHxZ=solver.cpml_c_h_z,
        invDy=solver.inv_dy,
        invDz=solver.inv_dz,
    ).launchRaw(
        blockSize=block_size,
        gridSize=runtime._adjoint_launch_shape(solver, "Hx", context.hx_mid),
    )
    forward_module.updateMagneticFieldHy3D(
        Hy=context.hy_mid,
        Ex=forward_state["Ex"],
        Ez=forward_state["Ez"],
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        PsiHyX=context.psi_hy_x_mid,
        PsiHyZ=context.psi_hy_z_mid,
        InvKappaHyX=solver.cpml_inv_kappa_h_x,
        ByHyX=solver.cpml_b_h_x,
        CyHyX=solver.cpml_c_h_x,
        InvKappaHyZ=solver.cpml_inv_kappa_h_z,
        ByHyZ=solver.cpml_b_h_z,
        CyHyZ=solver.cpml_c_h_z,
        invDx=solver.inv_dx,
        invDz=solver.inv_dz,
    ).launchRaw(
        blockSize=block_size,
        gridSize=runtime._adjoint_launch_shape(solver, "Hy", context.hy_mid),
    )
    forward_module.updateMagneticFieldHz3D(
        Hz=context.hz_mid,
        Ex=forward_state["Ex"],
        Ey=forward_state["Ey"],
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        PsiHzX=context.psi_hz_x_mid,
        PsiHzY=context.psi_hz_y_mid,
        InvKappaHzX=solver.cpml_inv_kappa_h_x,
        ByHzX=solver.cpml_b_h_x,
        CyHzX=solver.cpml_c_h_x,
        InvKappaHzY=solver.cpml_inv_kappa_h_y,
        ByHzY=solver.cpml_b_h_y,
        CyHzY=solver.cpml_c_h_y,
        invDx=solver.inv_dx,
        invDy=solver.inv_dy,
    ).launchRaw(
        blockSize=block_size,
        gridSize=runtime._adjoint_launch_shape(solver, "Hz", context.hz_mid),
    )
    return runtime._apply_resolved_magnetic_source_terms(
        {"Hx": context.hx_mid, "Hy": context.hy_mid, "Hz": context.hz_mid},
        solver=solver,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )


def prepare_dispersive_reverse_context(
    solver,
    forward_state,
    adjoint_state,
    *,
    eps_ex,
    eps_ey,
    eps_ez,
    profiler: _BackwardProfiler | None,
) -> _DispersiveReverseContext:
    runtime = _runtime()
    active_profiler = profiler if profiler is not None else _BackwardProfiler(enabled=False, device=None)
    dispersive_state = runtime._advance_dispersive_state(solver, forward_state)
    electric_source_adjoint, dispersive_output_adjoint, correction_grad_eps, source_adjoint_state = (
        runtime._reverse_dispersive_corrections(
            solver,
            adjoint_state,
            dispersive_state,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
        )
    )
    adjusted_adjoint_state = dict(adjoint_state)
    adjusted_adjoint_state.update(electric_source_adjoint)
    return _DispersiveReverseContext(
        runtime=runtime,
        active_profiler=active_profiler,
        adjoint_module=solver.fdtd_module,
        block_size=runtime._adjoint_kernel_block_size(solver),
        dispersive_output_adjoint=dispersive_output_adjoint,
        correction_grad_eps=correction_grad_eps,
        source_adjoint_state=source_adjoint_state,
        adjusted_adjoint_state=adjusted_adjoint_state,
    )


def finalize_dispersive_reverse_step(
    solver,
    forward_state,
    *,
    base_result: _ReverseStepResult,
    context: _DispersiveReverseContext,
    backend: str,
) -> _ReverseStepResult:
    runtime = context.runtime
    pre_step_adjoint = {
        name: tensor.detach().clone()
        for name, tensor in base_result.pre_step_adjoint.items()
    }
    for name in runtime.checkpoint_schema(solver).dispersive_state_names:
        if name not in pre_step_adjoint:
            pre_step_adjoint[name] = torch.zeros_like(forward_state[name])

    for component_name, model_name, index, _tensor_names, entry in runtime.iter_dispersive_state_specs(solver) or ():
        launch_shape = runtime._adjoint_launch_shape(solver, component_name, forward_state[component_name])
        if model_name == "debye":
            polarization_name = runtime.dispersive_state_name(component_name, model_name, index, "polarization")
            current_name = runtime.dispersive_state_name(component_name, model_name, index, "current")
            context.adjoint_module.reverseDebyeCurrent3D(
                AdjElectricPrev=pre_step_adjoint[component_name],
                AdjPolarizationPrev=pre_step_adjoint[polarization_name],
                AdjPolarizationPost=context.dispersive_output_adjoint[polarization_name],
                AdjCurrentPost=context.dispersive_output_adjoint[current_name],
                DebyeDrive=entry["drive"],
                decay=float(entry["decay"]),
                dt=float(solver.dt),
            ).launchRaw(
                blockSize=context.block_size,
                gridSize=launch_shape,
            )
            continue

        if model_name == "drude":
            current_name = runtime.dispersive_state_name(component_name, model_name, index, "current")
            context.adjoint_module.reverseDrudeCurrent3D(
                AdjElectricPrev=pre_step_adjoint[component_name],
                AdjCurrentPrev=pre_step_adjoint[current_name],
                AdjCurrentPost=context.dispersive_output_adjoint[current_name],
                DrudeDrive=entry["drive"],
                decay=float(entry["decay"]),
            ).launchRaw(
                blockSize=context.block_size,
                gridSize=launch_shape,
            )
            continue

        polarization_name = runtime.dispersive_state_name(component_name, model_name, index, "polarization")
        current_name = runtime.dispersive_state_name(component_name, model_name, index, "current")
        context.adjoint_module.reverseLorentzCurrent3D(
            AdjElectricPrev=pre_step_adjoint[component_name],
            AdjPolarizationPrev=pre_step_adjoint[polarization_name],
            AdjCurrentPrev=pre_step_adjoint[current_name],
            AdjPolarizationPost=context.dispersive_output_adjoint[polarization_name],
            AdjCurrentPost=context.dispersive_output_adjoint[current_name],
            LorentzDrive=entry["drive"],
            decay=float(entry["decay"]),
            restoring=float(entry["restoring"]),
            dt=float(solver.dt),
        ).launchRaw(
            blockSize=context.block_size,
            gridSize=launch_shape,
        )

    return finalize_reverse_step_result(
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=base_result.grad_eps_ex + context.correction_grad_eps["Ex"],
        grad_eps_ey=base_result.grad_eps_ey + context.correction_grad_eps["Ey"],
        grad_eps_ez=base_result.grad_eps_ez + context.correction_grad_eps["Ez"],
        backend=backend,
        source_adjoint_state=context.source_adjoint_state,
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
        invDy=solver.inv_dy,
        invDz=solver.inv_dz,
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
        invDx=solver.inv_dx,
        invDz=solver.inv_dz,
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
        invDx=solver.inv_dx,
        invDy=solver.inv_dy,
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
            invDy=solver.inv_dy,
            invDz=solver.inv_dz,
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
            invDx=solver.inv_dx,
            invDz=solver.inv_dz,
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
            invDx=solver.inv_dx,
            invDy=solver.inv_dy,
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


def finalize_reverse_step_result(
    *,
    pre_step_adjoint,
    grad_eps_ex,
    grad_eps_ey,
    grad_eps_ez,
    backend,
    magnetic_output_adjoint=None,
    source_adjoint_state=None,
):
    magnetic_output = None
    if magnetic_output_adjoint is not None:
        magnetic_output = {
            name: tensor.detach()
            for name, tensor in magnetic_output_adjoint.items()
        }
    return _ReverseStepResult(
        pre_step_adjoint={name: tensor.detach() for name, tensor in pre_step_adjoint.items()},
        grad_eps_ex=grad_eps_ex.detach(),
        grad_eps_ey=grad_eps_ey.detach(),
        grad_eps_ez=grad_eps_ez.detach(),
        backend=backend,
        magnetic_output_adjoint=magnetic_output,
        source_adjoint_state=source_adjoint_state,
    )
