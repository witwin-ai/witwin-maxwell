"""Native CUDA reverse-step runners for the FDTD adjoint.

These runners orchestrate the fused reverse-math CUDA kernels (registered in
``cuda/backend.py`` and parity-tested against the frozen Torch reference) into a
full per-step reverse pass that is bit-for-bit equivalent to the analytic Torch
reference backend it mirrors. Python only sequences the kernel launches; every
per-cell reverse computation (electric-adjoint -> mid-H adjoint, magnetic-adjoint
-> pre-step E adjoint + eps gradient, and the magnetic decay pullback) runs
inside the compiled kernels.

The mid-step magnetic field is still reconstructed with the shared Torch replay
helper (``_forward_magnetic_fields``); nativizing that replay is tracked
separately under the native-replay work. The reverse *math* itself is fully
native here.
"""

from __future__ import annotations

from ..cuda import backend as _cuda_backend
from .dispatch import (
    _NATIVE_REVERSE_LABELS,
    _ReverseBackend,
    register_native_reverse_backend,
)


def _scene_is_cuda(forward_state) -> bool:
    reference = forward_state.get("Ex")
    return bool(reference is not None and reference.is_cuda)


def _cuda_scene_native_qualifies(solver, forward_state) -> bool:
    """Native reverse runners only run on a CUDA scene with the compiled extension."""
    return bool(_scene_is_cuda(forward_state) and _cuda_backend.is_available())


def _reverse_step_standard_native(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
    profiler,
):
    """Fully native reverse step for the standard (open-boundary) configuration.

    Mirrors ``reverse_step_standard_python_reference`` exactly, launching the
    fused reverse kernels in a fixed order. The fused kernels *assign* (not
    accumulate) into their outputs, so the launch order is load-bearing: the
    electric->H kernels must fully populate the mid-H adjoints before the
    magnetic->E kernels (which read them) and before the decay pullback.
    """
    import torch

    from . import core as _adjoint
    from .reverse_common import dynamic_electric_curls

    # Mid-step H the forward electric update consumed (shared Torch replay).
    magnetic_fields = _adjoint._forward_magnetic_fields(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
    hx_mid = magnetic_fields["Hx"]
    hy_mid = magnetic_fields["Hy"]
    hz_mid = magnetic_fields["Hz"]

    # Dynamic electric curl coefficients (cast base curl by the eps leaf).
    ex_curl, ey_curl, ez_curl = dynamic_electric_curls(
        solver, eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez
    )

    # Phase 1: electric adjoint -> mid-H adjoint (the magnetic_output_adjoint).
    adj_hx_mid = torch.empty_like(forward_state["Hx"])
    adj_hy_mid = torch.empty_like(forward_state["Hy"])
    adj_hz_mid = torch.empty_like(forward_state["Hz"])
    _cuda_backend._reverse_electric_hx_standard(
        AdjHxMid=adj_hx_mid,
        AdjHxPost=adjoint_state["Hx"],
        AdjEyPost=adjoint_state["Ey"],
        AdjEzPost=adjoint_state["Ez"],
        EyCurl=ey_curl,
        EzCurl=ez_curl,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
    )
    _cuda_backend._reverse_electric_hy_standard(
        AdjHyMid=adj_hy_mid,
        AdjHyPost=adjoint_state["Hy"],
        AdjExPost=adjoint_state["Ex"],
        AdjEzPost=adjoint_state["Ez"],
        ExCurl=ex_curl,
        EzCurl=ez_curl,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
    )
    _cuda_backend._reverse_electric_hz_standard(
        AdjHzMid=adj_hz_mid,
        AdjHzPost=adjoint_state["Hz"],
        AdjExPost=adjoint_state["Ex"],
        AdjEyPost=adjoint_state["Ey"],
        ExCurl=ex_curl,
        EyCurl=ey_curl,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
    )

    # Phase 2: magnetic adjoint -> pre-step E adjoint + eps gradient. Each kernel
    # reconstructs its own curl(H) from the mid-H fields and assigns the complete
    # pre-step E adjoint (electric decay pullback + magnetic forward-diff).
    adj_ex_prev = torch.empty_like(forward_state["Ex"])
    adj_ey_prev = torch.empty_like(forward_state["Ey"])
    adj_ez_prev = torch.empty_like(forward_state["Ez"])
    grad_eps_ex = torch.empty_like(eps_ex)
    grad_eps_ey = torch.empty_like(eps_ey)
    grad_eps_ez = torch.empty_like(eps_ez)
    _cuda_backend._reverse_magnetic_ex_standard(
        AdjExPrev=adj_ex_prev,
        GradEpsEx=grad_eps_ex,
        AdjExPost=adjoint_state["Ex"],
        AdjHyMid=adj_hy_mid,
        AdjHzMid=adj_hz_mid,
        ExDecay=solver.cex_decay,
        ExCurl=ex_curl,
        EpsEx=eps_ex,
        HyMid=hy_mid,
        HzMid=hz_mid,
        HyCurl=solver.chy_curl,
        HzCurl=solver.chz_curl,
        invDyE=solver.inv_dy_e,
        invDzE=solver.inv_dz_e,
        invDyH=solver.inv_dy_h,
        invDzH=solver.inv_dz_h,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    )
    _cuda_backend._reverse_magnetic_ey_standard(
        AdjEyPrev=adj_ey_prev,
        GradEpsEy=grad_eps_ey,
        AdjEyPost=adjoint_state["Ey"],
        AdjHxMid=adj_hx_mid,
        AdjHzMid=adj_hz_mid,
        EyDecay=solver.cey_decay,
        EyCurl=ey_curl,
        EpsEy=eps_ey,
        HxMid=hx_mid,
        HzMid=hz_mid,
        HxCurl=solver.chx_curl,
        HzCurl=solver.chz_curl,
        invDxE=solver.inv_dx_e,
        invDzE=solver.inv_dz_e,
        invDxH=solver.inv_dx_h,
        invDzH=solver.inv_dz_h,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    )
    _cuda_backend._reverse_magnetic_ez_standard(
        AdjEzPrev=adj_ez_prev,
        GradEpsEz=grad_eps_ez,
        AdjEzPost=adjoint_state["Ez"],
        AdjHxMid=adj_hx_mid,
        AdjHyMid=adj_hy_mid,
        EzDecay=solver.cez_decay,
        EzCurl=ez_curl,
        EpsEz=eps_ez,
        HxMid=hx_mid,
        HyMid=hy_mid,
        HxCurl=solver.chx_curl,
        HyCurl=solver.chy_curl,
        invDxE=solver.inv_dx_e,
        invDyE=solver.inv_dy_e,
        invDxH=solver.inv_dx_h,
        invDyH=solver.inv_dy_h,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
    )

    # Phase 3: magnetic decay pullback -> pre-step H adjoint.
    adj_hx_prev = torch.empty_like(forward_state["Hx"])
    adj_hy_prev = torch.empty_like(forward_state["Hy"])
    adj_hz_prev = torch.empty_like(forward_state["Hz"])
    _cuda_backend._reverse_magnetic_hx_decay(
        AdjHxPrev=adj_hx_prev, AdjHxMid=adj_hx_mid, HxDecay=solver.chx_decay
    )
    _cuda_backend._reverse_magnetic_hy_decay(
        AdjHyPrev=adj_hy_prev, AdjHyMid=adj_hy_mid, HyDecay=solver.chy_decay
    )
    _cuda_backend._reverse_magnetic_hz_decay(
        AdjHzPrev=adj_hz_prev, AdjHzMid=adj_hz_mid, HzDecay=solver.chz_decay
    )

    step_result = _adjoint._ReverseStepResult(
        pre_step_adjoint={
            "Ex": adj_ex_prev,
            "Ey": adj_ey_prev,
            "Ez": adj_ez_prev,
            "Hx": adj_hx_prev,
            "Hy": adj_hy_prev,
            "Hz": adj_hz_prev,
        },
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_STANDARD],
        magnetic_output_adjoint={"Hx": adj_hx_mid, "Hy": adj_hy_mid, "Hz": adj_hz_mid},
    )
    # The native runner owns the full reverse-step contract, including the
    # analytic source-term eps-gradient accumulation the reference path applies.
    return _adjoint._accumulate_source_term_gradients(
        step_result,
        solver=solver,
        adjoint_state=adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def _reverse_step_cpml_native(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
    profiler,
):
    """Fully native reverse step for the CPML (absorbing-boundary) configuration.

    Mirrors ``reverse_step_cpml_python_reference`` exactly, launching the fused
    CPML reverse kernels in a fixed order. Every per-cell reverse computation
    (electric-adjoint -> pre-step E/psi_e adjoint + eps gradient + curl(H)
    derivative adjoint, and magnetic-decay/psi_h pullback -> pre-step H/psi_h
    adjoint + curl(E) derivative adjoint) runs inside the compiled kernels; Python
    only sequences the launches.

    Launch order is load-bearing. The fused CPML electric/magnetic kernels
    *assign* into their pre-step / eps-gradient / psi outputs, and the transposed
    difference accumulators *add* into the mid-step H adjoint and the pre-step E
    adjoint. So all three electric kernels (and their backward-difference folds)
    must fully populate the mid-step H adjoint before any magnetic kernel reads
    it, and the electric kernels must assign the pre-step E adjoint before the
    magnetic kernels add their forward-difference contributions into it.

    The coefficient argument lists are long and per-component permuted (the
    positive/negative CPML axis, its ``b``/``c``/``inv_kappa`` stretch vectors,
    and the pre-step psi state all rotate with the component); the wiring here is
    a line-for-line transcription of the Torch reference and is pinned against it
    by the step-level parity test.
    """
    from . import core as _adjoint
    from .reverse_common import allocate_cpml_reverse_context

    # Mid-step H the forward electric update consumed (shared Torch replay).
    magnetic_fields = _adjoint._forward_magnetic_fields(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
    hx_mid = magnetic_fields["Hx"]
    hy_mid = magnetic_fields["Hy"]
    hz_mid = magnetic_fields["Hz"]

    ctx = allocate_cpml_reverse_context(
        solver,
        forward_state,
        adjoint_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    pre = ctx.pre_step_adjoint
    adj_h_mid = ctx.magnetic_output_adjoint

    # Phase 1: electric adjoint -> pre-step E/psi_e adjoint + eps gradient, and
    # fold each curl(H)-derivative adjoint into the mid-step H adjoint.
    _cuda_backend._reverse_electric_cpml_ex(
        AdjExPrev=pre["Ex"],
        GradEpsEx=ctx.grad_eps_ex,
        AdjPsiPosPrev=pre["psi_ex_y"],
        AdjPsiNegPrev=pre["psi_ex_z"],
        AdjDPos=ctx.adj_d_hz_dy,
        AdjDNeg=ctx.adj_d_hy_dz,
        AdjExPost=adjoint_state["Ex"],
        AdjPsiPosPost=adjoint_state["psi_ex_y"],
        AdjPsiNegPost=adjoint_state["psi_ex_z"],
        ExDecay=solver.cex_decay,
        ExCurl=ctx.ex_curl,
        EpsEx=eps_ex,
        PsiPos=forward_state["psi_ex_y"],
        PsiNeg=forward_state["psi_ex_z"],
        BPos=solver.cpml_b_e_y,
        CPos=solver.cpml_c_e_y,
        InvKappaPos=solver.cpml_inv_kappa_e_y,
        BNeg=solver.cpml_b_e_z,
        CNeg=solver.cpml_c_e_z,
        InvKappaNeg=solver.cpml_inv_kappa_e_z,
        HyMid=hy_mid,
        HzMid=hz_mid,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    )
    _cuda_backend._accumulate_backward_diff_y(FieldGrad=adj_h_mid["Hz"], DiffGrad=ctx.adj_d_hz_dy, invDy=solver.inv_dy_e)
    _cuda_backend._accumulate_backward_diff_z(FieldGrad=adj_h_mid["Hy"], DiffGrad=ctx.adj_d_hy_dz, invDz=solver.inv_dz_e)

    _cuda_backend._reverse_electric_cpml_ey(
        AdjEyPrev=pre["Ey"],
        GradEpsEy=ctx.grad_eps_ey,
        AdjPsiPosPrev=pre["psi_ey_z"],
        AdjPsiNegPrev=pre["psi_ey_x"],
        AdjDPos=ctx.adj_d_hx_dz,
        AdjDNeg=ctx.adj_d_hz_dx,
        AdjEyPost=adjoint_state["Ey"],
        AdjPsiPosPost=adjoint_state["psi_ey_x"],
        AdjPsiNegPost=adjoint_state["psi_ey_z"],
        EyDecay=solver.cey_decay,
        EyCurl=ctx.ey_curl,
        EpsEy=eps_ey,
        PsiPos=forward_state["psi_ey_z"],
        PsiNeg=forward_state["psi_ey_x"],
        BPos=solver.cpml_b_e_z,
        CPos=solver.cpml_c_e_z,
        InvKappaPos=solver.cpml_inv_kappa_e_z,
        BNeg=solver.cpml_b_e_x,
        CNeg=solver.cpml_c_e_x,
        InvKappaNeg=solver.cpml_inv_kappa_e_x,
        HxMid=hx_mid,
        HzMid=hz_mid,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    )
    _cuda_backend._accumulate_backward_diff_z(FieldGrad=adj_h_mid["Hx"], DiffGrad=ctx.adj_d_hx_dz, invDz=solver.inv_dz_e)
    _cuda_backend._accumulate_backward_diff_x(FieldGrad=adj_h_mid["Hz"], DiffGrad=ctx.adj_d_hz_dx, invDx=solver.inv_dx_e)

    _cuda_backend._reverse_electric_cpml_ez(
        AdjEzPrev=pre["Ez"],
        GradEpsEz=ctx.grad_eps_ez,
        AdjPsiPosPrev=pre["psi_ez_x"],
        AdjPsiNegPrev=pre["psi_ez_y"],
        AdjDPos=ctx.adj_d_hy_dx,
        AdjDNeg=ctx.adj_d_hx_dy,
        AdjEzPost=adjoint_state["Ez"],
        AdjPsiPosPost=adjoint_state["psi_ez_x"],
        AdjPsiNegPost=adjoint_state["psi_ez_y"],
        EzDecay=solver.cez_decay,
        EzCurl=ctx.ez_curl,
        EpsEz=eps_ez,
        PsiPos=forward_state["psi_ez_x"],
        PsiNeg=forward_state["psi_ez_y"],
        BPos=solver.cpml_b_e_x,
        CPos=solver.cpml_c_e_x,
        InvKappaPos=solver.cpml_inv_kappa_e_x,
        BNeg=solver.cpml_b_e_y,
        CNeg=solver.cpml_c_e_y,
        InvKappaNeg=solver.cpml_inv_kappa_e_y,
        HxMid=hx_mid,
        HyMid=hy_mid,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
    )
    _cuda_backend._accumulate_backward_diff_x(FieldGrad=adj_h_mid["Hy"], DiffGrad=ctx.adj_d_hy_dx, invDx=solver.inv_dx_e)
    _cuda_backend._accumulate_backward_diff_y(FieldGrad=adj_h_mid["Hx"], DiffGrad=ctx.adj_d_hx_dy, invDy=solver.inv_dy_e)

    # Phase 2: magnetic-decay + psi_h pullback -> pre-step H/psi_h adjoint,
    # folding each curl(E)-derivative adjoint into the pre-step E adjoint.
    _cuda_backend._reverse_magnetic_cpml_hx(
        AdjHxPrev=pre["Hx"],
        AdjPsiPosPrev=pre["psi_hx_y"],
        AdjPsiNegPrev=pre["psi_hx_z"],
        AdjDPos=ctx.adj_d_ez_dy,
        AdjDNeg=ctx.adj_d_ey_dz,
        AdjHxPost=adj_h_mid["Hx"],
        AdjPsiPosPost=adjoint_state["psi_hx_y"],
        AdjPsiNegPost=adjoint_state["psi_hx_z"],
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        BPos=solver.cpml_b_h_y,
        CPos=solver.cpml_c_h_y,
        InvKappaPos=solver.cpml_inv_kappa_h_y,
        BNeg=solver.cpml_b_h_z,
        CNeg=solver.cpml_c_h_z,
        InvKappaNeg=solver.cpml_inv_kappa_h_z,
    )
    _cuda_backend._accumulate_forward_diff_y(FieldGrad=pre["Ez"], DiffGrad=ctx.adj_d_ez_dy, invDy=solver.inv_dy_h)
    _cuda_backend._accumulate_forward_diff_z(FieldGrad=pre["Ey"], DiffGrad=ctx.adj_d_ey_dz, invDz=solver.inv_dz_h)

    _cuda_backend._reverse_magnetic_cpml_hy(
        AdjHyPrev=pre["Hy"],
        AdjPsiPosPrev=pre["psi_hy_z"],
        AdjPsiNegPrev=pre["psi_hy_x"],
        AdjDPos=ctx.adj_d_ex_dz,
        AdjDNeg=ctx.adj_d_ez_dx,
        AdjHyPost=adj_h_mid["Hy"],
        AdjPsiPosPost=adjoint_state["psi_hy_x"],
        AdjPsiNegPost=adjoint_state["psi_hy_z"],
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        BPos=solver.cpml_b_h_z,
        CPos=solver.cpml_c_h_z,
        InvKappaPos=solver.cpml_inv_kappa_h_z,
        BNeg=solver.cpml_b_h_x,
        CNeg=solver.cpml_c_h_x,
        InvKappaNeg=solver.cpml_inv_kappa_h_x,
    )
    _cuda_backend._accumulate_forward_diff_z(FieldGrad=pre["Ex"], DiffGrad=ctx.adj_d_ex_dz, invDz=solver.inv_dz_h)
    _cuda_backend._accumulate_forward_diff_x(FieldGrad=pre["Ez"], DiffGrad=ctx.adj_d_ez_dx, invDx=solver.inv_dx_h)

    _cuda_backend._reverse_magnetic_cpml_hz(
        AdjHzPrev=pre["Hz"],
        AdjPsiPosPrev=pre["psi_hz_x"],
        AdjPsiNegPrev=pre["psi_hz_y"],
        AdjDPos=ctx.adj_d_ey_dx,
        AdjDNeg=ctx.adj_d_ex_dy,
        AdjHzPost=adj_h_mid["Hz"],
        AdjPsiPosPost=adjoint_state["psi_hz_x"],
        AdjPsiNegPost=adjoint_state["psi_hz_y"],
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        BPos=solver.cpml_b_h_x,
        CPos=solver.cpml_c_h_x,
        InvKappaPos=solver.cpml_inv_kappa_h_x,
        BNeg=solver.cpml_b_h_y,
        CNeg=solver.cpml_c_h_y,
        InvKappaNeg=solver.cpml_inv_kappa_h_y,
    )
    _cuda_backend._accumulate_forward_diff_x(FieldGrad=pre["Ey"], DiffGrad=ctx.adj_d_ey_dx, invDx=solver.inv_dx_h)
    _cuda_backend._accumulate_forward_diff_y(FieldGrad=pre["Ex"], DiffGrad=ctx.adj_d_ex_dy, invDy=solver.inv_dy_h)

    step_result = _adjoint._ReverseStepResult(
        pre_step_adjoint=pre,
        grad_eps_ex=ctx.grad_eps_ex,
        grad_eps_ey=ctx.grad_eps_ey,
        grad_eps_ez=ctx.grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_CPML],
        magnetic_output_adjoint=adj_h_mid,
    )
    # The native runner owns the full reverse-step contract, including the
    # analytic source-term eps-gradient accumulation the reference path applies.
    return _adjoint._accumulate_source_term_gradients(
        step_result,
        solver=solver,
        adjoint_state=adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def register_native_reverse_backends() -> None:
    """Register every available native CUDA reverse-step runner."""
    register_native_reverse_backend(
        _ReverseBackend.PYTHON_STANDARD,
        _reverse_step_standard_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.PYTHON_CPML,
        _reverse_step_cpml_native,
        qualifier=_cuda_scene_native_qualifies,
    )
