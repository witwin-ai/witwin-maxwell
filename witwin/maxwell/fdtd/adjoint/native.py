"""Native CUDA reverse-step runners for the FDTD adjoint.

These runners orchestrate the fused reverse-math CUDA kernels (registered in
``cuda/backend.py`` and parity-tested against the frozen Torch reference) into a
full per-step reverse pass that is bit-for-bit equivalent to the analytic Torch
reference backend it mirrors. Python only sequences the kernel launches; every
per-cell reverse computation (electric-adjoint -> mid-H adjoint, magnetic-adjoint
-> pre-step E adjoint + eps gradient, and the magnetic decay pullback) runs
inside the compiled kernels.

The mid-step magnetic field is still reconstructed with the shared Torch replay
helper (``_forward_magnetic_fields`` / ``_forward_magnetic_fields_complex`` for the
complex split-field Bloch runner); nativizing that replay is tracked separately
under the native-replay work. The reverse *math* itself is fully native here.
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


def _dispersive_native_qualifies(solver, forward_state) -> bool:
    """Gate the native dispersive runner to the electric-only ADE configuration.

    The dispersive runner nativizes the electric Debye/Drude/Lorentz current VJP
    and the 1/eps correction VJP on top of the native standard/CPML base reverse.
    The magnetic ADE (mu-pole) correction/state VJP is not nativized here, so a
    scene that also enables magnetic dispersion falls back to the analytic Torch
    reference in auto mode (and a forced ``native`` override raises).
    """
    if getattr(solver, "magnetic_dispersive_enabled", False):
        return False
    return _cuda_scene_native_qualifies(solver, forward_state)


def _reverse_step_standard_native_core(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
):
    """Fused native standard reverse math, without the source-term eps gradient.

    Returns the ``_ReverseStepResult`` the analytic standard reference produces
    *before* ``_accumulate_source_term_gradients`` runs. The public runner adds
    that accumulation; the dispersive runner reuses this core as its base reverse
    and defers the single source-term accumulation to the end of the full step.
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
    return step_result


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
    from . import core as _adjoint

    step_result = _reverse_step_standard_native_core(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
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


def _reverse_step_cpml_native_core(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
):
    """Fused native CPML reverse math, without the source-term eps gradient.

    Returns the ``_ReverseStepResult`` the analytic CPML reference produces
    *before* ``_accumulate_source_term_gradients`` runs. The public runner adds
    that accumulation; the dispersive runner reuses this core as its CPML base
    reverse and defers the single source-term accumulation to the end of the step.

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
    return step_result


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
    """Fully native reverse step for the CPML (absorbing-boundary) configuration."""
    from . import core as _adjoint

    step_result = _reverse_step_cpml_native_core(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
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


def _reverse_step_bloch_native(
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
    """Fully native reverse step for the complex split-field Bloch configuration.

    Mirrors ``reverse_step_bloch_python_reference`` exactly, launching the fused
    complex Bloch reverse kernels in a fixed order. Every per-cell reverse
    computation (electric-adjoint -> mid-H adjoint with the boundary wrap phase,
    magnetic-adjoint -> pre-step E adjoint + eps gradient, and the real/imag
    magnetic decay pullback) runs inside the compiled kernels; Python only
    sequences the launches and carries the three per-axis Bloch phase pairs.

    The fused kernels *assign* (not accumulate) into their outputs, so the launch
    order is load-bearing: the electric->H kernels must fully populate both real
    and imag mid-H adjoints before the magnetic->E kernels (which read them for the
    forward-diff fold) and before the decay pullback (which reads them for the
    pre-step H adjoint).
    """
    import torch

    from . import core as _adjoint
    from .reverse_common import dynamic_electric_curls

    # Mid-step complex H the forward electric update consumed (shared Torch replay).
    magnetic_fields = _adjoint._forward_magnetic_fields_complex(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )

    # Dynamic electric curl coefficients (cast base curl by the eps leaf).
    ex_curl, ey_curl, ez_curl = dynamic_electric_curls(
        solver, eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez
    )

    cos = solver.boundary_phase_cos
    sin = solver.boundary_phase_sin
    phase_cos_x, phase_sin_x = float(cos[0]), float(sin[0])
    phase_cos_y, phase_sin_y = float(cos[1]), float(sin[1])
    phase_cos_z, phase_sin_z = float(cos[2]), float(sin[2])

    # Phase 1: electric adjoint -> mid-H adjoint (complex), with the boundary wrap
    # phase on the transposed backward differences.
    adj_hx_mid_r = torch.empty_like(forward_state["Hx"])
    adj_hx_mid_i = torch.empty_like(forward_state["Hx"])
    adj_hy_mid_r = torch.empty_like(forward_state["Hy"])
    adj_hy_mid_i = torch.empty_like(forward_state["Hy"])
    adj_hz_mid_r = torch.empty_like(forward_state["Hz"])
    adj_hz_mid_i = torch.empty_like(forward_state["Hz"])
    _cuda_backend._reverse_electric_hx_bloch(
        AdjHxMidReal=adj_hx_mid_r,
        AdjHxMidImag=adj_hx_mid_i,
        AdjHxPostReal=adjoint_state["Hx"],
        AdjHxPostImag=adjoint_state["Hx_imag"],
        AdjEyPostReal=adjoint_state["Ey"],
        AdjEyPostImag=adjoint_state["Ey_imag"],
        AdjEzPostReal=adjoint_state["Ez"],
        AdjEzPostImag=adjoint_state["Ez_imag"],
        EyCurl=ey_curl,
        EzCurl=ez_curl,
        phaseCosY=phase_cos_y,
        phaseSinY=phase_sin_y,
        phaseCosZ=phase_cos_z,
        phaseSinZ=phase_sin_z,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
    )
    _cuda_backend._reverse_electric_hy_bloch(
        AdjHyMidReal=adj_hy_mid_r,
        AdjHyMidImag=adj_hy_mid_i,
        AdjHyPostReal=adjoint_state["Hy"],
        AdjHyPostImag=adjoint_state["Hy_imag"],
        AdjExPostReal=adjoint_state["Ex"],
        AdjExPostImag=adjoint_state["Ex_imag"],
        AdjEzPostReal=adjoint_state["Ez"],
        AdjEzPostImag=adjoint_state["Ez_imag"],
        ExCurl=ex_curl,
        EzCurl=ez_curl,
        phaseCosX=phase_cos_x,
        phaseSinX=phase_sin_x,
        phaseCosZ=phase_cos_z,
        phaseSinZ=phase_sin_z,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
    )
    _cuda_backend._reverse_electric_hz_bloch(
        AdjHzMidReal=adj_hz_mid_r,
        AdjHzMidImag=adj_hz_mid_i,
        AdjHzPostReal=adjoint_state["Hz"],
        AdjHzPostImag=adjoint_state["Hz_imag"],
        AdjExPostReal=adjoint_state["Ex"],
        AdjExPostImag=adjoint_state["Ex_imag"],
        AdjEyPostReal=adjoint_state["Ey"],
        AdjEyPostImag=adjoint_state["Ey_imag"],
        ExCurl=ex_curl,
        EyCurl=ey_curl,
        phaseCosX=phase_cos_x,
        phaseSinX=phase_sin_x,
        phaseCosY=phase_cos_y,
        phaseSinY=phase_sin_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
    )

    # Phase 2: magnetic adjoint -> pre-step E adjoint (complex) + eps gradient. Each
    # kernel reconstructs its own curl(H) from the mid-H fields (for the eps
    # gradient) and assigns the complete pre-step E adjoint (electric decay pullback
    # + magnetic forward-diff fold of the mid-H adjoints).
    adj_ex_prev_r = torch.empty_like(forward_state["Ex"])
    adj_ex_prev_i = torch.empty_like(forward_state["Ex"])
    adj_ey_prev_r = torch.empty_like(forward_state["Ey"])
    adj_ey_prev_i = torch.empty_like(forward_state["Ey"])
    adj_ez_prev_r = torch.empty_like(forward_state["Ez"])
    adj_ez_prev_i = torch.empty_like(forward_state["Ez"])
    grad_eps_ex = torch.empty_like(eps_ex)
    grad_eps_ey = torch.empty_like(eps_ey)
    grad_eps_ez = torch.empty_like(eps_ez)
    _cuda_backend._reverse_magnetic_ex_bloch(
        AdjExPrevReal=adj_ex_prev_r,
        AdjExPrevImag=adj_ex_prev_i,
        GradEpsEx=grad_eps_ex,
        AdjExPostReal=adjoint_state["Ex"],
        AdjExPostImag=adjoint_state["Ex_imag"],
        AdjHyMidReal=adj_hy_mid_r,
        AdjHyMidImag=adj_hy_mid_i,
        AdjHzMidReal=adj_hz_mid_r,
        AdjHzMidImag=adj_hz_mid_i,
        ExDecay=solver.cex_decay,
        ExCurl=ex_curl,
        EpsEx=eps_ex,
        HyMidReal=magnetic_fields["Hy"],
        HyMidImag=magnetic_fields["Hy_imag"],
        HzMidReal=magnetic_fields["Hz"],
        HzMidImag=magnetic_fields["Hz_imag"],
        HyCurl=solver.chy_curl,
        HzCurl=solver.chz_curl,
        phaseCosY=phase_cos_y,
        phaseSinY=phase_sin_y,
        phaseCosZ=phase_cos_z,
        phaseSinZ=phase_sin_z,
        invDyE=solver.inv_dy_e,
        invDzE=solver.inv_dz_e,
        invDyH=solver.inv_dy_h,
        invDzH=solver.inv_dz_h,
    )
    _cuda_backend._reverse_magnetic_ey_bloch(
        AdjEyPrevReal=adj_ey_prev_r,
        AdjEyPrevImag=adj_ey_prev_i,
        GradEpsEy=grad_eps_ey,
        AdjEyPostReal=adjoint_state["Ey"],
        AdjEyPostImag=adjoint_state["Ey_imag"],
        AdjHxMidReal=adj_hx_mid_r,
        AdjHxMidImag=adj_hx_mid_i,
        AdjHzMidReal=adj_hz_mid_r,
        AdjHzMidImag=adj_hz_mid_i,
        EyDecay=solver.cey_decay,
        EyCurl=ey_curl,
        EpsEy=eps_ey,
        HxMidReal=magnetic_fields["Hx"],
        HxMidImag=magnetic_fields["Hx_imag"],
        HzMidReal=magnetic_fields["Hz"],
        HzMidImag=magnetic_fields["Hz_imag"],
        HxCurl=solver.chx_curl,
        HzCurl=solver.chz_curl,
        phaseCosX=phase_cos_x,
        phaseSinX=phase_sin_x,
        phaseCosZ=phase_cos_z,
        phaseSinZ=phase_sin_z,
        invDxE=solver.inv_dx_e,
        invDzE=solver.inv_dz_e,
        invDxH=solver.inv_dx_h,
        invDzH=solver.inv_dz_h,
    )
    _cuda_backend._reverse_magnetic_ez_bloch(
        AdjEzPrevReal=adj_ez_prev_r,
        AdjEzPrevImag=adj_ez_prev_i,
        GradEpsEz=grad_eps_ez,
        AdjEzPostReal=adjoint_state["Ez"],
        AdjEzPostImag=adjoint_state["Ez_imag"],
        AdjHxMidReal=adj_hx_mid_r,
        AdjHxMidImag=adj_hx_mid_i,
        AdjHyMidReal=adj_hy_mid_r,
        AdjHyMidImag=adj_hy_mid_i,
        EzDecay=solver.cez_decay,
        EzCurl=ez_curl,
        EpsEz=eps_ez,
        HxMidReal=magnetic_fields["Hx"],
        HxMidImag=magnetic_fields["Hx_imag"],
        HyMidReal=magnetic_fields["Hy"],
        HyMidImag=magnetic_fields["Hy_imag"],
        HxCurl=solver.chx_curl,
        HyCurl=solver.chy_curl,
        phaseCosX=phase_cos_x,
        phaseSinX=phase_sin_x,
        phaseCosY=phase_cos_y,
        phaseSinY=phase_sin_y,
        invDxE=solver.inv_dx_e,
        invDyE=solver.inv_dy_e,
        invDxH=solver.inv_dx_h,
        invDyH=solver.inv_dy_h,
    )

    # Phase 3: magnetic decay pullback -> pre-step H adjoint (complex). The decay is
    # real, so the standard real decay kernel runs once per split-field half.
    adj_hx_prev_r = torch.empty_like(forward_state["Hx"])
    adj_hx_prev_i = torch.empty_like(forward_state["Hx"])
    adj_hy_prev_r = torch.empty_like(forward_state["Hy"])
    adj_hy_prev_i = torch.empty_like(forward_state["Hy"])
    adj_hz_prev_r = torch.empty_like(forward_state["Hz"])
    adj_hz_prev_i = torch.empty_like(forward_state["Hz"])
    _cuda_backend._reverse_magnetic_hx_decay(AdjHxPrev=adj_hx_prev_r, AdjHxMid=adj_hx_mid_r, HxDecay=solver.chx_decay)
    _cuda_backend._reverse_magnetic_hx_decay(AdjHxPrev=adj_hx_prev_i, AdjHxMid=adj_hx_mid_i, HxDecay=solver.chx_decay)
    _cuda_backend._reverse_magnetic_hy_decay(AdjHyPrev=adj_hy_prev_r, AdjHyMid=adj_hy_mid_r, HyDecay=solver.chy_decay)
    _cuda_backend._reverse_magnetic_hy_decay(AdjHyPrev=adj_hy_prev_i, AdjHyMid=adj_hy_mid_i, HyDecay=solver.chy_decay)
    _cuda_backend._reverse_magnetic_hz_decay(AdjHzPrev=adj_hz_prev_r, AdjHzMid=adj_hz_mid_r, HzDecay=solver.chz_decay)
    _cuda_backend._reverse_magnetic_hz_decay(AdjHzPrev=adj_hz_prev_i, AdjHzMid=adj_hz_mid_i, HzDecay=solver.chz_decay)

    pre_by_name = {
        "Ex": adj_ex_prev_r,
        "Ey": adj_ey_prev_r,
        "Ez": adj_ez_prev_r,
        "Hx": adj_hx_prev_r,
        "Hy": adj_hy_prev_r,
        "Hz": adj_hz_prev_r,
        "Ex_imag": adj_ex_prev_i,
        "Ey_imag": adj_ey_prev_i,
        "Ez_imag": adj_ez_prev_i,
        "Hx_imag": adj_hx_prev_i,
        "Hy_imag": adj_hy_prev_i,
        "Hz_imag": adj_hz_prev_i,
    }
    step_result = _adjoint._ReverseStepResult(
        pre_step_adjoint={name: pre_by_name[name] for name in forward_state},
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_BLOCH],
    )
    # The native runner owns the full reverse-step contract, including the analytic
    # source-term eps-gradient accumulation the reference path applies via ``finish``.
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


def _reverse_step_dispersive_native(
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
    """Fully native reverse step for the electric-dispersive (ADE) configuration.

    Mirrors ``reverse_step_dispersive_python_reference`` for the electric-only
    dispersive case. The reverse *math* runs entirely in native kernels:

    1. The mid-step post-update dispersive currents are reconstructed with the
       shared Torch ADE replay (``_advance_dispersive_state``); this is the
       once-per-step replay tracked under the native-replay work, same bar as the
       standard/CPML runners' mid-H replay.
    2. The native standard or CPML *core* reverse produces the pre-step E/H (and
       psi) adjoint plus the base eps gradient, consuming the post-step field
       adjoints directly (the reference's ``electric_source_adjoint`` is an
       identity clone of those, so no separate copy is needed).
    3. Per electric pole the fused ``reverseDispersiveCorrection3D`` kernel folds
       the 1/eps coupling: it assigns the corrected post-step current adjoint and
       accumulates the ``dt * J * adj_E / eps^2`` eps gradient onto the base eps
       gradient in place.
    4. The fused ``reverseDebye/Drude/LorentzCurrent3D`` kernels pull the corrected
       current (and raw polarization) adjoint back through the ADE update, adding
       the field-drive term into the pre-step E adjoint and assigning the pre-step
       polarization/current adjoints.

    Launch order is load-bearing: the base reverse assigns the pre-step E adjoint
    and base eps gradient first; the correction kernel then accumulates the eps
    gradient and produces the corrected current adjoint the ADE-state kernels read;
    the ADE-state kernels accumulate the field-drive term on top of the base
    pre-step E adjoint. The single source-term eps-gradient accumulation runs once
    at the end, matching the reference ``finish`` step.
    """
    import torch

    from . import core as _adjoint

    # Step 1: replay the post-update electric dispersive currents (Torch).
    dispersive_state = _adjoint._advance_dispersive_state(solver, forward_state)

    # Step 2: native base reverse (standard or CPML core), no source-term grads.
    if getattr(solver, "uses_cpml", False):
        base_result = _reverse_step_cpml_native_core(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms,
        )
    else:
        base_result = _reverse_step_standard_native_core(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms,
        )

    pre_step_adjoint = dict(base_result.pre_step_adjoint)
    for name in _adjoint.checkpoint_schema(solver).dispersive_state_names:
        if name not in pre_step_adjoint:
            pre_step_adjoint[name] = torch.zeros_like(forward_state[name])

    grad_eps_by_field = {
        "Ex": base_result.grad_eps_ex,
        "Ey": base_result.grad_eps_ey,
        "Ez": base_result.grad_eps_ez,
    }
    eps_by_field = {"Ex": eps_ex, "Ey": eps_ey, "Ez": eps_ez}
    dt = float(solver.dt)

    # Steps 3 + 4: per pole, correction VJP then ADE-state VJP.
    for component_name, model_name, index, _tensor_names, entry in _adjoint.iter_dispersive_state_specs(solver) or ():
        current_name = _adjoint.dispersive_state_name(component_name, model_name, index, "current")
        adj_current_corrected = torch.empty_like(forward_state[current_name])
        _cuda_backend._reverse_dispersive_correction(
            AdjCurrentCorrected=adj_current_corrected,
            GradEps=grad_eps_by_field[component_name],
            AdjCurrentPost=adjoint_state[current_name],
            AdjElectricPost=adjoint_state[component_name],
            Current=dispersive_state[current_name],
            Eps=eps_by_field[component_name],
            dt=dt,
        )
        if model_name == "debye":
            polarization_name = _adjoint.dispersive_state_name(component_name, model_name, index, "polarization")
            _cuda_backend._reverse_debye_current(
                AdjElectricPrev=pre_step_adjoint[component_name],
                AdjPolarizationPrev=pre_step_adjoint[polarization_name],
                AdjPolarizationPost=adjoint_state[polarization_name],
                AdjCurrentPost=adj_current_corrected,
                DebyeDrive=entry["drive"],
                decay=float(entry["decay"]),
                dt=dt,
            )
        elif model_name == "drude":
            _cuda_backend._reverse_drude_current(
                AdjElectricPrev=pre_step_adjoint[component_name],
                AdjCurrentPrev=pre_step_adjoint[current_name],
                AdjCurrentPost=adj_current_corrected,
                DrudeDrive=entry["drive"],
                decay=float(entry["decay"]),
            )
        else:
            polarization_name = _adjoint.dispersive_state_name(component_name, model_name, index, "polarization")
            _cuda_backend._reverse_lorentz_current(
                AdjElectricPrev=pre_step_adjoint[component_name],
                AdjPolarizationPrev=pre_step_adjoint[polarization_name],
                AdjCurrentPrev=pre_step_adjoint[current_name],
                AdjPolarizationPost=adjoint_state[polarization_name],
                AdjCurrentPost=adj_current_corrected,
                LorentzDrive=entry["drive"],
                decay=float(entry["decay"]),
                restoring=float(entry["restoring"]),
                dt=dt,
            )

    step_result = _adjoint._ReverseStepResult(
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=grad_eps_by_field["Ex"],
        grad_eps_ey=grad_eps_by_field["Ey"],
        grad_eps_ez=grad_eps_by_field["Ez"],
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.PYTHON_DISPERSIVE],
        source_adjoint_state=dict(adjoint_state),
    )
    # The reference dispersive path discards the base magnetic mid-step adjoint
    # (it returns no ``magnetic_output_adjoint``), so the single source-term
    # accumulation sees the same empty magnetic seed here.
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


def _accumulate_tfsf_sample_adjoint_native(adj_aux, terms, adjoint_fields, *, origin, ds):
    """Transpose the per-step TFSF incident injection into ``adj_aux`` via kernels.

    Iterates the injection terms exactly like the forward per-term apply, slicing
    the adjoint of the field the incident patch was written into and launching the
    matching native sample-adjoint kernel (scalar / line / interpolated). The
    ``adj_field_patch`` is materialized contiguous so the kernels can index it
    linearly against the term coefficient patch.
    """
    for term in terms:
        offset_i, offset_j, offset_k = (int(offset) for offset in term["offsets"])
        coeff_patch = term["coeff_patch"]
        shape_i, shape_j, shape_k = (int(length) for length in coeff_patch.shape)
        adj_field_patch = adjoint_fields[term["field_name"]][
            offset_i : offset_i + shape_i,
            offset_j : offset_j + shape_j,
            offset_k : offset_k + shape_k,
        ].contiguous()
        component_scale = float(term["component_scale"])
        if "sample_positions" in term:
            _cuda_backend._accumulate_tfsf_interpolated_sample_adjoint(
                AdjAuxField=adj_aux,
                AdjFieldPatch=adj_field_patch,
                CoeffPatch=coeff_patch,
                SamplePositions=term["sample_positions"],
                origin=origin,
                ds=ds,
                componentScale=component_scale,
            )
            continue
        if term["scalar_sample_index"] is not None:
            _cuda_backend._accumulate_tfsf_scalar_sample_adjoint(
                AdjAuxField=adj_aux,
                AdjFieldPatch=adj_field_patch,
                CoeffPatch=coeff_patch,
                sampleIndex=int(term["scalar_sample_index"]),
                componentScale=component_scale,
            )
            continue
        from ..excitation.tfsf_specs import reference_sample_axis_code

        _cuda_backend._accumulate_tfsf_line_sample_adjoint(
            AdjAuxField=adj_aux,
            AdjFieldPatch=adj_field_patch,
            CoeffPatch=coeff_patch,
            SampleIndices=term["sample_indices"],
            sampleAxisCode=int(reference_sample_axis_code(term)),
            componentScale=component_scale,
        )


def _reverse_tfsf_auxiliary_state_native(solver, forward_state, adjoint_state, *, magnetic_output_adjoint):
    """Native CUDA transpose of the per-step TFSF auxiliary update.

    Mirrors ``_reverse_tfsf_auxiliary_state_python_reference`` one-for-one, driving
    the fused 1D auxiliary reverse kernels and the sample-adjoint kernels in the
    load-bearing order: the electric-side incident injection and the electric
    auxiliary advance both accumulate into the advanced-magnetic adjoint before the
    magnetic auxiliary advance reads it; the magnetic incident injection reads the
    base reverse's full mid-step H adjoint. Returns the aux pre-step adjoints, or an
    empty dict when the active provider carries no 1D auxiliary state.
    """
    from . import core as _adjoint

    auxiliary_state = _adjoint._extract_tfsf_auxiliary_state(forward_state)
    if auxiliary_state is None:
        return {}
    tfsf_state = getattr(solver, "_tfsf_state", None)
    if tfsf_state is None:
        return {}
    auxiliary_grid = tfsf_state.get("auxiliary_grid")
    if auxiliary_grid is None:
        return {}

    import torch

    adj_electric_prev = torch.zeros_like(auxiliary_state["electric"])
    adj_magnetic_prev = torch.empty_like(auxiliary_state["magnetic"])
    adj_magnetic_after = adjoint_state["tfsf_aux_magnetic"].detach().clone()

    ds = float(auxiliary_grid.ds)
    origin_electric = float(auxiliary_grid.s_min)
    origin_magnetic = float(auxiliary_grid.s_min + 0.5 * ds)

    # (1) Electric incident injection samples the advanced magnetic grid.
    _accumulate_tfsf_sample_adjoint_native(
        adj_magnetic_after,
        tfsf_state.get("electric_terms", ()),
        adjoint_state,
        origin=origin_magnetic,
        ds=ds,
    )
    # (2) Electric auxiliary advance reverse.
    _cuda_backend._reverse_tfsf_auxiliary_electric(
        AdjElectricPrev=adj_electric_prev,
        AdjMagneticAfter=adj_magnetic_after,
        AdjElectricPost=adjoint_state["tfsf_aux_electric"],
        ElectricDecay=auxiliary_grid.electric_decay,
        ElectricCurl=auxiliary_grid.electric_curl,
        sourceIndex=int(auxiliary_grid.source_index),
    )
    # (3) Magnetic auxiliary advance reverse.
    _cuda_backend._reverse_tfsf_auxiliary_magnetic(
        AdjElectricPrev=adj_electric_prev,
        AdjMagneticPrev=adj_magnetic_prev,
        AdjMagneticAfter=adj_magnetic_after,
        MagneticDecay=auxiliary_grid.magnetic_decay,
        MagneticCurl=auxiliary_grid.magnetic_curl,
    )
    # (4) Magnetic incident injection samples the pre-step electric grid; it lands
    # on the mid-step H, so read the full mid-H adjoint from the base reverse.
    _accumulate_tfsf_sample_adjoint_native(
        adj_electric_prev,
        tfsf_state.get("magnetic_terms", ()),
        magnetic_output_adjoint,
        origin=origin_electric,
        ds=ds,
    )
    return {
        "tfsf_aux_electric": adj_electric_prev,
        "tfsf_aux_magnetic": adj_magnetic_prev,
    }


def _reverse_step_tfsf_native(
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
    """Fully native reverse step for the TFSF (total-field/scattered-field) case.

    Mirrors ``reference.reverse_step_tfsf``: the native standard or CPML *core*
    reverse produces the pre-step field/psi adjoint (its mid-step H replay consumes
    the materialized magnetic incident source terms, so the injected incident field
    is folded into the mid-H exactly as the forward step did), then the native TFSF
    auxiliary reverse kernels add the 1D auxiliary electric/magnetic pre-step
    adjoints. The TFSF incident terms are literal patches, so the reference path
    runs no source-term eps-gradient accumulation and neither does this runner.
    """
    from . import core as _adjoint

    tfsf_source_terms = _adjoint._tfsf_magnetic_source_terms(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )

    if getattr(solver, "uses_cpml", False):
        base_result = _reverse_step_cpml_native_core(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=tfsf_source_terms,
        )
    else:
        base_result = _reverse_step_standard_native_core(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=tfsf_source_terms,
        )

    pre_step_adjoint = dict(base_result.pre_step_adjoint)
    auxiliary_grads = _reverse_tfsf_auxiliary_state_native(
        solver,
        forward_state,
        adjoint_state,
        magnetic_output_adjoint=base_result.magnetic_output_adjoint,
    )
    pre_step_adjoint.update(auxiliary_grads)

    return _adjoint._ReverseStepResult(
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=base_result.grad_eps_ex,
        grad_eps_ey=base_result.grad_eps_ey,
        grad_eps_ez=base_result.grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.TFSF],
        magnetic_output_adjoint=base_result.magnetic_output_adjoint,
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
    register_native_reverse_backend(
        _ReverseBackend.PYTHON_BLOCH,
        _reverse_step_bloch_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.PYTHON_DISPERSIVE,
        _reverse_step_dispersive_native,
        qualifier=_dispersive_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.TFSF,
        _reverse_step_tfsf_native,
        qualifier=_cuda_scene_native_qualifies,
    )
