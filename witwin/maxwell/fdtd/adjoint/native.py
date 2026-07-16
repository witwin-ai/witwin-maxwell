"""Native CUDA reverse-step runners for the FDTD adjoint.

These runners orchestrate the fused reverse-math CUDA kernels registered in
``cuda/backend.py`` into a full per-step reverse pass. Python only sequences the
kernel launches; every
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
    if not _scene_is_cuda(forward_state) or not _cuda_backend.is_available():
        return False
    try:
        extension = _cuda_backend.get_compiled_extension()
    except (ImportError, OSError, RuntimeError):
        return False
    solver._fdtd_cuda_extension = extension
    return True


def _dispersive_native_qualifies(solver, forward_state) -> bool:
    """Gate the native ADE runner to CUDA scenes with the compiled extension."""
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
    magnetic_fields=None,
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
    if magnetic_fields is None:
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
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.STANDARD],
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

    Mirrors ``reverse_step_standard_native`` exactly, launching the
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
    magnetic_fields=None,
):
    """Fused native CPML reverse math, without the source-term eps gradient.

    Returns the ``_ReverseStepResult`` the analytic CPML reference produces
    *before* ``_accumulate_source_term_gradients`` runs. The public runner adds
    that accumulation; the dispersive runner reuses this core as its CPML base
    reverse and defers the single source-term accumulation to the end of the step.

    Mirrors ``reverse_step_cpml_native`` exactly, launching the fused
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
    if magnetic_fields is None:
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
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.CPML],
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


def _reverse_step_conductive_native(
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
    """Fully native reverse step for a static-conductive (sigma_e) CPML medium.

    Mirrors :func:`reverse_step_conductive_cpml_native`. The magnetic
    reverse, the curl(H)/curl(E) difference folds, and the pre-step field/psi
    accumulation are the linear CPML machinery unchanged; the only conduction
    specialization is the electric reverse kernel, which recomputes the eps
    sensitivity of the semi-implicit ``decay``/``curl`` pair analytically instead
    of applying the linear rule. Launch order is load-bearing: the conductive
    electric kernels *assign* the pre-step E/psi + eps gradient and the
    difference folds *accumulate* into the mid-step H adjoint, so all three
    electric kernels (and their folds) must complete before any magnetic kernel
    reads the mid-step H adjoint.
    """
    from . import core as _adjoint
    from .reverse_common import allocate_cpml_reverse_context

    magnetic_fields = _adjoint._forward_magnetic_fields(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
    hx_mid = magnetic_fields["Hx"]
    hy_mid = magnetic_fields["Hy"]
    hz_mid = magnetic_fields["Hz"]

    coeffs = _adjoint._conductive_reverse_coefficients(
        solver, eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez
    )
    decay_ex, curl_ex, half_ex = coeffs["Ex"]
    decay_ey, curl_ey, half_ey = coeffs["Ey"]
    decay_ez, curl_ez, half_ez = coeffs["Ez"]
    dt = float(solver.dt)

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

    # Phase 1: conductive electric adjoint -> pre-step E/psi_e adjoint + eps
    # gradient, folding each curl(H)-derivative adjoint into the mid-step H.
    _cuda_backend._reverse_electric_cpml_conductive_ex(
        AdjExPrev=pre["Ex"],
        GradEpsEx=ctx.grad_eps_ex,
        AdjPsiPosPrev=pre["psi_ex_y"],
        AdjPsiNegPrev=pre["psi_ex_z"],
        AdjDPos=ctx.adj_d_hz_dy,
        AdjDNeg=ctx.adj_d_hy_dz,
        AdjExPost=adjoint_state["Ex"],
        AdjPsiPosPost=adjoint_state["psi_ex_y"],
        AdjPsiNegPost=adjoint_state["psi_ex_z"],
        ExDecay=decay_ex,
        ExCurl=curl_ex,
        ExHalf=half_ex,
        ExPrev=forward_state["Ex"],
        EpsEx=eps_ex,
        Dt=dt,
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

    _cuda_backend._reverse_electric_cpml_conductive_ey(
        AdjEyPrev=pre["Ey"],
        GradEpsEy=ctx.grad_eps_ey,
        AdjPsiPosPrev=pre["psi_ey_z"],
        AdjPsiNegPrev=pre["psi_ey_x"],
        AdjDPos=ctx.adj_d_hx_dz,
        AdjDNeg=ctx.adj_d_hz_dx,
        AdjEyPost=adjoint_state["Ey"],
        AdjPsiPosPost=adjoint_state["psi_ey_x"],
        AdjPsiNegPost=adjoint_state["psi_ey_z"],
        EyDecay=decay_ey,
        EyCurl=curl_ey,
        EyHalf=half_ey,
        EyPrev=forward_state["Ey"],
        EpsEy=eps_ey,
        Dt=dt,
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

    _cuda_backend._reverse_electric_cpml_conductive_ez(
        AdjEzPrev=pre["Ez"],
        GradEpsEz=ctx.grad_eps_ez,
        AdjPsiPosPrev=pre["psi_ez_x"],
        AdjPsiNegPrev=pre["psi_ez_y"],
        AdjDPos=ctx.adj_d_hy_dx,
        AdjDNeg=ctx.adj_d_hx_dy,
        AdjEzPost=adjoint_state["Ez"],
        AdjPsiPosPost=adjoint_state["psi_ez_x"],
        AdjPsiNegPost=adjoint_state["psi_ez_y"],
        EzDecay=decay_ez,
        EzCurl=curl_ez,
        EzHalf=half_ez,
        EzPrev=forward_state["Ez"],
        EpsEz=eps_ez,
        Dt=dt,
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
    # folding each curl(E)-derivative adjoint into the pre-step E adjoint. This is
    # the linear CPML magnetic reverse unchanged (conduction is electric-only).
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
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.CONDUCTIVE],
        magnetic_output_adjoint=adj_h_mid,
    )
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


def _reverse_cpml_magnetic_phase_native(solver, ctx, adjoint_state, *, suffix=""):
    """Run the shared linear magnetic CPML pullback after an electric variant."""
    pre = ctx.pre_step_adjoint
    adj_h_mid = ctx.magnetic_output_adjoint
    for component, pos_name, neg_name, d_pos, d_neg, curl, b_pos, c_pos, k_pos, b_neg, c_neg, k_neg in (
        ("Hx", "psi_hx_y", "psi_hx_z", ctx.adj_d_ez_dy, ctx.adj_d_ey_dz, solver.chx_curl,
         solver.cpml_b_h_y, solver.cpml_c_h_y, solver.cpml_inv_kappa_h_y,
         solver.cpml_b_h_z, solver.cpml_c_h_z, solver.cpml_inv_kappa_h_z),
        ("Hy", "psi_hy_z", "psi_hy_x", ctx.adj_d_ex_dz, ctx.adj_d_ez_dx, solver.chy_curl,
         solver.cpml_b_h_z, solver.cpml_c_h_z, solver.cpml_inv_kappa_h_z,
         solver.cpml_b_h_x, solver.cpml_c_h_x, solver.cpml_inv_kappa_h_x),
        ("Hz", "psi_hz_x", "psi_hz_y", ctx.adj_d_ey_dx, ctx.adj_d_ex_dy, solver.chz_curl,
         solver.cpml_b_h_x, solver.cpml_c_h_x, solver.cpml_inv_kappa_h_x,
         solver.cpml_b_h_y, solver.cpml_c_h_y, solver.cpml_inv_kappa_h_y),
    ):
        field_key = component + suffix
        pos_key, neg_key = pos_name + suffix, neg_name + suffix
        getattr(_cuda_backend, f"_reverse_magnetic_cpml_{component.lower()}")(
            **{f"Adj{component}Prev": pre[field_key]},
            AdjPsiPosPrev=pre[pos_key], AdjPsiNegPrev=pre[neg_key],
            AdjDPos=d_pos, AdjDNeg=d_neg,
            **{f"Adj{component}Post": adj_h_mid[field_key]},
            AdjPsiPosPost=adjoint_state[pos_key], AdjPsiNegPost=adjoint_state[neg_key],
            **{f"{component}Decay": getattr(solver, f"c{component.lower()}_decay")},
            **{f"{component}Curl": curl},
            BPos=b_pos, CPos=c_pos, InvKappaPos=k_pos,
            BNeg=b_neg, CNeg=c_neg, InvKappaNeg=k_neg,
        )
    _cuda_backend._accumulate_forward_diff_y(FieldGrad=pre["Ez" + suffix], DiffGrad=ctx.adj_d_ez_dy, invDy=solver.inv_dy_h)
    _cuda_backend._accumulate_forward_diff_z(FieldGrad=pre["Ey" + suffix], DiffGrad=ctx.adj_d_ey_dz, invDz=solver.inv_dz_h)
    _cuda_backend._accumulate_forward_diff_z(FieldGrad=pre["Ex" + suffix], DiffGrad=ctx.adj_d_ex_dz, invDz=solver.inv_dz_h)
    _cuda_backend._accumulate_forward_diff_x(FieldGrad=pre["Ez" + suffix], DiffGrad=ctx.adj_d_ez_dx, invDx=solver.inv_dx_h)
    _cuda_backend._accumulate_forward_diff_x(FieldGrad=pre["Ey" + suffix], DiffGrad=ctx.adj_d_ey_dx, invDx=solver.inv_dx_h)
    _cuda_backend._accumulate_forward_diff_y(FieldGrad=pre["Ex" + suffix], DiffGrad=ctx.adj_d_ex_dy, invDy=solver.inv_dy_h)


def _reverse_step_kerr_native(
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
    """Fully native reverse step for an instantaneous Kerr (chi3) CPML medium.

    Mirrors :func:`reverse_step_kerr_cpml_native`. The Kerr forward
    rewrites only the electric ``curl`` coefficient each step from the pre-update
    fields (``curl = (dt / eff) * decay`` with ``eff = eps + eps0 * chi3 * |E|^2``),
    so the reverse math splits into: (1) a native forward collocation of the frozen
    fields to the per-edge ``|E|^2`` (``fsq``); (2) the fused Kerr electric reverse
    kernels, which assign the pre-step E/psi adjoint (decay pullback) plus the
    grad_eps / grad_chi3 coefficient sensitivities and emit the per-edge ``|E|^2``
    cotangent ``g_fsq``, while folding curl(H) into the mid-step H adjoint; (3) the
    linear CPML magnetic reverse (the Kerr term is electric-only); and (4) the
    shared ``collocation_transpose`` scattering ``g_fsq`` back onto the pre-step
    fields. Launch order is load-bearing: the electric kernels *assign* the pre-step
    E adjoint before the magnetic forward-diff folds and the collocation transpose
    *accumulate* onto it, and all three electric kernels (and their curl(H) folds)
    must complete before any magnetic kernel reads the mid-step H adjoint.
    """
    import torch

    from . import core as _adjoint
    from .reverse_common import allocate_cpml_reverse_context

    magnetic_fields = _adjoint._forward_magnetic_fields(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
    hx_mid = magnetic_fields["Hx"]
    hy_mid = magnetic_fields["Hy"]
    hz_mid = magnetic_fields["Hz"]

    eps0 = float(solver.eps0)
    dt = float(solver.dt)

    # Per-edge |E|^2 collocation of the frozen checkpoint fields (native forward).
    fsq_ex = torch.empty_like(eps_ex)
    fsq_ey = torch.empty_like(eps_ey)
    fsq_ez = torch.empty_like(eps_ez)
    _cuda_backend._collocate_field_square(
        FsqEx=fsq_ex,
        FsqEy=fsq_ey,
        FsqEz=fsq_ez,
        Ex=forward_state["Ex"],
        Ey=forward_state["Ey"],
        Ez=forward_state["Ez"],
    )

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

    grad_chi3_ex = torch.empty_like(eps_ex)
    grad_chi3_ey = torch.empty_like(eps_ey)
    grad_chi3_ez = torch.empty_like(eps_ez)
    g_fsq_ex = torch.empty_like(eps_ex)
    g_fsq_ey = torch.empty_like(eps_ey)
    g_fsq_ez = torch.empty_like(eps_ez)

    # Phase 1: Kerr electric adjoint -> pre-step E/psi adjoint + grad_eps/grad_chi3
    # + |E|^2 cotangent, folding each curl(H) derivative into the mid-step H.
    _cuda_backend._reverse_electric_cpml_kerr_ex(
        AdjExPrev=pre["Ex"],
        GradEpsEx=ctx.grad_eps_ex,
        GradChi3Ex=grad_chi3_ex,
        GFsqEx=g_fsq_ex,
        AdjPsiPosPrev=pre["psi_ex_y"],
        AdjPsiNegPrev=pre["psi_ex_z"],
        AdjDPos=ctx.adj_d_hz_dy,
        AdjDNeg=ctx.adj_d_hy_dz,
        AdjExPost=adjoint_state["Ex"],
        AdjPsiPosPost=adjoint_state["psi_ex_y"],
        AdjPsiNegPost=adjoint_state["psi_ex_z"],
        ExDecay=solver.cex_decay,
        EpsEx=eps_ex,
        Chi3Ex=solver.kerr_chi3_Ex,
        FsqEx=fsq_ex,
        Dt=dt,
        Eps0=eps0,
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

    _cuda_backend._reverse_electric_cpml_kerr_ey(
        AdjEyPrev=pre["Ey"],
        GradEpsEy=ctx.grad_eps_ey,
        GradChi3Ey=grad_chi3_ey,
        GFsqEy=g_fsq_ey,
        AdjPsiPosPrev=pre["psi_ey_z"],
        AdjPsiNegPrev=pre["psi_ey_x"],
        AdjDPos=ctx.adj_d_hx_dz,
        AdjDNeg=ctx.adj_d_hz_dx,
        AdjEyPost=adjoint_state["Ey"],
        AdjPsiPosPost=adjoint_state["psi_ey_x"],
        AdjPsiNegPost=adjoint_state["psi_ey_z"],
        EyDecay=solver.cey_decay,
        EpsEy=eps_ey,
        Chi3Ey=solver.kerr_chi3_Ey,
        FsqEy=fsq_ey,
        Dt=dt,
        Eps0=eps0,
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

    _cuda_backend._reverse_electric_cpml_kerr_ez(
        AdjEzPrev=pre["Ez"],
        GradEpsEz=ctx.grad_eps_ez,
        GradChi3Ez=grad_chi3_ez,
        GFsqEz=g_fsq_ez,
        AdjPsiPosPrev=pre["psi_ez_x"],
        AdjPsiNegPrev=pre["psi_ez_y"],
        AdjDPos=ctx.adj_d_hy_dx,
        AdjDNeg=ctx.adj_d_hx_dy,
        AdjEzPost=adjoint_state["Ez"],
        AdjPsiPosPost=adjoint_state["psi_ez_x"],
        AdjPsiNegPost=adjoint_state["psi_ez_y"],
        EzDecay=solver.cez_decay,
        EpsEz=eps_ez,
        Chi3Ez=solver.kerr_chi3_Ez,
        FsqEz=fsq_ez,
        Dt=dt,
        Eps0=eps0,
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

    # Phase 2: linear CPML magnetic reverse (the Kerr term is electric-only).
    _reverse_cpml_magnetic_phase_native(solver, ctx, adjoint_state)

    # Phase 3: scatter the |E|^2 cotangent back onto the pre-step fields (the exact
    # transpose of the collocation, own-axis 2*E plus the 4-point off-axis average).
    # It accumulates onto the pre-step E adjoint the electric kernels already
    # assigned and the magnetic folds added, so it runs last.
    _cuda_backend._collocation_transpose(
        AdjEx=pre["Ex"],
        AdjEy=pre["Ey"],
        AdjEz=pre["Ez"],
        GEx=g_fsq_ex,
        GEy=g_fsq_ey,
        GEz=g_fsq_ez,
        Ex=forward_state["Ex"],
        Ey=forward_state["Ey"],
        Ez=forward_state["Ez"],
    )

    step_result = _adjoint._ReverseStepResult(
        pre_step_adjoint=pre,
        grad_eps_ex=ctx.grad_eps_ex,
        grad_eps_ey=ctx.grad_eps_ey,
        grad_eps_ez=ctx.grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.KERR],
        magnetic_output_adjoint=adj_h_mid,
        grad_chi3_ex=grad_chi3_ex,
        grad_chi3_ey=grad_chi3_ey,
        grad_chi3_ez=grad_chi3_ez,
    )
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


def _reverse_step_nonlinear_native(
    solver, forward_state, adjoint_state, *, time_value, eps_ex, eps_ey, eps_ez,
    resolved_source_terms, profiler,
):
    """Native reverse for chi2/chi3/two-photon absorption on standard or CPML grids."""
    import torch

    from . import core as _adjoint
    from .reverse_common import allocate_cpml_reverse_context

    magnetic_fields = _adjoint._forward_magnetic_fields(
        solver, forward_state, time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
    original_state_names = tuple(forward_state)
    if not getattr(solver, "uses_cpml", False):
        psi_components = {
            "psi_ex_y": "Ex", "psi_ex_z": "Ex",
            "psi_ey_x": "Ey", "psi_ey_z": "Ey",
            "psi_ez_x": "Ez", "psi_ez_y": "Ez",
            "psi_hx_y": "Hx", "psi_hx_z": "Hx",
            "psi_hy_x": "Hy", "psi_hy_z": "Hy",
            "psi_hz_x": "Hz", "psi_hz_y": "Hz",
        }
        forward_state = dict(forward_state)
        adjoint_state = dict(adjoint_state)
        for psi_name, component_name in psi_components.items():
            forward_state[psi_name] = torch.zeros_like(forward_state[component_name])
            adjoint_state[psi_name] = torch.zeros_like(adjoint_state[component_name])
        for stagger in ("e", "h"):
            for axis in ("x", "y", "z"):
                vector = getattr(solver, f"inv_d{axis}_{stagger}")
                setattr(solver, f"cpml_b_{stagger}_{axis}", torch.ones_like(vector))
                setattr(solver, f"cpml_c_{stagger}_{axis}", torch.zeros_like(vector))
                setattr(solver, f"cpml_inv_kappa_{stagger}_{axis}", torch.ones_like(vector))
    dispersive_state = _adjoint._advance_dispersive_state(solver, forward_state)
    fsq = {name: torch.empty_like(eps) for name, eps in (
        ("Ex", eps_ex), ("Ey", eps_ey), ("Ez", eps_ez)
    )}
    _cuda_backend._collocate_field_square(
        FsqEx=fsq["Ex"], FsqEy=fsq["Ey"], FsqEz=fsq["Ez"],
        Ex=forward_state["Ex"], Ey=forward_state["Ey"], Ez=forward_state["Ez"],
    )
    ctx = allocate_cpml_reverse_context(
        solver, forward_state, adjoint_state,
        eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
    )
    pre, adj_h = ctx.pre_step_adjoint, ctx.magnetic_output_adjoint
    grad_chi2 = {name: torch.empty_like(value) for name, value in fsq.items()}
    grad_chi3 = {name: torch.empty_like(value) for name, value in fsq.items()}
    grad_tpa = {name: torch.empty_like(value) for name, value in fsq.items()}
    g_fsq = {name: torch.empty_like(value) for name, value in fsq.items()}
    configs = (
        (0, "Ex", eps_ex, "psi_ex_y", "psi_ex_z", ctx.adj_d_hz_dy, ctx.adj_d_hy_dz,
         magnetic_fields["Hz"], magnetic_fields["Hy"], solver.inv_dy_e, solver.inv_dz_e,
         solver.cpml_b_e_y, solver.cpml_c_e_y, solver.cpml_inv_kappa_e_y,
         solver.cpml_b_e_z, solver.cpml_c_e_z, solver.cpml_inv_kappa_e_z,
         solver.boundary_y_low_code, solver.boundary_y_high_code,
         solver.boundary_z_low_code, solver.boundary_z_high_code),
        (1, "Ey", eps_ey, "psi_ey_z", "psi_ey_x", ctx.adj_d_hx_dz, ctx.adj_d_hz_dx,
         magnetic_fields["Hx"], magnetic_fields["Hz"], solver.inv_dz_e, solver.inv_dx_e,
         solver.cpml_b_e_z, solver.cpml_c_e_z, solver.cpml_inv_kappa_e_z,
         solver.cpml_b_e_x, solver.cpml_c_e_x, solver.cpml_inv_kappa_e_x,
         solver.boundary_x_low_code, solver.boundary_x_high_code,
         solver.boundary_z_low_code, solver.boundary_z_high_code),
        (2, "Ez", eps_ez, "psi_ez_x", "psi_ez_y", ctx.adj_d_hy_dx, ctx.adj_d_hx_dy,
         magnetic_fields["Hy"], magnetic_fields["Hx"], solver.inv_dx_e, solver.inv_dy_e,
         solver.cpml_b_e_x, solver.cpml_c_e_x, solver.cpml_inv_kappa_e_x,
         solver.cpml_b_e_y, solver.cpml_c_e_y, solver.cpml_inv_kappa_e_y,
         solver.boundary_x_low_code, solver.boundary_x_high_code,
         solver.boundary_y_low_code, solver.boundary_y_high_code),
    )
    grad_eps = {"Ex": ctx.grad_eps_ex, "Ey": ctx.grad_eps_ey, "Ez": ctx.grad_eps_ez}
    for (component, name, eps, psi_pos, psi_neg, adj_d_pos, adj_d_neg,
         h_pos, h_neg, inv_pos, inv_neg, b_pos, c_pos, k_pos, b_neg, c_neg, k_neg,
         low_a, high_a, low_b, high_b) in configs:
        # Ey's reference wiring stores the two post-psi cotangents in the
        # historical x/z order; retain it for exact forward/checkpoint parity.
        post_pos, post_neg = (psi_neg, psi_pos) if name == "Ey" else (psi_pos, psi_neg)
        suffix = name[-1]
        _cuda_backend._reverse_electric_cpml_nonlinear(
            component=component, AdjPrev=pre[name], GradEps=grad_eps[name],
            GradChi2=grad_chi2[name], GradChi3=grad_chi3[name], GradTpa=grad_tpa[name],
            GFsq=g_fsq[name], AdjPsiPosPrev=pre[psi_pos], AdjPsiNegPrev=pre[psi_neg],
            AdjDPos=adj_d_pos, AdjDNeg=adj_d_neg, AdjPost=adjoint_state[name],
            AdjPsiPosPost=adjoint_state[post_pos], AdjPsiNegPost=adjoint_state[post_neg],
            EPrev=forward_state[name], ExternalDecay=getattr(solver, f"ce{suffix.lower()}_decay_external"),
            Eps=eps, Chi2=getattr(solver, f"nonlinear_chi2_{name}"),
            Chi3=getattr(solver, f"kerr_chi3_{name}"), Tpa=getattr(solver, f"tpa_sigma_{name}"),
            SigmaStatic=getattr(solver, f"sigma_e_{name}"), Fsq=fsq[name],
            Dt=float(solver.dt), Eps0=float(solver.eps0),
            PsiPos=forward_state[psi_pos], PsiNeg=forward_state[psi_neg],
            BPos=b_pos, CPos=c_pos, InvKappaPos=k_pos,
            BNeg=b_neg, CNeg=c_neg, InvKappaNeg=k_neg,
            HPosMid=h_pos, HNegMid=h_neg, InvPos=inv_pos, InvNeg=inv_neg,
            LowModeA=low_a, HighModeA=high_a, LowModeB=low_b, HighModeB=high_b,
        )
    _cuda_backend._accumulate_backward_diff_y(FieldGrad=adj_h["Hz"], DiffGrad=ctx.adj_d_hz_dy, invDy=solver.inv_dy_e)
    _cuda_backend._accumulate_backward_diff_z(FieldGrad=adj_h["Hy"], DiffGrad=ctx.adj_d_hy_dz, invDz=solver.inv_dz_e)
    _cuda_backend._accumulate_backward_diff_z(FieldGrad=adj_h["Hx"], DiffGrad=ctx.adj_d_hx_dz, invDz=solver.inv_dz_e)
    _cuda_backend._accumulate_backward_diff_x(FieldGrad=adj_h["Hz"], DiffGrad=ctx.adj_d_hz_dx, invDx=solver.inv_dx_e)
    _cuda_backend._accumulate_backward_diff_x(FieldGrad=adj_h["Hy"], DiffGrad=ctx.adj_d_hy_dx, invDx=solver.inv_dx_e)
    _cuda_backend._accumulate_backward_diff_y(FieldGrad=adj_h["Hx"], DiffGrad=ctx.adj_d_hx_dy, invDy=solver.inv_dy_e)
    _reverse_cpml_magnetic_phase_native(solver, ctx, adjoint_state)
    _cuda_backend._collocation_transpose(
        AdjEx=pre["Ex"], AdjEy=pre["Ey"], AdjEz=pre["Ez"],
        GEx=g_fsq["Ex"], GEy=g_fsq["Ey"], GEz=g_fsq["Ez"],
        Ex=forward_state["Ex"], Ey=forward_state["Ey"], Ez=forward_state["Ez"],
    )
    if dispersive_state:
        eps_by_field = {"Ex": eps_ex, "Ey": eps_ey, "Ez": eps_ez}
        grad_eps = {"Ex": ctx.grad_eps_ex, "Ey": ctx.grad_eps_ey, "Ez": ctx.grad_eps_ez}
        dt = float(solver.dt)
        for component, model, index, _tensor_names, entry in (
            _adjoint.iter_dispersive_state_specs(solver) or ()
        ):
            current = _adjoint.dispersive_state_name(component, model, index, "current")
            adj_current = torch.empty_like(forward_state[current])
            _cuda_backend._reverse_dispersive_correction(
                AdjCurrentCorrected=adj_current, GradEps=grad_eps[component],
                AdjCurrentPost=adjoint_state[current], AdjElectricPost=adjoint_state[component],
                Current=dispersive_state[current], Eps=eps_by_field[component], dt=dt,
            )
            if model == "debye":
                polarization = _adjoint.dispersive_state_name(component, model, index, "polarization")
                _cuda_backend._reverse_debye_current(
                    AdjElectricPrev=pre[component], AdjPolarizationPrev=pre[polarization],
                    AdjPolarizationPost=adjoint_state[polarization], AdjCurrentPost=adj_current,
                    DebyeDrive=entry["drive"], decay=float(entry["decay"]), dt=dt,
                )
            elif model == "drude":
                _cuda_backend._reverse_drude_current(
                    AdjElectricPrev=pre[component], AdjCurrentPrev=pre[current],
                    AdjCurrentPost=adj_current, DrudeDrive=entry["drive"],
                    decay=float(entry["decay"]),
                )
            else:
                polarization = _adjoint.dispersive_state_name(component, model, index, "polarization")
                _cuda_backend._reverse_lorentz_current(
                    AdjElectricPrev=pre[component], AdjPolarizationPrev=pre[polarization],
                    AdjCurrentPrev=pre[current], AdjPolarizationPost=adjoint_state[polarization],
                    AdjCurrentPost=adj_current, LorentzDrive=entry["drive"],
                    decay=float(entry["decay"]), restoring=float(entry["restoring"]), dt=dt,
                )
    result = _adjoint._ReverseStepResult(
        pre_step_adjoint={name: pre[name] for name in original_state_names}, grad_eps_ex=ctx.grad_eps_ex,
        grad_eps_ey=ctx.grad_eps_ey, grad_eps_ez=ctx.grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.GENERAL_NONLINEAR],
        magnetic_output_adjoint=adj_h,
        grad_chi2_ex=grad_chi2["Ex"], grad_chi2_ey=grad_chi2["Ey"], grad_chi2_ez=grad_chi2["Ez"],
        grad_chi3_ex=grad_chi3["Ex"], grad_chi3_ey=grad_chi3["Ey"], grad_chi3_ez=grad_chi3["Ez"],
        grad_tpa_ex=grad_tpa["Ex"], grad_tpa_ey=grad_tpa["Ey"], grad_tpa_ez=grad_tpa["Ez"],
    )
    return _adjoint._accumulate_source_term_gradients(
        result, solver=solver, adjoint_state=adjoint_state, time_value=time_value,
        eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )


def _reverse_step_bloch_native_core(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms,
    magnetic_fields=None,
    magnetic_cpml=False,
    magnetic_adjoint_additions=None,
):
    """Fused native complex-Bloch reverse math, without the source-term eps gradient.

    Returns the ``_ReverseStepResult`` the analytic Bloch reference produces
    *before* ``_accumulate_source_term_gradients`` runs, with ``pre_step_adjoint``
    holding the twelve real/imag field adjoints. The public runner adds that
    accumulation; the Bloch+dispersive runner reuses this core as its complex base
    reverse and folds the electric ADE VJP on top before the single accumulation.

    Mirrors ``reverse_step_bloch_native`` exactly, launching the fused
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
    if magnetic_fields is None:
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
    if magnetic_adjoint_additions:
        for name, target in (
            ("Hx", adj_hx_mid_r), ("Hy", adj_hy_mid_r), ("Hz", adj_hz_mid_r),
            ("Hx_imag", adj_hx_mid_i), ("Hy_imag", adj_hy_mid_i), ("Hz_imag", adj_hz_mid_i),
        ):
            addition = magnetic_adjoint_additions.get(name)
            if addition is not None:
                _cuda_backend._accumulate_in_place(dst=target, src=addition)
    if magnetic_cpml:
        zero_hx_r, zero_hx_i = torch.zeros_like(adj_hx_mid_r), torch.zeros_like(adj_hx_mid_i)
        zero_hy_r, zero_hy_i = torch.zeros_like(adj_hy_mid_r), torch.zeros_like(adj_hy_mid_i)
        zero_hz_r, zero_hz_i = torch.zeros_like(adj_hz_mid_r), torch.zeros_like(adj_hz_mid_i)
    else:
        zero_hx_r, zero_hx_i = adj_hx_mid_r, adj_hx_mid_i
        zero_hy_r, zero_hy_i = adj_hy_mid_r, adj_hy_mid_i
        zero_hz_r, zero_hz_i = adj_hz_mid_r, adj_hz_mid_i

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
        AdjHyMidReal=zero_hy_r,
        AdjHyMidImag=zero_hy_i,
        AdjHzMidReal=zero_hz_r,
        AdjHzMidImag=zero_hz_i,
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
        AdjHxMidReal=zero_hx_r,
        AdjHxMidImag=zero_hx_i,
        AdjHzMidReal=zero_hz_r,
        AdjHzMidImag=zero_hz_i,
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
        AdjHxMidReal=zero_hx_r,
        AdjHxMidImag=zero_hx_i,
        AdjHyMidReal=zero_hy_r,
        AdjHyMidImag=zero_hy_i,
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

    pre_by_name = {
        "Ex": adj_ex_prev_r,
        "Ey": adj_ey_prev_r,
        "Ez": adj_ez_prev_r,
        "Ex_imag": adj_ex_prev_i,
        "Ey_imag": adj_ey_prev_i,
        "Ez_imag": adj_ez_prev_i,
    }
    if magnetic_cpml:
        from types import SimpleNamespace
        for name in ("Hx", "Hy", "Hz", "Hx_imag", "Hy_imag", "Hz_imag"):
            pre_by_name[name] = torch.empty_like(forward_state[name])
        for name in _adjoint.checkpoint_schema(solver).cpml_state_names:
            pre_by_name[name] = torch.zeros_like(forward_state[name])
        def make_ctx(suffix, adj_h_values):
            return SimpleNamespace(
                pre_step_adjoint=pre_by_name,
                magnetic_output_adjoint=adj_h_values,
                adj_d_ez_dy=torch.empty_like(forward_state["Hx"]),
                adj_d_ey_dz=torch.empty_like(forward_state["Hx"]),
                adj_d_ex_dz=torch.empty_like(forward_state["Hy"]),
                adj_d_ez_dx=torch.empty_like(forward_state["Hy"]),
                adj_d_ey_dx=torch.empty_like(forward_state["Hz"]),
                adj_d_ex_dy=torch.empty_like(forward_state["Hz"]),
            )
        real_ctx = make_ctx("", {"Hx": adj_hx_mid_r, "Hy": adj_hy_mid_r, "Hz": adj_hz_mid_r})
        imag_ctx = make_ctx("_imag", {
            "Hx_imag": adj_hx_mid_i, "Hy_imag": adj_hy_mid_i, "Hz_imag": adj_hz_mid_i
        })
        _reverse_cpml_magnetic_phase_native(solver, real_ctx, adjoint_state)
        _reverse_cpml_magnetic_phase_native(solver, imag_ctx, adjoint_state, suffix="_imag")
    else:
        for component, adj_r, adj_i in (
            ("Hx", adj_hx_mid_r, adj_hx_mid_i),
            ("Hy", adj_hy_mid_r, adj_hy_mid_i),
            ("Hz", adj_hz_mid_r, adj_hz_mid_i),
        ):
            prev_r, prev_i = torch.empty_like(adj_r), torch.empty_like(adj_i)
            getattr(_cuda_backend, f"_reverse_magnetic_{component.lower()}_decay")(
                **{f"Adj{component}Prev": prev_r, f"Adj{component}Mid": adj_r,
                   f"{component}Decay": getattr(solver, f"c{component.lower()}_decay")}
            )
            getattr(_cuda_backend, f"_reverse_magnetic_{component.lower()}_decay")(
                **{f"Adj{component}Prev": prev_i, f"Adj{component}Mid": adj_i,
                   f"{component}Decay": getattr(solver, f"c{component.lower()}_decay")}
            )
            pre_by_name[component] = prev_r
            pre_by_name[component + "_imag"] = prev_i
    pre_by_name = {name: pre_by_name[name] for name in forward_state if name in pre_by_name}
    return _adjoint._ReverseStepResult(
        pre_step_adjoint=pre_by_name,
        grad_eps_ex=grad_eps_ex,
        grad_eps_ey=grad_eps_ey,
        grad_eps_ez=grad_eps_ez,
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.BLOCH],
        magnetic_output_adjoint={
            "Hx": adj_hx_mid_r, "Hy": adj_hy_mid_r, "Hz": adj_hz_mid_r,
            "Hx_imag": adj_hx_mid_i, "Hy_imag": adj_hy_mid_i, "Hz_imag": adj_hz_mid_i,
        },
    )


def _reverse_step_mixed_bloch_cpml_native(
    solver, forward_state, adjoint_state, *, time_value, eps_ex, eps_ey, eps_ez,
    resolved_source_terms, profiler,
):
    """Native reverse of one-PML-axis/two-Bloch-axis complex CPML."""
    import dataclasses
    import torch

    from . import core as _adjoint
    from .reverse_common import dynamic_electric_curls
    from ..runtime.stepping import _bloch_cpml_pml_axis

    pml_axis_name = _bloch_cpml_pml_axis(solver)
    if pml_axis_name is None:
        raise RuntimeError("Native mixed Bloch/CPML reverse requires one PML axis and two Bloch axes.")
    captured = []
    _adjoint._step_state(
        solver, forward_state, time_value=time_value,
        eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
        capture_magnetic=captured,
    )
    magnetic_fields = {name: tensor.contiguous() for name, tensor in captured[0].items()}
    dispersive_state = _adjoint._advance_dispersive_state(solver, forward_state)
    magnetic_dispersive_state = _adjoint._advance_magnetic_dispersive_state(
        solver, forward_state
    )
    curls = dict(zip(("Ex", "Ey", "Ez"), dynamic_electric_curls(
        solver, eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez
    )))
    additions = {
        name: torch.zeros_like(forward_state[name])
        for name in ("Hx", "Hy", "Hz", "Hx_imag", "Hy_imag", "Hz_imag")
    }
    electric_psi_names = (
        "psi_ex_y", "psi_ex_z", "psi_ey_x", "psi_ey_z", "psi_ez_x", "psi_ez_y"
    )
    psi_pre = {
        name + suffix: adjoint_state[name + suffix].detach().clone()
        for suffix in ("", "_imag") for name in electric_psi_names
    }
    axis = {"x": 0, "y": 1, "z": 2}[pml_axis_name]
    # component, psi, derivative target H, sign, tangent axis
    correction_specs = {
        "z": (("Ex", "psi_ex_z", "Hy", -1.0, 1), ("Ey", "psi_ey_z", "Hx", 1.0, 0)),
        "y": (("Ex", "psi_ex_y", "Hz", 1.0, 2), ("Ez", "psi_ez_y", "Hx", -1.0, 0)),
        "x": (("Ey", "psi_ey_x", "Hz", -1.0, 2), ("Ez", "psi_ez_x", "Hy", 1.0, 1)),
    }[pml_axis_name]
    inv_delta = (solver.inv_dx_e, solver.inv_dy_e, solver.inv_dz_e)[axis]
    b = (solver.cpml_b_e_x, solver.cpml_b_e_y, solver.cpml_b_e_z)[axis]
    c = (solver.cpml_c_e_x, solver.cpml_c_e_y, solver.cpml_c_e_z)[axis]
    inv_kappa = (
        solver.cpml_inv_kappa_e_x, solver.cpml_inv_kappa_e_y, solver.cpml_inv_kappa_e_z
    )[axis]
    for suffix in ("", "_imag"):
        for component, psi_name, h_name, sign, tangent_axis in correction_specs:
            field_key, psi_key, h_key = component + suffix, psi_name + suffix, h_name + suffix
            adj_derivative = torch.empty_like(forward_state[field_key])
            _cuda_backend._reverse_cpml_correction(
                AdjPsiPrev=psi_pre[psi_key], AdjDerivative=adj_derivative,
                AdjField=adjoint_state[field_key], AdjPsiPost=adjoint_state[psi_key],
                Curl=curls[component], B=b, C=c, InvKappa=inv_kappa,
                NormalAxis=axis, TangentAxis=tangent_axis,
                TangentLowMode=getattr(solver, f"boundary_{'xyz'[tangent_axis]}_low_code"),
                TangentHighMode=getattr(solver, f"boundary_{'xyz'[tangent_axis]}_high_code"),
                Sign=sign,
            )
            accumulate = (
                _cuda_backend._accumulate_backward_diff_x,
                _cuda_backend._accumulate_backward_diff_y,
                _cuda_backend._accumulate_backward_diff_z,
            )[axis]
            accumulate(**{
                "FieldGrad": additions[h_key], "DiffGrad": adj_derivative,
                ("invDx", "invDy", "invDz")[axis]: inv_delta,
            })
    result = _reverse_step_bloch_native_core(
        solver, forward_state, adjoint_state, time_value=time_value,
        eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
        magnetic_fields=magnetic_fields, magnetic_cpml=True,
        magnetic_adjoint_additions=additions,
    )
    pre = dict(result.pre_step_adjoint)
    pre.update(psi_pre)
    for name in _adjoint.checkpoint_schema(solver).dispersive_state_names:
        pre[name] = torch.zeros_like(forward_state[name])
    for name in _adjoint.checkpoint_schema(solver).magnetic_dispersive_state_names:
        pre[name] = torch.zeros_like(forward_state[name])
    if dispersive_state:
        grad_eps = {"Ex": result.grad_eps_ex, "Ey": result.grad_eps_ey, "Ez": result.grad_eps_ez}
        eps_by_field = {"Ex": eps_ex, "Ey": eps_ey, "Ez": eps_ez}
        dt = float(solver.dt)
        for suffix in ("", "_imag"):
            for component, model, index, _tensor_names, entry in (
                _adjoint.iter_dispersive_state_specs(solver) or ()
            ):
                current = _adjoint.dispersive_state_name(component, model, index, "current") + suffix
                adj_current = torch.empty_like(forward_state[current])
                _cuda_backend._reverse_dispersive_correction(
                    AdjCurrentCorrected=adj_current, GradEps=grad_eps[component],
                    AdjCurrentPost=adjoint_state[current],
                    AdjElectricPost=adjoint_state[component + suffix],
                    Current=dispersive_state[current], Eps=eps_by_field[component], dt=dt,
                )
                if model == "debye":
                    polarization = _adjoint.dispersive_state_name(
                        component, model, index, "polarization"
                    ) + suffix
                    _cuda_backend._reverse_debye_current(
                        AdjElectricPrev=pre[component + suffix],
                        AdjPolarizationPrev=pre[polarization],
                        AdjPolarizationPost=adjoint_state[polarization],
                        AdjCurrentPost=adj_current, DebyeDrive=entry["drive"],
                        decay=float(entry["decay"]), dt=dt,
                    )
                elif model == "drude":
                    _cuda_backend._reverse_drude_current(
                        AdjElectricPrev=pre[component + suffix], AdjCurrentPrev=pre[current],
                        AdjCurrentPost=adj_current, DrudeDrive=entry["drive"],
                        decay=float(entry["decay"]),
                    )
                else:
                    polarization = _adjoint.dispersive_state_name(
                        component, model, index, "polarization"
                    ) + suffix
                    _cuda_backend._reverse_lorentz_current(
                        AdjElectricPrev=pre[component + suffix],
                        AdjPolarizationPrev=pre[polarization], AdjCurrentPrev=pre[current],
                        AdjPolarizationPost=adjoint_state[polarization], AdjCurrentPost=adj_current,
                        LorentzDrive=entry["drive"], decay=float(entry["decay"]),
                        restoring=float(entry["restoring"]), dt=dt,
                    )
    _apply_magnetic_ade_reverse_native(
        solver, forward_state, adjoint_state, magnetic_dispersive_state,
        result.magnetic_output_adjoint, pre,
    )
    pre = {name: pre[name] for name in forward_state}
    backend = (
        _ReverseBackend.GRATING_TFSF if getattr(solver, "tfsf_enabled", False)
        else _ReverseBackend.MIXED_BLOCH_CPML
    )
    result = dataclasses.replace(
        result, pre_step_adjoint=pre, backend=_NATIVE_REVERSE_LABELS[backend]
    )
    return _adjoint._accumulate_source_term_gradients(
        result, solver=solver, adjoint_state=adjoint_state, time_value=time_value,
        eps_ex=eps_ex, eps_ey=eps_ey, eps_ez=eps_ez,
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
    """Fully native reverse step for the complex split-field Bloch configuration."""
    from . import core as _adjoint

    step_result = _reverse_step_bloch_native_core(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
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


def _apply_magnetic_ade_reverse_native(
    solver, forward_state, adjoint_state, magnetic_state, magnetic_output_adjoint,
    pre_step_adjoint,
):
    """Apply native magnetic-current correction and ADE-state pullbacks."""
    import torch

    from . import core as _adjoint

    if not magnetic_state:
        return
    mu = {"Hx": solver.mu_Hx, "Hy": solver.mu_Hy, "Hz": solver.mu_Hz}
    suffixes = ("", "_imag") if getattr(solver, "complex_fields_enabled", False) else ("",)
    for suffix in suffixes:
        for component, model, index, _tensor_names, entry in (
            _adjoint.iter_magnetic_dispersive_state_specs(solver) or ()
        ):
            current = _adjoint.dispersive_state_name(component, model, index, "current") + suffix
            adj_current = torch.empty_like(forward_state[current])
            discarded_grad_mu = torch.zeros_like(mu[component])
            _cuda_backend._reverse_dispersive_correction(
                AdjCurrentCorrected=adj_current, GradEps=discarded_grad_mu,
                AdjCurrentPost=adjoint_state[current],
                AdjElectricPost=magnetic_output_adjoint[component + suffix],
                Current=magnetic_state[current], Eps=mu[component], dt=float(solver.dt),
            )
            if model == "debye":
                polarization = _adjoint.dispersive_state_name(
                    component, model, index, "polarization"
                ) + suffix
                _cuda_backend._reverse_debye_current(
                    AdjElectricPrev=pre_step_adjoint[component + suffix],
                    AdjPolarizationPrev=pre_step_adjoint[polarization],
                    AdjPolarizationPost=adjoint_state[polarization], AdjCurrentPost=adj_current,
                    DebyeDrive=entry["drive"], decay=float(entry["decay"]), dt=float(solver.dt),
                )
            elif model == "drude":
                _cuda_backend._reverse_drude_current(
                    AdjElectricPrev=pre_step_adjoint[component + suffix],
                    AdjCurrentPrev=pre_step_adjoint[current], AdjCurrentPost=adj_current,
                    DrudeDrive=entry["drive"], decay=float(entry["decay"]),
                )
            else:
                polarization = _adjoint.dispersive_state_name(
                    component, model, index, "polarization"
                ) + suffix
                _cuda_backend._reverse_lorentz_current(
                    AdjElectricPrev=pre_step_adjoint[component + suffix],
                    AdjPolarizationPrev=pre_step_adjoint[polarization],
                    AdjCurrentPrev=pre_step_adjoint[current],
                    AdjPolarizationPost=adjoint_state[polarization], AdjCurrentPost=adj_current,
                    LorentzDrive=entry["drive"], decay=float(entry["decay"]),
                    restoring=float(entry["restoring"]), dt=float(solver.dt),
                )


def _reverse_step_bloch_dispersive_native(
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
    """Fully native reverse step for a Bloch (complex-field) + electric-dispersive medium.

    Mirrors :func:`reverse_step_bloch_dispersive_native`. The complex
    Bloch base reverse produces the twelve real/imag field adjoints and the base
    eps gradient; on top of it the electric ADE VJP runs on the real and imaginary
    halves independently, reusing the same fused dispersive-correction and
    ADE-current kernels the pure-real dispersive runner uses. Both halves share the
    real eps, so both correction launches *accumulate* into the same eps gradient;
    both ADE-current launches *accumulate* the field-drive term onto their half's
    pre-step E adjoint and *assign* their half's pre-step polarization/current
    adjoint. Launch order is load-bearing: the Bloch base assigns the pre-step E
    adjoint first, then per half the correction produces the corrected current
    adjoint the ADE-current kernels read, and those accumulate onto the base E
    adjoint. The single source-term accumulation runs once at the end.
    """
    import torch

    from . import core as _adjoint

    # Replay the post-update electric dispersive currents (Torch), real + imag.
    dispersive_state = _adjoint._advance_dispersive_state(solver, forward_state)
    magnetic_dispersive_state = _adjoint._advance_magnetic_dispersive_state(
        solver, forward_state
    )
    magnetic_fields = None
    if magnetic_dispersive_state:
        magnetic_fields = _adjoint._apply_magnetic_dispersive_corrections(
            solver,
            _adjoint._forward_magnetic_fields_complex(
                solver, forward_state, time_value=time_value,
                resolved_source_terms=resolved_source_terms,
            ),
            magnetic_dispersive_state,
        )

    # Native complex Bloch base reverse (no source-term grads yet).
    base_result = _reverse_step_bloch_native_core(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
        magnetic_fields=magnetic_fields,
    )

    pre_step_adjoint = dict(base_result.pre_step_adjoint)
    for name in _adjoint.checkpoint_schema(solver).dispersive_state_names:
        if name not in pre_step_adjoint:
            pre_step_adjoint[name] = torch.zeros_like(forward_state[name])
    for name in _adjoint.checkpoint_schema(solver).magnetic_dispersive_state_names:
        pre_step_adjoint[name] = torch.zeros_like(forward_state[name])

    grad_eps_by_field = {
        "Ex": base_result.grad_eps_ex,
        "Ey": base_result.grad_eps_ey,
        "Ez": base_result.grad_eps_ez,
    }
    eps_by_field = {"Ex": eps_ex, "Ey": eps_ey, "Ez": eps_ez}
    dt = float(solver.dt)

    # Per half (real, imaginary), per pole: correction VJP then ADE-state VJP. The
    # coefficients are identical across halves; only the field/state suffix differs.
    for suffix in ("", "_imag"):
        for component_name, model_name, index, _tensor_names, entry in _adjoint.iter_dispersive_state_specs(solver) or ():
            current_name = _adjoint.dispersive_state_name(component_name, model_name, index, "current") + suffix
            adj_current_corrected = torch.empty_like(forward_state[current_name])
            _cuda_backend._reverse_dispersive_correction(
                AdjCurrentCorrected=adj_current_corrected,
                GradEps=grad_eps_by_field[component_name],
                AdjCurrentPost=adjoint_state[current_name],
                AdjElectricPost=adjoint_state[component_name + suffix],
                Current=dispersive_state[current_name],
                Eps=eps_by_field[component_name],
                dt=dt,
            )
            if model_name == "debye":
                polarization_name = _adjoint.dispersive_state_name(component_name, model_name, index, "polarization") + suffix
                _cuda_backend._reverse_debye_current(
                    AdjElectricPrev=pre_step_adjoint[component_name + suffix],
                    AdjPolarizationPrev=pre_step_adjoint[polarization_name],
                    AdjPolarizationPost=adjoint_state[polarization_name],
                    AdjCurrentPost=adj_current_corrected,
                    DebyeDrive=entry["drive"],
                    decay=float(entry["decay"]),
                    dt=dt,
                )
            elif model_name == "drude":
                _cuda_backend._reverse_drude_current(
                    AdjElectricPrev=pre_step_adjoint[component_name + suffix],
                    AdjCurrentPrev=pre_step_adjoint[current_name],
                    AdjCurrentPost=adj_current_corrected,
                    DrudeDrive=entry["drive"],
                    decay=float(entry["decay"]),
                )
            else:
                polarization_name = _adjoint.dispersive_state_name(component_name, model_name, index, "polarization") + suffix
                _cuda_backend._reverse_lorentz_current(
                    AdjElectricPrev=pre_step_adjoint[component_name + suffix],
                    AdjPolarizationPrev=pre_step_adjoint[polarization_name],
                    AdjCurrentPrev=pre_step_adjoint[current_name],
                    AdjPolarizationPost=adjoint_state[polarization_name],
                    AdjCurrentPost=adj_current_corrected,
                    LorentzDrive=entry["drive"],
                    decay=float(entry["decay"]),
                    restoring=float(entry["restoring"]),
                    dt=dt,
                )

    _apply_magnetic_ade_reverse_native(
        solver, forward_state, adjoint_state, magnetic_dispersive_state,
        base_result.magnetic_output_adjoint, pre_step_adjoint,
    )
    step_result = _adjoint._ReverseStepResult(
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=grad_eps_by_field["Ex"],
        grad_eps_ey=grad_eps_by_field["Ey"],
        grad_eps_ez=grad_eps_by_field["Ez"],
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.BLOCH_DISPERSIVE],
        source_adjoint_state=dict(adjoint_state),
    )
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

    Mirrors ``reverse_step_dispersive_native`` for the electric-only
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

    # Step 1: replay the post-update electric and magnetic ADE currents.  Replay
    # is orchestration; all pullbacks below are compiled CUDA kernels.
    dispersive_state = _adjoint._advance_dispersive_state(solver, forward_state)
    magnetic_dispersive_state = _adjoint._advance_magnetic_dispersive_state(
        solver, forward_state
    )
    magnetic_fields = None
    if magnetic_dispersive_state:
        magnetic_fields = _adjoint._apply_magnetic_dispersive_corrections(
            solver,
            _adjoint._forward_magnetic_fields(
                solver,
                forward_state,
                time_value=time_value,
                resolved_source_terms=resolved_source_terms,
            ),
            magnetic_dispersive_state,
        )

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
            magnetic_fields=magnetic_fields,
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
            magnetic_fields=magnetic_fields,
        )

    pre_step_adjoint = dict(base_result.pre_step_adjoint)
    for name in _adjoint.checkpoint_schema(solver).dispersive_state_names:
        if name not in pre_step_adjoint:
            pre_step_adjoint[name] = torch.zeros_like(forward_state[name])
    for name in _adjoint.checkpoint_schema(solver).magnetic_dispersive_state_names:
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
            polarization_name = _adjoint.dispersive_state_name(
                component_name, model_name, index, "polarization"
            )
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

    # Magnetic ADE is the same local state recurrence on H.  The correction
    # kernel is reused with mu as its denominator; its accumulated mu gradient is
    # intentionally not exposed by the current public material compiler, matching
    # the existing static-mu adjoint contract.
    if magnetic_dispersive_state:
        mu_by_field = {
            "Hx": solver.mu_Hx,
            "Hy": solver.mu_Hy,
            "Hz": solver.mu_Hz,
        }
        magnetic_output_adjoint = base_result.magnetic_output_adjoint
        for component_name, model_name, index, _tensor_names, entry in (
            _adjoint.iter_magnetic_dispersive_state_specs(solver) or ()
        ):
            current_name = _adjoint.dispersive_state_name(
                component_name, model_name, index, "current"
            )
            adj_current_corrected = torch.empty_like(forward_state[current_name])
            discarded_grad_mu = torch.zeros_like(mu_by_field[component_name])
            _cuda_backend._reverse_dispersive_correction(
                AdjCurrentCorrected=adj_current_corrected,
                GradEps=discarded_grad_mu,
                AdjCurrentPost=adjoint_state[current_name],
                AdjElectricPost=magnetic_output_adjoint[component_name],
                Current=magnetic_dispersive_state[current_name],
                Eps=mu_by_field[component_name],
                dt=dt,
            )
            if model_name == "debye":
                polarization_name = _adjoint.dispersive_state_name(
                    component_name, model_name, index, "polarization"
                )
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
                polarization_name = _adjoint.dispersive_state_name(
                    component_name, model_name, index, "polarization"
                )
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
        backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.DISPERSIVE],
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

    Mirrors ``_reverse_tfsf_auxiliary_state_native`` one-for-one, driving
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

    The native standard or CPML core reverse produces the pre-step field/psi
    adjoint (its mid-step H replay consumes
    the materialized magnetic incident source terms, so the injected incident field
    is folded into the mid-H exactly as the forward step did), then the native TFSF
    auxiliary reverse kernels add the 1D auxiliary electric/magnetic pre-step
    adjoints. TFSF incident terms are literal patches, so this runner needs no
    source-term permittivity-gradient accumulation.
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


def _full_aniso_neg_inv_deltas(solver):
    """Cache the negated electric inverse spacings the off-diagonal fold needs.

    The off-diagonal reverse folds each ``adj_curl`` into two mid-step H adjoints
    with opposite signs (``curl = D_a - D_b``). The shared backward-difference
    transpose only accumulates ``+D^T``, so the subtractive fold passes a negated
    inverse-spacing vector (the transpose is linear in it). These 1D vectors are
    static for the whole reverse pass, so they are negated once and cached on the
    solver rather than per step, keeping the per-cell reverse math fully native.
    """
    cached = getattr(solver, "_full_aniso_neg_inv_deltas", None)
    if cached is None:
        cached = {
            "x": -solver.inv_dx_e,
            "y": -solver.inv_dy_e,
            "z": -solver.inv_dz_e,
        }
        try:
            solver._full_aniso_neg_inv_deltas = cached
        except (AttributeError, TypeError):
            pass
    return cached


def _reverse_step_full_aniso_native(
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
    """Fully native reverse step for a full (off-diagonal) anisotropic CPML medium.

    The off-diagonal coupling only feeds the mid-step magnetic adjoint the diagonal
    CPML reverse already consumes, so this runner (1) forms the off-diagonal
    curl(H) cotangents with the native ``full_aniso_curl_adjoint`` op, (2) folds
    them into a clone of the post-step H adjoint with the shared native
    backward-difference transpose, and (3) delegates to the linear CPML native core
    on that augmented adjoint state. Every per-cell reverse computation runs inside
    the compiled kernels; Python only sequences the launches (matching
    ``reverse_step_full_aniso_cpml_native`` exactly).
    """
    import dataclasses

    import torch

    from . import core as _adjoint

    adj_ex = adjoint_state["Ex"]
    adj_ey = adjoint_state["Ey"]
    adj_ez = adjoint_state["Ez"]

    # (1) Off-diagonal curl(H) cotangents on their own electric-edge grids (assign).
    adj_curl_x = torch.empty_like(adj_ex)
    adj_curl_y = torch.empty_like(adj_ey)
    adj_curl_z = torch.empty_like(adj_ez)
    _cuda_backend._full_aniso_curl_adjoint(
        AdjCurlX=adj_curl_x,
        AdjCurlY=adj_curl_y,
        AdjCurlZ=adj_curl_z,
        AdjEx=adj_ex,
        AdjEy=adj_ey,
        AdjEz=adj_ez,
        CoeffExY=solver.cex_aniso_y,
        CoeffExZ=solver.cex_aniso_z,
        CoeffEyX=solver.cey_aniso_x,
        CoeffEyZ=solver.cey_aniso_z,
        CoeffEzX=solver.cez_aniso_x,
        CoeffEzY=solver.cez_aniso_y,
    )

    # (2) Fold into a clone of the post-step H adjoint (native accumulate). The
    # +/- sign of each curl derivative matches the diagonal curl fold; the minus
    # folds pass a negated inverse spacing (the transpose is linear in it).
    neg = _full_aniso_neg_inv_deltas(solver)
    aug_hx = adjoint_state["Hx"].detach().clone()
    aug_hy = adjoint_state["Hy"].detach().clone()
    aug_hz = adjoint_state["Hz"].detach().clone()
    # curl_x = dHz/dy - dHy/dz  ->  Hz += Dy^T(curl_x); Hy -= Dz^T(curl_x)
    _cuda_backend._accumulate_backward_diff_y(FieldGrad=aug_hz, DiffGrad=adj_curl_x, invDy=solver.inv_dy_e)
    _cuda_backend._accumulate_backward_diff_z(FieldGrad=aug_hy, DiffGrad=adj_curl_x, invDz=neg["z"])
    # curl_y = dHx/dz - dHz/dx  ->  Hx += Dz^T(curl_y); Hz -= Dx^T(curl_y)
    _cuda_backend._accumulate_backward_diff_z(FieldGrad=aug_hx, DiffGrad=adj_curl_y, invDz=solver.inv_dz_e)
    _cuda_backend._accumulate_backward_diff_x(FieldGrad=aug_hz, DiffGrad=adj_curl_y, invDx=neg["x"])
    # curl_z = dHy/dx - dHx/dy  ->  Hy += Dx^T(curl_z); Hx -= Dy^T(curl_z)
    _cuda_backend._accumulate_backward_diff_x(FieldGrad=aug_hy, DiffGrad=adj_curl_z, invDx=solver.inv_dx_e)
    _cuda_backend._accumulate_backward_diff_y(FieldGrad=aug_hx, DiffGrad=adj_curl_z, invDy=neg["y"])

    adjoint_state_aug = dict(adjoint_state)
    adjoint_state_aug["Hx"] = aug_hx
    adjoint_state_aug["Hy"] = aug_hy
    adjoint_state_aug["Hz"] = aug_hz

    # (3) Diagonal CPML reverse on the augmented adjoint state.
    step_result = _reverse_step_cpml_native_core(
        solver,
        forward_state,
        adjoint_state_aug,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )
    step_result = dataclasses.replace(
        step_result, backend=_NATIVE_REVERSE_LABELS[_ReverseBackend.FULL_ANISO]
    )
    # Source-term eps-gradient accumulation reads the electric adjoint only, so it
    # runs on the original (unaugmented) adjoint state.
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


def _reverse_step_wire_native(
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
    cpml,
):
    """Compose the fused Yee reverse with the sparse wire-network transpose."""

    import dataclasses

    from . import core as _adjoint
    from ..wire import reverse_wire_step

    base_core = (
        _reverse_step_cpml_native_core if cpml else _reverse_step_standard_native_core
    )
    step_result = base_core(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=resolved_source_terms,
    )
    wire_result = reverse_wire_step(
        solver,
        forward_state,
        adjoint_state,
        eps_by_field={"Ex": eps_ex, "Ey": eps_ey, "Ez": eps_ez},
    )
    pre_step_adjoint = dict(step_result.pre_step_adjoint)
    for name in ("Ex", "Ey", "Ez"):
        pre_step_adjoint[name] = pre_step_adjoint[name] + wire_result.field_adjoint[name]
    pre_step_adjoint["wire_current"] = wire_result.pre_current
    pre_step_adjoint["wire_charge"] = wire_result.pre_charge
    backend = (
        _ReverseBackend.WIRE_CPML if cpml else _ReverseBackend.WIRE_STANDARD
    )
    step_result = dataclasses.replace(
        step_result,
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=step_result.grad_eps_ex + wire_result.grad_eps["Ex"],
        grad_eps_ey=step_result.grad_eps_ey + wire_result.grad_eps["Ey"],
        grad_eps_ez=step_result.grad_eps_ez + wire_result.grad_eps["Ez"],
        grad_wire_inductance=wire_result.grad_inductance,
        grad_wire_capacitance=wire_result.grad_node_capacitance,
        backend=_NATIVE_REVERSE_LABELS[backend],
    )
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


def _reverse_step_wire_standard_native(solver, forward_state, adjoint_state, **kwargs):
    return _reverse_step_wire_native(
        solver, forward_state, adjoint_state, cpml=False, **kwargs
    )


def _reverse_step_wire_cpml_native(solver, forward_state, adjoint_state, **kwargs):
    return _reverse_step_wire_native(
        solver, forward_state, adjoint_state, cpml=True, **kwargs
    )


def register_native_reverse_backends() -> None:
    """Register every available native CUDA reverse-step runner."""
    register_native_reverse_backend(
        _ReverseBackend.GENERAL_NONLINEAR,
        _reverse_step_nonlinear_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.MIXED_BLOCH_CPML,
        _reverse_step_mixed_bloch_cpml_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.GRATING_TFSF,
        _reverse_step_mixed_bloch_cpml_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.STANDARD,
        _reverse_step_standard_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.CPML,
        _reverse_step_cpml_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.WIRE_STANDARD,
        _reverse_step_wire_standard_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.WIRE_CPML,
        _reverse_step_wire_cpml_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.CONDUCTIVE,
        _reverse_step_conductive_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.KERR,
        _reverse_step_kerr_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.FULL_ANISO,
        _reverse_step_full_aniso_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.BLOCH,
        _reverse_step_bloch_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.BLOCH_DISPERSIVE,
        _reverse_step_bloch_dispersive_native,
        qualifier=_cuda_scene_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.DISPERSIVE,
        _reverse_step_dispersive_native,
        qualifier=_dispersive_native_qualifies,
    )
    register_native_reverse_backend(
        _ReverseBackend.TFSF,
        _reverse_step_tfsf_native,
        qualifier=_cuda_scene_native_qualifies,
    )
