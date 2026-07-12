from __future__ import annotations

import importlib

import torch

from .core import (
    _BackwardProfiler,
    _ReverseStepResult,
    _accumulate_backward_diff_adjoint,
    _accumulate_bloch_backward_diff_adjoint,
    _accumulate_forward_diff_adjoint,
    _advance_dispersive_state,
    _advance_magnetic_dispersive_state,
    _apply_magnetic_dispersive_corrections,
    _apply_resolved_magnetic_source_terms,
    _backward_diff,
    _bloch_backward_diff,
    _forward_magnetic_fields,
    _forward_magnetic_fields_complex,
    _reverse_dispersive_corrections,
    _reverse_dispersive_state_python_reference,
    _reverse_electric_component_bloch,
    _reverse_electric_component_cpml,
    _reverse_electric_component_standard,
    _reverse_magnetic_component_cpml,
    _reverse_magnetic_component_standard,
    _reverse_magnetic_dispersive_corrections,
    _reverse_magnetic_dispersive_state_python_reference,
    _reverse_tfsf_auxiliary_state_python_reference,
    _tfsf_magnetic_source_terms,
)

__all__ = [
    "reverse_step_bloch_python_reference",
    "reverse_step_cpml_python_reference",
    "reverse_step_dispersive_python_reference",
    "reverse_step_grating_tfsf",
    "reverse_step_standard_python_reference",
    "reverse_step_tfsf",
    "reverse_step_torch_vjp",
]


def reverse_step_standard_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
    magnetic_fields=None,
):
    if magnetic_fields is None:
        magnetic_fields = _forward_magnetic_fields(
            solver,
            forward_state,
            time_value=time_value,
            resolved_source_terms=resolved_source_terms,
        )
    grad_eps_ex = torch.zeros_like(eps_ex)
    grad_eps_ey = torch.zeros_like(eps_ey)
    grad_eps_ez = torch.zeros_like(eps_ez)

    pre_step_adjoint = {
        name: torch.zeros_like(tensor)
        for name, tensor in forward_state.items()
    }
    magnetic_output_adjoint = {
        "Hx": adjoint_state["Hx"].clone(),
        "Hy": adjoint_state["Hy"].clone(),
        "Hz": adjoint_state["Hz"].clone(),
    }

    d_hz_dy = _backward_diff(magnetic_fields["Hz"], axis=1, inv_delta=solver.inv_dy_e)
    d_hy_dz = _backward_diff(magnetic_fields["Hy"], axis=2, inv_delta=solver.inv_dz_e)
    adj_ex, adj_d_hz_dy, adj_d_hy_dz, grad_eps_ex_increment = _reverse_electric_component_standard(
        adjoint_state["Ex"],
        forward_state["Ex"],
        d_pos=d_hz_dy,
        d_neg=d_hy_dz,
        decay=solver.cex_decay,
        curl_prefactor=solver.cex_curl * solver.eps_Ex,
        eps=eps_ex,
        low_mode_pos=solver.boundary_y_low_code,
        high_mode_pos=solver.boundary_y_high_code,
        low_mode_neg=solver.boundary_z_low_code,
        high_mode_neg=solver.boundary_z_high_code,
        axis_pos=1,
        axis_neg=2,
    )
    pre_step_adjoint["Ex"] = pre_step_adjoint["Ex"] + adj_ex
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hz"], adj_d_hz_dy, axis=1, inv_delta=solver.inv_dy_e)
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hy"], adj_d_hy_dz, axis=2, inv_delta=solver.inv_dz_e)
    grad_eps_ex = grad_eps_ex + grad_eps_ex_increment

    d_hx_dz = _backward_diff(magnetic_fields["Hx"], axis=2, inv_delta=solver.inv_dz_e)
    d_hz_dx = _backward_diff(magnetic_fields["Hz"], axis=0, inv_delta=solver.inv_dx_e)
    adj_ey, adj_d_hx_dz, adj_d_hz_dx, grad_eps_ey_increment = _reverse_electric_component_standard(
        adjoint_state["Ey"],
        forward_state["Ey"],
        d_pos=d_hx_dz,
        d_neg=d_hz_dx,
        decay=solver.cey_decay,
        curl_prefactor=solver.cey_curl * solver.eps_Ey,
        eps=eps_ey,
        low_mode_pos=solver.boundary_z_low_code,
        high_mode_pos=solver.boundary_z_high_code,
        low_mode_neg=solver.boundary_x_low_code,
        high_mode_neg=solver.boundary_x_high_code,
        axis_pos=2,
        axis_neg=0,
    )
    pre_step_adjoint["Ey"] = pre_step_adjoint["Ey"] + adj_ey
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hx"], adj_d_hx_dz, axis=2, inv_delta=solver.inv_dz_e)
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hz"], adj_d_hz_dx, axis=0, inv_delta=solver.inv_dx_e)
    grad_eps_ey = grad_eps_ey + grad_eps_ey_increment

    d_hy_dx = _backward_diff(magnetic_fields["Hy"], axis=0, inv_delta=solver.inv_dx_e)
    d_hx_dy = _backward_diff(magnetic_fields["Hx"], axis=1, inv_delta=solver.inv_dy_e)
    adj_ez, adj_d_hy_dx, adj_d_hx_dy, grad_eps_ez_increment = _reverse_electric_component_standard(
        adjoint_state["Ez"],
        forward_state["Ez"],
        d_pos=d_hy_dx,
        d_neg=d_hx_dy,
        decay=solver.cez_decay,
        curl_prefactor=solver.cez_curl * solver.eps_Ez,
        eps=eps_ez,
        low_mode_pos=solver.boundary_x_low_code,
        high_mode_pos=solver.boundary_x_high_code,
        low_mode_neg=solver.boundary_y_low_code,
        high_mode_neg=solver.boundary_y_high_code,
        axis_pos=0,
        axis_neg=1,
    )
    pre_step_adjoint["Ez"] = pre_step_adjoint["Ez"] + adj_ez
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hy"], adj_d_hy_dx, axis=0, inv_delta=solver.inv_dx_e)
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hx"], adj_d_hx_dy, axis=1, inv_delta=solver.inv_dy_e)
    grad_eps_ez = grad_eps_ez + grad_eps_ez_increment

    adj_hx, adj_d_ez_dy, adj_d_ey_dz = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hx"],
        decay=solver.chx_decay,
        curl=solver.chx_curl,
    )
    pre_step_adjoint["Hx"] = pre_step_adjoint["Hx"] + adj_hx
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez"], adj_d_ez_dy, axis=1, inv_delta=solver.inv_dy_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey"], adj_d_ey_dz, axis=2, inv_delta=solver.inv_dz_h)

    adj_hy, adj_d_ex_dz, adj_d_ez_dx = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hy"],
        decay=solver.chy_decay,
        curl=solver.chy_curl,
    )
    pre_step_adjoint["Hy"] = pre_step_adjoint["Hy"] + adj_hy
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex"], adj_d_ex_dz, axis=2, inv_delta=solver.inv_dz_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez"], adj_d_ez_dx, axis=0, inv_delta=solver.inv_dx_h)

    adj_hz, adj_d_ey_dx, adj_d_ex_dy = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hz"],
        decay=solver.chz_decay,
        curl=solver.chz_curl,
    )
    pre_step_adjoint["Hz"] = pre_step_adjoint["Hz"] + adj_hz
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey"], adj_d_ey_dx, axis=0, inv_delta=solver.inv_dx_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex"], adj_d_ex_dy, axis=1, inv_delta=solver.inv_dy_h)

    return _ReverseStepResult(
        pre_step_adjoint={name: tensor.detach() for name, tensor in pre_step_adjoint.items()},
        grad_eps_ex=grad_eps_ex.detach(),
        grad_eps_ey=grad_eps_ey.detach(),
        grad_eps_ez=grad_eps_ez.detach(),
        backend="python_reference_standard",
        magnetic_output_adjoint={name: tensor.detach() for name, tensor in magnetic_output_adjoint.items()},
    )


def reverse_step_bloch_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    magnetic_fields = _forward_magnetic_fields_complex(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )

    grad_eps_ex = torch.zeros_like(eps_ex)
    grad_eps_ey = torch.zeros_like(eps_ey)
    grad_eps_ez = torch.zeros_like(eps_ez)
    pre_step_adjoint = {
        name: torch.zeros_like(tensor)
        for name, tensor in forward_state.items()
    }
    magnetic_output_adjoint = {
        "Hx": torch.complex(adjoint_state["Hx"], adjoint_state["Hx_imag"]),
        "Hy": torch.complex(adjoint_state["Hy"], adjoint_state["Hy_imag"]),
        "Hz": torch.complex(adjoint_state["Hz"], adjoint_state["Hz_imag"]),
    }

    hy_complex = torch.complex(magnetic_fields["Hy"], magnetic_fields["Hy_imag"])
    hz_complex = torch.complex(magnetic_fields["Hz"], magnetic_fields["Hz_imag"])
    d_hz_dy = _bloch_backward_diff(
        hz_complex,
        axis=1,
        inv_delta=solver.inv_dy_e,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    d_hy_dz = _bloch_backward_diff(
        hy_complex,
        axis=2,
        inv_delta=solver.inv_dz_e,
        phase_cos=float(solver.boundary_phase_cos[2]),
        phase_sin=float(solver.boundary_phase_sin[2]),
    )
    adj_ex, adj_d_hz_dy, adj_d_hy_dz, grad_eps_ex_increment = _reverse_electric_component_bloch(
        torch.complex(adjoint_state["Ex"], adjoint_state["Ex_imag"]),
        d_pos=d_hz_dy,
        d_neg=d_hy_dz,
        decay=solver.cex_decay,
        curl_prefactor=solver.cex_curl * solver.eps_Ex,
        eps=eps_ex,
    )
    pre_step_adjoint["Ex"] = pre_step_adjoint["Ex"] + adj_ex.real
    pre_step_adjoint["Ex_imag"] = pre_step_adjoint["Ex_imag"] + adj_ex.imag
    _accumulate_bloch_backward_diff_adjoint(
        magnetic_output_adjoint["Hz"],
        adj_d_hz_dy,
        axis=1,
        inv_delta=solver.inv_dy_e,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    _accumulate_bloch_backward_diff_adjoint(
        magnetic_output_adjoint["Hy"],
        adj_d_hy_dz,
        axis=2,
        inv_delta=solver.inv_dz_e,
        phase_cos=float(solver.boundary_phase_cos[2]),
        phase_sin=float(solver.boundary_phase_sin[2]),
    )
    grad_eps_ex = grad_eps_ex + grad_eps_ex_increment

    hx_complex = torch.complex(magnetic_fields["Hx"], magnetic_fields["Hx_imag"])
    d_hx_dz = _bloch_backward_diff(
        hx_complex,
        axis=2,
        inv_delta=solver.inv_dz_e,
        phase_cos=float(solver.boundary_phase_cos[2]),
        phase_sin=float(solver.boundary_phase_sin[2]),
    )
    d_hz_dx = _bloch_backward_diff(
        hz_complex,
        axis=0,
        inv_delta=solver.inv_dx_e,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    adj_ey, adj_d_hx_dz, adj_d_hz_dx, grad_eps_ey_increment = _reverse_electric_component_bloch(
        torch.complex(adjoint_state["Ey"], adjoint_state["Ey_imag"]),
        d_pos=d_hx_dz,
        d_neg=d_hz_dx,
        decay=solver.cey_decay,
        curl_prefactor=solver.cey_curl * solver.eps_Ey,
        eps=eps_ey,
    )
    pre_step_adjoint["Ey"] = pre_step_adjoint["Ey"] + adj_ey.real
    pre_step_adjoint["Ey_imag"] = pre_step_adjoint["Ey_imag"] + adj_ey.imag
    _accumulate_bloch_backward_diff_adjoint(
        magnetic_output_adjoint["Hx"],
        adj_d_hx_dz,
        axis=2,
        inv_delta=solver.inv_dz_e,
        phase_cos=float(solver.boundary_phase_cos[2]),
        phase_sin=float(solver.boundary_phase_sin[2]),
    )
    _accumulate_bloch_backward_diff_adjoint(
        magnetic_output_adjoint["Hz"],
        adj_d_hz_dx,
        axis=0,
        inv_delta=solver.inv_dx_e,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    grad_eps_ey = grad_eps_ey + grad_eps_ey_increment

    d_hy_dx = _bloch_backward_diff(
        hy_complex,
        axis=0,
        inv_delta=solver.inv_dx_e,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    d_hx_dy = _bloch_backward_diff(
        hx_complex,
        axis=1,
        inv_delta=solver.inv_dy_e,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    adj_ez, adj_d_hy_dx, adj_d_hx_dy, grad_eps_ez_increment = _reverse_electric_component_bloch(
        torch.complex(adjoint_state["Ez"], adjoint_state["Ez_imag"]),
        d_pos=d_hy_dx,
        d_neg=d_hx_dy,
        decay=solver.cez_decay,
        curl_prefactor=solver.cez_curl * solver.eps_Ez,
        eps=eps_ez,
    )
    pre_step_adjoint["Ez"] = pre_step_adjoint["Ez"] + adj_ez.real
    pre_step_adjoint["Ez_imag"] = pre_step_adjoint["Ez_imag"] + adj_ez.imag
    _accumulate_bloch_backward_diff_adjoint(
        magnetic_output_adjoint["Hy"],
        adj_d_hy_dx,
        axis=0,
        inv_delta=solver.inv_dx_e,
        phase_cos=float(solver.boundary_phase_cos[0]),
        phase_sin=float(solver.boundary_phase_sin[0]),
    )
    _accumulate_bloch_backward_diff_adjoint(
        magnetic_output_adjoint["Hx"],
        adj_d_hx_dy,
        axis=1,
        inv_delta=solver.inv_dy_e,
        phase_cos=float(solver.boundary_phase_cos[1]),
        phase_sin=float(solver.boundary_phase_sin[1]),
    )
    grad_eps_ez = grad_eps_ez + grad_eps_ez_increment

    adj_hx, adj_d_ez_dy, adj_d_ey_dz = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hx"].real,
        decay=solver.chx_decay,
        curl=solver.chx_curl,
    )
    pre_step_adjoint["Hx"] = pre_step_adjoint["Hx"] + adj_hx
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez"], adj_d_ez_dy, axis=1, inv_delta=solver.inv_dy_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey"], adj_d_ey_dz, axis=2, inv_delta=solver.inv_dz_h)
    adj_hx_imag, adj_d_ez_imag_dy, adj_d_ey_imag_dz = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hx"].imag,
        decay=solver.chx_decay,
        curl=solver.chx_curl,
    )
    pre_step_adjoint["Hx_imag"] = pre_step_adjoint["Hx_imag"] + adj_hx_imag
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez_imag"], adj_d_ez_imag_dy, axis=1, inv_delta=solver.inv_dy_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey_imag"], adj_d_ey_imag_dz, axis=2, inv_delta=solver.inv_dz_h)

    adj_hy, adj_d_ex_dz, adj_d_ez_dx = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hy"].real,
        decay=solver.chy_decay,
        curl=solver.chy_curl,
    )
    pre_step_adjoint["Hy"] = pre_step_adjoint["Hy"] + adj_hy
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex"], adj_d_ex_dz, axis=2, inv_delta=solver.inv_dz_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez"], adj_d_ez_dx, axis=0, inv_delta=solver.inv_dx_h)
    adj_hy_imag, adj_d_ex_imag_dz, adj_d_ez_imag_dx = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hy"].imag,
        decay=solver.chy_decay,
        curl=solver.chy_curl,
    )
    pre_step_adjoint["Hy_imag"] = pre_step_adjoint["Hy_imag"] + adj_hy_imag
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex_imag"], adj_d_ex_imag_dz, axis=2, inv_delta=solver.inv_dz_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez_imag"], adj_d_ez_imag_dx, axis=0, inv_delta=solver.inv_dx_h)

    adj_hz, adj_d_ey_dx, adj_d_ex_dy = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hz"].real,
        decay=solver.chz_decay,
        curl=solver.chz_curl,
    )
    pre_step_adjoint["Hz"] = pre_step_adjoint["Hz"] + adj_hz
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey"], adj_d_ey_dx, axis=0, inv_delta=solver.inv_dx_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex"], adj_d_ex_dy, axis=1, inv_delta=solver.inv_dy_h)
    adj_hz_imag, adj_d_ey_imag_dx, adj_d_ex_imag_dy = _reverse_magnetic_component_standard(
        magnetic_output_adjoint["Hz"].imag,
        decay=solver.chz_decay,
        curl=solver.chz_curl,
    )
    pre_step_adjoint["Hz_imag"] = pre_step_adjoint["Hz_imag"] + adj_hz_imag
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey_imag"], adj_d_ey_imag_dx, axis=0, inv_delta=solver.inv_dx_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex_imag"], adj_d_ex_imag_dy, axis=1, inv_delta=solver.inv_dy_h)

    return _ReverseStepResult(
        pre_step_adjoint={name: tensor.detach() for name, tensor in pre_step_adjoint.items()},
        grad_eps_ex=grad_eps_ex.detach(),
        grad_eps_ey=grad_eps_ey.detach(),
        grad_eps_ez=grad_eps_ez.detach(),
        backend="python_reference_bloch",
    )


def reverse_step_dispersive_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
):
    dispersive_state = _advance_dispersive_state(solver, forward_state)
    electric_source_adjoint, dispersive_output_adjoint, correction_grad_eps, source_adjoint_state = (
        _reverse_dispersive_corrections(
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

    # Magnetic ADE mirror: the corrected mid-step H is what the electric update
    # consumed, so recompute it once here and hand it to the base reverse.
    magnetic_dispersive_state = _advance_magnetic_dispersive_state(solver, forward_state)
    magnetic_fields = None
    if magnetic_dispersive_state:
        magnetic_fields = _apply_magnetic_dispersive_corrections(
            solver,
            _forward_magnetic_fields(
                solver,
                forward_state,
                time_value=time_value,
                resolved_source_terms=resolved_source_terms,
            ),
            magnetic_dispersive_state,
        )

    if solver.uses_cpml:
        base_result = reverse_step_cpml_python_reference(
            solver,
            forward_state,
            adjusted_adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms,
            magnetic_fields=magnetic_fields,
        )
        backend = "python_reference_dispersive_cpml"
    else:
        base_result = reverse_step_standard_python_reference(
            solver,
            forward_state,
            adjusted_adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=resolved_source_terms,
            magnetic_fields=magnetic_fields,
        )
        backend = "python_reference_dispersive_standard"

    electric_prev_adjoint, pre_step_dispersive_adjoint = _reverse_dispersive_state_python_reference(
        solver,
        forward_state,
        dispersive_output_adjoint,
    )
    pre_step_adjoint = {
        name: tensor.detach().clone()
        for name, tensor in base_result.pre_step_adjoint.items()
    }
    for component_name, grad in electric_prev_adjoint.items():
        pre_step_adjoint[component_name] = pre_step_adjoint[component_name] + grad
    for name, grad in pre_step_dispersive_adjoint.items():
        pre_step_adjoint[name] = pre_step_adjoint[name] + grad

    if magnetic_dispersive_state:
        # The base reverse's magnetic_output_adjoint is the adjoint of the
        # corrected H (direct post-step seed + electric-update contribution);
        # the correction VJP seeds the post-step magnetic dispersive currents
        # and the state reversal propagates them to the pre-step H and state.
        magnetic_output_adjoint = _reverse_magnetic_dispersive_corrections(
            solver,
            adjoint_state,
            base_result.magnetic_output_adjoint,
        )
        magnetic_prev_adjoint, pre_step_magnetic_adjoint = _reverse_magnetic_dispersive_state_python_reference(
            solver,
            forward_state,
            magnetic_output_adjoint,
        )
        for component_name, grad in magnetic_prev_adjoint.items():
            pre_step_adjoint[component_name] = pre_step_adjoint[component_name] + grad
        for name, grad in pre_step_magnetic_adjoint.items():
            pre_step_adjoint[name] = pre_step_adjoint[name] + grad

    return _ReverseStepResult(
        pre_step_adjoint={name: tensor.detach() for name, tensor in pre_step_adjoint.items()},
        grad_eps_ex=(base_result.grad_eps_ex + correction_grad_eps["Ex"]).detach(),
        grad_eps_ey=(base_result.grad_eps_ey + correction_grad_eps["Ey"]).detach(),
        grad_eps_ez=(base_result.grad_eps_ez + correction_grad_eps["Ez"]).detach(),
        backend=backend,
        source_adjoint_state=source_adjoint_state,
    )


def reverse_step_cpml_python_reference(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value=0.0,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
    magnetic_fields=None,
):
    if magnetic_fields is None:
        magnetic_fields = _forward_magnetic_fields(
            solver,
            forward_state,
            time_value=time_value,
            resolved_source_terms=resolved_source_terms,
        )
    grad_eps_ex = torch.zeros_like(eps_ex)
    grad_eps_ey = torch.zeros_like(eps_ey)
    grad_eps_ez = torch.zeros_like(eps_ez)
    pre_step_adjoint = {
        name: torch.zeros_like(tensor)
        for name, tensor in forward_state.items()
    }
    magnetic_output_adjoint = {
        "Hx": adjoint_state["Hx"].clone(),
        "Hy": adjoint_state["Hy"].clone(),
        "Hz": adjoint_state["Hz"].clone(),
    }

    d_hz_dy = _backward_diff(magnetic_fields["Hz"], axis=1, inv_delta=solver.inv_dy_e)
    d_hy_dz = _backward_diff(magnetic_fields["Hy"], axis=2, inv_delta=solver.inv_dz_e)
    adj_ex, adj_d_hz_dy, adj_d_hy_dz, grad_eps_ex_increment, adj_psi_ex_y, adj_psi_ex_z = _reverse_electric_component_cpml(
        adjoint_state["Ex"],
        adjoint_state["psi_ex_y"],
        adjoint_state["psi_ex_z"],
        forward_state["Ex"],
        d_pos=d_hz_dy,
        d_neg=d_hy_dz,
        decay=solver.cex_decay,
        curl_prefactor=solver.cex_curl * solver.eps_Ex,
        eps=eps_ex,
        low_mode_pos=solver.boundary_y_low_code,
        high_mode_pos=solver.boundary_y_high_code,
        low_mode_neg=solver.boundary_z_low_code,
        high_mode_neg=solver.boundary_z_high_code,
        axis_pos=1,
        axis_neg=2,
        psi_pos=forward_state["psi_ex_y"],
        psi_neg=forward_state["psi_ex_z"],
        b_pos=solver.cpml_b_e_y,
        c_pos=solver.cpml_c_e_y,
        inv_kappa_pos=solver.cpml_inv_kappa_e_y,
        b_neg=solver.cpml_b_e_z,
        c_neg=solver.cpml_c_e_z,
        inv_kappa_neg=solver.cpml_inv_kappa_e_z,
    )
    pre_step_adjoint["Ex"] = pre_step_adjoint["Ex"] + adj_ex
    pre_step_adjoint["psi_ex_y"] = pre_step_adjoint["psi_ex_y"] + adj_psi_ex_y
    pre_step_adjoint["psi_ex_z"] = pre_step_adjoint["psi_ex_z"] + adj_psi_ex_z
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hz"], adj_d_hz_dy, axis=1, inv_delta=solver.inv_dy_e)
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hy"], adj_d_hy_dz, axis=2, inv_delta=solver.inv_dz_e)
    grad_eps_ex = grad_eps_ex + grad_eps_ex_increment

    d_hx_dz = _backward_diff(magnetic_fields["Hx"], axis=2, inv_delta=solver.inv_dz_e)
    d_hz_dx = _backward_diff(magnetic_fields["Hz"], axis=0, inv_delta=solver.inv_dx_e)
    adj_ey, adj_d_hx_dz, adj_d_hz_dx, grad_eps_ey_increment, adj_psi_ey_z, adj_psi_ey_x = _reverse_electric_component_cpml(
        adjoint_state["Ey"],
        adjoint_state["psi_ey_x"],
        adjoint_state["psi_ey_z"],
        forward_state["Ey"],
        d_pos=d_hx_dz,
        d_neg=d_hz_dx,
        decay=solver.cey_decay,
        curl_prefactor=solver.cey_curl * solver.eps_Ey,
        eps=eps_ey,
        low_mode_pos=solver.boundary_z_low_code,
        high_mode_pos=solver.boundary_z_high_code,
        low_mode_neg=solver.boundary_x_low_code,
        high_mode_neg=solver.boundary_x_high_code,
        axis_pos=2,
        axis_neg=0,
        psi_pos=forward_state["psi_ey_z"],
        psi_neg=forward_state["psi_ey_x"],
        b_pos=solver.cpml_b_e_z,
        c_pos=solver.cpml_c_e_z,
        inv_kappa_pos=solver.cpml_inv_kappa_e_z,
        b_neg=solver.cpml_b_e_x,
        c_neg=solver.cpml_c_e_x,
        inv_kappa_neg=solver.cpml_inv_kappa_e_x,
    )
    pre_step_adjoint["Ey"] = pre_step_adjoint["Ey"] + adj_ey
    pre_step_adjoint["psi_ey_z"] = pre_step_adjoint["psi_ey_z"] + adj_psi_ey_z
    pre_step_adjoint["psi_ey_x"] = pre_step_adjoint["psi_ey_x"] + adj_psi_ey_x
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hx"], adj_d_hx_dz, axis=2, inv_delta=solver.inv_dz_e)
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hz"], adj_d_hz_dx, axis=0, inv_delta=solver.inv_dx_e)
    grad_eps_ey = grad_eps_ey + grad_eps_ey_increment

    d_hy_dx = _backward_diff(magnetic_fields["Hy"], axis=0, inv_delta=solver.inv_dx_e)
    d_hx_dy = _backward_diff(magnetic_fields["Hx"], axis=1, inv_delta=solver.inv_dy_e)
    adj_ez, adj_d_hy_dx, adj_d_hx_dy, grad_eps_ez_increment, adj_psi_ez_x, adj_psi_ez_y = _reverse_electric_component_cpml(
        adjoint_state["Ez"],
        adjoint_state["psi_ez_x"],
        adjoint_state["psi_ez_y"],
        forward_state["Ez"],
        d_pos=d_hy_dx,
        d_neg=d_hx_dy,
        decay=solver.cez_decay,
        curl_prefactor=solver.cez_curl * solver.eps_Ez,
        eps=eps_ez,
        low_mode_pos=solver.boundary_x_low_code,
        high_mode_pos=solver.boundary_x_high_code,
        low_mode_neg=solver.boundary_y_low_code,
        high_mode_neg=solver.boundary_y_high_code,
        axis_pos=0,
        axis_neg=1,
        psi_pos=forward_state["psi_ez_x"],
        psi_neg=forward_state["psi_ez_y"],
        b_pos=solver.cpml_b_e_x,
        c_pos=solver.cpml_c_e_x,
        inv_kappa_pos=solver.cpml_inv_kappa_e_x,
        b_neg=solver.cpml_b_e_y,
        c_neg=solver.cpml_c_e_y,
        inv_kappa_neg=solver.cpml_inv_kappa_e_y,
    )
    pre_step_adjoint["Ez"] = pre_step_adjoint["Ez"] + adj_ez
    pre_step_adjoint["psi_ez_x"] = pre_step_adjoint["psi_ez_x"] + adj_psi_ez_x
    pre_step_adjoint["psi_ez_y"] = pre_step_adjoint["psi_ez_y"] + adj_psi_ez_y
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hy"], adj_d_hy_dx, axis=0, inv_delta=solver.inv_dx_e)
    _accumulate_backward_diff_adjoint(magnetic_output_adjoint["Hx"], adj_d_hx_dy, axis=1, inv_delta=solver.inv_dy_e)
    grad_eps_ez = grad_eps_ez + grad_eps_ez_increment

    adj_hx, adj_d_ez_dy, adj_d_ey_dz, adj_psi_hx_y, adj_psi_hx_z = _reverse_magnetic_component_cpml(
        magnetic_output_adjoint["Hx"],
        adjoint_state["psi_hx_y"],
        adjoint_state["psi_hx_z"],
        decay=solver.chx_decay,
        curl=solver.chx_curl,
        b_pos=solver.cpml_b_h_y,
        c_pos=solver.cpml_c_h_y,
        inv_kappa_pos=solver.cpml_inv_kappa_h_y,
        b_neg=solver.cpml_b_h_z,
        c_neg=solver.cpml_c_h_z,
        inv_kappa_neg=solver.cpml_inv_kappa_h_z,
        axis_pos=1,
        axis_neg=2,
    )
    pre_step_adjoint["Hx"] = pre_step_adjoint["Hx"] + adj_hx
    pre_step_adjoint["psi_hx_y"] = pre_step_adjoint["psi_hx_y"] + adj_psi_hx_y
    pre_step_adjoint["psi_hx_z"] = pre_step_adjoint["psi_hx_z"] + adj_psi_hx_z
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez"], adj_d_ez_dy, axis=1, inv_delta=solver.inv_dy_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey"], adj_d_ey_dz, axis=2, inv_delta=solver.inv_dz_h)

    adj_hy, adj_d_ex_dz, adj_d_ez_dx, adj_psi_hy_z, adj_psi_hy_x = _reverse_magnetic_component_cpml(
        magnetic_output_adjoint["Hy"],
        adjoint_state["psi_hy_x"],
        adjoint_state["psi_hy_z"],
        decay=solver.chy_decay,
        curl=solver.chy_curl,
        b_pos=solver.cpml_b_h_z,
        c_pos=solver.cpml_c_h_z,
        inv_kappa_pos=solver.cpml_inv_kappa_h_z,
        b_neg=solver.cpml_b_h_x,
        c_neg=solver.cpml_c_h_x,
        inv_kappa_neg=solver.cpml_inv_kappa_h_x,
        axis_pos=2,
        axis_neg=0,
    )
    pre_step_adjoint["Hy"] = pre_step_adjoint["Hy"] + adj_hy
    pre_step_adjoint["psi_hy_z"] = pre_step_adjoint["psi_hy_z"] + adj_psi_hy_z
    pre_step_adjoint["psi_hy_x"] = pre_step_adjoint["psi_hy_x"] + adj_psi_hy_x
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex"], adj_d_ex_dz, axis=2, inv_delta=solver.inv_dz_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ez"], adj_d_ez_dx, axis=0, inv_delta=solver.inv_dx_h)

    adj_hz, adj_d_ey_dx, adj_d_ex_dy, adj_psi_hz_x, adj_psi_hz_y = _reverse_magnetic_component_cpml(
        magnetic_output_adjoint["Hz"],
        adjoint_state["psi_hz_x"],
        adjoint_state["psi_hz_y"],
        decay=solver.chz_decay,
        curl=solver.chz_curl,
        b_pos=solver.cpml_b_h_x,
        c_pos=solver.cpml_c_h_x,
        inv_kappa_pos=solver.cpml_inv_kappa_h_x,
        b_neg=solver.cpml_b_h_y,
        c_neg=solver.cpml_c_h_y,
        inv_kappa_neg=solver.cpml_inv_kappa_h_y,
        axis_pos=0,
        axis_neg=1,
    )
    pre_step_adjoint["Hz"] = pre_step_adjoint["Hz"] + adj_hz
    pre_step_adjoint["psi_hz_x"] = pre_step_adjoint["psi_hz_x"] + adj_psi_hz_x
    pre_step_adjoint["psi_hz_y"] = pre_step_adjoint["psi_hz_y"] + adj_psi_hz_y
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ey"], adj_d_ey_dx, axis=0, inv_delta=solver.inv_dx_h)
    _accumulate_forward_diff_adjoint(pre_step_adjoint["Ex"], adj_d_ex_dy, axis=1, inv_delta=solver.inv_dy_h)

    return _ReverseStepResult(
        pre_step_adjoint={name: tensor.detach() for name, tensor in pre_step_adjoint.items()},
        grad_eps_ex=grad_eps_ex.detach(),
        grad_eps_ey=grad_eps_ey.detach(),
        grad_eps_ez=grad_eps_ez.detach(),
        backend="python_reference_cpml",
        magnetic_output_adjoint={name: tensor.detach() for name, tensor in magnetic_output_adjoint.items()},
    )


def reverse_step_tfsf(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    resolved_source_terms=None,
    profiler: _BackwardProfiler | None = None,
):
    tfsf_source_terms = _tfsf_magnetic_source_terms(
        solver,
        forward_state,
        time_value=time_value,
        resolved_source_terms=resolved_source_terms,
    )
    if getattr(solver, "uses_cpml", False):
        base_result = reverse_step_cpml_python_reference(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=tfsf_source_terms,
        )
        backend = "python_reference_tfsf_cpml"
    else:
        base_result = reverse_step_standard_python_reference(
            solver,
            forward_state,
            adjoint_state,
            time_value=time_value,
            eps_ex=eps_ex,
            eps_ey=eps_ey,
            eps_ez=eps_ez,
            resolved_source_terms=tfsf_source_terms,
        )
        backend = "python_reference_tfsf_standard"

    auxiliary_grads = _reverse_tfsf_auxiliary_state_python_reference(
        solver,
        forward_state,
        adjoint_state,
        magnetic_output_adjoint=base_result.magnetic_output_adjoint,
    )
    if not auxiliary_grads:
        return _ReverseStepResult(
            pre_step_adjoint=base_result.pre_step_adjoint,
            grad_eps_ex=base_result.grad_eps_ex,
            grad_eps_ey=base_result.grad_eps_ey,
            grad_eps_ez=base_result.grad_eps_ez,
            backend=backend,
            magnetic_output_adjoint=base_result.magnetic_output_adjoint,
        )

    pre_step_adjoint = {
        name: tensor.detach().clone()
        for name, tensor in base_result.pre_step_adjoint.items()
    }
    for name, grad in auxiliary_grads.items():
        pre_step_adjoint[name] = pre_step_adjoint[name] + grad

    return _ReverseStepResult(
        pre_step_adjoint={name: tensor.detach() for name, tensor in pre_step_adjoint.items()},
        grad_eps_ex=base_result.grad_eps_ex.detach(),
        grad_eps_ey=base_result.grad_eps_ey.detach(),
        grad_eps_ez=base_result.grad_eps_ez.detach(),
        backend=backend,
        magnetic_output_adjoint=base_result.magnetic_output_adjoint,
    )


def reverse_step_torch_vjp(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    chi3_ex=None,
    chi3_ey=None,
    chi3_ez=None,
    chi2_ex=None,
    chi2_ey=None,
    chi2_ez=None,
    tpa_ex=None,
    tpa_ey=None,
    tpa_ez=None,
    profiler: _BackwardProfiler | None = None,
    backend: str = "torch_vjp",
):
    _adjoint = importlib.import_module(__package__)

    chi3_leaves = (chi3_ex, chi3_ey, chi3_ez)
    has_chi3_leaves = all(leaf is not None for leaf in chi3_leaves)
    general_leaves = (chi2_ex, chi2_ey, chi2_ez, tpa_ex, tpa_ey, tpa_ez)
    has_general_leaves = all(leaf is not None for leaf in general_leaves)
    active_profiler = profiler if profiler is not None else _adjoint._BackwardProfiler(enabled=False, device=None)
    with torch.enable_grad():
        with active_profiler.section("state_clone"):
            state_inputs = {
                name: tensor.detach().clone().requires_grad_(True)
                for name, tensor in forward_state.items()
            }
        with active_profiler.section("step_forward"):
            chi3_kwargs = (
                {"chi3_ex": chi3_ex, "chi3_ey": chi3_ey, "chi3_ez": chi3_ez}
                if has_chi3_leaves
                else {}
            )
            general_kwargs = (
                {
                    "chi2_ex": chi2_ex,
                    "chi2_ey": chi2_ey,
                    "chi2_ez": chi2_ez,
                    "tpa_ex": tpa_ex,
                    "tpa_ey": tpa_ey,
                    "tpa_ez": tpa_ez,
                }
                if has_general_leaves
                else {}
            )
            next_state = _adjoint._step_state(
                solver,
                state_inputs,
                time_value=time_value,
                eps_ex=eps_ex,
                eps_ey=eps_ey,
                eps_ez=eps_ez,
                **chi3_kwargs,
                **general_kwargs,
            )
            objective = next_state["Ex"].new_zeros(())
            for name in forward_state:
                objective = objective + torch.sum(next_state[name] * adjoint_state[name])
        with active_profiler.section("step_vjp"):
            leaves = tuple(state_inputs.values()) + (eps_ex, eps_ey, eps_ez)
            if has_chi3_leaves:
                leaves = leaves + chi3_leaves
            if has_general_leaves:
                leaves = leaves + general_leaves
            gradients = torch.autograd.grad(
                objective,
                leaves,
                allow_unused=True,
            )

    state_grads = gradients[: len(state_inputs)]
    pre_step_adjoint = {
        name: _adjoint._safe_grad(grad, forward_state[name]).detach()
        for (name, _), grad in zip(state_inputs.items(), state_grads)
    }
    eps_offset = len(state_inputs)
    cursor = eps_offset + 3
    grad_chi3 = {}
    if has_chi3_leaves:
        grad_chi3 = {
            "grad_chi3_ex": _adjoint._safe_grad(gradients[cursor], chi3_ex).detach(),
            "grad_chi3_ey": _adjoint._safe_grad(gradients[cursor + 1], chi3_ey).detach(),
            "grad_chi3_ez": _adjoint._safe_grad(gradients[cursor + 2], chi3_ez).detach(),
        }
        cursor += 3
    grad_general = {}
    if has_general_leaves:
        grad_general = {
            "grad_chi2_ex": _adjoint._safe_grad(gradients[cursor], chi2_ex).detach(),
            "grad_chi2_ey": _adjoint._safe_grad(gradients[cursor + 1], chi2_ey).detach(),
            "grad_chi2_ez": _adjoint._safe_grad(gradients[cursor + 2], chi2_ez).detach(),
            "grad_tpa_ex": _adjoint._safe_grad(gradients[cursor + 3], tpa_ex).detach(),
            "grad_tpa_ey": _adjoint._safe_grad(gradients[cursor + 4], tpa_ey).detach(),
            "grad_tpa_ez": _adjoint._safe_grad(gradients[cursor + 5], tpa_ez).detach(),
        }
        cursor += 6
    return _adjoint._ReverseStepResult(
        pre_step_adjoint=pre_step_adjoint,
        grad_eps_ex=_adjoint._safe_grad(gradients[eps_offset], eps_ex).detach(),
        grad_eps_ey=_adjoint._safe_grad(gradients[eps_offset + 1], eps_ey).detach(),
        grad_eps_ez=_adjoint._safe_grad(gradients[eps_offset + 2], eps_ez).detach(),
        backend=backend,
        **grad_chi3,
        **grad_general,
    )


def reverse_step_grating_tfsf(
    solver,
    forward_state,
    adjoint_state,
    *,
    time_value,
    eps_ex,
    eps_ey,
    eps_ez,
    profiler: _BackwardProfiler | None = None,
):
    return reverse_step_torch_vjp(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        profiler=profiler,
        backend="python_reference_grating_tfsf",
    )
