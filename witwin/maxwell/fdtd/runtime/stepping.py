from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm
from ..thin_sheet import begin_thin_sheet_step, advance_thin_sheet_magnetic, advance_thin_sheet_electric
from ..copper_foil import (
    initialize_copper_foils, begin_copper_step, correct_copper_magnetic,
    advance_copper_electric, prepare_copper_spectra, accumulate_copper_spectra,
    validate_copper_port_coupling,
)

from ...visualization import visualize_slice
from ..boundary import (
    BOUNDARY_BLOCH,
    BOUNDARY_NONE,
    BOUNDARY_PEC,
    BOUNDARY_PERIODIC,
    BOUNDARY_PML,
    has_complex_fields,
    initialize_boundary_state,
)
from ..boundary.mur import apply_extended_one_way_boundaries, has_extended_one_way_boundaries
from ..excitation.temporal import injection_time
from ..excitation import (
    advance_tfsf_auxiliary_electric,
    advance_tfsf_auxiliary_magnetic,
    apply_tfsf_e_correction,
    apply_tfsf_h_correction,
    inject_electric_surface_source_terms,
    inject_magnetic_surface_source_terms,
    initialize_source_terms,
    initialize_tfsf_state,
    tfsf_incident_is_gpu_driven,
)
from ..ports import (
    _accumulate_embedded_port_observers_gpu,
    accumulate_port_observers,
    apply_port_runtimes,
    complete_port_observer_graph,
    complete_port_spectral_normalization,
    finalize_port_data,
    make_port_observer_runner,
    prepare_port_drive_tables,
    prepare_port_runtimes,
    prepare_port_spectral_accumulators,
)
from ..circuits import (
    finalize_circuit_data,
    prepare_circuit_graph_runners,
    prepare_circuit_time_series,
)
from ..networks import (
    apply_network_runtimes,
    finalize_embedded_networks,
    make_network_runner,
    prepare_network_runtimes,
)
from ..wire import (
    accumulate_wire_monitors,
    complete_wire_monitor_normalization,
    deposit_wire_current,
    finalize_wire_data,
    initialize_wire_runtime,
    prepare_wire_monitors,
    sample_and_update_wire,
)
from .breakdown import advance_breakdown_state, initialize_breakdown_runtime


def iter_cpml_memory_regions(solver, attr_name):
    layout = getattr(solver, "_cpml_memory_layouts", {}).get(attr_name)
    if layout is None:
        return
    axis = int(layout["axis"])
    tensor = getattr(solver, attr_name)
    for region in layout["regions"]:
        length = int(region["length"])
        if length <= 0:
            continue
        local_start = int(region["local_start"])
        global_start = int(region["global_start"])
        psi_region = tensor.narrow(axis, local_start, length)
        offsets = [0, 0, 0]
        offsets[axis] = global_start
        yield {
            "axis": axis,
            "length": length,
            "offsets": tuple(offsets),
            "psi": psi_region,
        }


def cpml_layout_params(solver, attr_name):
    cached = solver._cpml_layout_cache.get(attr_name)
    if cached is not None:
        return cached
    layout = getattr(solver, "_cpml_memory_layouts", {}).get(attr_name)
    if layout is None:
        result = (0, 0, 0)
    else:
        low = next((region for region in layout["regions"] if region["side"] == "low"), None)
        high = next((region for region in layout["regions"] if region["side"] == "high"), None)
        low_length = 0 if low is None else int(low["length"])
        high_start = 0 if high is None else int(high["global_start"])
        high_length = 0 if high is None else int(high["length"])
        result = (low_length, high_start, high_length)
    solver._cpml_layout_cache[attr_name] = result
    return result


def _cpml_memory_attr(attr_name, *, imag=False):
    return f"{attr_name}_imag" if imag else attr_name


def _cpml_memory_tensor(solver, attr_name, *, imag=False):
    return getattr(solver, _cpml_memory_attr(attr_name, imag=imag))


def update_magnetic_fields_cpml_dense(solver, hx, hy, hz, ex, ey, ez, *, imag=False):
    solver.fdtd_module.updateMagneticFieldHx3D(
        Hx=hx,
        Ey=ey,
        Ez=ez,
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        PsiHxY=_cpml_memory_tensor(solver, "psi_hx_y", imag=imag),
        PsiHxZ=_cpml_memory_tensor(solver, "psi_hx_z", imag=imag),
        InvKappaHxY=solver.cpml_inv_kappa_h_y,
        ByHxY=solver.cpml_b_h_y,
        CyHxY=solver.cpml_c_h_y,
        InvKappaHxZ=solver.cpml_inv_kappa_h_z,
        ByHxZ=solver.cpml_b_h_z,
        CyHxZ=solver.cpml_c_h_z,
        invDy=solver.inv_dy_h,
        invDz=solver.inv_dz_h,
        uniformDecay=solver._coefficient_uniformity["chx_decay"],
        uniformCurl=solver._coefficient_uniformity["chx_curl"],
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHy3D(
        Hy=hy,
        Ex=ex,
        Ez=ez,
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        PsiHyX=_cpml_memory_tensor(solver, "psi_hy_x", imag=imag),
        PsiHyZ=_cpml_memory_tensor(solver, "psi_hy_z", imag=imag),
        InvKappaHyX=solver.cpml_inv_kappa_h_x,
        ByHyX=solver.cpml_b_h_x,
        CyHyX=solver.cpml_c_h_x,
        InvKappaHyZ=solver.cpml_inv_kappa_h_z,
        ByHyZ=solver.cpml_b_h_z,
        CyHyZ=solver.cpml_c_h_z,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
        uniformDecay=solver._coefficient_uniformity["chy_decay"],
        uniformCurl=solver._coefficient_uniformity["chy_curl"],
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHz3D(
        Hz=hz,
        Ex=ex,
        Ey=ey,
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        PsiHzX=_cpml_memory_tensor(solver, "psi_hz_x", imag=imag),
        PsiHzY=_cpml_memory_tensor(solver, "psi_hz_y", imag=imag),
        InvKappaHzX=solver.cpml_inv_kappa_h_x,
        ByHzX=solver.cpml_b_h_x,
        CyHzX=solver.cpml_c_h_x,
        InvKappaHzY=solver.cpml_inv_kappa_h_y,
        ByHzY=solver.cpml_b_h_y,
        CyHzY=solver.cpml_c_h_y,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
        uniformDecay=solver._coefficient_uniformity["chz_decay"],
        uniformCurl=solver._coefficient_uniformity["chz_curl"],
    ).launchRaw()


def update_magnetic_fields_cpml_compressed(solver, hx, hy, hz, ex, ey, ez, *, imag=False):
    psi_hx_y_attr = _cpml_memory_attr("psi_hx_y", imag=imag)
    psi_hx_z_attr = _cpml_memory_attr("psi_hx_z", imag=imag)
    psi_hy_x_attr = _cpml_memory_attr("psi_hy_x", imag=imag)
    psi_hy_z_attr = _cpml_memory_attr("psi_hy_z", imag=imag)
    psi_hz_x_attr = _cpml_memory_attr("psi_hz_x", imag=imag)
    psi_hz_y_attr = _cpml_memory_attr("psi_hz_y", imag=imag)
    psi_hx_y_low, psi_hx_y_high_start, psi_hx_y_high = cpml_layout_params(solver, psi_hx_y_attr)
    psi_hx_z_low, psi_hx_z_high_start, psi_hx_z_high = cpml_layout_params(solver, psi_hx_z_attr)
    psi_hy_x_low, psi_hy_x_high_start, psi_hy_x_high = cpml_layout_params(solver, psi_hy_x_attr)
    psi_hy_z_low, psi_hy_z_high_start, psi_hy_z_high = cpml_layout_params(solver, psi_hy_z_attr)
    psi_hz_x_low, psi_hz_x_high_start, psi_hz_x_high = cpml_layout_params(solver, psi_hz_x_attr)
    psi_hz_y_low, psi_hz_y_high_start, psi_hz_y_high = cpml_layout_params(solver, psi_hz_y_attr)

    solver.fdtd_module.updateMagneticFieldHxCpmlCompressed3D(
        Hx=hx,
        Ey=ey,
        Ez=ez,
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        PsiHxY=_cpml_memory_tensor(solver, "psi_hx_y", imag=imag),
        PsiHxZ=_cpml_memory_tensor(solver, "psi_hx_z", imag=imag),
        InvKappaHxY=solver.cpml_inv_kappa_h_y,
        ByHxY=solver.cpml_b_h_y,
        CyHxY=solver.cpml_c_h_y,
        InvKappaHxZ=solver.cpml_inv_kappa_h_z,
        ByHxZ=solver.cpml_b_h_z,
        CyHxZ=solver.cpml_c_h_z,
        invDy=solver.inv_dy_h,
        invDz=solver.inv_dz_h,
        psiHxYLowLength=psi_hx_y_low,
        psiHxYHighStart=psi_hx_y_high_start,
        psiHxYHighLength=psi_hx_y_high,
        psiHxZLowLength=psi_hx_z_low,
        psiHxZHighStart=psi_hx_z_high_start,
        psiHxZHighLength=psi_hx_z_high,
        uniformDecay=solver._coefficient_uniformity["chx_decay"],
        uniformCurl=solver._coefficient_uniformity["chx_curl"],
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHyCpmlCompressed3D(
        Hy=hy,
        Ex=ex,
        Ez=ez,
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        PsiHyX=_cpml_memory_tensor(solver, "psi_hy_x", imag=imag),
        PsiHyZ=_cpml_memory_tensor(solver, "psi_hy_z", imag=imag),
        InvKappaHyX=solver.cpml_inv_kappa_h_x,
        ByHyX=solver.cpml_b_h_x,
        CyHyX=solver.cpml_c_h_x,
        InvKappaHyZ=solver.cpml_inv_kappa_h_z,
        ByHyZ=solver.cpml_b_h_z,
        CyHyZ=solver.cpml_c_h_z,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
        psiHyXLowLength=psi_hy_x_low,
        psiHyXHighStart=psi_hy_x_high_start,
        psiHyXHighLength=psi_hy_x_high,
        psiHyZLowLength=psi_hy_z_low,
        psiHyZHighStart=psi_hy_z_high_start,
        psiHyZHighLength=psi_hy_z_high,
        uniformDecay=solver._coefficient_uniformity["chy_decay"],
        uniformCurl=solver._coefficient_uniformity["chy_curl"],
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHzCpmlCompressed3D(
        Hz=hz,
        Ex=ex,
        Ey=ey,
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        PsiHzX=_cpml_memory_tensor(solver, "psi_hz_x", imag=imag),
        PsiHzY=_cpml_memory_tensor(solver, "psi_hz_y", imag=imag),
        InvKappaHzX=solver.cpml_inv_kappa_h_x,
        ByHzX=solver.cpml_b_h_x,
        CyHzX=solver.cpml_c_h_x,
        InvKappaHzY=solver.cpml_inv_kappa_h_y,
        ByHzY=solver.cpml_b_h_y,
        CyHzY=solver.cpml_c_h_y,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
        psiHzXLowLength=psi_hz_x_low,
        psiHzXHighStart=psi_hz_x_high_start,
        psiHzXHighLength=psi_hz_x_high,
        psiHzYLowLength=psi_hz_y_low,
        psiHzYHighStart=psi_hz_y_high_start,
        psiHzYHighLength=psi_hz_y_high,
        uniformDecay=solver._coefficient_uniformity["chz_decay"],
        uniformCurl=solver._coefficient_uniformity["chz_curl"],
    ).launchRaw()


def update_magnetic_fields_cpml(solver, hx, hy, hz, ex, ey, ez, *, imag=False):
    if solver._cpml_memory_mode == "dense":
        update_magnetic_fields_cpml_dense(solver, hx, hy, hz, ex, ey, ez, imag=imag)
        return
    update_magnetic_fields_cpml_compressed(solver, hx, hy, hz, ex, ey, ez, imag=imag)


def update_magnetic_fields_standard(solver, hx, hy, hz, ex, ey, ez):
    solver.fdtd_module.updateMagneticFieldHxStandard3D(
        Hx=hx,
        Ey=ey,
        Ez=ez,
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        invDy=solver.inv_dy_h,
        invDz=solver.inv_dz_h,
        uniformDecay=solver._coefficient_uniformity["chx_decay"],
        uniformCurl=solver._coefficient_uniformity["chx_curl"],
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHyStandard3D(
        Hy=hy,
        Ex=ex,
        Ez=ez,
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
        uniformDecay=solver._coefficient_uniformity["chy_decay"],
        uniformCurl=solver._coefficient_uniformity["chy_curl"],
    ).launchRaw()
    solver.fdtd_module.updateMagneticFieldHzStandard3D(
        Hz=hz,
        Ex=ex,
        Ey=ey,
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
        uniformDecay=solver._coefficient_uniformity["chz_decay"],
        uniformCurl=solver._coefficient_uniformity["chz_curl"],
    ).launchRaw()


def update_magnetic_fields(solver, hx, hy, hz, ex, ey, ez, *, imag=False):
    if getattr(solver, "_core_runtime", None) is not None:
        solver._core_runtime.magnetic()
        return
    if solver.uses_cpml:
        update_magnetic_fields_cpml(solver, hx, hy, hz, ex, ey, ez, imag=imag)
    else:
        update_magnetic_fields_standard(solver, hx, hy, hz, ex, ey, ez)


def _electric_curl_tensors(solver):
    if getattr(solver, "nonlinear_enabled", False):
        return solver.cex_curl_dynamic, solver.cey_curl_dynamic, solver.cez_curl_dynamic
    return solver.cex_curl, solver.cey_curl, solver.cez_curl


def _electric_decay_tensors(solver):
    # The general nonlinear kernel (chi2 / field-dependent conductivity)
    # rewrites the decay coefficients per step; pure-chi3 Kerr keeps the
    # static decay tensors.
    if getattr(solver, "nonlinear_general_enabled", False):
        return solver.cex_decay_dynamic, solver.cey_decay_dynamic, solver.cez_decay_dynamic
    return solver.cex_decay, solver.cey_decay, solver.cez_decay


def update_electric_fields_cpml_dense(solver, ex, ey, ez, hx, hy, hz):
    ex_curl, ey_curl, ez_curl = _electric_curl_tensors(solver)
    ex_decay, ey_decay, ez_decay = _electric_decay_tensors(solver)
    solver.fdtd_module.updateElectricFieldExCpml3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        PsiExY=solver.psi_ex_y,
        PsiExZ=solver.psi_ex_z,
        InvKappaExY=solver.cpml_inv_kappa_e_y,
        BExY=solver.cpml_b_e_y,
        CExY=solver.cpml_c_e_y,
        InvKappaExZ=solver.cpml_inv_kappa_e_z,
        BExZ=solver.cpml_b_e_z,
        CExZ=solver.cpml_c_e_z,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        uniformDecay=None if ex_decay is not solver.cex_decay else solver._coefficient_uniformity["cex_decay"],
        uniformCurl=None if ex_curl is not solver.cex_curl else solver._coefficient_uniformity["cex_curl"],
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyCpml3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        PsiEyX=solver.psi_ey_x,
        PsiEyZ=solver.psi_ey_z,
        InvKappaEyX=solver.cpml_inv_kappa_e_x,
        BEyX=solver.cpml_b_e_x,
        CEyX=solver.cpml_c_e_x,
        InvKappaEyZ=solver.cpml_inv_kappa_e_z,
        BEyZ=solver.cpml_b_e_z,
        CEyZ=solver.cpml_c_e_z,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        uniformDecay=None if ey_decay is not solver.cey_decay else solver._coefficient_uniformity["cey_decay"],
        uniformCurl=None if ey_curl is not solver.cey_curl else solver._coefficient_uniformity["cey_curl"],
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzCpml3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        PsiEzX=solver.psi_ez_x,
        PsiEzY=solver.psi_ez_y,
        InvKappaEzX=solver.cpml_inv_kappa_e_x,
        BEzX=solver.cpml_b_e_x,
        CEzX=solver.cpml_c_e_x,
        InvKappaEzY=solver.cpml_inv_kappa_e_y,
        BEzY=solver.cpml_b_e_y,
        CEzY=solver.cpml_c_e_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        uniformDecay=None if ez_decay is not solver.cez_decay else solver._coefficient_uniformity["cez_decay"],
        uniformCurl=None if ez_curl is not solver.cez_curl else solver._coefficient_uniformity["cez_curl"],
    ).launchRaw()


def update_electric_fields_cpml_compressed(solver, ex, ey, ez, hx, hy, hz):
    psi_ex_y_low, psi_ex_y_high_start, psi_ex_y_high = cpml_layout_params(solver, "psi_ex_y")
    psi_ex_z_low, psi_ex_z_high_start, psi_ex_z_high = cpml_layout_params(solver, "psi_ex_z")
    psi_ey_x_low, psi_ey_x_high_start, psi_ey_x_high = cpml_layout_params(solver, "psi_ey_x")
    psi_ey_z_low, psi_ey_z_high_start, psi_ey_z_high = cpml_layout_params(solver, "psi_ey_z")
    psi_ez_x_low, psi_ez_x_high_start, psi_ez_x_high = cpml_layout_params(solver, "psi_ez_x")
    psi_ez_y_low, psi_ez_y_high_start, psi_ez_y_high = cpml_layout_params(solver, "psi_ez_y")
    ex_curl, ey_curl, ez_curl = _electric_curl_tensors(solver)
    ex_decay, ey_decay, ez_decay = _electric_decay_tensors(solver)

    solver.fdtd_module.updateElectricFieldExCpmlCompressed3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        PsiExY=solver.psi_ex_y,
        PsiExZ=solver.psi_ex_z,
        InvKappaExY=solver.cpml_inv_kappa_e_y,
        BExY=solver.cpml_b_e_y,
        CExY=solver.cpml_c_e_y,
        InvKappaExZ=solver.cpml_inv_kappa_e_z,
        BExZ=solver.cpml_b_e_z,
        CExZ=solver.cpml_c_e_z,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        psiExYLowLength=psi_ex_y_low,
        psiExYHighStart=psi_ex_y_high_start,
        psiExYHighLength=psi_ex_y_high,
        psiExZLowLength=psi_ex_z_low,
        psiExZHighStart=psi_ex_z_high_start,
        psiExZHighLength=psi_ex_z_high,
        uniformDecay=None if ex_decay is not solver.cex_decay else solver._coefficient_uniformity["cex_decay"],
        uniformCurl=None if ex_curl is not solver.cex_curl else solver._coefficient_uniformity["cex_curl"],
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyCpmlCompressed3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        PsiEyX=solver.psi_ey_x,
        PsiEyZ=solver.psi_ey_z,
        InvKappaEyX=solver.cpml_inv_kappa_e_x,
        BEyX=solver.cpml_b_e_x,
        CEyX=solver.cpml_c_e_x,
        InvKappaEyZ=solver.cpml_inv_kappa_e_z,
        BEyZ=solver.cpml_b_e_z,
        CEyZ=solver.cpml_c_e_z,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        psiEyXLowLength=psi_ey_x_low,
        psiEyXHighStart=psi_ey_x_high_start,
        psiEyXHighLength=psi_ey_x_high,
        psiEyZLowLength=psi_ey_z_low,
        psiEyZHighStart=psi_ey_z_high_start,
        psiEyZHighLength=psi_ey_z_high,
        uniformDecay=None if ey_decay is not solver.cey_decay else solver._coefficient_uniformity["cey_decay"],
        uniformCurl=None if ey_curl is not solver.cey_curl else solver._coefficient_uniformity["cey_curl"],
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzCpmlCompressed3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        PsiEzX=solver.psi_ez_x,
        PsiEzY=solver.psi_ez_y,
        InvKappaEzX=solver.cpml_inv_kappa_e_x,
        BEzX=solver.cpml_b_e_x,
        CEzX=solver.cpml_c_e_x,
        InvKappaEzY=solver.cpml_inv_kappa_e_y,
        BEzY=solver.cpml_b_e_y,
        CEzY=solver.cpml_c_e_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        psiEzXLowLength=psi_ez_x_low,
        psiEzXHighStart=psi_ez_x_high_start,
        psiEzXHighLength=psi_ez_x_high,
        psiEzYLowLength=psi_ez_y_low,
        psiEzYHighStart=psi_ez_y_high_start,
        psiEzYHighLength=psi_ez_y_high,
        uniformDecay=None if ez_decay is not solver.cez_decay else solver._coefficient_uniformity["cez_decay"],
        uniformCurl=None if ez_curl is not solver.cez_curl else solver._coefficient_uniformity["cez_curl"],
    ).launchRaw()


def update_electric_fields_cpml(solver, ex, ey, ez, hx, hy, hz):
    if solver._cpml_memory_mode == "dense":
        update_electric_fields_cpml_dense(solver, ex, ey, ez, hx, hy, hz)
        return
    update_electric_fields_cpml_compressed(solver, ex, ey, ez, hx, hy, hz)


def electric_update_support(solver, component):
    """Return half-open per-axis bounds where the E curl update can evolve.

    Standard and CPML kernels in ``cuda/kernels/electric.cu`` test
    ``boundary_inactive`` and ``boundary_pec`` on the two derivative axes:
    none/PML rows are frozen and PEC rows are zeroed (also enforced by the
    tangential PEC clamp). Mur maps to NONE: its separate post-source ABC
    overwrites these rows rather than evolving an injected current there.
    PMC uses mirrored derivatives; periodic/Bloch use wrapped derivatives, so
    their wall rows stay active. Mixed Bloch/CPML restores the same PML/PEC
    zeros. The component axis is half-staggered and has no wall-node exclusion.
    """
    if component not in ("Ex", "Ey", "Ez"):
        raise ValueError(f"Expected an electric field component, got {component!r}.")
    support = []
    for axis, size in zip("xyz", getattr(solver, component).shape):
        lo, hi = 0, int(size)
        if axis != component[-1]:
            if getattr(solver, f"boundary_{axis}_low_code") in (BOUNDARY_NONE, BOUNDARY_PML, BOUNDARY_PEC):
                lo += 1
            if getattr(solver, f"boundary_{axis}_high_code") in (BOUNDARY_NONE, BOUNDARY_PML, BOUNDARY_PEC):
                hi -= 1
        support.append((lo, hi))
    return tuple(support)


def update_electric_fields_standard(solver, ex, ey, ez, hx, hy, hz):
    ex_curl, ey_curl, ez_curl = _electric_curl_tensors(solver)
    ex_decay, ey_decay, ez_decay = _electric_decay_tensors(solver)
    solver.fdtd_module.updateElectricFieldExStandard3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        uniformDecay=None if ex_decay is not solver.cex_decay else solver._coefficient_uniformity["cex_decay"],
        uniformCurl=None if ex_curl is not solver.cex_curl else solver._coefficient_uniformity["cex_curl"],
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyStandard3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        uniformDecay=None if ey_decay is not solver.cey_decay else solver._coefficient_uniformity["cey_decay"],
        uniformCurl=None if ey_curl is not solver.cey_curl else solver._coefficient_uniformity["cey_curl"],
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzStandard3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        uniformDecay=None if ez_decay is not solver.cez_decay else solver._coefficient_uniformity["cez_decay"],
        uniformCurl=None if ez_curl is not solver.cez_curl else solver._coefficient_uniformity["cez_curl"],
    ).launchRaw()


def update_electric_fields_modulated_standard(solver, ex, ey, ez, hx, hy, hz, time_value):
    # Composing modulation with an instantaneous nonlinearity: the modulation
    # scales the same field-dependent decay/curl coefficients the nonlinear kernel
    # rewrote this step (static tensors when no nonlinearity is present).
    ex_curl, ey_curl, ez_curl = _electric_curl_tensors(solver)
    ex_decay, ey_decay, ez_decay = _electric_decay_tensors(solver)
    solver.fdtd_module.updateElectricFieldExModulated3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        ModCos=solver.mod_cos_Ex,
        ModSin=solver.mod_sin_Ex,
        ModOmega=solver.mod_omega_Ex,
        ModulationTime=solver._modulation_time,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyModulated3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        ModCos=solver.mod_cos_Ey,
        ModSin=solver.mod_sin_Ey,
        ModOmega=solver.mod_omega_Ey,
        ModulationTime=solver._modulation_time,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzModulated3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        ModCos=solver.mod_cos_Ez,
        ModSin=solver.mod_sin_Ez,
        ModOmega=solver.mod_omega_Ez,
        ModulationTime=solver._modulation_time,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
    ).launchRaw()


def update_electric_fields_modulated_cpml(solver, ex, ey, ez, hx, hy, hz, time_value):
    if solver._cpml_memory_mode == "dense":
        update_electric_fields_modulated_cpml_dense(solver, ex, ey, ez, hx, hy, hz, time_value)
        return
    update_electric_fields_modulated_cpml_compressed(solver, ex, ey, ez, hx, hy, hz, time_value)


def update_electric_fields_modulated_cpml_dense(solver, ex, ey, ez, hx, hy, hz, time_value):
    ex_curl, ey_curl, ez_curl = _electric_curl_tensors(solver)
    ex_decay, ey_decay, ez_decay = _electric_decay_tensors(solver)
    solver.fdtd_module.updateElectricFieldExCpmlModulated3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        ModCos=solver.mod_cos_Ex,
        ModSin=solver.mod_sin_Ex,
        ModOmega=solver.mod_omega_Ex,
        ModulationTime=solver._modulation_time,
        PsiExY=solver.psi_ex_y,
        PsiExZ=solver.psi_ex_z,
        InvKappaExY=solver.cpml_inv_kappa_e_y,
        BExY=solver.cpml_b_e_y,
        CExY=solver.cpml_c_e_y,
        InvKappaExZ=solver.cpml_inv_kappa_e_z,
        BExZ=solver.cpml_b_e_z,
        CExZ=solver.cpml_c_e_z,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyCpmlModulated3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        ModCos=solver.mod_cos_Ey,
        ModSin=solver.mod_sin_Ey,
        ModOmega=solver.mod_omega_Ey,
        ModulationTime=solver._modulation_time,
        PsiEyX=solver.psi_ey_x,
        PsiEyZ=solver.psi_ey_z,
        InvKappaEyX=solver.cpml_inv_kappa_e_x,
        BEyX=solver.cpml_b_e_x,
        CEyX=solver.cpml_c_e_x,
        InvKappaEyZ=solver.cpml_inv_kappa_e_z,
        BEyZ=solver.cpml_b_e_z,
        CEyZ=solver.cpml_c_e_z,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzCpmlModulated3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        ModCos=solver.mod_cos_Ez,
        ModSin=solver.mod_sin_Ez,
        ModOmega=solver.mod_omega_Ez,
        ModulationTime=solver._modulation_time,
        PsiEzX=solver.psi_ez_x,
        PsiEzY=solver.psi_ez_y,
        InvKappaEzX=solver.cpml_inv_kappa_e_x,
        BEzX=solver.cpml_b_e_x,
        CEzX=solver.cpml_c_e_x,
        InvKappaEzY=solver.cpml_inv_kappa_e_y,
        BEzY=solver.cpml_b_e_y,
        CEzY=solver.cpml_c_e_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
    ).launchRaw()


def update_electric_fields_modulated_cpml_compressed(solver, ex, ey, ez, hx, hy, hz, time_value):
    # Compressed (slab) psi layout: thread the per-side narrow()/offset bookkeeping
    # (cpml_layout_params) through the modulated kernels, exactly as the plain
    # compressed CPML update does, with the modulation phase factors applied on top.
    psi_ex_y_low, psi_ex_y_high_start, psi_ex_y_high = cpml_layout_params(solver, "psi_ex_y")
    psi_ex_z_low, psi_ex_z_high_start, psi_ex_z_high = cpml_layout_params(solver, "psi_ex_z")
    psi_ey_x_low, psi_ey_x_high_start, psi_ey_x_high = cpml_layout_params(solver, "psi_ey_x")
    psi_ey_z_low, psi_ey_z_high_start, psi_ey_z_high = cpml_layout_params(solver, "psi_ey_z")
    psi_ez_x_low, psi_ez_x_high_start, psi_ez_x_high = cpml_layout_params(solver, "psi_ez_x")
    psi_ez_y_low, psi_ez_y_high_start, psi_ez_y_high = cpml_layout_params(solver, "psi_ez_y")
    ex_curl, ey_curl, ez_curl = _electric_curl_tensors(solver)
    ex_decay, ey_decay, ez_decay = _electric_decay_tensors(solver)
    solver.fdtd_module.updateElectricFieldExCpmlModulatedCompressed3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        ModCos=solver.mod_cos_Ex,
        ModSin=solver.mod_sin_Ex,
        ModOmega=solver.mod_omega_Ex,
        ModulationTime=solver._modulation_time,
        PsiExY=solver.psi_ex_y,
        PsiExZ=solver.psi_ex_z,
        InvKappaExY=solver.cpml_inv_kappa_e_y,
        BExY=solver.cpml_b_e_y,
        CExY=solver.cpml_c_e_y,
        InvKappaExZ=solver.cpml_inv_kappa_e_z,
        BExZ=solver.cpml_b_e_z,
        CExZ=solver.cpml_c_e_z,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        psiExYLowLength=psi_ex_y_low,
        psiExYHighStart=psi_ex_y_high_start,
        psiExYHighLength=psi_ex_y_high,
        psiExZLowLength=psi_ex_z_low,
        psiExZHighStart=psi_ex_z_high_start,
        psiExZHighLength=psi_ex_z_high,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyCpmlModulatedCompressed3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        ModCos=solver.mod_cos_Ey,
        ModSin=solver.mod_sin_Ey,
        ModOmega=solver.mod_omega_Ey,
        ModulationTime=solver._modulation_time,
        PsiEyX=solver.psi_ey_x,
        PsiEyZ=solver.psi_ey_z,
        InvKappaEyX=solver.cpml_inv_kappa_e_x,
        BEyX=solver.cpml_b_e_x,
        CEyX=solver.cpml_c_e_x,
        InvKappaEyZ=solver.cpml_inv_kappa_e_z,
        BEyZ=solver.cpml_b_e_z,
        CEyZ=solver.cpml_c_e_z,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
        psiEyXLowLength=psi_ey_x_low,
        psiEyXHighStart=psi_ey_x_high_start,
        psiEyXHighLength=psi_ey_x_high,
        psiEyZLowLength=psi_ey_z_low,
        psiEyZHighStart=psi_ey_z_high_start,
        psiEyZHighLength=psi_ey_z_high,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzCpmlModulatedCompressed3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        ModCos=solver.mod_cos_Ez,
        ModSin=solver.mod_sin_Ez,
        ModOmega=solver.mod_omega_Ez,
        ModulationTime=solver._modulation_time,
        PsiEzX=solver.psi_ez_x,
        PsiEzY=solver.psi_ez_y,
        InvKappaEzX=solver.cpml_inv_kappa_e_x,
        BEzX=solver.cpml_b_e_x,
        CEzX=solver.cpml_c_e_x,
        InvKappaEzY=solver.cpml_inv_kappa_e_y,
        BEzY=solver.cpml_b_e_y,
        CEzY=solver.cpml_c_e_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        psiEzXLowLength=psi_ez_x_low,
        psiEzXHighStart=psi_ez_x_high_start,
        psiEzXHighLength=psi_ez_x_high,
        psiEzYLowLength=psi_ez_y_low,
        psiEzYHighStart=psi_ez_y_high_start,
        psiEzYHighLength=psi_ez_y_high,
    ).launchRaw()


def update_electric_fields(solver, ex, ey, ez, hx, hy, hz, *, time_value=None):
    if getattr(solver, "_core_runtime", None) is not None:
        solver._core_runtime.electric()
        return
    if getattr(solver, "modulation_enabled", False):
        if time_value is None:
            raise RuntimeError(
                "The time-modulated electric update requires the current step time_value."
            )
        if solver.uses_cpml:
            update_electric_fields_modulated_cpml(solver, ex, ey, ez, hx, hy, hz, time_value)
        else:
            update_electric_fields_modulated_standard(solver, ex, ey, ez, hx, hy, hz, time_value)
        return
    if solver.uses_cpml:
        update_electric_fields_cpml(solver, ex, ey, ez, hx, hy, hz)
    else:
        update_electric_fields_standard(solver, ex, ey, ez, hx, hy, hz)


def _full_aniso_periodic_flags(solver):
    return (
        int(solver.boundary_x_low_code == BOUNDARY_PERIODIC and solver.boundary_x_high_code == BOUNDARY_PERIODIC),
        int(solver.boundary_y_low_code == BOUNDARY_PERIODIC and solver.boundary_y_high_code == BOUNDARY_PERIODIC),
        int(solver.boundary_z_low_code == BOUNDARY_PERIODIC and solver.boundary_z_high_code == BOUNDARY_PERIODIC),
    )


def apply_full_aniso_corrections(solver):
    """Add the off-diagonal anisotropic coupling terms to the E update.

    Each kernel sums the four independently weighted off-axis curl(H) samples.
    The weights scatter a shared node inverse tensor through reciprocal Yee
    gathers, scaled by ``dt/eps0``, and vanish outside anisotropic structures. Periodic axes
    wrap the collocation stencil; other boundaries skip out-of-range samples.
    Under CPML the coordinate-stretched variant is used so the coupling is
    absorbed consistently where an anisotropic structure reaches the boundary.
    """
    if getattr(solver, "uses_cpml", False):
        apply_full_aniso_corrections_cpml(solver)
        return
    periodic_x, periodic_y, periodic_z = _full_aniso_periodic_flags(solver)
    solver.fdtd_module.updateElectricFieldExFullAniso3D(
        Ex=solver.Ex,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        CoeffY=solver.cex_aniso_y,
        CoeffZ=solver.cex_aniso_z,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyFullAniso3D(
        Ey=solver.Ey,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        CoeffX=solver.cey_aniso_x,
        CoeffZ=solver.cey_aniso_z,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzFullAniso3D(
        Ez=solver.Ez,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        CoeffX=solver.cez_aniso_x,
        CoeffY=solver.cez_aniso_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw()


def apply_full_aniso_corrections_cpml(solver):
    """CPML-consistent off-diagonal anisotropic coupling for absorber overlap.

    Each spatial derivative that enters the collocated off-axis curl(H) is
    coordinate-stretched with its own recursive-convolution memory owned by the
    target E edge. The two transverse directions use the E-field (node) CPML
    profiles at the edge index; the edge's own direction sits on a half point and
    uses the H-field (half) profile. Outside the absorber the profiles are neutral
    and the psi buffers stay zero, so the correction matches the raw update.
    """
    periodic_x, periodic_y, periodic_z = _full_aniso_periodic_flags(solver)
    solver.fdtd_module.updateElectricFieldExFullAnisoCpml3D(
        Ex=solver.Ex,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        CoeffY=solver.cex_aniso_y,
        CoeffZ=solver.cex_aniso_z,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        InvKappaX=solver.cpml_inv_kappa_h_x,
        BX=solver.cpml_b_h_x,
        CX=solver.cpml_c_h_x,
        InvKappaY=solver.cpml_inv_kappa_e_y,
        BY=solver.cpml_b_e_y,
        CY=solver.cpml_c_e_y,
        InvKappaZ=solver.cpml_inv_kappa_e_z,
        BZ=solver.cpml_b_e_z,
        CZ=solver.cpml_c_e_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
        PsiX=solver.psi_ex_aniso_x,
        PsiY=solver.psi_ex_aniso_y,
        PsiZ=solver.psi_ex_aniso_z,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyFullAnisoCpml3D(
        Ey=solver.Ey,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        CoeffX=solver.cey_aniso_x,
        CoeffZ=solver.cey_aniso_z,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        InvKappaX=solver.cpml_inv_kappa_e_x,
        BX=solver.cpml_b_e_x,
        CX=solver.cpml_c_e_x,
        InvKappaY=solver.cpml_inv_kappa_h_y,
        BY=solver.cpml_b_h_y,
        CY=solver.cpml_c_h_y,
        InvKappaZ=solver.cpml_inv_kappa_e_z,
        BZ=solver.cpml_b_e_z,
        CZ=solver.cpml_c_e_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
        PsiX=solver.psi_ey_aniso_x,
        PsiY=solver.psi_ey_aniso_y,
        PsiZ=solver.psi_ey_aniso_z,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzFullAnisoCpml3D(
        Ez=solver.Ez,
        Hx=solver.Hx,
        Hy=solver.Hy,
        Hz=solver.Hz,
        CoeffX=solver.cez_aniso_x,
        CoeffY=solver.cez_aniso_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        InvKappaX=solver.cpml_inv_kappa_e_x,
        BX=solver.cpml_b_e_x,
        CX=solver.cpml_c_e_x,
        InvKappaY=solver.cpml_inv_kappa_e_y,
        BY=solver.cpml_b_e_y,
        CY=solver.cpml_c_e_y,
        InvKappaZ=solver.cpml_inv_kappa_h_z,
        BZ=solver.cpml_b_h_z,
        CZ=solver.cpml_c_h_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
        PsiX=solver.psi_ez_aniso_x,
        PsiY=solver.psi_ez_aniso_y,
        PsiZ=solver.psi_ez_aniso_z,
    ).launchRaw()


def _full_aniso_conductive(solver) -> bool:
    return getattr(solver, "full_aniso_enabled", False) and getattr(solver, "_aniso_cond_current", None) is not None


def capture_aniso_conduction_currents(solver):
    """Snapshot the conduction current sigma . E^n before the electric update.

    For a full (off-diagonal) anisotropic conductive medium the exact semi-implicit
    update is ``E^{n+1} = E^n + B . (curl H - sigma . E^n)`` with
    ``B = dt (eps_inf + dt/2 diag(sigma))^-1``. The diagonal ``-B_ii sigma_i E_i^n``
    term is folded into the decay coefficient, but the off-diagonal
    ``-B_ij sigma_j E_j^n`` (i != j) terms need the pre-update field, which the base
    update overwrites, so they are captured here into ``_aniso_cond_current``.
    """
    if not _full_aniso_conductive(solver):
        return
    currents = solver._aniso_cond_current
    solver.fdtd_module.captureAnisoConductionCurrent3D(
        SigmaX=solver.sigma_e_Ex,
        SigmaY=solver.sigma_e_Ey,
        SigmaZ=solver.sigma_e_Ez,
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        Jx=currents["Ex"],
        Jy=currents["Ey"],
        Jz=currents["Ez"],
    ).launchRaw()


def apply_full_aniso_conduction(solver):
    """Subtract the off-diagonal conduction current through the inverse tensor.

    Applies the same four reciprocal cross-component weights to ``sigma E^n``
    as to curl(H). The conduction current is pointwise
    (no spatial derivative), so it needs no CPML coordinate stretch and reuses the
    plain off-diagonal current kernels for every boundary configuration.
    """
    if not _full_aniso_conductive(solver):
        return
    currents = solver._aniso_cond_current
    periodic_x, periodic_y, periodic_z = _full_aniso_periodic_flags(solver)
    solver.fdtd_module.applyAnisoOffdiagCurrentEx3D(
        Ex=solver.Ex,
        Jy=currents["Ey"],
        Jz=currents["Ez"],
        CoeffY=solver.cex_aniso_y,
        CoeffZ=solver.cex_aniso_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw()
    solver.fdtd_module.applyAnisoOffdiagCurrentEy3D(
        Ey=solver.Ey,
        Jx=currents["Ex"],
        Jz=currents["Ez"],
        CoeffX=solver.cey_aniso_x,
        CoeffZ=solver.cey_aniso_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw()
    solver.fdtd_module.applyAnisoOffdiagCurrentEz3D(
        Ez=solver.Ez,
        Jx=currents["Ex"],
        Jy=currents["Ey"],
        CoeffX=solver.cez_aniso_x,
        CoeffY=solver.cez_aniso_y,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw()


def update_electric_fields_bloch(solver):
    if getattr(solver, "nonlinear_enabled", False):
        raise NotImplementedError(
            "FDTD instantaneous nonlinear media cannot evolve on the complex Bloch fields: the "
            "|E|^2 Kerr/chi2/TPA polarization couples the single-wavevector Bloch field to the 2k/3k "
            "harmonic sectors a complex-Bloch run does not carry, so the nonlinear response is not "
            "closed on the real update kernels. Use a real-field boundary (PML/PEC/PMC/periodic)."
        )
    solver.fdtd_module.updateElectricFieldExBloch3D(
        ExReal=solver.Ex,
        ExImag=solver.Ex_imag,
        HyReal=solver.Hy,
        HyImag=solver.Hy_imag,
        HzReal=solver.Hz,
        HzImag=solver.Hz_imag,
        ExDecay=solver.cex_decay,
        ExCurl=solver.cex_curl,
        phaseCosY=solver.boundary_phase_cos[1],
        phaseSinY=solver.boundary_phase_sin[1],
        phaseCosZ=solver.boundary_phase_cos[2],
        phaseSinZ=solver.boundary_phase_sin[2],
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyBloch3D(
        EyReal=solver.Ey,
        EyImag=solver.Ey_imag,
        HxReal=solver.Hx,
        HxImag=solver.Hx_imag,
        HzReal=solver.Hz,
        HzImag=solver.Hz_imag,
        EyDecay=solver.cey_decay,
        EyCurl=solver.cey_curl,
        phaseCosX=solver.boundary_phase_cos[0],
        phaseSinX=solver.boundary_phase_sin[0],
        phaseCosZ=solver.boundary_phase_cos[2],
        phaseSinZ=solver.boundary_phase_sin[2],
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzBloch3D(
        EzReal=solver.Ez,
        EzImag=solver.Ez_imag,
        HxReal=solver.Hx,
        HxImag=solver.Hx_imag,
        HyReal=solver.Hy,
        HyImag=solver.Hy_imag,
        EzDecay=solver.cez_decay,
        EzCurl=solver.cez_curl,
        phaseCosX=solver.boundary_phase_cos[0],
        phaseSinX=solver.boundary_phase_sin[0],
        phaseCosY=solver.boundary_phase_cos[1],
        phaseSinY=solver.boundary_phase_sin[1],
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
    ).launchRaw()


def update_electric_fields_bloch_xy_standard_z(solver):
    if getattr(solver, "nonlinear_enabled", False):
        raise NotImplementedError(
            "FDTD instantaneous nonlinear media cannot evolve on the complex Bloch fields: the "
            "|E|^2 Kerr/chi2/TPA polarization couples the single-wavevector Bloch field to the 2k/3k "
            "harmonic sectors a complex-Bloch run does not carry, so the nonlinear response is not "
            "closed on the real update kernels. Use a real-field boundary (PML/PEC/PMC/periodic)."
        )
    solver.fdtd_module.updateElectricFieldExBlochYStandardZ3D(
        ExReal=solver.Ex,
        ExImag=solver.Ex_imag,
        HyReal=solver.Hy,
        HyImag=solver.Hy_imag,
        HzReal=solver.Hz,
        HzImag=solver.Hz_imag,
        ExDecay=solver.cex_decay,
        ExCurl=solver.cex_curl,
        phaseCosY=solver.boundary_phase_cos[1],
        phaseSinY=solver.boundary_phase_sin[1],
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEyBlochXStandardZ3D(
        EyReal=solver.Ey,
        EyImag=solver.Ey_imag,
        HxReal=solver.Hx,
        HxImag=solver.Hx_imag,
        HzReal=solver.Hz,
        HzImag=solver.Hz_imag,
        EyDecay=solver.cey_decay,
        EyCurl=solver.cey_curl,
        phaseCosX=solver.boundary_phase_cos[0],
        phaseSinX=solver.boundary_phase_sin[0],
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    solver.fdtd_module.updateElectricFieldEzBloch3D(
        EzReal=solver.Ez,
        EzImag=solver.Ez_imag,
        HxReal=solver.Hx,
        HxImag=solver.Hx_imag,
        HyReal=solver.Hy,
        HyImag=solver.Hy_imag,
        EzDecay=solver.cez_decay,
        EzCurl=solver.cez_curl,
        phaseCosX=solver.boundary_phase_cos[0],
        phaseSinX=solver.boundary_phase_sin[0],
        phaseCosY=solver.boundary_phase_cos[1],
        phaseSinY=solver.boundary_phase_sin[1],
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
    ).launchRaw()


def _bloch_cpml_pml_axis(solver):
    """Return the single PML axis (``"x"``/``"y"``/``"z"``) of a mixed Bloch+CPML
    update whose faces are exactly one axis of paired PML and two axes of paired
    Bloch, or ``None`` for any other face configuration.

    The complex split-field Bloch update carries the transverse wrap phase on the
    two periodic axes and runs the recursive-convolution CPML stretch on the one
    absorbing axis; that decomposition exists only for the three single-PML-axis
    permutations."""
    axes = ("x", "y", "z")
    codes = {
        axis: (
            int(getattr(solver, f"boundary_{axis}_low_code")),
            int(getattr(solver, f"boundary_{axis}_high_code")),
        )
        for axis in axes
    }
    pml_axes = [axis for axis in axes if codes[axis] == (BOUNDARY_PML, BOUNDARY_PML)]
    bloch_axes = [axis for axis in axes if codes[axis] == (BOUNDARY_BLOCH, BOUNDARY_BLOCH)]
    if len(pml_axes) == 1 and len(bloch_axes) == 2:
        return pml_axes[0]
    return None


def _validate_bloch_cpml_update_layout(solver):
    pml_axis = _bloch_cpml_pml_axis(solver)
    if pml_axis is None:
        raise NotImplementedError(
            "Mixed Bloch/CPML FDTD updates require exactly one PML axis (both faces PML) and two "
            "Bloch axes (both faces Bloch). The complex split-field Bloch update carries the "
            "transverse wrap phase on the two periodic axes while the single absorbing axis runs the "
            "recursive-convolution CPML stretch, so a face layout that mixes Bloch and PML on one axis "
            "or leaves more than one absorbing axis is not expressible on that split."
        )
    return pml_axis


def _iter_cpml_field_regions(solver, field, curl, attr_name):
    layout = getattr(solver, "_cpml_memory_layouts", {}).get(attr_name)
    if layout is None:
        yield {
            "field": field,
            "curl": curl,
            "psi": getattr(solver, attr_name),
            "offsets": (0, 0, 0),
        }
        return

    axis = int(layout["axis"])
    for region in iter_cpml_memory_regions(solver, attr_name):
        length = int(region["length"])
        start = int(region["offsets"][axis])
        yield {
            "field": field.narrow(axis, start, length),
            "curl": curl.narrow(axis, start, length),
            "psi": region["psi"],
            "offsets": region["offsets"],
        }


def _apply_electric_z_cpml_corrections(solver, ex, ey, hx, hy, *, imag=False):
    psi_ex_z_attr = _cpml_memory_attr("psi_ex_z", imag=imag)
    psi_ey_z_attr = _cpml_memory_attr("psi_ey_z", imag=imag)
    for region in _iter_cpml_field_regions(solver, ex, solver.cex_curl, psi_ex_z_attr):
        offset_i, offset_j, offset_k = region["offsets"]
        solver.fdtd_module.applyElectricFieldExCpmlZCorrection3D(
            Ex=region["field"],
            Hy=hy,
            ExCurl=region["curl"],
            PsiExZ=region["psi"],
            InvKappaExZ=solver.cpml_inv_kappa_e_z,
            BExZ=solver.cpml_b_e_z,
            CExZ=solver.cpml_c_e_z,
            invDz=solver.inv_dz_e,
            offsetI=offset_i,
            offsetJ=offset_j,
            offsetK=offset_k,
            yLowBoundaryMode=solver.boundary_y_low_code,
            yHighBoundaryMode=solver.boundary_y_high_code,
            fullSizeY=solver.Ex.shape[1],
            fullSizeZ=solver.Ex.shape[2],
        ).launchRaw()

    for region in _iter_cpml_field_regions(solver, ey, solver.cey_curl, psi_ey_z_attr):
        offset_i, offset_j, offset_k = region["offsets"]
        solver.fdtd_module.applyElectricFieldEyCpmlZCorrection3D(
            Ey=region["field"],
            Hx=hx,
            EyCurl=region["curl"],
            PsiEyZ=region["psi"],
            InvKappaEyZ=solver.cpml_inv_kappa_e_z,
            BEyZ=solver.cpml_b_e_z,
            CEyZ=solver.cpml_c_e_z,
            invDz=solver.inv_dz_e,
            offsetI=offset_i,
            offsetJ=offset_j,
            offsetK=offset_k,
            xLowBoundaryMode=solver.boundary_x_low_code,
            xHighBoundaryMode=solver.boundary_x_high_code,
            fullSizeX=solver.Ey.shape[0],
            fullSizeZ=solver.Ey.shape[2],
        ).launchRaw()


def _apply_electric_y_cpml_corrections(solver, ex, ez, hx, hz, *, imag=False):
    # PML axis y: Ex takes the recursive-convolution stretch on its d(Hz)/dy term
    # (tangent Bloch axis z), Ez on its d(Hx)/dy term (tangent Bloch axis x). The
    # base full-Bloch update already applied the un-stretched derivatives.
    psi_ex_y_attr = _cpml_memory_attr("psi_ex_y", imag=imag)
    psi_ez_y_attr = _cpml_memory_attr("psi_ez_y", imag=imag)
    for region in _iter_cpml_field_regions(solver, ex, solver.cex_curl, psi_ex_y_attr):
        offset_i, offset_j, offset_k = region["offsets"]
        solver.fdtd_module.applyElectricFieldExCpmlYCorrection3D(
            Ex=region["field"],
            Hz=hz,
            ExCurl=region["curl"],
            PsiExY=region["psi"],
            InvKappaExY=solver.cpml_inv_kappa_e_y,
            BExY=solver.cpml_b_e_y,
            CExY=solver.cpml_c_e_y,
            invDy=solver.inv_dy_e,
            offsetI=offset_i,
            offsetJ=offset_j,
            offsetK=offset_k,
            zLowBoundaryMode=solver.boundary_z_low_code,
            zHighBoundaryMode=solver.boundary_z_high_code,
            fullSizeZ=solver.Ex.shape[2],
            fullSizeY=solver.Ex.shape[1],
        ).launchRaw()

    for region in _iter_cpml_field_regions(solver, ez, solver.cez_curl, psi_ez_y_attr):
        offset_i, offset_j, offset_k = region["offsets"]
        solver.fdtd_module.applyElectricFieldEzCpmlYCorrection3D(
            Ez=region["field"],
            Hx=hx,
            EzCurl=region["curl"],
            PsiEzY=region["psi"],
            InvKappaEzY=solver.cpml_inv_kappa_e_y,
            BEzY=solver.cpml_b_e_y,
            CEzY=solver.cpml_c_e_y,
            invDy=solver.inv_dy_e,
            offsetI=offset_i,
            offsetJ=offset_j,
            offsetK=offset_k,
            xLowBoundaryMode=solver.boundary_x_low_code,
            xHighBoundaryMode=solver.boundary_x_high_code,
            fullSizeX=solver.Ez.shape[0],
            fullSizeY=solver.Ez.shape[1],
        ).launchRaw()


def _apply_electric_x_cpml_corrections(solver, ey, ez, hy, hz, *, imag=False):
    # PML axis x: Ey takes the stretch on its d(Hz)/dx term (tangent Bloch axis z),
    # Ez on its d(Hy)/dx term (tangent Bloch axis y).
    psi_ey_x_attr = _cpml_memory_attr("psi_ey_x", imag=imag)
    psi_ez_x_attr = _cpml_memory_attr("psi_ez_x", imag=imag)
    for region in _iter_cpml_field_regions(solver, ey, solver.cey_curl, psi_ey_x_attr):
        offset_i, offset_j, offset_k = region["offsets"]
        solver.fdtd_module.applyElectricFieldEyCpmlXCorrection3D(
            Ey=region["field"],
            Hz=hz,
            EyCurl=region["curl"],
            PsiEyX=region["psi"],
            InvKappaEyX=solver.cpml_inv_kappa_e_x,
            BEyX=solver.cpml_b_e_x,
            CEyX=solver.cpml_c_e_x,
            invDx=solver.inv_dx_e,
            offsetI=offset_i,
            offsetJ=offset_j,
            offsetK=offset_k,
            zLowBoundaryMode=solver.boundary_z_low_code,
            zHighBoundaryMode=solver.boundary_z_high_code,
            fullSizeZ=solver.Ey.shape[2],
            fullSizeX=solver.Ey.shape[0],
        ).launchRaw()

    for region in _iter_cpml_field_regions(solver, ez, solver.cez_curl, psi_ez_x_attr):
        offset_i, offset_j, offset_k = region["offsets"]
        solver.fdtd_module.applyElectricFieldEzCpmlXCorrection3D(
            Ez=region["field"],
            Hy=hy,
            EzCurl=region["curl"],
            PsiEzX=region["psi"],
            InvKappaEzX=solver.cpml_inv_kappa_e_x,
            BEzX=solver.cpml_b_e_x,
            CEzX=solver.cpml_c_e_x,
            invDx=solver.inv_dx_e,
            offsetI=offset_i,
            offsetJ=offset_j,
            offsetK=offset_k,
            yLowBoundaryMode=solver.boundary_y_low_code,
            yHighBoundaryMode=solver.boundary_y_high_code,
            fullSizeY=solver.Ez.shape[1],
            fullSizeX=solver.Ez.shape[0],
        ).launchRaw()


def _bloch_cpml_multi_axis_layout(solver):
    """Return the PML axes of a generalized Bloch+CPML face layout, or ``None``.

    Every axis must be paired Bloch, paired PML or paired PEC, with at least one
    Bloch axis and one PML axis. The single-PML-axis/two-Bloch-axis permutations
    keep their dedicated split and return ``None`` here."""
    if _bloch_cpml_pml_axis(solver) is not None:
        return None
    pairs = {
        axis: (
            int(getattr(solver, f"boundary_{axis}_low_code")),
            int(getattr(solver, f"boundary_{axis}_high_code")),
        )
        for axis in ("x", "y", "z")
    }
    allowed = {(BOUNDARY_BLOCH, BOUNDARY_BLOCH), (BOUNDARY_PML, BOUNDARY_PML), (BOUNDARY_PEC, BOUNDARY_PEC)}
    if any(pair not in allowed for pair in pairs.values()):
        return None
    pml_axes = tuple(axis for axis, pair in pairs.items() if pair == (BOUNDARY_PML, BOUNDARY_PML))
    bloch_axes = tuple(axis for axis, pair in pairs.items() if pair == (BOUNDARY_BLOCH, BOUNDARY_BLOCH))
    if not pml_axes or not bloch_axes:
        return None
    return pml_axes


def update_electric_fields_bloch_multi_cpml(solver, pml_axes):
    """Generalized Bloch+CPML electric update (e.g. one Bloch axis with two PML
    axes, or Bloch with a PEC axis).

    The base full-Bloch kernel carries the wrap phase on the Bloch axes and the
    unit phase elsewhere; the recursive-convolution stretch is then folded in
    along every PML axis for the components with a derivative along it, exactly
    as the single-PML-axis split does for its one axis. The outermost tangential
    planes of every PML face and the tangential planes of every PEC face are held
    at zero in both the real and imaginary fields."""
    update_electric_fields_bloch(solver)
    for axis in pml_axes:
        if axis == "x":
            _apply_electric_x_cpml_corrections(solver, solver.Ey, solver.Ez, solver.Hy, solver.Hz, imag=False)
            _apply_electric_x_cpml_corrections(
                solver, solver.Ey_imag, solver.Ez_imag, solver.Hy_imag, solver.Hz_imag, imag=True
            )
        elif axis == "y":
            _apply_electric_y_cpml_corrections(solver, solver.Ex, solver.Ez, solver.Hx, solver.Hz, imag=False)
            _apply_electric_y_cpml_corrections(
                solver, solver.Ex_imag, solver.Ez_imag, solver.Hx_imag, solver.Hz_imag, imag=True
            )
        else:
            _apply_electric_z_cpml_corrections(solver, solver.Ex, solver.Ey, solver.Hx, solver.Hy, imag=False)
            _apply_electric_z_cpml_corrections(
                solver, solver.Ex_imag, solver.Ey_imag, solver.Hx_imag, solver.Hy_imag, imag=True
            )
    # The standard CPML update never advances the outermost tangential planes of
    # a PML face, and PEC faces carry zero tangential E in both the real and the
    # imaginary Bloch field; restore both after the wrapped base update (the
    # post-update PEC pass clamps the real field only).
    tangential = {"x": ("Ey", "Ez"), "y": ("Ex", "Ez"), "z": ("Ex", "Ey")}
    pec_axes = tuple(
        axis
        for axis in ("x", "y", "z")
        if int(getattr(solver, f"boundary_{axis}_low_code")) == BOUNDARY_PEC
    )
    for axis in pml_axes + pec_axes:
        axis_index = "xyz".index(axis)
        for name in tangential[axis]:
            for field in (getattr(solver, name), getattr(solver, name + "_imag")):
                clamp_field_face(solver, field, axis_index, "low")
                clamp_field_face(solver, field, axis_index, "high")


def update_electric_fields_bloch_cpml(solver):
    multi_pml_axes = _bloch_cpml_multi_axis_layout(solver)
    if multi_pml_axes is not None:
        update_electric_fields_bloch_multi_cpml(solver, multi_pml_axes)
        return
    pml_axis = _validate_bloch_cpml_update_layout(solver)
    if pml_axis == "z":
        # Grating-style x/y Bloch + z PML: standard boundary treatment on z keeps
        # the outer plane exactly zero (mirrored by the adjoint replay).
        update_electric_fields_bloch_xy_standard_z(solver)
        _apply_electric_z_cpml_corrections(solver, solver.Ex, solver.Ey, solver.Hx, solver.Hy, imag=False)
        _apply_electric_z_cpml_corrections(
            solver,
            solver.Ex_imag,
            solver.Ey_imag,
            solver.Hx_imag,
            solver.Hy_imag,
            imag=True,
        )
        return
    # General single-PML-axis permutation (x or y): the base full-Bloch update
    # carries the true wrap phase on both periodic axes and the natural unit phase
    # on the absorbing axis, then the recursive-convolution stretch is folded in
    # along that axis for the two components with an absorbing-axis derivative.
    update_electric_fields_bloch(solver)
    if pml_axis == "y":
        _apply_electric_y_cpml_corrections(solver, solver.Ex, solver.Ez, solver.Hx, solver.Hz, imag=False)
        _apply_electric_y_cpml_corrections(
            solver, solver.Ex_imag, solver.Ez_imag, solver.Hx_imag, solver.Hz_imag, imag=True
        )
    else:
        _apply_electric_x_cpml_corrections(solver, solver.Ey, solver.Ez, solver.Hy, solver.Hz, imag=False)
        _apply_electric_x_cpml_corrections(
            solver, solver.Ey_imag, solver.Ez_imag, solver.Hy_imag, solver.Hz_imag, imag=True
        )


def clamp_field_face(solver, field, axis, side):
    side_code = 0 if side == "low" else 1
    module = getattr(solver, "fdtd_module", None)
    if module is not None and module.__class__.__name__ == "NativeFDTDModule":
        module.clampFieldFace3D(field=field, axis=int(axis), side=side_code).launchRaw()
        return
    index = 0 if side_code == 0 else field.shape[axis] - 1
    field.select(axis, index).zero_()


def _launch_native_clamp(solver, kernel_name, **kwargs):
    module = getattr(solver, "fdtd_module", None)
    if module is None or module.__class__.__name__ != "NativeFDTDModule":
        return False
    getattr(module, kernel_name)(**kwargs).launchRaw()
    return True


def _enforce_native_pec_field(solver, field, axis_a, low_a, high_a, axis_b, low_b, high_b):
    sides_a = (low_a == BOUNDARY_PEC, high_a == BOUNDARY_PEC)
    sides_b = (low_b == BOUNDARY_PEC, high_b == BOUNDARY_PEC)
    if not any(sides_a) and not any(sides_b):
        return True

    if all(sides_a) and all(sides_b):
        return _launch_native_clamp(solver, "clampPecBoundary3D", field=field, axisA=axis_a, axisB=axis_b)
    if all(sides_a):
        if not _launch_native_clamp(solver, "clampPecBoundary3D", field=field, axisA=axis_a, axisB=axis_a):
            return False
    else:
        for side_code, is_pec in enumerate(sides_a):
            if is_pec and not _launch_native_clamp(
                solver, "clampFieldFace3D", field=field, axis=axis_a, side=side_code
            ):
                return False

    if all(sides_b):
        return _launch_native_clamp(solver, "clampPecBoundary3D", field=field, axisA=axis_b, axisB=axis_b)
    for side_code, is_pec in enumerate(sides_b):
        if is_pec and not _launch_native_clamp(solver, "clampFieldFace3D", field=field, axis=axis_b, side=side_code):
            return False
    return True


def enforce_pec_boundaries(solver):
    if not getattr(solver, "has_pec_faces", False):
        return
    if has_complex_fields(solver):
        _enforce_imag_pec_boundaries(solver)

    if _enforce_native_pec_field(
        solver,
        solver.Ex,
        1,
        solver.boundary_y_low_code,
        solver.boundary_y_high_code,
        2,
        solver.boundary_z_low_code,
        solver.boundary_z_high_code,
    ) and _enforce_native_pec_field(
        solver,
        solver.Ey,
        0,
        solver.boundary_x_low_code,
        solver.boundary_x_high_code,
        2,
        solver.boundary_z_low_code,
        solver.boundary_z_high_code,
    ) and _enforce_native_pec_field(
        solver,
        solver.Ez,
        0,
        solver.boundary_x_low_code,
        solver.boundary_x_high_code,
        1,
        solver.boundary_y_low_code,
        solver.boundary_y_high_code,
    ):
        return

    face_specs = (
        (solver.boundary_x_low_code, "low", (("Ey", 0), ("Ez", 0))),
        (solver.boundary_x_high_code, "high", (("Ey", 0), ("Ez", 0))),
        (solver.boundary_y_low_code, "low", (("Ex", 1), ("Ez", 1))),
        (solver.boundary_y_high_code, "high", (("Ex", 1), ("Ez", 1))),
        (solver.boundary_z_low_code, "low", (("Ex", 2), ("Ey", 2))),
        (solver.boundary_z_high_code, "high", (("Ex", 2), ("Ey", 2))),
    )
    for boundary_code, side, targets in face_specs:
        if boundary_code != BOUNDARY_PEC:
            continue
        for field_name, axis in targets:
            clamp_field_face(solver, getattr(solver, field_name), axis, side)


def _enforce_imag_pec_boundaries(solver):
    # The complex Bloch electric update writes every node, PEC faces included,
    # so the imaginary split field needs the same tangential clamp as the real one.
    face_specs = (
        (solver.boundary_x_low_code, "low", (("Ey_imag", 0), ("Ez_imag", 0))),
        (solver.boundary_x_high_code, "high", (("Ey_imag", 0), ("Ez_imag", 0))),
        (solver.boundary_y_low_code, "low", (("Ex_imag", 1), ("Ez_imag", 1))),
        (solver.boundary_y_high_code, "high", (("Ex_imag", 1), ("Ez_imag", 1))),
        (solver.boundary_z_low_code, "low", (("Ex_imag", 2), ("Ey_imag", 2))),
        (solver.boundary_z_high_code, "high", (("Ex_imag", 2), ("Ey_imag", 2))),
    )
    for boundary_code, side, targets in face_specs:
        if boundary_code != BOUNDARY_PEC:
            continue
        for field_name, axis in targets:
            clamp_field_face(solver, getattr(solver, field_name), axis, side)


def apply_surface_impedance(solver):
    """Override the tangential E on every exposed surface-impedance face.

    Each write realizes the Leontovich relation ``E_t = Z_s * (n_hat x H)`` from the
    vacuum-side tangential H (index-aligned on the Yee grid by
    ``_configure_surface_impedance``). Two realizations of the same contract share this
    loop:

    * a narrowband good-conductor face is pure resistance ``Z_s = R``: a full-plane
      face uses the fused native kernel ``E_surface = sign * R * H`` (bit-identical to
      the generic ``D * (sign * H)`` scalar path, since ``sign`` is exactly +/-1); a
      finite (sub-plane) face writes only its transverse window with torch slicing.
    * a generic rational face is a passive Z-form ADE: per edge, output
      ``E = C x + D u`` then state advance ``x <- A x + B u`` with input ``u = sign * H``
      (the trapezoidal ``|z| < 1`` discretization from the shared fitter).

    The metal interior is masked to zero by the coefficient setup, so each surface acts
    as a good-conductor termination without resolving the skin depth. Writes run in the
    layout's deterministic owner order (minimum-rank owner writes last), so a corner
    edge shared by two faces has a single deterministic owner.
    """
    state = getattr(solver, "_surface_impedance", None)
    if state is None:
        return
    for face in state["writes"]:
        electric = getattr(solver, face["e_name"])
        magnetic = getattr(solver, face["h_name"])
        axis = face["axis"]
        mask = face.get("mask")
        if mask is not None:
            # Staircased (voxelized) face: overwrite the exposed transverse footprint of
            # the surface E plane with the Leontovich value, leaving lateral (interior)
            # edges at their coefficient-masked zero. ``torch.where`` is static-shape so it
            # captures into the field-update graph.
            e_index = _surface_plane_index(electric.dim(), axis, face["electric_index"])
            h_index = _surface_plane_index(magnetic.dim(), axis, face["magnetic_index"])
            electric[e_index] = torch.where(
                mask, face["sign"] * face["surface_r"] * magnetic[h_index], electric[e_index]
            )
            continue
        ade = face.get("ade")
        if ade is None and face["full_plane"]:
            solver.fdtd_module.applySibcSurface3D(
                electric=electric,
                magnetic=magnetic,
                axis=axis,
                electricIndex=face["electric_index"],
                magneticIndex=face["magnetic_index"],
                sign=face["sign"],
                surfaceR=face["surface_r"],
            ).launchRaw()
            continue
        e_index = _surface_face_index(
            electric.dim(), axis, face["electric_index"], face
        )
        h_index = _surface_face_index(
            magnetic.dim(), axis, face["magnetic_index"], face
        )
        h_sub = magnetic[h_index]
        if ade is None:
            electric[e_index] = face["sign"] * face["surface_r"] * h_sub
            continue
        u = (face["sign"] * h_sub).reshape(-1)
        electric[e_index] = surface_ade_step(ade, u).reshape(h_sub.shape)


def surface_ade_step(ade, u):
    """Advance a batch of per-edge Z-form surface ADEs and return the tangential E.

    ``ade`` holds the shared discrete state space ``(A, B, C, D)`` (scalar input/output)
    and a per-edge state ``x`` of shape ``[order, M]``; ``u`` is the length-``M`` input
    ``sign * H``. Per edge this is exactly the shared
    ``DiscreteStateSpaceNetwork.step``: output ``E = C x + D u`` uses the pre-update
    state, then the state advances ``x <- A x + B u`` in place (so the recurrence is
    captured into the CUDA graph). Returns ``E`` of shape ``[M]``.
    """
    x = ade["state"]
    u_row = u.reshape(1, -1)
    output = (ade["C"] @ x).reshape(-1) + ade["D"].reshape(()) * u
    x.copy_(ade["A"] @ x + ade["B"] @ u_row)
    return output


def _surface_face_index(ndim, axis, axis_index, face):
    index = [slice(None)] * ndim
    index[axis] = int(axis_index)
    index[face["b_axis"]] = face["b_slice"]
    index[face["c_axis"]] = face["c_slice"]
    return tuple(index)


def _surface_plane_index(ndim, axis, axis_index):
    """Index tuple selecting the whole transverse plane at ``axis_index`` along ``axis``."""
    index = [slice(None)] * ndim
    index[axis] = int(axis_index)
    return tuple(index)


def clamp_pec_boundaries(solver):
    enforce_pec_boundaries(solver)


def apply_mur_boundaries(solver):
    # First-order Mur absorbing boundary on the CUDA E-field tensors after the
    # interior E-update. For an outer face:
    #   E_boundary(n+1) = E_adjacent(n) + coef * (E_adjacent(n+1) - E_boundary(n))
    # with coef = (c*dt - d) / (c*dt + d) and d the normal cell size. Each face is
    # a single native-CUDA kernel launch that updates its persistent boundary /
    # first-interior plane buffers in place, so it stays capturable into the tail
    # CUDA graph and carries no per-step host arithmetic or allocation.
    if not getattr(solver, "has_mur_faces", False):
        return

    for entry in solver._mur_state:
        solver.fdtd_module.applyMurBoundary3D(
            field=getattr(solver, entry["field"]),
            axis=entry["axis"],
            boundaryIndex=entry["boundary_index"],
            adjacentIndex=entry["adjacent_index"],
            coef=entry["coef"],
            prevBoundary=entry["prev_boundary"],
            prevAdjacent=entry["prev_adjacent"],
        ).launchRaw()
    if has_extended_one_way_boundaries(solver):
        apply_extended_one_way_boundaries(solver)


def init_field(solver):
    solver.Ex = torch.zeros((solver.Nx - 1, solver.Ny, solver.Nz), device=solver.device, dtype=solver.scene.x.dtype)
    solver.Ey = torch.zeros((solver.Nx, solver.Ny - 1, solver.Nz), device=solver.device, dtype=solver.scene.x.dtype)
    solver.Ez = torch.zeros((solver.Nx, solver.Ny, solver.Nz - 1), device=solver.device, dtype=solver.scene.x.dtype)
    solver.Hx = torch.zeros((solver.Nx, solver.Ny - 1, solver.Nz - 1), device=solver.device, dtype=solver.scene.x.dtype)
    solver.Hy = torch.zeros((solver.Nx - 1, solver.Ny, solver.Nz - 1), device=solver.device, dtype=solver.scene.x.dtype)
    solver.Hz = torch.zeros((solver.Nx - 1, solver.Ny - 1, solver.Nz), device=solver.device, dtype=solver.scene.x.dtype)

    solver.build_materials(solver.scene)
    # Wire preparation may conservatively tighten dt using the joint
    # Maxwell-plus-wire spectral bound. All dt-dependent boundary and field
    # coefficients must therefore be built after it.
    initialize_wire_runtime(solver)
    from ..thin_sheet import prepare_thin_sheet_time_step
    prepare_thin_sheet_time_step(solver)
    from .materials import build_material_update_coefficients

    build_material_update_coefficients(solver)
    solver._time_step_metadata["dt"] = float(solver.dt)
    initialize_boundary_state(solver)
    solver._build_update_coefficients()
    initialize_copper_foils(solver)
    # Compile the deterministic breakdown state machine after the base update
    # coefficients exist (they seed the exact intact-edge reconstruction). A scene
    # with no breakdown material leaves solver.breakdown_enabled False and adds no
    # per-step machinery.
    initialize_breakdown_runtime(solver)
    solver._initialize_dispersive_state()
    solver._initialize_magnetic_dispersive_state()
    solver._initialize_gyromagnetic_state()
    initialize_tfsf_state(solver)
    initialize_source_terms(solver)
    prepare_port_runtimes(
        solver,
        getattr(solver, "_requested_port_frequencies", (solver.source_frequency,)),
        getattr(solver, "_port_excitations", ()),
    )
    prepare_network_runtimes(solver)
    validate_copper_port_coupling(solver)
    from ..cut_surface import initialize_cut_surfaces
    initialize_cut_surfaces(solver)
    from ..thin_sheet import initialize_thin_sheets
    initialize_thin_sheets(solver)


def _compute_shutoff_min_step(solver, shutoff_check_interval: int) -> int:
    import math

    plan = getattr(solver, "_time_plan", None)
    if plan is not None:
        if not plan["shutoff_allowed"]:
            return math.inf
        return max(plan["shutoff_min_step"], plan["spectral_start_step"]) + 2 * shutoff_check_interval

    source_time = getattr(solver, "_source_time", None)
    settling_time = 0.0
    if source_time is not None:
        if isinstance(source_time, dict):
            settling_time = float(source_time.get("settling_time", 0.0))
        else:
            settling_time = float(getattr(source_time, "settling_time", 0.0))
    settling_step = int(math.ceil(settling_time / solver.dt)) if settling_time > 0.0 else 0

    dft_start_step = getattr(solver, "dft_start_step", None)

    observer_entries = getattr(solver, "_observer_spectral_entries", None)
    observer_start = 0
    if observer_entries:
        observer_start = min(int(entry["start_step"]) for entry in observer_entries)

    floor = max(settling_step, dft_start_step or 0, observer_start or 0)
    return floor + 2 * shutoff_check_interval


def _electric_field_energy(solver) -> torch.Tensor:
    energy = (
        (solver.eps_Ex * solver.Ex * solver.Ex).sum()
        + (solver.eps_Ey * solver.Ey * solver.Ey).sum()
        + (solver.eps_Ez * solver.Ez * solver.Ez).sum()
    )
    if has_complex_fields(solver):
        energy = (
            energy
            + (solver.eps_Ex * solver.Ex_imag * solver.Ex_imag).sum()
            + (solver.eps_Ey * solver.Ey_imag * solver.Ey_imag).sum()
            + (solver.eps_Ez * solver.Ez_imag * solver.Ez_imag).sum()
        )
    return energy


def _planned_window_normalization(window_type: str, start_step: int, end_step: int) -> float:
    # Sum the spectral window weights over the full planned [start_step, end_step)
    # range, matching compute_window_weight / accumulate_dft exactly for a full run.
    total = int(end_step) - int(start_step)
    if total <= 0:
        return 0.0
    if window_type == "none":
        return float(total)
    positions = np.arange(total, dtype=np.float64) / total
    if window_type == "hanning":
        weights = 0.5 * (1.0 - np.cos(2.0 * np.pi * positions))
        return float(weights.sum())
    if window_type == "ramp":
        ramp_fraction = 0.1
        weights = np.ones_like(positions)
        ramp_mask = positions < ramp_fraction
        weights[ramp_mask] = 0.5 * (1.0 - np.cos(np.pi * positions[ramp_mask] / ramp_fraction))
        return float(weights.sum())
    return float(total)


def _complete_spectral_normalization(solver, time_steps: int) -> None:
    # Restore the historical planned CW denominator. Pulse integral scales use
    # only dt, so these diagnostic weight sums cannot rescale a stopped pulse.
    if getattr(solver, "dft_enabled", False) and getattr(solver, "_dft_entries", None):
        for index in range(len(solver._dft_entries)):
            start_step = int(solver._dft_start_steps[index])
            end_step = int(solver._dft_end_steps[index])
            if end_step < 0:
                end_step = int(time_steps)
            solver._dft_window_normalization_values[index] = _planned_window_normalization(
                solver.dft_window_type, start_step, end_step
            )
    if getattr(solver, "observers_enabled", False):
        for entry in solver._observer_spectral_entries:
            end_step = entry["end_step"] if entry["end_step"] is not None else int(time_steps)
            entry["window_normalization"] = _planned_window_normalization(
                solver.observer_window_type, int(entry["start_step"]), int(end_step)
            )
    complete_port_spectral_normalization(solver)


def _field_update_block(solver, time_value):
    """One step of the time-marching field-update core: the magnetic and electric
    Yee updates plus their in-place dispersive/Kerr/CPML/Bloch state advances.

    This is exactly the contiguous, time-marching part of the step that carries no
    per-step host input when TFSF and magnetic sources are absent, so it can be
    captured once into a CUDA graph and replayed (additive electric sources and
    the running DFT stay outside the graph; native SIBC remains inside). Kept as
    a single function so the normal and graph-captured paths execute an identical
    kernel sequence.
    """
    begin_copper_step(solver)
    begin_thin_sheet_step(solver)
    if getattr(solver, "modulation_enabled", False):
        solver.fdtd_module.advanceModulationTime3D(
            ModulationTime=solver._modulation_time,
            ModulationStep=solver._modulation_step,
            dt=solver.dt,
        ).launchRaw()
    solver._advance_magnetic_dispersive_state()
    # Record the pre-update (leapfrog) transverse H so the post-update coupled step
    # can form the time-centred magnetization drive (H_pre + H_tmp)/2. The coupled
    # implicit-midpoint solve + correction runs after the plain magnetic update.
    solver._snapshot_gyromagnetic_drive()
    update_magnetic_fields(solver, solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
    if has_complex_fields(solver):
        update_magnetic_fields(
            solver,
            solver.Hx_imag,
            solver.Hy_imag,
            solver.Hz_imag,
            solver.Ex_imag,
            solver.Ey_imag,
            solver.Ez_imag,
            imag=True,
        )
    correct_copper_magnetic(solver)
    advance_thin_sheet_magnetic(solver)
    if solver.tfsf_enabled:
        apply_tfsf_h_correction(solver, time_value)
        advance_tfsf_auxiliary_magnetic(solver)
    if solver._magnetic_source_terms:
        inject_magnetic_surface_source_terms(
            solver, time_value=injection_time("magnetic_current", time_value, solver.dt)
        )
    solver._apply_magnetic_dispersive_corrections()
    # Non-reciprocal gyromagnetic step: the coupled implicit-midpoint magnetization
    # advance plus the correction H -= dM/mu_inf (magnetic mirror of the electric-side
    # full-anisotropy correction). Passive (discretely non-growing at zero damping),
    # unlike an explicit pre-update advance / post-update correct split.
    solver._step_gyromagnetic_coupled()
    if solver._wire_runtime is not None:
        sample_and_update_wire(solver)

    if has_complex_fields(solver):
        solver._advance_dispersive_state()
        if solver.uses_cpml:
            update_electric_fields_bloch_cpml(solver)
        else:
            update_electric_fields_bloch(solver)
    else:
        solver._advance_dispersive_state()
        if solver.nonlinear_enabled:
            solver._update_nonlinear_electric_coefficients()
        # Snapshot sigma . E^n before the base update overwrites E (no-op unless the
        # medium is both full-anisotropic and conductive).
        capture_aniso_conduction_currents(solver)
        update_electric_fields(
            solver,
            solver.Ex,
            solver.Ey,
            solver.Ez,
            solver.Hx,
            solver.Hy,
            solver.Hz,
            time_value=time_value,
        )
        if getattr(solver, "full_aniso_enabled", False):
            apply_full_aniso_corrections(solver)
            apply_full_aniso_conduction(solver)
    if solver._wire_runtime is not None:
        deposit_wire_current(solver)
    apply_surface_impedance(solver)
    advance_copper_electric(solver)
    advance_thin_sheet_electric(solver)
    accumulate_copper_spectra(solver)


def _make_field_update_runner(solver, use_cuda_graph: bool):
    """Return ``run(time_value)`` for the field-update block, capturing it into a
    CUDA graph when requested and safe. Falls back to the direct call for TFSF /
    magnetic-source scenes (whose block carries per-step host input) or if capture
    is unavailable. All carried state is restored after capture or failure,
    including material memory when the initial field is nonzero.
    """
    solver._cuda_graph_active = False

    def normal(time_value):
        _field_update_block(solver, time_value)

    # v1 scope: the standard real-field path (optionally conductive), linear
    # electric/magnetic dispersion, the instantaneous nonlinear family (Kerr chi3,
    # chi2, and two-photon-absorption loss), the complex split-field Bloch path
    # (optionally with a single CPML axis), and reference-provider TFSF injection
    # (axis-aligned / x-Ez plane waves, in both the six-face box and the two-face
    # non-periodic slab injection mode). Magnetic surface sources put a
    # host-evaluated signal scalar inside the block, so those scenes stay on the
    # eager path.
    #
    # A reference-provider TFSF scene is safe to capture: its in-block H correction
    # and auxiliary magnetic advance read the incident field from the
    # device-resident auxiliary 1D grid (aux.electric / aux.magnetic) at fixed
    # integer sample indices, so the block carries no per-step host input, and the
    # reference injection kernel's warp-aggregated writes are collision-free for the
    # axis-aligned face layout (bit-reproducible run to run). The 1D source waveform
    # is evaluated outside capture: entry-plane boundary values are staged
    # before the block and the E-side hard seed is written after it. The captured H
    # correction reads whatever aux.electric holds at replay -- exactly matching the
    # eager path. The block mutates only aux.magnetic among the auxiliary tensors;
    # it is snapshotted below alongside aux.electric so warmup/capture on the zero
    # field leave the incident line untouched for the physical run, and the incident
    # line is a zero fixed point during capture (aux.electric stays zero until the
    # eager E-side advance injects the source after capture). The oblique
    # auxiliary-line provider is excluded (its interpolated injection kernel is not
    # bit-reproducible under cross-warp atomics), and the oblique-CW / analytic
    # providers evaluate the waveform on the host each step; all stay eager
    # (see excitation.tfsf_incident_is_gpu_driven).
    #
    # Linear dispersion is safe to capture: the Debye/Drude/Lorentz ADE advance and
    # the polarization-current subtraction launch persistent per-pole state tensors
    # (polarization/current, allocated once at init) with fixed dt/decay/restoring
    # scalars, so they carry no per-step host input. The ADE is linear with zero
    # forcing at E = H = 0, so with all pole state starting at zero the zero initial
    # field is still a fixed point of the extended block; warmup/capture leaves the
    # pole state at zero and does not perturb the physical run.
    #
    # Instantaneous nonlinearity is safe to capture too: the Kerr curl-only kernel
    # and the general chi2/TPA coefficient kernel recompute the dynamic curl/decay
    # tensors in place from the current field, the persistent chi3/chi2/tpa maps,
    # and fixed dt/eps0 scalars, with no per-step host input. Unlike dispersion they
    # carry no accumulated auxiliary state: the dynamic coefficients are a pure
    # function of the live field, rewritten before they are read every step, so the
    # fields+psi snapshot already covers everything the block mutates. Every
    # nonlinear polarization vanishes at E = 0 (P ~ chi2 E^2, chi3 |E|^2 E, and
    # sigma_NL ~ |E|^2), collapsing the dynamic coefficients to their linear values,
    # so the zero field remains a fixed point through warmup/capture.
    #
    # The complex (split-field Bloch) path is safe to capture too: a Bloch-periodic
    # scene marches a second real FDTD copy (the imaginary split field) through the
    # same magnetic/electric curl+decay kernels, coupled only by a fixed per-axis
    # wrap phase (cos/sin of the Bloch wavevector, constant for the whole run), so
    # the block carries no per-step host input. The extra state it evolves -- the
    # imaginary E/H fields and their CPML psi memory -- is snapshotted alongside the
    # real fields below (the imaginary psi tensors already match the ``psi`` prefix),
    # and the split update is linear with zero forcing at E = H = 0, so the zero
    # field (real and imaginary) stays a fixed point through warmup/capture. The
    # complex running DFT keeps the host tail (build_dft_step_tables declines the GPU
    # weight table for complex fields), so only the field-update block is captured.
    graphable = (
        use_cuda_graph
        and torch.cuda.is_available()
        and torch.device(solver.device).type == "cuda"
        # TFSF is capturable only for the reference providers, whose in-block
        # correction reads the device-resident incident line through the
        # bit-reproducible integer-indexed reference kernel and carries no per-step
        # host input (see _make_field_update_runner docstring above).
        and (not solver.tfsf_enabled or tfsf_incident_is_gpu_driven(solver))
        # A magnetic surface source injects a host-evaluated signal scalar inside
        # the block, which a captured graph would freeze at its capture-time value.
        and not solver._magnetic_source_terms
    )
    if not graphable:
        return normal
    from ..cuda.runtime.graph import CudaGraphRunner

    from ..checkpoint import graph_state_tensors

    state = graph_state_tensors(solver)
    saved = {k: v.clone() for k, v in state.items()}

    def _restore():
        for k, v in saved.items():
            state[k].copy_(v)

    try:
        runner = CudaGraphRunner(device=solver.device, enabled=True, warmup_steps=3)
        replay = runner.capture(lambda: _field_update_block(solver, 0.0))
    except Exception:
        # Any capture failure (e.g. a non-capturable kernel) degrades to the
        # normal path rather than breaking the solve.
        _restore()
        return normal
    _restore()
    solver._cuda_graph_active = True
    return lambda time_value: replay()


def _post_source_block(solver):
    """The time-independent tail after source injection: dispersive corrections,
    PEC clamp, Mur ABC, and the GPU-driven running-DFT accumulation. Captured as
    a second graph so its kernel launches collapse to one replay; only valid when
    the DFT is GPU-driven (``build_dft_step_tables`` succeeded)."""
    solver._apply_dispersive_corrections()
    if not solver.tfsf_enabled:
        enforce_pec_boundaries(solver)
    apply_mur_boundaries(solver)
    from ..cut_surface import apply_cut_surface_constraints
    apply_cut_surface_constraints(solver)
    solver.accumulate_dft_gpu()


def _make_tail_runner(solver, use_gpu_dft: bool):
    """Capture the post-source tail (PEC + Mur + GPU-DFT) into a graph. Returns a
    replay callable, or ``None`` to keep the inline tail. Snapshots the state the
    tail mutates (fields, Mur history, device counters and DFT accumulators)
    around capture so nonzero initial states are preserved too."""
    if not (use_gpu_dft and getattr(solver, "_cuda_graph_active", False)):
        return None
    cosine = getattr(solver, "_dft_weighted_cos_table", None)
    sine = getattr(solver, "_dft_weighted_sin_table", None)
    counter = getattr(solver, "_dft_step", None)
    if (not isinstance(cosine, torch.Tensor) or not isinstance(sine, torch.Tensor)
        or cosine.ndim != 2 or sine.shape != cosine.shape or cosine.shape[1] == 0
        or cosine.device != sine.device or cosine.dtype != torch.float32 or sine.dtype != torch.float32
        or not isinstance(counter, torch.Tensor) or counter.shape != (1,)
        or counter.dtype != torch.int64 or counter.device != cosine.device):
        raise ValueError("Tail capture requires matching float32 DFT tables and an int64[1] device counter.")
    # This single host read happens during preparation, never in capture/replay.
    current_step = int(counter.item())
    rows = int(cosine.shape[0])
    if current_step < 0 or current_step > rows:
        raise ValueError("Tail capture DFT counter is outside its precomputed table horizon.")
    remaining = rows - current_step
    if remaining == 0:
        return None
    # Conservatively reserve a valid row for capture preparation. Recording a
    # CUDA graph does not physically advance the GPU state until replay.
    warmup_steps = min(3, remaining - 1)
    from ..cuda.runtime.graph import CudaGraphRunner

    from ..checkpoint import graph_state_tensors

    state = graph_state_tensors(solver, tail=True)
    if isinstance(getattr(solver, "_dft_step", None), torch.Tensor):
        state[("step",)] = solver._dft_step
    for comp, buffers in getattr(solver, "_dft_batched_fields", {}).items():
        for key in ("real", "imag"):
            tensor = buffers.get(key)
            if isinstance(tensor, torch.Tensor):
                state[("dft", comp, key)] = tensor
    saved = {k: v.clone() for k, v in state.items()}

    def _restore():
        for k, v in saved.items():
            state[k].copy_(v)

    try:
        replay = CudaGraphRunner(device=solver.device, enabled=True, warmup_steps=warmup_steps).capture(
            lambda: _post_source_block(solver)
        )
    except Exception:
        _restore()
        return None
    _restore()
    return replay


def _port_step_block(solver, step: torch.Tensor | None, *, post_source=None) -> None:
    """The post-source step phase of prepared ports, driven by a device step counter.

    Applies the port terminations with the tabulated drive (see
    ``prepare_port_drive_tables``), then the network feedback, the post-source
    tail and the port V/I DFT: the eager order of ``solve_steps``. It carries no
    per-step host input, so the single-run port graph and the native batch step
    chunks capture this one sequence. ``step`` is not advanced here.
    """
    apply_port_runtimes(solver, drive_step=step)
    for sheet in getattr(solver, "_thin_sheets", ()):
        if sheet.feeds:
            sheet.electric(solver)
    apply_network_runtimes(solver, native_lu=True)
    if post_source is None:
        _post_source_block(solver)
    else:
        post_source()
    accumulate_port_observers(solver, step=step)


def _make_port_step_runner(solver, *, use_cuda_graph: bool, use_gpu_dft: bool, time_steps: int):
    """Graph-capture the post-source phase of a run with lumped and gap ports.

    Returns ``None``, keeping the eager phase, unless no port runtime is bound
    to a circuit or an embedded network (driven or terminated Yee-edge, coaxial
    and wire-bound ports, and passive V/I observation ports, all qualify), the
    scene has no circuits or networks, and the running DFT (if any) is
    GPU-driven. Otherwise one call per step runs
    ``_port_step_block`` from the drive tables and ``solver._port_observer_step``.
    The first call executes eagerly, which also initializes every kernel and
    library handle; the second call captures the block (recording only, so no
    state needs restoring) and every later call replays that one graph. If
    capture fails the block keeps running eagerly with identical numerics.
    """
    runtimes = tuple(getattr(solver, "_port_runtimes", ()))
    graphable = (
        use_cuda_graph
        and torch.cuda.is_available()
        and torch.device(solver.device).type == "cuda"
        and bool(runtimes)
        and all(
            runtime.embedded_network_name is None and runtime.circuit_port is None
            for runtime in runtimes
        )
        and not getattr(solver, "_circuit_runtimes", ())
        and not getattr(solver, "_network_runtimes", ())
        and (not solver.dft_enabled or use_gpu_dft)
    )
    if not graphable:
        return None
    prepare_port_drive_tables(solver, time_steps)
    counter = solver._port_observer_step
    counter.fill_(runtimes[0].sample_index)
    # Port spectra are now sampled from the device counter; the host sample
    # count is restored by complete_port_observer_graph after the loop.
    solver._port_observer_graph_active = True

    def step() -> None:
        _port_step_block(solver, counter)
        counter.add_(1)

    from ..cuda.runtime.graph import CudaGraphRunner

    replay = None
    calls = 0

    def run() -> None:
        nonlocal replay, calls
        if replay is not None:
            replay()
            return
        calls += 1
        if calls == 2:
            try:
                replay = CudaGraphRunner(device=solver.device, enabled=True, warmup_steps=0).capture(step)
            except Exception as error:
                solver._port_graph_capture_error = repr(error)
                replay = None
            else:
                solver._port_cuda_graph_active = True
                replay()
                return
        step()

    return run


def _make_full_embedded_step_runner(solver, *, use_cuda_graph: bool):
    """Capture a source-free field/network/terminal-observer step as one graph."""

    runtimes = tuple(getattr(solver, "_network_runtimes", ()))
    port_runtimes = tuple(getattr(solver, "_port_runtimes", ()))
    graphable = (
        use_cuda_graph
        and torch.cuda.is_available()
        and torch.device(solver.device).type == "cuda"
        and bool(runtimes)
        and bool(port_runtimes)
        and all(runtime.delay_runtime is None for runtime in runtimes)
        and all(
            runtime.lumped is not None
            and runtime.embedded_network_name is not None
            and runtime.excitation is None
            and runtime.drive_accumulator is None
            for runtime in port_runtimes
        )
        and all(runtime.window_weights.shape[0] >= 4 for runtime in port_runtimes)
        and not solver._source_terms
        and not solver._electric_source_terms
        and not solver._magnetic_source_terms
        and not solver.tfsf_enabled
        and not solver.dft_enabled
        and not solver.observers
        and not getattr(solver, "time_observers", ())
        and not getattr(solver, "dispersive_enabled", False)
        and not getattr(solver, "nonlinear_enabled", False)
        and not getattr(solver, "nonlinear_general_enabled", False)
        and not getattr(solver, "modulation_enabled", False)
        and not getattr(solver, "has_mur_faces", False)
        and not getattr(solver, "surface_impedance_enabled", False)
    )
    if not graphable:
        return None

    from ..cuda.runtime.graph import CudaGraphRunner

    def step() -> None:
        _field_update_block(solver, 0.0)
        apply_port_runtimes(solver)
        apply_network_runtimes(solver, native_lu=True)
        solver._apply_dispersive_corrections()
        if not solver.tfsf_enabled:
            enforce_pec_boundaries(solver)
        apply_mur_boundaries(solver)
        from ..cut_surface import apply_cut_surface_constraints
        apply_cut_surface_constraints(solver)
        _accumulate_embedded_port_observers_gpu(solver)

    from ..checkpoint import graph_state_tensors

    mutable_tensors = list(graph_state_tensors(solver).values())
    for runtime in runtimes:
        mutable_tensors.extend(
            (
                runtime.state,
                runtime.next_state,
                runtime.state_drive,
                runtime.free_voltage,
                runtime.raw_free_voltage,
                runtime.carried_voltage,
                runtime.network_voltage,
                runtime.voltage_after,
                runtime.branch_current,
                runtime.output_buffer,
                runtime.direct_drive,
                runtime.net_power,
                runtime.power_buffer,
                runtime.port_energy,
                runtime.absorbed_increment,
                runtime.generated_increment,
                runtime.absorbed_energy,
                runtime.generated_energy,
            )
        )
        for group in runtime.terminal_groups:
            mutable_tensors.extend((group.edge_buffer, group.correction_buffer))
    observer_tensors = [solver._port_observer_step]
    for runtime in port_runtimes:
        accumulator = runtime.accumulator
        observer_tensors.extend(
            (
                runtime.observer_current_buffer,
                runtime.observer_window_buffer,
                runtime.observer_voltage_kernel_buffer,
                runtime.observer_current_kernel_buffer,
                runtime.electric_time,
                runtime.magnetic_time,
                accumulator._voltage_sum,
                accumulator._current_sum,
                accumulator._window_weight_sum,
                accumulator._voltage_term,
                accumulator._current_term,
            )
        )
    mutable_tensors.extend(observer_tensors)
    unique_tensors = []
    seen = set()
    for tensor in mutable_tensors:
        if tensor.data_ptr() not in seen:
            unique_tensors.append(tensor)
            seen.add(tensor.data_ptr())
    saved = [tensor.clone() for tensor in unique_tensors]

    def restore() -> None:
        for tensor, value in zip(unique_tensors, saved):
            tensor.copy_(value)

    try:
        replay = CudaGraphRunner(device=solver.device, enabled=True, warmup_steps=3).capture(step)
    except Exception:
        restore()
        return None
    restore()
    solver._cuda_graph_active = True
    solver._network_cuda_graph_active = True
    solver._port_observer_graph_active = True
    return replay


def solve(solver, *args, **kwargs):
    """Execute the ordinary solve without suspending between field updates."""
    iterator = solve_steps(solver, *args, **kwargs)
    try:
        next(iterator)
    except StopIteration as done:
        return done.value
    raise RuntimeError("An ordinary FDTD solve unexpectedly suspended.")


def solve_steps(
    solver,
    time_steps: int,
    dft_frequency: float = None,
    enable_plot: bool = False,
    dft_window: str = "hanning",
    full_field_dft: bool = True,
    normalize_source: bool = False,
    shutoff: float = 0.0,
    shutoff_check_interval: int = 100,
    use_cuda_graph: bool = False,
    resume_from=None,
    stop_step: int | None = None,
    cooperative: bool = False,
    cooperative_observer=None,
    cooperative_driver: bool = False,
    dft_components=("Ex", "Ey", "Ez"),
):
    """Shared solve lifecycle, optionally yielding before each field update.

    A cooperative caller performs the complete field block before resuming the
    generator. Source injection, boundary clamps, observation and normalization
    remain identical to the ordinary solve. The caller must validate that its
    field block supports every enabled physical capability.
    """
    if cooperative and use_cuda_graph:
        raise ValueError("Cooperative execution owns its field graph policy.")
    if solver.verbose:
        print(f"Starting 3D FDTD simulation (Yee grid), grid size: {solver.Nx}x{solver.Ny}x{solver.Nz}")
    from .timeplan import sources_share_spectrum
    if normalize_source and not sources_share_spectrum(getattr(solver, "_compiled_sources", ())):
        raise NotImplementedError("normalize_source requires all sources to share one source-time spectrum.")
    checkpoint_execution = resume_from is not None or stop_step is not None
    if checkpoint_execution and bool(getattr(solver, "breakdown_observers", ())):
        raise NotImplementedError("FDTD resume does not yet preserve breakdown observers.")
    if stop_step is not None:
        if isinstance(stop_step, bool) or not isinstance(stop_step, int):
            raise TypeError("stop_step must be an integer when provided.")
        if not 0 <= stop_step <= time_steps:
            raise ValueError("stop_step must satisfy 0 <= stop_step <= time_steps.")
    solver._normalize_source = normalize_source
    solver._synchronize_device()
    solve_start = time.perf_counter()

    if dft_frequency is not None and full_field_dft:
        solver.enable_dft(dft_frequency, window_type=dft_window, end_step=time_steps,
                          components=dft_components, reuse_buffers=cooperative_driver)
    else:
        solver.dft_enabled = False
        solver._dft_entries = []
        solver._sync_dft_legacy_state()

    observer_frequency = dft_frequency if dft_frequency is not None else solver.source_frequency
    if solver.observers:
        solver._prepare_observers(observer_frequency, dft_window, time_steps)
    if getattr(solver, "time_observers", None):
        solver._prepare_time_observers(time_steps)
    if getattr(solver, "breakdown_observers", None):
        solver._prepare_breakdown_observers()
    from ..observers import make_single_observer_engine
    run_spectral_observers = make_single_observer_engine(
        solver, enabled=use_cuda_graph and cooperative_observer is None)
    prepare_port_spectral_accumulators(solver, time_steps, dft_window)
    prepare_copper_spectra(solver, time_steps, observer_frequency, dft_window)
    prepare_circuit_time_series(solver, time_steps)
    circuit_runtimes = tuple(getattr(solver, "_circuit_runtimes", ()))
    if circuit_runtimes:
        prepare_circuit_graph_runners(solver, use_cuda_graph)
    prepare_wire_monitors(solver, time_steps, dft_window)

    # Deterministic breakdown mutates the electric update coefficients in place
    # between steps from an accumulated per-cell state; a captured CUDA graph would
    # freeze the pre-breakdown coefficient reads, so keep the eager path. The core
    # whole-step graph reads live coefficients and keeps the caller's request; its
    # per-step fallback below stays eager.
    core_use_cuda_graph = use_cuda_graph
    if getattr(solver, "breakdown_enabled", False):
        use_cuda_graph = False

    from .passivity import make_passivity_state, sample_passivity, total_energy, SAMPLE_INTERVAL

    solver._passivity_state = make_passivity_state(solver)
    solver._shutoff_triggered = False
    solver._shutoff_step = None
    solver._shutoff_peak = torch.zeros((), device=solver.device, dtype=solver.Ex.dtype)
    shutoff_min_step = _compute_shutoff_min_step(solver, shutoff_check_interval)

    run_full_embedded_step = _make_full_embedded_step_runner(
        solver,
        use_cuda_graph=use_cuda_graph,
    )
    if run_full_embedded_step is None:
        run_field_update = None if cooperative else _make_field_update_runner(solver, use_cuda_graph)
        run_network_updates = make_network_runner(
            solver,
            use_cuda_graph=use_cuda_graph,
        )
        run_port_observers = make_port_observer_runner(
            solver,
            use_cuda_graph=use_cuda_graph,
        )
    else:
        run_field_update = None
        run_network_updates = None
        run_port_observers = None

    # Drive the running DFT from a precomputed GPU weight table indexed by a
    # device step counter whenever the table can be built, dropping the per-step
    # host arithmetic and host->device transfer of the DFT weights. This is
    # independent of graph capture: the table rows are bit-identical to the
    # per-step host path, so eager runs take the same fast path. The host path
    # remains only as the fallback for the complex-field (split-field Bloch)
    # DFT, which the table builder declines.
    use_gpu_dft = False
    if getattr(solver, "dft_enabled", False):
        use_gpu_dft = solver.build_dft_step_tables(time_steps)
    run_tail = _make_tail_runner(solver, use_gpu_dft)
    solver._tail_graph_active = run_tail is not None

    start_step = 0
    if resume_from is not None:
        from ..resume import restore_resume_checkpoint

        start_step = restore_resume_checkpoint(
            solver,
            resume_from,
            total_steps=time_steps,
        )
    end_step = time_steps if stop_step is None else stop_step
    if end_step < start_step:
        raise ValueError(
            f"stop_step={end_step} precedes resume step {start_step}."
        )
    # Built after the resume restore: the drive tables start from the restored
    # port clocks and the device counter from the restored sample index.
    solver._port_cuda_graph_active = False
    run_port_step = None
    if run_full_embedded_step is None and end_step > start_step:
        run_port_step = _make_port_step_runner(
            solver,
            use_cuda_graph=use_cuda_graph,
            use_gpu_dft=use_gpu_dft,
            time_steps=time_steps,
        )

    measure_step_loop = bool(
        circuit_runtimes or getattr(solver, "_port_runtimes", ())
    )
    time_on_device = measure_step_loop and torch.device(solver.device).type == "cuda"
    step_start_event = torch.cuda.Event(enable_timing=True) if time_on_device else None
    step_end_event = torch.cuda.Event(enable_timing=True) if time_on_device else None
    if step_start_event is not None:
        step_start_event.record()
    host_step_start = time.perf_counter() if measure_step_loop and not time_on_device else None
    iterator = range(start_step, end_step)
    pbar = None
    if solver.verbose:
        pbar = tqdm(
            iterator,
            desc="FDTD",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        iterator = pbar

    steps_run = 0
    if cooperative_driver:
        steps_run = yield {"gpu_dft": use_gpu_dft, "shutoff_min_step": shutoff_min_step}
        iterator = ()
    if getattr(solver, "_core_runtime", None) is not None:
        core_steps = solver._core_runtime.run_chunks(
            solver, time_steps=time_steps, start_step=start_step, end_step=end_step,
            use_cuda_graph=core_use_cuda_graph, use_gpu_dft=use_gpu_dft,
            cooperative=cooperative, cooperative_driver=cooperative_driver,
            cooperative_observer=cooperative_observer, resume_from=resume_from,
            shutoff=shutoff, shutoff_check_interval=shutoff_check_interval,
            shutoff_min_step=shutoff_min_step, enable_plot=enable_plot, pbar=pbar,
        )
        if core_steps is not None:
            steps_run = core_steps
            iterator = ()
    for n in iterator:
        steps_run = n + 1
        time_value = n * solver.dt
        if solver.tfsf_enabled and solver._tfsf_state.get("auxiliary_grid") is not None:
            # Stage the physical entry-plane ghost samples outside field-graph
            # replay. The captured block reads persistent E/H boundary tensors.
            # Auxiliary-line protocol: every auxiliary incident line must implement
            # prepare_incident_boundary(values=None). It may be a no-op when the
            # line's incident_reference is None (no entry-plane reference).
            solver._tfsf_state["auxiliary_grid"].prepare_incident_boundary()
        if run_full_embedded_step is not None:
            run_full_embedded_step()
        else:
            if cooperative:
                yield n
            else:
                run_field_update(time_value)

            if solver.tfsf_enabled:
                apply_tfsf_e_correction(solver, time_value)
                advance_tfsf_auxiliary_electric(solver)
            if solver._electric_source_terms:
                # The electric equivalent current enters the E update through the
                # incident H field, which lives at the Yee half step.  Sample it at
                # the same n + 1/2 instant as the TFSF electric correction.
                inject_electric_surface_source_terms(
                    solver,
                    time_value=injection_time("equivalent_electric_current", time_value, solver.dt),
                )

            if solver._source_terms and cooperative_observer is None:
                solver.add_source(time_value=injection_time("electric_current", time_value, solver.dt))
            if run_port_step is not None:
                # Ports, networks, tail and port DFT in this same order.
                run_port_step()
            else:
                def post_source():
                    if run_tail is not None:
                        run_tail()
                    else:
                        solver._apply_dispersive_corrections()
                        if not solver.tfsf_enabled:
                            enforce_pec_boundaries(solver)
                        apply_mur_boundaries(solver)
                        from ..cut_surface import apply_cut_surface_constraints
                        apply_cut_surface_constraints(solver)
                        if use_gpu_dft:
                            solver.accumulate_dft_gpu()
                        else:
                            solver.accumulate_dft(n)

                if any(r.sheet_feed is not None for r in solver._port_runtimes):
                    _port_step_block(solver, None, post_source=post_source)
                else:
                    apply_port_runtimes(solver)
                    run_network_updates()
                    post_source()
                    run_port_observers()
        if getattr(solver, "breakdown_enabled", False):
            # After the full E-update (sources, PEC clamp, and tail applied), run the
            # per-cell breakdown state machine and scatter the updated conductivity
            # into the electric coefficients used by the next step.
            advance_breakdown_state(solver, n)
        if run_spectral_observers is not None:
            run_spectral_observers()
        elif cooperative_observer is None:
            solver.accumulate_observers(n)
        else:
            yield ("observers", n)
        solver.accumulate_time_observers(n)
        solver.accumulate_breakdown_observers(n)
        accumulate_wire_monitors(solver, n)

        if solver._passivity_state is not None and (n + 1) % SAMPLE_INTERVAL == 0:
            sample_passivity(solver._passivity_state, n + 1, total_energy(solver))
            if solver._passivity_state["warning"] is not None:
                solver.passivity_guard_status = "warned: complex fixed-k growth"
                solver.passivity_guard_warning = solver._passivity_state["warning"]

        if shutoff > 0 and (n + 1) % shutoff_check_interval == 0:
            e_energy = _electric_field_energy(solver)
            solver._shutoff_peak = torch.maximum(solver._shutoff_peak, e_energy)
            should_stop = (
                n >= shutoff_min_step
                and bool(((solver._shutoff_peak > 0.0) & (e_energy < shutoff * solver._shutoff_peak)).item())
            )
            if should_stop:
                solver._shutoff_triggered = True
                solver._shutoff_step = n
                break

        if pbar is not None:
            should_update_progress = ((n + 1) % solver.progress_update_interval == 0 or n == end_step - 1)
            if should_update_progress and solver.dft_enabled and solver.dft_start_step is not None:
                if n < solver.dft_start_step:
                    pbar.set_postfix({"status": "transient"})
                else:
                    pbar.set_postfix({"status": "DFT accumulation", "samples": solver.dft_sample_count})

        if enable_plot and n % solver.plot_interval == 0:
            visualize_slice(solver, n)

    if measure_step_loop:
        completed_steps = (
            solver._shutoff_step + 1 - start_step
            if solver._shutoff_triggered
            else end_step - start_step
        )
        if time_on_device:
            step_end_event.record()
            step_end_event.synchronize()
            solver.last_step_loop_elapsed_s = step_start_event.elapsed_time(step_end_event) * 1.0e-3
        else:
            solver.last_step_loop_elapsed_s = time.perf_counter() - host_step_start
        solver.last_step_loop_steps = completed_steps
    # steps_run, not n + 1: the loop variable is unbound for time_steps == 0 and
    # the graph-driven observer step counter must match the steps actually run,
    # including an early shutoff break.
    if cooperative_observer is not None:
        engine, lane = cooperative_observer
        engine.finalize(lane)
    if getattr(solver, "_single_observer_engine", None) is not None:
        solver._single_observer_engine.finalize(0)
        # The engine's slot holds the solver; drop the back-reference so a finished
        # solver and its field/CPML/DFT buffers are freed without the cyclic GC.
        solver._single_observer_engine = None
    complete_port_observer_graph(solver, steps_run)
    solver._synchronize_device()
    from .timestep import validate_nonlinear_field_envelope

    validate_nonlinear_field_envelope(solver)
    solver.last_solve_elapsed_s = time.perf_counter() - solve_start
    if end_step < time_steps:
        return None
    if solver._shutoff_triggered:
        _complete_spectral_normalization(solver, time_steps)
        complete_wire_monitor_normalization(solver, time_steps)
    if solver.dft_enabled:
        solver._sync_dft_legacy_state()
    if solver.observers_enabled:
        solver._sync_observer_legacy_state()
    if solver.verbose:
        print(
            f"FDTD solve elapsed: {solver.last_solve_elapsed_s:.3f}s "
            f"({solver.last_solve_elapsed_s * 1e3 / max(time_steps, 1):.3f} ms/step)"
        )

    output = {}
    if solver.dft_enabled:
        output.update(solver.get_frequency_solution(all_frequencies=True))
    monitors = solver.get_observer_results() if solver.observers_enabled else {}
    if getattr(solver, "time_observers_enabled", False):
        monitors = dict(monitors)
        monitors.update(solver.get_time_observer_results())
    if getattr(solver, "breakdown_observers_enabled", False):
        monitors = dict(monitors)
        monitors.update(solver.get_breakdown_observer_results())
    wire_monitors = finalize_wire_data(solver)
    if wire_monitors:
        monitors = dict(monitors)
        monitors.update(wire_monitors)
    if monitors:
        output["observers"] = monitors
    ports = finalize_port_data(solver)
    if ports:
        output["ports"] = ports
    circuits = finalize_circuit_data(solver)
    if circuits:
        output["circuits"] = circuits
    embedded_networks = finalize_embedded_networks(solver, ports)
    if embedded_networks:
        output["embedded_networks"] = embedded_networks
    return output or None
