from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from ...visualization import visualize_slice
from ..boundary import (
    BOUNDARY_BLOCH,
    BOUNDARY_PEC,
    BOUNDARY_PERIODIC,
    BOUNDARY_PML,
    has_complex_fields,
    initialize_boundary_state,
)
from ..excitation import (
    advance_tfsf_auxiliary_electric,
    advance_tfsf_auxiliary_magnetic,
    apply_tfsf_e_correction,
    apply_tfsf_h_correction,
    inject_electric_surface_source_terms,
    inject_magnetic_surface_source_terms,
    initialize_source_terms,
    initialize_tfsf_state,
)


def compute_linear_launch_shape(solver, length):
    threads_per_block = (
        solver.kernel_block_size[0] * solver.kernel_block_size[1] * solver.kernel_block_size[2]
    )
    grid_x = max(1, (int(length) + threads_per_block - 1) // threads_per_block)
    return (grid_x, 1, 1)


def compute_face_launch_shape(solver, field, axis):
    sizes = tuple(int(dim) for dim in field.shape)
    face_sizes = (
        sizes[1] * sizes[2],
        sizes[0] * sizes[2],
        sizes[0] * sizes[1],
    )
    return compute_linear_launch_shape(solver, face_sizes[int(axis)])


def refresh_launch_shapes(solver):
    solver._field_launch_shapes = {
        "Ex": compute_linear_launch_shape(solver, int(solver.Ex.numel())),
        "Ey": compute_linear_launch_shape(solver, int(solver.Ey.numel())),
        "Ez": compute_linear_launch_shape(solver, int(solver.Ez.numel())),
        "Hx": compute_linear_launch_shape(solver, int(solver.Hx.numel())),
        "Hy": compute_linear_launch_shape(solver, int(solver.Hy.numel())),
        "Hz": compute_linear_launch_shape(solver, int(solver.Hz.numel())),
    }
    electric_numel = max(int(solver.Ex.numel()), int(solver.Ey.numel()), int(solver.Ez.numel()))
    solver._spectral_launch_shapes = {"electric": compute_linear_launch_shape(solver, electric_numel)}


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
            "grid": compute_linear_launch_shape(solver, int(psi_region.numel())),
        }


def cpml_layout_params(solver, attr_name):
    layout = getattr(solver, "_cpml_memory_layouts", {}).get(attr_name)
    if layout is None:
        return 0, 0, 0
    low = next((region for region in layout["regions"] if region["side"] == "low"), None)
    high = next((region for region in layout["regions"] if region["side"] == "high"), None)
    low_length = 0 if low is None else int(low["length"])
    high_start = 0 if high is None else int(high["global_start"])
    high_length = 0 if high is None else int(high["length"])
    return low_length, high_start, high_length


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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hx"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hy"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hz"])


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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hx"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hy"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hz"])


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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hx"])
    solver.fdtd_module.updateMagneticFieldHyStandard3D(
        Hy=hy,
        Ex=ex,
        Ez=ez,
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hy"])
    solver.fdtd_module.updateMagneticFieldHzStandard3D(
        Hz=hz,
        Ex=ex,
        Ey=ey,
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Hz"])


def update_magnetic_fields(solver, hx, hy, hz, ex, ey, ez, *, imag=False):
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def update_electric_fields_cpml(solver, ex, ey, ez, hx, hy, hz):
    if solver._cpml_memory_mode == "dense":
        update_electric_fields_cpml_dense(solver, ex, ey, ez, hx, hy, hz)
        return
    update_electric_fields_cpml_compressed(solver, ex, ey, ez, hx, hy, hz)


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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def _modulation_phase_factors(solver, time_value):
    """Host-side ``(cos, sin)`` pairs of the modulation phase at the previous and
    new E-field time instants. The E update advances E from ``time_value - dt`` to
    ``time_value`` (source injection at ``time_value`` follows the update)."""
    omega = float(solver.modulation_angular_frequency)
    t_next = float(time_value)
    t_prev = t_next - float(solver.dt)
    return (
        float(np.cos(omega * t_prev)),
        float(np.sin(omega * t_prev)),
        float(np.cos(omega * t_next)),
        float(np.sin(omega * t_next)),
    )


def update_electric_fields_modulated_standard(solver, ex, ey, ez, hx, hy, hz, time_value):
    cos_prev, sin_prev, cos_next, sin_next = _modulation_phase_factors(solver, time_value)
    solver.fdtd_module.updateElectricFieldExModulated3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=solver.cex_decay,
        ExCurl=solver.cex_curl,
        ModCos=solver.mod_cos_Ex,
        ModSin=solver.mod_sin_Ex,
        cosPrev=cos_prev,
        sinPrev=sin_prev,
        cosNext=cos_next,
        sinNext=sin_next,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
    solver.fdtd_module.updateElectricFieldEyModulated3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=solver.cey_decay,
        EyCurl=solver.cey_curl,
        ModCos=solver.mod_cos_Ey,
        ModSin=solver.mod_sin_Ey,
        cosPrev=cos_prev,
        sinPrev=sin_prev,
        cosNext=cos_next,
        sinNext=sin_next,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
    solver.fdtd_module.updateElectricFieldEzModulated3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=solver.cez_decay,
        EzCurl=solver.cez_curl,
        ModCos=solver.mod_cos_Ez,
        ModSin=solver.mod_sin_Ez,
        cosPrev=cos_prev,
        sinPrev=sin_prev,
        cosNext=cos_next,
        sinNext=sin_next,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def update_electric_fields_modulated_cpml(solver, ex, ey, ez, hx, hy, hz, time_value):
    if solver._cpml_memory_mode != "dense":
        raise RuntimeError(
            "FDTD time-modulated media require the dense CPML memory mode; "
            "the boundary initialization should have forced it."
        )
    cos_prev, sin_prev, cos_next, sin_next = _modulation_phase_factors(solver, time_value)
    solver.fdtd_module.updateElectricFieldExCpmlModulated3D(
        Ex=ex,
        Hy=hy,
        Hz=hz,
        ExDecay=solver.cex_decay,
        ExCurl=solver.cex_curl,
        ModCos=solver.mod_cos_Ex,
        ModSin=solver.mod_sin_Ex,
        cosPrev=cos_prev,
        sinPrev=sin_prev,
        cosNext=cos_next,
        sinNext=sin_next,
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
    solver.fdtd_module.updateElectricFieldEyCpmlModulated3D(
        Ey=ey,
        Hx=hx,
        Hz=hz,
        EyDecay=solver.cey_decay,
        EyCurl=solver.cey_curl,
        ModCos=solver.mod_cos_Ey,
        ModSin=solver.mod_sin_Ey,
        cosPrev=cos_prev,
        sinPrev=sin_prev,
        cosNext=cos_next,
        sinNext=sin_next,
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
    solver.fdtd_module.updateElectricFieldEzCpmlModulated3D(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        EzDecay=solver.cez_decay,
        EzCurl=solver.cez_curl,
        ModCos=solver.mod_cos_Ez,
        ModSin=solver.mod_sin_Ez,
        cosPrev=cos_prev,
        sinPrev=sin_prev,
        cosNext=cos_next,
        sinNext=sin_next,
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def update_electric_fields(solver, ex, ey, ez, hx, hy, hz, *, time_value=None):
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

    Each kernel neighbor-averages the two off-axis curl(H) components onto the
    target Yee edge and accumulates ``coeff * <curlH>``; the coefficients are the
    off-diagonal entries of the per-edge inverse permittivity tensor scaled by
    ``dt/eps0`` and vanish outside the anisotropic structures. Periodic axes
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def update_electric_fields_bloch(solver):
    if getattr(solver, "nonlinear_enabled", False):
        raise NotImplementedError("FDTD nonlinear media are not implemented for Bloch / complex-field runs.")
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def update_electric_fields_bloch_xy_standard_z(solver):
    if getattr(solver, "nonlinear_enabled", False):
        raise NotImplementedError("FDTD nonlinear media are not implemented for Bloch / complex-field runs.")
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
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
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def _validate_bloch_cpml_update_layout(solver):
    if tuple(getattr(solver, "has_bloch_axes", ())) != ("x", "y"):
        raise NotImplementedError("Mixed Bloch/CPML FDTD updates currently support x/y Bloch with z PML only.")
    if (
        solver.boundary_x_low_code != BOUNDARY_BLOCH
        or solver.boundary_x_high_code != BOUNDARY_BLOCH
        or solver.boundary_y_low_code != BOUNDARY_BLOCH
        or solver.boundary_y_high_code != BOUNDARY_BLOCH
        or solver.boundary_z_low_code != BOUNDARY_PML
        or solver.boundary_z_high_code != BOUNDARY_PML
    ):
        raise NotImplementedError("Mixed Bloch/CPML FDTD updates currently support paired x/y Bloch faces and z PML faces only.")


def _iter_cpml_field_regions(solver, field, curl, attr_name):
    layout = getattr(solver, "_cpml_memory_layouts", {}).get(attr_name)
    if layout is None:
        field_key = "Ex" if attr_name.startswith("psi_ex_z") else "Ey"
        yield {
            "field": field,
            "curl": curl,
            "psi": getattr(solver, attr_name),
            "offsets": (0, 0, 0),
            "grid": solver._field_launch_shapes[field_key],
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
            "grid": region["grid"],
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
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=region["grid"])

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
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=region["grid"])


def update_electric_fields_bloch_cpml(solver):
    _validate_bloch_cpml_update_layout(solver)
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


def clamp_field_face(solver, field, axis, side):
    side_code = 0 if side == "low" else 1
    module = getattr(solver, "fdtd_module", None)
    if module is not None and module.__class__.__name__ == "NativeFDTDModule":
        module.clampFieldFace3D(field=field, axis=int(axis), side=side_code).launchRaw(
            blockSize=getattr(solver, "kernel_block_size", None),
            gridSize=None,
        )
        return
    index = 0 if side_code == 0 else field.shape[axis] - 1
    field.select(axis, index).zero_()


def _launch_native_clamp(solver, kernel_name, **kwargs):
    module = getattr(solver, "fdtd_module", None)
    if module is None or module.__class__.__name__ != "NativeFDTDModule":
        return False
    getattr(module, kernel_name)(**kwargs).launchRaw(
        blockSize=getattr(solver, "kernel_block_size", None),
        gridSize=None,
    )
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
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=None)


def init_field(solver):
    solver.Ex = torch.zeros((solver.Nx - 1, solver.Ny, solver.Nz), device=solver.device, dtype=torch.float32)
    solver.Ey = torch.zeros((solver.Nx, solver.Ny - 1, solver.Nz), device=solver.device, dtype=torch.float32)
    solver.Ez = torch.zeros((solver.Nx, solver.Ny, solver.Nz - 1), device=solver.device, dtype=torch.float32)
    solver.Hx = torch.zeros((solver.Nx, solver.Ny - 1, solver.Nz - 1), device=solver.device, dtype=torch.float32)
    solver.Hy = torch.zeros((solver.Nx - 1, solver.Ny, solver.Nz - 1), device=solver.device, dtype=torch.float32)
    solver.Hz = torch.zeros((solver.Nx - 1, solver.Ny - 1, solver.Nz), device=solver.device, dtype=torch.float32)
    refresh_launch_shapes(solver)

    solver.build_materials(solver.scene)
    initialize_boundary_state(solver)
    solver._build_update_coefficients()
    solver._initialize_dispersive_state()
    solver._initialize_magnetic_dispersive_state()
    initialize_tfsf_state(solver)
    initialize_source_terms(solver)


def _compute_shutoff_min_step(solver, shutoff_check_interval: int) -> int:
    import math

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


def _electric_field_energy(solver) -> float:
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
    return float(energy.item())


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
    # After an early auto-shutoff the omitted tail steps carry negligible field, so
    # the running-DFT / observer numerators are already complete. Restore each
    # spectral normalizer to its planned full-window value so the
    # 2 / window_normalization scale is not inflated by the shortened run.
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


def _field_update_block(solver, time_value):
    """One step of the time-marching field-update core: the magnetic and electric
    Yee updates plus their in-place dispersive/Kerr/CPML/Bloch state advances.

    This is exactly the contiguous, time-marching part of the step that carries no
    per-step host input when TFSF and magnetic sources are absent, so it can be
    captured once into a CUDA graph and replayed (the additive/surface source and
    the running DFT stay outside the graph). Kept as a single function so the
    normal and graph-captured paths execute an identical kernel sequence.
    """
    solver._advance_magnetic_dispersive_state()
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
    if solver.tfsf_enabled:
        apply_tfsf_h_correction(solver, time_value)
        advance_tfsf_auxiliary_magnetic(solver)
    if solver._magnetic_source_terms:
        inject_magnetic_surface_source_terms(solver, time_value=time_value)
    solver._apply_magnetic_dispersive_corrections()

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


def _make_field_update_runner(solver, use_cuda_graph: bool):
    """Return ``run(time_value)`` for the field-update block, capturing it into a
    CUDA graph when requested and safe. Falls back to the direct call for TFSF /
    magnetic-source scenes (whose block carries per-step host input) or if capture
    is unavailable. Capture happens on the zero initial field, a fixed point of
    the source-free block, so it does not perturb the physical run.
    """
    solver._cuda_graph_active = False

    def normal(time_value):
        _field_update_block(solver, time_value)

    # v1 scope: the standard real-field path (optionally conductive). TFSF and
    # magnetic sources put per-step host input inside the block; complex/Kerr/
    # dispersive paths carry extra evolving state left to a later iteration.
    graphable = (
        use_cuda_graph
        and torch.cuda.is_available()
        and not solver.tfsf_enabled
        and not solver._magnetic_source_terms
        and not has_complex_fields(solver)
        and not getattr(solver, "nonlinear_enabled", False)
        and not getattr(solver, "dispersive_enabled", False)
        # The modulated E update consumes per-step host phase scalars, which a
        # captured CUDA graph would freeze at their capture-time values.
        and not getattr(solver, "modulation_enabled", False)
    )
    if not graphable:
        return normal
    from ..cuda.runtime.graph import CudaGraphRunner

    # Snapshot the state this block mutates (fields + CPML psi) so warmup/capture
    # do not perturb the physical run, then restore before stepping begins.
    state = {
        k: v
        for k, v in vars(solver).items()
        if isinstance(v, torch.Tensor)
        and (k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") or k.startswith("psi"))
    }
    saved = {k: v.clone() for k, v in state.items()}

    def _restore():
        for k, v in saved.items():
            state[k].copy_(v)

    try:
        runner = CudaGraphRunner(enabled=True, warmup_steps=3)
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
    solver.accumulate_dft_gpu()


def _make_tail_runner(solver, use_gpu_dft: bool):
    """Capture the post-source tail (PEC + Mur + GPU-DFT) into a graph. Returns a
    replay callable, or ``None`` to keep the inline tail. Snapshots the state the
    tail mutates (fields, the device step counter, the DFT accumulators) around
    capture so warmup/capture on the zero field do not perturb the run."""
    if not (use_gpu_dft and getattr(solver, "_cuda_graph_active", False)):
        return None
    from ..cuda.runtime.graph import CudaGraphRunner

    state = {}
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        tensor = getattr(solver, name, None)
        if isinstance(tensor, torch.Tensor):
            state[("field", name)] = tensor
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
        replay = CudaGraphRunner(enabled=True, warmup_steps=3).capture(
            lambda: _post_source_block(solver)
        )
    except Exception:
        _restore()
        return None
    _restore()
    return replay


def solve(
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
):
    if solver.verbose:
        print(f"Starting 3D FDTD simulation (Yee grid), grid size: {solver.Nx}x{solver.Ny}x{solver.Nz}")
    tfsf_state = getattr(solver, "_tfsf_state", None)
    if isinstance(tfsf_state, dict) and tfsf_state.get("runtime_pending", False):
        raise NotImplementedError("TFSF slab forward runtime support is not implemented yet.")
    if normalize_source and len(getattr(solver, "_compiled_sources", ())) != 1:
        raise NotImplementedError("normalize_source currently requires exactly one compiled source.")
    solver._normalize_source = normalize_source
    solver._synchronize_device()
    solve_start = time.perf_counter()

    if dft_frequency is not None and full_field_dft:
        solver.enable_dft(dft_frequency, window_type=dft_window, end_step=time_steps)
    else:
        solver.dft_enabled = False
        solver._dft_entries = []
        solver._sync_dft_legacy_state()

    observer_frequency = dft_frequency if dft_frequency is not None else solver.source_frequency
    if solver.observers:
        solver._prepare_observers(observer_frequency, dft_window, time_steps)
    if getattr(solver, "time_observers", None):
        solver._prepare_time_observers(time_steps)

    solver._shutoff_triggered = False
    solver._shutoff_step = None
    solver._shutoff_peak = 0.0
    shutoff_min_step = _compute_shutoff_min_step(solver, shutoff_check_interval)

    run_field_update = _make_field_update_runner(solver, use_cuda_graph)

    # When the field-update graph is active, drive the running DFT from a
    # precomputed GPU weight table indexed by a device step counter, dropping the
    # per-step host arithmetic and host->device transfer of the DFT weights.
    use_gpu_dft = False
    if getattr(solver, "_cuda_graph_active", False) and getattr(solver, "dft_enabled", False):
        use_gpu_dft = solver.build_dft_step_tables(time_steps)
    run_tail = _make_tail_runner(solver, use_gpu_dft)
    solver._tail_graph_active = run_tail is not None

    iterator = range(time_steps)
    pbar = None
    if solver.verbose:
        pbar = tqdm(
            iterator,
            desc="FDTD",
            unit="step",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        iterator = pbar

    for n in iterator:
        time_value = n * solver.dt
        run_field_update(time_value)

        if solver.tfsf_enabled:
            apply_tfsf_e_correction(solver, time_value)
            advance_tfsf_auxiliary_electric(solver)
        if solver._electric_source_terms:
            inject_electric_surface_source_terms(solver, time_value=time_value)

        if solver._source_terms:
            solver.add_source(time_value=time_value)
        if run_tail is not None:
            run_tail()
        else:
            solver._apply_dispersive_corrections()
            if not solver.tfsf_enabled:
                enforce_pec_boundaries(solver)
            apply_mur_boundaries(solver)
            if use_gpu_dft:
                solver.accumulate_dft_gpu()
            else:
                solver.accumulate_dft(n)
        solver.accumulate_observers(n)
        solver.accumulate_time_observers(n)

        if shutoff > 0 and (n + 1) % shutoff_check_interval == 0:
            e_energy = _electric_field_energy(solver)
            solver._shutoff_peak = max(solver._shutoff_peak, e_energy)
            if (
                solver._shutoff_peak > 0.0
                and n >= shutoff_min_step
                and e_energy < shutoff * solver._shutoff_peak
            ):
                solver._shutoff_triggered = True
                solver._shutoff_step = n
                break

        if pbar is not None:
            should_update_progress = ((n + 1) % solver.progress_update_interval == 0 or n == time_steps - 1)
            if should_update_progress and solver.dft_enabled and solver.dft_start_step is not None:
                if n < solver.dft_start_step:
                    pbar.set_postfix({"status": "transient"})
                else:
                    pbar.set_postfix({"status": "DFT accumulation", "samples": solver.dft_sample_count})

        if enable_plot and n % solver.plot_interval == 0:
            visualize_slice(solver, n)

    solver._synchronize_device()
    solver.last_solve_elapsed_s = time.perf_counter() - solve_start
    if solver._shutoff_triggered:
        _complete_spectral_normalization(solver, time_steps)
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
    if monitors:
        output["observers"] = monitors
    return output or None
