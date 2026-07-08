from __future__ import annotations

import numpy as np
import torch

from ...compiler.materials import (
    evaluate_material_permeability,
    evaluate_material_permittivity,
)
from ..boundary import has_complex_fields


def _any_component_nonzero(components: dict[str, torch.Tensor]) -> bool:
    return any(torch.any(components[axis] != 0).item() for axis in ("x", "y", "z"))


def build_materials(solver, scene):
    material_model = scene.compile_materials()
    if scene.boundary.uses_kind("bloch") and torch.any(material_model["kerr_chi3"] != 0):
        raise NotImplementedError("FDTD Kerr media are not implemented for Bloch / complex-field runs.")

    solver._compiled_material_model = material_model
    eps_components = material_model["eps_components"]
    mu_components = material_model["mu_components"]
    solver.epsilon_r_components = eps_components
    solver.mu_r_components = mu_components
    solver.epsilon_r = material_model["eps_r"]
    solver.mu_r = material_model["mu_r"]
    solver.material_eps_r = evaluate_material_permittivity(material_model, solver.source_frequency)
    solver.material_mu_r = evaluate_material_permeability(material_model, solver.source_frequency)

    eps_x_node = eps_components["x"] * solver.eps0
    eps_y_node = eps_components["y"] * solver.eps0
    eps_z_node = eps_components["z"] * solver.eps0
    mu_x_node = mu_components["x"] * solver.mu0
    mu_y_node = mu_components["y"] * solver.mu0
    mu_z_node = mu_components["z"] * solver.mu0

    solver.eps_Ex = average_node_to_component(solver, eps_x_node, "Ex")
    solver.eps_Ey = average_node_to_component(solver, eps_y_node, "Ey")
    solver.eps_Ez = average_node_to_component(solver, eps_z_node, "Ez")
    sigma_e_components = material_model["sigma_e_components"]
    solver.sigma_e_Ex = average_node_to_component(solver, sigma_e_components["x"], "Ex")
    solver.sigma_e_Ey = average_node_to_component(solver, sigma_e_components["y"], "Ey")
    solver.sigma_e_Ez = average_node_to_component(solver, sigma_e_components["z"], "Ez")
    solver.conductive_enabled = bool(_any_component_nonzero(sigma_e_components))
    solver.mu_Hx = average_node_to_magnetic_component(solver, mu_x_node, "Hx")
    solver.mu_Hy = average_node_to_magnetic_component(solver, mu_y_node, "Hy")
    solver.mu_Hz = average_node_to_magnetic_component(solver, mu_z_node, "Hz")

    solver.kerr_enabled = bool(torch.any(material_model["kerr_chi3"] != 0).item())
    solver.kerr_chi3 = material_model["kerr_chi3"]
    if solver.kerr_enabled:
        solver.kerr_chi3_Ex = average_node_to_component(solver, solver.kerr_chi3, "Ex")
        solver.kerr_chi3_Ey = average_node_to_component(solver, solver.kerr_chi3, "Ey")
        solver.kerr_chi3_Ez = average_node_to_component(solver, solver.kerr_chi3, "Ez")
        solver.cex_curl_dynamic = torch.empty_like(solver.eps_Ex)
        solver.cey_curl_dynamic = torch.empty_like(solver.eps_Ey)
        solver.cez_curl_dynamic = torch.empty_like(solver.eps_Ez)
    else:
        solver.kerr_chi3_Ex = None
        solver.kerr_chi3_Ey = None
        solver.kerr_chi3_Ez = None
        solver.cex_curl_dynamic = None
        solver.cey_curl_dynamic = None
        solver.cez_curl_dynamic = None

    build_dispersive_templates(solver, material_model)
    build_magnetic_dispersive_templates(solver, material_model)
    solver.dispersive_enabled = solver.electric_dispersive_enabled or solver.magnetic_dispersive_enabled


def average_node_to_component(solver, node_tensor, component_name):
    del solver
    if component_name == "Ex":
        return (0.5 * (node_tensor[:-1, :, :] + node_tensor[1:, :, :])).contiguous()
    if component_name == "Ey":
        return (0.5 * (node_tensor[:, :-1, :] + node_tensor[:, 1:, :])).contiguous()
    if component_name == "Ez":
        return (0.5 * (node_tensor[:, :, :-1] + node_tensor[:, :, 1:])).contiguous()
    raise ValueError(f"Unsupported electric field component {component_name!r}.")


def average_node_to_magnetic_component(solver, node_tensor, component_name):
    del solver
    if component_name == "Hx":
        return (
            0.25
            * (
                node_tensor[:, :-1, :-1]
                + node_tensor[:, :-1, 1:]
                + node_tensor[:, 1:, :-1]
                + node_tensor[:, 1:, 1:]
            )
        ).contiguous()
    if component_name == "Hy":
        return (
            0.25
            * (
                node_tensor[:-1, :, :-1]
                + node_tensor[:-1, :, 1:]
                + node_tensor[1:, :, :-1]
                + node_tensor[1:, :, 1:]
            )
        ).contiguous()
    if component_name == "Hz":
        return (
            0.25
            * (
                node_tensor[:-1, :-1, :]
                + node_tensor[:-1, 1:, :]
                + node_tensor[1:, :-1, :]
                + node_tensor[1:, 1:, :]
            )
        ).contiguous()
    raise ValueError(f"Unsupported magnetic field component {component_name!r}.")


def build_dispersive_templates(solver, material_model):
    templates = {
        "Ex": {"inv_eps": (1.0 / solver.eps_Ex).contiguous(), "debye": [], "drude": [], "lorentz": []},
        "Ey": {"inv_eps": (1.0 / solver.eps_Ey).contiguous(), "debye": [], "drude": [], "lorentz": []},
        "Ez": {"inv_eps": (1.0 / solver.eps_Ez).contiguous(), "debye": [], "drude": [], "lorentz": []},
    }

    for entry in material_model["debye_poles"]:
        pole = entry["pole"]
        decay = (2.0 * pole.tau - solver.dt) / (2.0 * pole.tau + solver.dt)
        base_scale = 2.0 * solver.eps0 * pole.delta_eps * solver.dt / (2.0 * pole.tau + solver.dt)
        for component_name in ("Ex", "Ey", "Ez"):
            weight = average_node_to_component(solver, entry["weight"], component_name)
            templates[component_name]["debye"].append(
                {
                    "decay": float(decay),
                    "drive": (weight * base_scale).contiguous(),
                }
            )

    for entry in material_model["drude_poles"]:
        pole = entry["pole"]
        omega_p = 2.0 * np.pi * pole.plasma_frequency
        gamma = 2.0 * np.pi * pole.gamma
        denom = 2.0 + gamma * solver.dt
        decay = (2.0 - gamma * solver.dt) / denom
        base_scale = 2.0 * solver.eps0 * omega_p * omega_p * solver.dt / denom
        for component_name in ("Ex", "Ey", "Ez"):
            weight = average_node_to_component(solver, entry["weight"], component_name)
            templates[component_name]["drude"].append(
                {
                    "decay": float(decay),
                    "drive": (weight * base_scale).contiguous(),
                }
            )

    for entry in material_model["lorentz_poles"]:
        pole = entry["pole"]
        omega_0 = 2.0 * np.pi * pole.resonance_frequency
        gamma = 2.0 * np.pi * pole.gamma
        denom = 2.0 + gamma * solver.dt
        decay = (2.0 - gamma * solver.dt) / denom
        restoring = 2.0 * omega_0 * omega_0 * solver.dt / denom
        base_scale = 2.0 * solver.eps0 * pole.delta_eps * omega_0 * omega_0 * solver.dt / denom
        for component_name in ("Ex", "Ey", "Ez"):
            weight = average_node_to_component(solver, entry["weight"], component_name)
            templates[component_name]["lorentz"].append(
                {
                    "decay": float(decay),
                    "restoring": float(restoring),
                    "drive": (weight * base_scale).contiguous(),
                }
            )

    solver._dispersive_templates = templates
    solver.electric_dispersive_enabled = any(
        templates[component_name][model_name]
        for component_name in ("Ex", "Ey", "Ez")
        for model_name in ("debye", "drude", "lorentz")
    )


def build_magnetic_dispersive_templates(solver, material_model):
    templates = {
        "Hx": {"inv_mu": (1.0 / solver.mu_Hx).contiguous(), "debye": [], "drude": [], "lorentz": []},
        "Hy": {"inv_mu": (1.0 / solver.mu_Hy).contiguous(), "debye": [], "drude": [], "lorentz": []},
        "Hz": {"inv_mu": (1.0 / solver.mu_Hz).contiguous(), "debye": [], "drude": [], "lorentz": []},
    }

    for entry in material_model["mu_debye_poles"]:
        pole = entry["pole"]
        decay = (2.0 * pole.tau - solver.dt) / (2.0 * pole.tau + solver.dt)
        base_scale = 2.0 * solver.mu0 * pole.delta_eps * solver.dt / (2.0 * pole.tau + solver.dt)
        for component_name in ("Hx", "Hy", "Hz"):
            weight = average_node_to_magnetic_component(solver, entry["weight"], component_name)
            templates[component_name]["debye"].append(
                {
                    "decay": float(decay),
                    "drive": (weight * base_scale).contiguous(),
                }
            )

    for entry in material_model["mu_drude_poles"]:
        pole = entry["pole"]
        omega_p = 2.0 * np.pi * pole.plasma_frequency
        gamma = 2.0 * np.pi * pole.gamma
        denom = 2.0 + gamma * solver.dt
        decay = (2.0 - gamma * solver.dt) / denom
        base_scale = 2.0 * solver.mu0 * omega_p * omega_p * solver.dt / denom
        for component_name in ("Hx", "Hy", "Hz"):
            weight = average_node_to_magnetic_component(solver, entry["weight"], component_name)
            templates[component_name]["drude"].append(
                {
                    "decay": float(decay),
                    "drive": (weight * base_scale).contiguous(),
                }
            )

    for entry in material_model["mu_lorentz_poles"]:
        pole = entry["pole"]
        omega_0 = 2.0 * np.pi * pole.resonance_frequency
        gamma = 2.0 * np.pi * pole.gamma
        denom = 2.0 + gamma * solver.dt
        decay = (2.0 - gamma * solver.dt) / denom
        restoring = 2.0 * omega_0 * omega_0 * solver.dt / denom
        base_scale = 2.0 * solver.mu0 * pole.delta_eps * omega_0 * omega_0 * solver.dt / denom
        for component_name in ("Hx", "Hy", "Hz"):
            weight = average_node_to_magnetic_component(solver, entry["weight"], component_name)
            templates[component_name]["lorentz"].append(
                {
                    "decay": float(decay),
                    "restoring": float(restoring),
                    "drive": (weight * base_scale).contiguous(),
                }
            )

    solver._magnetic_dispersive_templates = templates
    solver.magnetic_dispersive_enabled = any(
        templates[component_name][model_name]
        for component_name in ("Hx", "Hy", "Hz")
        for model_name in ("debye", "drude", "lorentz")
    )


def initialize_dispersive_state(solver):
    if not solver._dispersive_templates:
        solver.electric_dispersive_enabled = False
        return

    for component_name, field in (("Ex", solver.Ex), ("Ey", solver.Ey), ("Ez", solver.Ez)):
        component_templates = solver._dispersive_templates[component_name]
        for entry in component_templates["debye"]:
            entry["polarization"] = torch.zeros_like(field)
            entry["current"] = torch.zeros_like(field)
            entry["polarization_imag"] = torch.zeros_like(field) if has_complex_fields(solver) else None
            entry["current_imag"] = torch.zeros_like(field) if has_complex_fields(solver) else None
        for entry in component_templates["drude"]:
            entry["current"] = torch.zeros_like(field)
            entry["current_imag"] = torch.zeros_like(field) if has_complex_fields(solver) else None
        for entry in component_templates["lorentz"]:
            entry["polarization"] = torch.zeros_like(field)
            entry["current"] = torch.zeros_like(field)
            if has_complex_fields(solver):
                entry["polarization_imag"] = torch.zeros_like(field)
                entry["current_imag"] = torch.zeros_like(field)
            else:
                entry["polarization_imag"] = None
                entry["current_imag"] = None


def initialize_magnetic_dispersive_state(solver):
    if not solver._magnetic_dispersive_templates:
        solver.magnetic_dispersive_enabled = False
        return

    for component_name, field in (("Hx", solver.Hx), ("Hy", solver.Hy), ("Hz", solver.Hz)):
        component_templates = solver._magnetic_dispersive_templates[component_name]
        for entry in component_templates["debye"]:
            entry["polarization"] = torch.zeros_like(field)
            entry["current"] = torch.zeros_like(field)
            entry["polarization_imag"] = torch.zeros_like(field) if has_complex_fields(solver) else None
            entry["current_imag"] = torch.zeros_like(field) if has_complex_fields(solver) else None
        for entry in component_templates["drude"]:
            entry["current"] = torch.zeros_like(field)
            entry["current_imag"] = torch.zeros_like(field) if has_complex_fields(solver) else None
        for entry in component_templates["lorentz"]:
            entry["polarization"] = torch.zeros_like(field)
            entry["current"] = torch.zeros_like(field)
            if has_complex_fields(solver):
                entry["polarization_imag"] = torch.zeros_like(field)
                entry["current_imag"] = torch.zeros_like(field)
            else:
                entry["polarization_imag"] = None
                entry["current_imag"] = None


def advance_component_dispersive_state(solver, component_name, field, *, imag=False):
    if not solver.electric_dispersive_enabled:
        return

    component_templates = solver._dispersive_templates.get(component_name)
    if not component_templates:
        return
    launch_shape = solver._field_launch_shapes[component_name]

    for entry in component_templates["debye"]:
        polarization = entry["polarization_imag"] if imag else entry["polarization"]
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.updateDebyeCurrent3D(
            ElectricField=field,
            Polarization=polarization,
            PolarizationCurrent=current,
            DebyeDrive=entry["drive"],
            decay=entry["decay"],
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["drude"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.updateDrudeCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            DrudeDrive=entry["drive"],
            decay=entry["decay"],
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["lorentz"]:
        polarization = entry["polarization_imag"] if imag else entry["polarization"]
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.updateLorentzCurrent3D(
            ElectricField=field,
            Polarization=polarization,
            PolarizationCurrent=current,
            LorentzDrive=entry["drive"],
            decay=entry["decay"],
            restoring=entry["restoring"],
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)


def advance_dispersive_state(solver):
    if not solver.electric_dispersive_enabled:
        return

    advance_component_dispersive_state(solver, "Ex", solver.Ex, imag=False)
    advance_component_dispersive_state(solver, "Ey", solver.Ey, imag=False)
    advance_component_dispersive_state(solver, "Ez", solver.Ez, imag=False)
    if has_complex_fields(solver):
        advance_component_dispersive_state(solver, "Ex", solver.Ex_imag, imag=True)
        advance_component_dispersive_state(solver, "Ey", solver.Ey_imag, imag=True)
        advance_component_dispersive_state(solver, "Ez", solver.Ez_imag, imag=True)


def advance_magnetic_component_dispersive_state(solver, component_name, field, *, imag=False):
    if not solver.magnetic_dispersive_enabled:
        return

    component_templates = solver._magnetic_dispersive_templates.get(component_name)
    if not component_templates:
        return
    launch_shape = solver._field_launch_shapes[component_name]

    for entry in component_templates["debye"]:
        polarization = entry["polarization_imag"] if imag else entry["polarization"]
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.updateDebyeCurrent3D(
            ElectricField=field,
            Polarization=polarization,
            PolarizationCurrent=current,
            DebyeDrive=entry["drive"],
            decay=entry["decay"],
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["drude"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.updateDrudeCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            DrudeDrive=entry["drive"],
            decay=entry["decay"],
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["lorentz"]:
        polarization = entry["polarization_imag"] if imag else entry["polarization"]
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.updateLorentzCurrent3D(
            ElectricField=field,
            Polarization=polarization,
            PolarizationCurrent=current,
            LorentzDrive=entry["drive"],
            decay=entry["decay"],
            restoring=entry["restoring"],
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)


def advance_magnetic_dispersive_state(solver):
    if not solver.magnetic_dispersive_enabled:
        return

    advance_magnetic_component_dispersive_state(solver, "Hx", solver.Hx, imag=False)
    advance_magnetic_component_dispersive_state(solver, "Hy", solver.Hy, imag=False)
    advance_magnetic_component_dispersive_state(solver, "Hz", solver.Hz, imag=False)
    if has_complex_fields(solver):
        advance_magnetic_component_dispersive_state(solver, "Hx", solver.Hx_imag, imag=True)
        advance_magnetic_component_dispersive_state(solver, "Hy", solver.Hy_imag, imag=True)
        advance_magnetic_component_dispersive_state(solver, "Hz", solver.Hz_imag, imag=True)


def apply_component_dispersive_currents(solver, component_name, field, *, imag=False):
    if not solver.electric_dispersive_enabled:
        return

    component_templates = solver._dispersive_templates.get(component_name)
    if not component_templates:
        return
    launch_shape = solver._field_launch_shapes[component_name]
    inv_eps = component_templates["inv_eps"]

    for entry in component_templates["debye"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_eps,
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["drude"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_eps,
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["lorentz"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_eps,
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)


def apply_dispersive_corrections(solver):
    if not solver.electric_dispersive_enabled:
        return

    apply_component_dispersive_currents(solver, "Ex", solver.Ex, imag=False)
    apply_component_dispersive_currents(solver, "Ey", solver.Ey, imag=False)
    apply_component_dispersive_currents(solver, "Ez", solver.Ez, imag=False)
    if has_complex_fields(solver):
        apply_component_dispersive_currents(solver, "Ex", solver.Ex_imag, imag=True)
        apply_component_dispersive_currents(solver, "Ey", solver.Ey_imag, imag=True)
        apply_component_dispersive_currents(solver, "Ez", solver.Ez_imag, imag=True)


def apply_magnetic_component_dispersive_currents(solver, component_name, field, *, imag=False):
    if not solver.magnetic_dispersive_enabled:
        return

    component_templates = solver._magnetic_dispersive_templates.get(component_name)
    if not component_templates:
        return
    launch_shape = solver._field_launch_shapes[component_name]
    inv_mu = component_templates["inv_mu"]

    for entry in component_templates["debye"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_mu,
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["drude"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_mu,
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["lorentz"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_mu,
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)


def apply_magnetic_dispersive_corrections(solver):
    if not solver.magnetic_dispersive_enabled:
        return

    apply_magnetic_component_dispersive_currents(solver, "Hx", solver.Hx, imag=False)
    apply_magnetic_component_dispersive_currents(solver, "Hy", solver.Hy, imag=False)
    apply_magnetic_component_dispersive_currents(solver, "Hz", solver.Hz, imag=False)
    if has_complex_fields(solver):
        apply_magnetic_component_dispersive_currents(solver, "Hx", solver.Hx_imag, imag=True)
        apply_magnetic_component_dispersive_currents(solver, "Hy", solver.Hy_imag, imag=True)
        apply_magnetic_component_dispersive_currents(solver, "Hz", solver.Hz_imag, imag=True)


def _electric_update_coefficients(solver, eps, sigma_e, pml_decay):
    """Compose the semi-implicit lossy-dielectric decay/curl factors for one E component.

    ``eps`` is the absolute permittivity (eps_r * eps0) and ``sigma_e`` the static
    electric conductivity averaged onto the same Yee component. When conductivity is
    inactive the material factors reduce to (decay=1, curl=dt/eps), so non-conductive
    scenes keep the exact existing coefficients. ``pml_decay`` folds the CPML decay in
    multiplicatively when present.
    """
    if solver.conductive_enabled:
        half = 0.5 * sigma_e * solver.dt / eps
        denom = 1.0 + half
        material_decay = (1.0 - half) / denom
        curl = (solver.dt / eps) / denom
    else:
        material_decay = None
        curl = solver.dt / eps

    if pml_decay is None:
        decay = material_decay if material_decay is not None else torch.ones_like(eps)
    else:
        decay = pml_decay if material_decay is None else material_decay * pml_decay
        curl = curl * pml_decay
    return decay.contiguous(), curl.contiguous()


def build_update_coefficients(solver):
    if solver.active_absorber_type not in ("pml", "absorber"):
        solver.cex_decay, solver.cex_curl = _electric_update_coefficients(solver, solver.eps_Ex, solver.sigma_e_Ex, None)
        solver.cey_decay, solver.cey_curl = _electric_update_coefficients(solver, solver.eps_Ey, solver.sigma_e_Ey, None)
        solver.cez_decay, solver.cez_curl = _electric_update_coefficients(solver, solver.eps_Ez, solver.sigma_e_Ez, None)
        solver.chx_decay = torch.ones_like(solver.mu_Hx).contiguous()
        solver.chx_curl = (solver.dt / solver.mu_Hx).contiguous()
        solver.chy_decay = torch.ones_like(solver.mu_Hy).contiguous()
        solver.chy_curl = (solver.dt / solver.mu_Hy).contiguous()
        solver.chz_decay = torch.ones_like(solver.mu_Hz).contiguous()
        solver.chz_curl = (solver.dt / solver.mu_Hz).contiguous()
        return

    ex_sigma_y = 0.5 * (solver.sigma_y[:-1, :, :] + solver.sigma_y[1:, :, :])
    ex_sigma_z = 0.5 * (solver.sigma_z[:-1, :, :] + solver.sigma_z[1:, :, :])
    ex_decay = 1.0 / (1.0 + solver.dt * (ex_sigma_y + ex_sigma_z))
    solver.cex_decay, solver.cex_curl = _electric_update_coefficients(solver, solver.eps_Ex, solver.sigma_e_Ex, ex_decay)

    ey_sigma_x = 0.5 * (solver.sigma_x[:, :-1, :] + solver.sigma_x[:, 1:, :])
    ey_sigma_z = 0.5 * (solver.sigma_z[:, :-1, :] + solver.sigma_z[:, 1:, :])
    ey_decay = 1.0 / (1.0 + solver.dt * (ey_sigma_x + ey_sigma_z))
    solver.cey_decay, solver.cey_curl = _electric_update_coefficients(solver, solver.eps_Ey, solver.sigma_e_Ey, ey_decay)

    ez_sigma_x = 0.5 * (solver.sigma_x[:, :, :-1] + solver.sigma_x[:, :, 1:])
    ez_sigma_y = 0.5 * (solver.sigma_y[:, :, :-1] + solver.sigma_y[:, :, 1:])
    ez_decay = 1.0 / (1.0 + solver.dt * (ez_sigma_x + ez_sigma_y))
    solver.cez_decay, solver.cez_curl = _electric_update_coefficients(solver, solver.eps_Ez, solver.sigma_e_Ez, ez_decay)

    hx_sigma_y = 0.25 * (
        solver.sigma_y[:, :-1, :-1]
        + solver.sigma_y[:, 1:, :-1]
        + solver.sigma_y[:, :-1, 1:]
        + solver.sigma_y[:, 1:, 1:]
    )
    hx_sigma_z = 0.25 * (
        solver.sigma_z[:, :-1, :-1]
        + solver.sigma_z[:, 1:, :-1]
        + solver.sigma_z[:, :-1, 1:]
        + solver.sigma_z[:, 1:, 1:]
    )
    hx_decay = 1.0 / (1.0 + solver.dt * (hx_sigma_y + hx_sigma_z))
    solver.chx_decay = hx_decay.contiguous()
    solver.chx_curl = ((solver.dt / solver.mu_Hx) * hx_decay).contiguous()

    hy_sigma_x = 0.25 * (
        solver.sigma_x[:-1, :, :-1]
        + solver.sigma_x[1:, :, :-1]
        + solver.sigma_x[:-1, :, 1:]
        + solver.sigma_x[1:, :, 1:]
    )
    hy_sigma_z = 0.25 * (
        solver.sigma_z[:-1, :, :-1]
        + solver.sigma_z[1:, :, :-1]
        + solver.sigma_z[:-1, :, 1:]
        + solver.sigma_z[1:, :, 1:]
    )
    hy_decay = 1.0 / (1.0 + solver.dt * (hy_sigma_x + hy_sigma_z))
    solver.chy_decay = hy_decay.contiguous()
    solver.chy_curl = ((solver.dt / solver.mu_Hy) * hy_decay).contiguous()

    hz_sigma_x = 0.25 * (
        solver.sigma_x[:-1, :-1, :]
        + solver.sigma_x[1:, :-1, :]
        + solver.sigma_x[:-1, 1:, :]
        + solver.sigma_x[1:, 1:, :]
    )
    hz_sigma_y = 0.25 * (
        solver.sigma_y[:-1, :-1, :]
        + solver.sigma_y[1:, :-1, :]
        + solver.sigma_y[:-1, 1:, :]
        + solver.sigma_y[1:, 1:, :]
    )
    hz_decay = 1.0 / (1.0 + solver.dt * (hz_sigma_x + hz_sigma_y))
    solver.chz_decay = hz_decay.contiguous()
    solver.chz_curl = ((solver.dt / solver.mu_Hz) * hz_decay).contiguous()


def update_kerr_electric_curls(solver):
    if not solver.kerr_enabled:
        return

    solver.fdtd_module.updateKerrElectricFieldExCurl3D(
        DynamicCurl=solver.cex_curl_dynamic,
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        LinearPermittivity=solver.eps_Ex,
        ExDecay=solver.cex_decay,
        KerrChi3=solver.kerr_chi3_Ex,
        dt=solver.dt,
        eps0=solver.eps0,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
    solver.fdtd_module.updateKerrElectricFieldEyCurl3D(
        DynamicCurl=solver.cey_curl_dynamic,
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        LinearPermittivity=solver.eps_Ey,
        EyDecay=solver.cey_decay,
        KerrChi3=solver.kerr_chi3_Ey,
        dt=solver.dt,
        eps0=solver.eps0,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
    solver.fdtd_module.updateKerrElectricFieldEzCurl3D(
        DynamicCurl=solver.cez_curl_dynamic,
        Ex=solver.Ex,
        Ey=solver.Ey,
        Ez=solver.Ez,
        LinearPermittivity=solver.eps_Ez,
        EzDecay=solver.cez_decay,
        KerrChi3=solver.kerr_chi3_Ez,
        dt=solver.dt,
        eps0=solver.eps0,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])
