from __future__ import annotations

import numpy as np
import torch

from ...compiler.materials import (
    evaluate_material_permeability,
    evaluate_material_permittivity,
    material_model_has_full_anisotropy,
    material_model_has_modulation,
    material_model_has_nonlinearity,
)
from ..boundary import BOUNDARY_PERIODIC, has_complex_fields


def _any_component_nonzero(components: dict[str, torch.Tensor]) -> bool:
    return any(torch.any(components[axis] != 0).item() for axis in ("x", "y", "z"))


def build_materials(solver, scene):
    material_model = scene.compile_materials()
    if scene.boundary.uses_kind("bloch") and material_model_has_nonlinearity(material_model):
        raise NotImplementedError("FDTD nonlinear media are not implemented for Bloch / complex-field runs.")
    if scene.boundary.uses_kind("bloch") and material_model_has_full_anisotropy(material_model):
        raise NotImplementedError(
            "FDTD full (off-diagonal) anisotropic media are not implemented for Bloch / complex-field runs."
        )
    if scene.boundary.uses_kind("bloch") and material_model_has_modulation(material_model):
        raise NotImplementedError(
            "FDTD time-modulated media are not implemented for Bloch / complex-field runs."
        )

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

    build_nonlinear_channels(solver, material_model)

    build_full_anisotropy(solver, material_model)

    build_modulation(solver, material_model)

    build_dispersive_templates(solver, material_model)
    build_magnetic_dispersive_templates(solver, material_model)
    solver.dispersive_enabled = solver.electric_dispersive_enabled or solver.magnetic_dispersive_enabled
    # Full (off-diagonal) anisotropic permittivity now composes with dispersion:
    # electric poles enter isotropically and the ADE polarization current is folded
    # through the same per-edge inverse permittivity tensor as curl(H) (see
    # _apply_aniso_dispersive_corrections); magnetic dispersion is an independent
    # H-side channel and does not touch the electric tensor.
    if solver.modulation_enabled:
        if solver.nonlinear_enabled:
            raise NotImplementedError(
                "FDTD time-modulated media cannot be combined with nonlinear media in the same Scene yet."
            )
        if solver.full_aniso_enabled:
            raise NotImplementedError(
                "FDTD time-modulated media cannot be combined with full (off-diagonal) anisotropic media in the same Scene yet."
            )
        if solver.dispersive_enabled:
            raise NotImplementedError(
                "FDTD time-modulated media cannot be combined with dispersive materials in the same Scene yet."
            )


def build_nonlinear_channels(solver, material_model):
    """Set the nonlinear feature flags and edge-averaged nonlinear channel fields.

    ``kerr_enabled`` marks the instantaneous chi3 (Kerr) channel and keeps the
    existing dynamic-curl fast path when it is the only nonlinearity.
    ``nonlinear_general_enabled`` switches to the general nonlinear coefficient
    kernel, which additionally handles the chi2 channel and a field-dependent
    conductivity and therefore rewrites both the decay and curl coefficients
    every step.
    """
    solver.kerr_enabled = bool(torch.any(material_model["kerr_chi3"] != 0).item())
    solver.chi2_enabled = bool(torch.any(material_model["chi2"] != 0).item())
    solver.tpa_enabled = bool(torch.any(material_model["tpa_sigma"] != 0).item())
    solver.nonlinear_enabled = solver.kerr_enabled or solver.chi2_enabled or solver.tpa_enabled
    solver.nonlinear_general_enabled = solver.chi2_enabled or solver.tpa_enabled
    solver.kerr_chi3 = material_model["kerr_chi3"]
    if solver.nonlinear_enabled:
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
    if solver.nonlinear_general_enabled:
        chi2 = material_model["chi2"]
        solver.nonlinear_chi2_Ex = average_node_to_component(solver, chi2, "Ex")
        solver.nonlinear_chi2_Ey = average_node_to_component(solver, chi2, "Ey")
        solver.nonlinear_chi2_Ez = average_node_to_component(solver, chi2, "Ez")
        # Field-dependent conductivity channel of the general nonlinear kernel:
        # sigma_NL = tpa_sigma * |E|^2, from TwoPhotonAbsorption descriptors.
        tpa_sigma = material_model["tpa_sigma"]
        solver.tpa_sigma_Ex = average_node_to_component(solver, tpa_sigma, "Ex")
        solver.tpa_sigma_Ey = average_node_to_component(solver, tpa_sigma, "Ey")
        solver.tpa_sigma_Ez = average_node_to_component(solver, tpa_sigma, "Ez")
        solver.cex_decay_dynamic = torch.empty_like(solver.eps_Ex)
        solver.cey_decay_dynamic = torch.empty_like(solver.eps_Ey)
        solver.cez_decay_dynamic = torch.empty_like(solver.eps_Ez)
    else:
        solver.nonlinear_chi2_Ex = None
        solver.nonlinear_chi2_Ey = None
        solver.nonlinear_chi2_Ez = None
        solver.tpa_sigma_Ex = None
        solver.tpa_sigma_Ey = None
        solver.tpa_sigma_Ez = None
        solver.cex_decay_dynamic = None
        solver.cey_decay_dynamic = None
        solver.cez_decay_dynamic = None


def build_modulation(solver, material_model):
    """Set the space-time modulation flag and its per-edge quadrature fields.

    The compiled ``modulation_cos`` / ``modulation_sin`` node fields
    (``amplitude*cos(phase)`` and ``amplitude*sin(phase)``) are averaged onto the
    Yee E edges once here; the per-step modulation factor
    ``m(t) = 1 + cos_field*cos(Omega*t) - sin_field*sin(Omega*t)`` is evaluated
    inside the modulated E-update kernels from these static fields plus two
    host-side ``(cos, sin)`` scalar pairs, so no coefficient tensor is rebuilt
    per step.
    """
    solver.modulation_enabled = bool(material_model_has_modulation(material_model))
    if not solver.modulation_enabled:
        solver.modulation_angular_frequency = None
        solver.mod_cos_Ex = None
        solver.mod_cos_Ey = None
        solver.mod_cos_Ez = None
        solver.mod_sin_Ex = None
        solver.mod_sin_Ey = None
        solver.mod_sin_Ez = None
        return
    solver.modulation_angular_frequency = 2.0 * np.pi * float(material_model["modulation_frequency"])
    modulation_cos = material_model["modulation_cos"]
    modulation_sin = material_model["modulation_sin"]
    solver.mod_cos_Ex = average_node_to_component(solver, modulation_cos, "Ex")
    solver.mod_cos_Ey = average_node_to_component(solver, modulation_cos, "Ey")
    solver.mod_cos_Ez = average_node_to_component(solver, modulation_cos, "Ez")
    solver.mod_sin_Ex = average_node_to_component(solver, modulation_sin, "Ex")
    solver.mod_sin_Ey = average_node_to_component(solver, modulation_sin, "Ey")
    solver.mod_sin_Ez = average_node_to_component(solver, modulation_sin, "Ez")


_ANISO_ROW_PAIRS = {
    # Inverse-tensor row entries coupling each E component to the two off-axis
    # curl(H) components, in (first, second) order matching the update
    # E_i += dt/eps0 * (inv_ij * <curlH_j> + inv_ik * <curlH_k>).
    "Ex": ("xy", "xz"),
    "Ey": ("xy", "yz"),
    "Ez": ("xz", "yz"),
}


def _symmetric3_inverse(a, b, c, d, e, f):
    """Closed-form inverse of the symmetric 3x3 tensor [[a,d,e],[d,b,f],[e,f,c]].

    Returns the six independent entries of the (symmetric) inverse as a dict
    keyed by ``xx, yy, zz, xy, xz, yz`` plus the determinant for validation.
    """
    det = a * (b * c - f * f) - d * (d * c - f * e) + e * (d * f - b * e)
    inv_det = 1.0 / det
    return {
        "xx": (b * c - f * f) * inv_det,
        "yy": (a * c - e * e) * inv_det,
        "zz": (a * b - d * d) * inv_det,
        "xy": (e * f - d * c) * inv_det,
        "xz": (d * f - b * e) * inv_det,
        "yz": (d * e - a * f) * inv_det,
    }, det


def build_full_anisotropy(solver, material_model):
    """Precompute per-edge inverse-permittivity tensor rows for full anisotropy.

    For each E component the six relative-permittivity node grids (diagonal +
    off-diagonal) are averaged onto that component's Yee edges, the symmetric
    3x3 tensor is inverted per edge in closed form, and the row belonging to the
    component is stored: the diagonal entry as an effective scalar permittivity
    (consumed by ``build_update_coefficients`` for the base curl coefficient)
    and the two off-diagonal entries as coupling coefficients for the
    neighbor-averaged off-axis curl(H) correction kernel.
    """
    solver.full_aniso_enabled = bool(material_model_has_full_anisotropy(material_model))
    solver._aniso_eps_eff = None
    solver._aniso_inverse_rows = None
    # Effective inverse permittivity (1/eps_eff) reused for the diagonal ADE
    # polarization-current subtraction, and the accumulated per-component
    # polarization current buffers used by the off-diagonal tensor subtraction.
    solver._aniso_disp_inv_eps = None
    solver._aniso_disp_current = None
    # Per-component conduction-current buffers (sigma . E at the pre-update field),
    # subtracted through the off-diagonal inverse tensor when the anisotropic
    # medium is also conductive; None otherwise.
    solver._aniso_cond_current = None
    solver.cex_aniso_y = None
    solver.cex_aniso_z = None
    solver.cey_aniso_x = None
    solver.cey_aniso_z = None
    solver.cez_aniso_x = None
    solver.cez_aniso_y = None
    # Per-direction CPML psi accumulators for the off-diagonal coupling, allocated
    # only when the anisotropic structure runs under a CPML absorber; kept as
    # attributes so a graph capture snapshots them like the base CPML psi.
    for component in ("ex", "ey", "ez"):
        for axis in ("x", "y", "z"):
            setattr(solver, f"psi_{component}_aniso_{axis}", None)
    solver._full_aniso_cpml_overlap = False
    if not solver.full_aniso_enabled:
        return

    if solver.nonlinear_enabled:
        raise NotImplementedError(
            "FDTD full (off-diagonal) anisotropic media cannot be combined with nonlinear media."
        )

    eps_components = material_model["eps_components"]
    offdiag = material_model["eps_offdiag_components"]
    sigma_components = material_model["sigma_e_components"]
    node = {
        "xx": eps_components["x"],
        "yy": eps_components["y"],
        "zz": eps_components["z"],
        "xy": offdiag["xy"],
        "xz": offdiag["xz"],
        "yz": offdiag["yz"],
    }
    diagonal_key = {"Ex": "xx", "Ey": "yy", "Ez": "zz"}
    # Semi-implicit conductive fold: invert (eps_inf + dt/2 * diag(sigma)) instead of
    # eps_inf so the same per-edge inverse tensor B = dt * (...)^-1 that couples
    # curl(H) also carries the exact tensor conduction damping. In relative units
    # the diagonal shift is (dt / (2 eps0)) * sigma_abs; adding a nonnegative
    # diagonal to an SPD tensor keeps it SPD, so the inverse stays well defined.
    conductive = solver.conductive_enabled
    conductive_shift = solver.dt / (2.0 * solver.eps0)
    solver._aniso_eps_eff = {}
    solver._aniso_inverse_rows = {}
    for component_name in ("Ex", "Ey", "Ez"):
        edge = {
            key: average_node_to_component(solver, tensor, component_name)
            for key, tensor in node.items()
        }
        if conductive:
            for diag_key, axis in (("xx", "x"), ("yy", "y"), ("zz", "z")):
                sigma_edge = average_node_to_component(solver, sigma_components[axis], component_name)
                edge[diag_key] = edge[diag_key] + conductive_shift * sigma_edge
        inverse, det = _symmetric3_inverse(
            edge["xx"], edge["yy"], edge["zz"], edge["xy"], edge["xz"], edge["yz"]
        )
        diagonal_inverse = inverse[diagonal_key[component_name]]
        if bool(torch.any(det <= 0).item()) or bool(torch.any(diagonal_inverse <= 0).item()):
            raise ValueError(
                "The blended anisotropic permittivity tensor is not positive-definite on the "
                f"{component_name} Yee edges; check overlapping structures and material regions."
            )
        # Effective scalar permittivity reproducing the diagonal inverse entry:
        # the base curl coefficient dt/eps_eff == dt * inv_diag / eps0 == B_ii, and
        # (when conductive) the diagonal decay 1 - sigma * B_ii is folded in by
        # _electric_update_coefficients through the aniso_shifted branch.
        solver._aniso_eps_eff[component_name] = (solver.eps0 / diagonal_inverse).contiguous()
        first_pair, second_pair = _ANISO_ROW_PAIRS[component_name]
        solver._aniso_inverse_rows[component_name] = (
            inverse[first_pair].contiguous(),
            inverse[second_pair].contiguous(),
        )

    if conductive:
        # Buffers for the pre-update conduction current sigma . E^n, subtracted
        # through the off-diagonal inverse tensor each step (the diagonal part is
        # already folded into the decay coefficient).
        solver._aniso_cond_current = {
            "Ex": torch.zeros_like(solver.Ex),
            "Ey": torch.zeros_like(solver.Ey),
            "Ez": torch.zeros_like(solver.Ez),
        }


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


_ELECTRIC_COMPONENT_AXIS = {"Ex": "x", "Ey": "y", "Ez": "z"}


def _electric_pole_components(entry):
    """The E components a pole entry drives (sheet poles restrict to tangential axes)."""
    axes = entry.get("axes")
    if axes is None:
        return ("Ex", "Ey", "Ez")
    return tuple(
        component_name
        for component_name, axis in _ELECTRIC_COMPONENT_AXIS.items()
        if axis in axes
    )


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
        for component_name in _electric_pole_components(entry):
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
        for component_name in _electric_pole_components(entry):
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
        for component_name in _electric_pole_components(entry):
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

    if getattr(solver, "full_aniso_enabled", False) and solver.electric_dispersive_enabled:
        # Accumulator for the total per-component polarization current, summed over
        # all electric poles, consumed by the coupled tensor ADE subtraction.
        solver._aniso_disp_current = {
            "Ex": torch.zeros_like(solver.Ex),
            "Ey": torch.zeros_like(solver.Ey),
            "Ez": torch.zeros_like(solver.Ez),
        }

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


def _dispersive_current_coefficient(solver, component_name, component_templates):
    """The ``(coefficient, dt)`` pair for the subtraction ``E -= dt * J_p * coefficient``.

    In the linear path the polarization current is divided by the static
    permittivity (``coefficient = 1/eps_lin``, ``dt = solver.dt``). When an
    instantaneous nonlinearity is active the same Yee edge's displacement-current
    term in the E update is scaled by the field-dependent ``c*_curl_dynamic``
    coefficient (``external * (dt / eps_eff) / (1 + half)``); reusing that
    coefficient here (with ``dt = 1``) subtracts the ADE current against the SAME
    effective permittivity the ``curl(H)`` term saw this step, so the
    instantaneous-nonlinear index shift and the dispersive response stay mutually
    consistent (dividing the ADE current by ``eps_lin`` instead would detune the
    dispersion by the nonlinear index shift). In a purely dispersive cell the
    dynamic curl reduces to ``dt / eps_lin``, so the linear path is unchanged.
    """
    if getattr(solver, "nonlinear_enabled", False):
        dynamic_curl = {
            "Ex": solver.cex_curl_dynamic,
            "Ey": solver.cey_curl_dynamic,
            "Ez": solver.cez_curl_dynamic,
        }[component_name]
        return dynamic_curl, 1.0
    return component_templates["inv_eps"], solver.dt


def apply_component_dispersive_currents(solver, component_name, field, *, imag=False):
    if not solver.electric_dispersive_enabled:
        return

    component_templates = solver._dispersive_templates.get(component_name)
    if not component_templates:
        return
    launch_shape = solver._field_launch_shapes[component_name]
    inv_eps, apply_dt = _dispersive_current_coefficient(solver, component_name, component_templates)

    for entry in component_templates["debye"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_eps,
            dt=apply_dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["drude"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_eps,
            dt=apply_dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)

    for entry in component_templates["lorentz"]:
        current = entry["current_imag"] if imag else entry["current"]
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=current,
            InvPermittivity=inv_eps,
            dt=apply_dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=launch_shape)


def _aniso_periodic_flags(solver):
    return (
        int(solver.boundary_x_low_code == BOUNDARY_PERIODIC and solver.boundary_x_high_code == BOUNDARY_PERIODIC),
        int(solver.boundary_y_low_code == BOUNDARY_PERIODIC and solver.boundary_y_high_code == BOUNDARY_PERIODIC),
        int(solver.boundary_z_low_code == BOUNDARY_PERIODIC and solver.boundary_z_high_code == BOUNDARY_PERIODIC),
    )


def _apply_aniso_dispersive_corrections(solver):
    """Subtract the ADE polarization current through the full inverse permittivity tensor.

    For a full (off-diagonal) anisotropic permittivity the instantaneous
    constitutive relation is ``D = eps_inf . E + P``, so the Ampere update is
    ``E += dt * eps_inf^-1 . (curl H - J_p)``. The base and off-diagonal curl
    kernels already apply ``eps_inf^-1`` (diagonal via ``dt/eps_eff`` and
    off-diagonal via ``c*_aniso_*``) to ``curl H``; this applies the same operator
    to the accumulated polarization current ``J_p``. Electric poles enter
    isotropically (``P_i`` is driven by ``E_i``), so the per-component ADE state
    advance is unchanged and only the subtraction becomes a tensor contraction.
    Full anisotropy runs on real fields only (Bloch/complex is rejected upstream).
    """
    templates = solver._dispersive_templates
    totals = solver._aniso_disp_current
    for component_name in ("Ex", "Ey", "Ez"):
        buffer = totals[component_name]
        buffer.zero_()
        component_templates = templates[component_name]
        for model_name in ("debye", "drude", "lorentz"):
            for entry in component_templates[model_name]:
                buffer.add_(entry["current"])

    inv_eps = solver._aniso_disp_inv_eps
    for component_name, field in (("Ex", solver.Ex), ("Ey", solver.Ey), ("Ez", solver.Ez)):
        solver.fdtd_module.applyPolarizationCurrent3D(
            ElectricField=field,
            PolarizationCurrent=totals[component_name],
            InvPermittivity=inv_eps[component_name],
            dt=solver.dt,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes[component_name])

    periodic_x, periodic_y, periodic_z = _aniso_periodic_flags(solver)
    solver.fdtd_module.applyAnisoOffdiagCurrentEx3D(
        Ex=solver.Ex,
        Jy=totals["Ey"],
        Jz=totals["Ez"],
        CoeffY=solver.cex_aniso_y,
        CoeffZ=solver.cex_aniso_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ex"])
    solver.fdtd_module.applyAnisoOffdiagCurrentEy3D(
        Ey=solver.Ey,
        Jx=totals["Ex"],
        Jz=totals["Ez"],
        CoeffX=solver.cey_aniso_x,
        CoeffZ=solver.cey_aniso_z,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ey"])
    solver.fdtd_module.applyAnisoOffdiagCurrentEz3D(
        Ez=solver.Ez,
        Jx=totals["Ex"],
        Jy=totals["Ey"],
        CoeffX=solver.cez_aniso_x,
        CoeffY=solver.cez_aniso_y,
        periodicX=periodic_x,
        periodicY=periodic_y,
        periodicZ=periodic_z,
    ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes["Ez"])


def apply_dispersive_corrections(solver):
    if not solver.electric_dispersive_enabled:
        return

    if getattr(solver, "full_aniso_enabled", False):
        _apply_aniso_dispersive_corrections(solver)
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


def _electric_update_coefficients(solver, eps, sigma_e, pml_decay, *, aniso_shifted=False):
    """Compose the semi-implicit lossy-dielectric decay/curl factors for one E component.

    ``eps`` is the absolute permittivity (eps_r * eps0) and ``sigma_e`` the static
    electric conductivity averaged onto the same Yee component. When conductivity is
    inactive the material factors reduce to (decay=1, curl=dt/eps), so non-conductive
    scenes keep the exact existing coefficients. ``pml_decay`` folds the CPML decay in
    multiplicatively when present.

    ``aniso_shifted`` selects the full-tensor conductive path: ``eps`` is then the
    effective permittivity ``eps0 / [(eps_inf + dt/2 diag(sigma))^-1]_ii`` whose
    reciprocal is the diagonal of the conductively-shifted inverse tensor, so the
    diagonal curl is already the semi-implicit ``B_ii = dt/eps`` and the decay is
    ``1 - sigma * B_ii`` (the same ``decay = 1 - sigma * curl`` identity the scalar
    lossy fold satisfies). The off-diagonal conduction coupling is applied separately
    through ``B_ij`` by the conduction-current subtraction.
    """
    if solver.conductive_enabled:
        if aniso_shifted:
            curl = solver.dt / eps
            material_decay = 1.0 - sigma_e * curl
        else:
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


def _pec_edge_open_fractions(solver):
    """Per-E-edge open fractions ``1 - fill`` from the PEC occupancy, or ``None``.

    The PEC fill on each E edge is the two-endpoint node->edge average of the PEC
    occupancy (same stencil as eps, keeping it consistent and differentiable). In
    ``staircase`` mode the fill is hard-thresholded at 0.5; in ``conformal`` mode the
    fractional fill is kept so the effective PEC wall sits at the sub-cell crossing.
    """
    model = getattr(solver, "_compiled_material_model", None)
    pec_occupancy = None if model is None else model.get("pec_occupancy")
    if pec_occupancy is None:
        return None
    mode = model.get("pec_mode", "staircase")
    open_fractions = {}
    for component_name in ("Ex", "Ey", "Ez"):
        fill = average_node_to_component(solver, pec_occupancy, component_name)
        if mode == "staircase":
            fill = (fill >= 0.5).to(fill.dtype)
        open_fractions[component_name] = 1.0 - fill
    return open_fractions


def _apply_pec_edge_suppression(solver):
    """Fold the PEC open fraction into the electric decay/curl coefficients.

    Scaling both ``decay`` and ``curl`` by ``open = 1 - fill`` turns the update into
    ``E_new = open * (decay*E_old + curl*(curlH - J))``, so a fully covered edge
    (``fill = 1``) keeps tangential E exactly zero while a fractional edge acts as a
    soft short with sub-cell wall placement. Never amplifies (``open <= 1``), so the
    scheme is unconditionally stable and needs no area floor.
    """
    open_fractions = _pec_edge_open_fractions(solver)
    if open_fractions is None:
        return
    solver.cex_decay = (solver.cex_decay * open_fractions["Ex"]).contiguous()
    solver.cex_curl = (solver.cex_curl * open_fractions["Ex"]).contiguous()
    solver.cey_decay = (solver.cey_decay * open_fractions["Ey"]).contiguous()
    solver.cey_curl = (solver.cey_curl * open_fractions["Ey"]).contiguous()
    solver.cez_decay = (solver.cez_decay * open_fractions["Ez"]).contiguous()
    solver.cez_curl = (solver.cez_curl * open_fractions["Ez"]).contiguous()
    if getattr(solver, "cex_decay_external", None) is not None:
        # The general nonlinear kernel recomposes decay/curl from this external
        # factor each step, so the PEC open fraction must fold into it too.
        solver.cex_decay_external = (solver.cex_decay_external * open_fractions["Ex"]).contiguous()
        solver.cey_decay_external = (solver.cey_decay_external * open_fractions["Ey"]).contiguous()
        solver.cez_decay_external = (solver.cez_decay_external * open_fractions["Ez"]).contiguous()
    if getattr(solver, "full_aniso_enabled", False):
        # The off-diagonal coupling terms are additive corrections to the same
        # E edges, so PEC-covered edges must suppress them identically.
        solver.cex_aniso_y = (solver.cex_aniso_y * open_fractions["Ex"]).contiguous()
        solver.cex_aniso_z = (solver.cex_aniso_z * open_fractions["Ex"]).contiguous()
        solver.cey_aniso_x = (solver.cey_aniso_x * open_fractions["Ey"]).contiguous()
        solver.cey_aniso_z = (solver.cey_aniso_z * open_fractions["Ey"]).contiguous()
        solver.cez_aniso_x = (solver.cez_aniso_x * open_fractions["Ez"]).contiguous()
        solver.cez_aniso_y = (solver.cez_aniso_y * open_fractions["Ez"]).contiguous()


def _store_nonlinear_external_decay(solver, ex_decay, ey_decay, ez_decay):
    """Keep the material-independent (PML) decay factors for the nonlinear kernel.

    The general nonlinear coefficient kernel recomposes the semi-implicit lossy
    decay/curl factors from the linear permittivity, the static conductivity, and
    the field-dependent nonlinear terms every step, so it needs the external
    (PML split-field) decay separately from the static material decay that
    ``_electric_update_coefficients`` folds into ``c*_decay``. The PEC open
    fractions are folded in afterwards by ``_apply_pec_edge_suppression``.
    """
    if not getattr(solver, "nonlinear_general_enabled", False):
        solver.cex_decay_external = None
        solver.cey_decay_external = None
        solver.cez_decay_external = None
        return
    solver.cex_decay_external = (
        torch.ones_like(solver.eps_Ex) if ex_decay is None else ex_decay.clone()
    ).contiguous()
    solver.cey_decay_external = (
        torch.ones_like(solver.eps_Ey) if ey_decay is None else ey_decay.clone()
    ).contiguous()
    solver.cez_decay_external = (
        torch.ones_like(solver.eps_Ez) if ez_decay is None else ez_decay.clone()
    ).contiguous()


def _electric_epsilon_tensors(solver):
    """The absolute permittivity tensors feeding the electric curl coefficients.

    Full anisotropy swaps in the effective scalar permittivity that reproduces
    the diagonal entry of the per-edge inverse tensor; the plain per-axis
    average is kept otherwise (byte-identical to the base path).
    """
    if getattr(solver, "full_aniso_enabled", False):
        eps_eff = solver._aniso_eps_eff
        return eps_eff["Ex"], eps_eff["Ey"], eps_eff["Ez"]
    return solver.eps_Ex, solver.eps_Ey, solver.eps_Ez


def _build_full_aniso_curl_coefficients(solver):
    """Materialize the off-diagonal coupling coefficient tensors.

    ``coeff = dt * inv_ij / eps0`` multiplies the neighbor-averaged off-axis
    curl(H) component in the correction kernel. The coefficients are zero
    outside the anisotropic structures, so the correction is a no-op there.
    """
    if not getattr(solver, "full_aniso_enabled", False):
        return
    scale = solver.dt / solver.eps0
    rows = solver._aniso_inverse_rows
    solver.cex_aniso_y = (scale * rows["Ex"][0]).contiguous()
    solver.cex_aniso_z = (scale * rows["Ex"][1]).contiguous()
    solver.cey_aniso_x = (scale * rows["Ey"][0]).contiguous()
    solver.cey_aniso_z = (scale * rows["Ey"][1]).contiguous()
    solver.cez_aniso_x = (scale * rows["Ez"][0]).contiguous()
    solver.cez_aniso_y = (scale * rows["Ez"][1]).contiguous()
    # The diagonal ADE polarization-current subtraction must divide by the same
    # effective permittivity the base curl coefficient uses (dt/eps_eff), not the
    # plain per-axis average, so the dispersive response and the tensor curl stay
    # consistent on off-diagonal edges (they coincide for a diagonal tensor).
    if getattr(solver, "electric_dispersive_enabled", False):
        eps_eff = solver._aniso_eps_eff
        solver._aniso_disp_inv_eps = {
            component_name: (1.0 / eps_eff[component_name]).contiguous()
            for component_name in ("Ex", "Ey", "Ez")
        }
    _handle_full_aniso_absorber_overlap(solver)


def _absorbing_region_node_mask(solver):
    """Boolean node mask of the absorbing-boundary region, or ``None``.

    Covers both the graded-sigma split-field absorbers (3D ``sigma_*`` node
    profiles) and CPML/stable-PML (1D stretching profiles per axis).
    """
    if solver.active_absorber_type in ("pml", "absorber"):
        return (solver.sigma_x != 0) | (solver.sigma_y != 0) | (solver.sigma_z != 0)
    if getattr(solver, "uses_cpml", False):
        masks = []
        for axis_name, view in (("x", (-1, 1, 1)), ("y", (1, -1, 1)), ("z", (1, 1, -1))):
            c_e = getattr(solver, f"cpml_c_e_{axis_name}", None)
            inv_kappa = getattr(solver, f"cpml_inv_kappa_e_{axis_name}", None)
            axis_mask = None
            if c_e is not None:
                axis_mask = c_e != 0
            if inv_kappa is not None:
                kappa_mask = inv_kappa != 1
                axis_mask = kappa_mask if axis_mask is None else (axis_mask | kappa_mask)
            if axis_mask is not None:
                masks.append(axis_mask.reshape(view))
        if not masks:
            return None
        mask = masks[0]
        for axis_mask in masks[1:]:
            mask = mask | axis_mask
        return mask.expand(solver.Nx, solver.Ny, solver.Nz)
    return None


def _full_aniso_offdiag_overlaps_absorber(solver):
    """Whether any off-diagonal coupling coefficient reaches into the absorber."""
    mask = _absorbing_region_node_mask(solver)
    if mask is None:
        return False
    mask = mask.to(dtype=torch.float32)
    component_coefficients = (
        ("Ex", (solver.cex_aniso_y, solver.cex_aniso_z)),
        ("Ey", (solver.cey_aniso_x, solver.cey_aniso_z)),
        ("Ez", (solver.cez_aniso_x, solver.cez_aniso_y)),
    )
    # The soft SDF occupancy leaves a vanishing (denormal-scale) tail far outside
    # the structure, so compare against a relative threshold instead of != 0.
    peak = max(
        float(coefficient.abs().max())
        for _, coefficients in component_coefficients
        for coefficient in coefficients
    )
    if peak == 0.0:
        return False
    threshold = 1.0e-6 * peak
    for component_name, coefficients in component_coefficients:
        edge_mask = average_node_to_component(solver, mask, component_name) > 0
        for coefficient in coefficients:
            if bool(torch.any(edge_mask & (coefficient.abs() > threshold)).item()):
                return True
    return False


def _initialize_full_aniso_cpml_psi(solver):
    """Allocate the per-direction off-diagonal CPML psi accumulators.

    The CPML off-diagonal correction kernel coordinate-stretches each of the
    three spatial derivative directions with an independent recursive-convolution
    memory owned by the target E edge. The buffers share the E-field shapes and
    start at zero, so a structure that never reaches the absorber leaves them at
    zero and the correction is byte-consistent with the raw (no-stretch) update.
    """
    for field_name, component in (("Ex", "ex"), ("Ey", "ey"), ("Ez", "ez")):
        field = getattr(solver, field_name)
        for axis in ("x", "y", "z"):
            setattr(solver, f"psi_{component}_aniso_{axis}", torch.zeros_like(field))


def _handle_full_aniso_absorber_overlap(solver):
    """Route an anisotropic structure that overlaps the absorbing boundary.

    Under CPML the dedicated off-diagonal kernel coordinate-stretches the tensor
    coupling with its own psi memory (allocated here), so the overlap is
    supported; the overlap is recorded because the reverse adjoint replay does
    not yet stretch the off-diagonal, and the split-field graded-sigma absorbers
    ('pml'/'absorber') carry no per-direction auxiliary memory to stretch the
    coupling, so an overlap there is rejected with a physical reason.
    """
    solver._full_aniso_cpml_overlap = False
    if getattr(solver, "uses_cpml", False):
        _initialize_full_aniso_cpml_psi(solver)
        solver._full_aniso_cpml_overlap = _full_aniso_offdiag_overlaps_absorber(solver)
        return
    if _full_aniso_offdiag_overlaps_absorber(solver):
        raise NotImplementedError(
            "FDTD full (off-diagonal) anisotropic media overlapping a split-field "
            "PML/absorber layer are not supported: the graded-sigma absorber carries no "
            "per-direction auxiliary memory to coordinate-stretch the off-diagonal tensor "
            "coupling. Use the default CPML absorber for anisotropic structures that reach "
            "the boundary region."
        )


def build_update_coefficients(solver):
    eps_ex, eps_ey, eps_ez = _electric_epsilon_tensors(solver)
    # Under full anisotropy eps_e* is the conductively-shifted effective permittivity,
    # so the diagonal decay/curl must use the aniso_shifted semi-implicit identity
    # rather than re-applying the scalar (1 + half) denominator.
    aniso_shifted = getattr(solver, "full_aniso_enabled", False)
    if solver.active_absorber_type not in ("pml", "absorber"):
        solver.cex_decay, solver.cex_curl = _electric_update_coefficients(solver, eps_ex, solver.sigma_e_Ex, None, aniso_shifted=aniso_shifted)
        solver.cey_decay, solver.cey_curl = _electric_update_coefficients(solver, eps_ey, solver.sigma_e_Ey, None, aniso_shifted=aniso_shifted)
        solver.cez_decay, solver.cez_curl = _electric_update_coefficients(solver, eps_ez, solver.sigma_e_Ez, None, aniso_shifted=aniso_shifted)
        solver.chx_decay = torch.ones_like(solver.mu_Hx).contiguous()
        solver.chx_curl = (solver.dt / solver.mu_Hx).contiguous()
        solver.chy_decay = torch.ones_like(solver.mu_Hy).contiguous()
        solver.chy_curl = (solver.dt / solver.mu_Hy).contiguous()
        solver.chz_decay = torch.ones_like(solver.mu_Hz).contiguous()
        solver.chz_curl = (solver.dt / solver.mu_Hz).contiguous()
        _store_nonlinear_external_decay(solver, None, None, None)
        _build_full_aniso_curl_coefficients(solver)
        _apply_pec_edge_suppression(solver)
        return

    ex_sigma_y = 0.5 * (solver.sigma_y[:-1, :, :] + solver.sigma_y[1:, :, :])
    ex_sigma_z = 0.5 * (solver.sigma_z[:-1, :, :] + solver.sigma_z[1:, :, :])
    ex_decay = 1.0 / (1.0 + solver.dt * (ex_sigma_y + ex_sigma_z))
    solver.cex_decay, solver.cex_curl = _electric_update_coefficients(solver, eps_ex, solver.sigma_e_Ex, ex_decay, aniso_shifted=aniso_shifted)

    ey_sigma_x = 0.5 * (solver.sigma_x[:, :-1, :] + solver.sigma_x[:, 1:, :])
    ey_sigma_z = 0.5 * (solver.sigma_z[:, :-1, :] + solver.sigma_z[:, 1:, :])
    ey_decay = 1.0 / (1.0 + solver.dt * (ey_sigma_x + ey_sigma_z))
    solver.cey_decay, solver.cey_curl = _electric_update_coefficients(solver, eps_ey, solver.sigma_e_Ey, ey_decay, aniso_shifted=aniso_shifted)

    ez_sigma_x = 0.5 * (solver.sigma_x[:, :, :-1] + solver.sigma_x[:, :, 1:])
    ez_sigma_y = 0.5 * (solver.sigma_y[:, :, :-1] + solver.sigma_y[:, :, 1:])
    ez_decay = 1.0 / (1.0 + solver.dt * (ez_sigma_x + ez_sigma_y))
    solver.cez_decay, solver.cez_curl = _electric_update_coefficients(solver, eps_ez, solver.sigma_e_Ez, ez_decay, aniso_shifted=aniso_shifted)

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

    _store_nonlinear_external_decay(solver, ex_decay, ey_decay, ez_decay)
    _build_full_aniso_curl_coefficients(solver)
    _apply_pec_edge_suppression(solver)


def update_nonlinear_electric_coefficients(solver):
    """Recompute the field-dependent electric update coefficients for this step.

    Pure-chi3 (Kerr) scenes keep the existing curl-only kernel; scenes with a
    chi2 channel or field-dependent conductivity use the general nonlinear
    kernel that rewrites both the decay and curl coefficients.
    """
    if not getattr(solver, "nonlinear_enabled", False):
        return
    if getattr(solver, "nonlinear_general_enabled", False):
        update_general_nonlinear_coefficients(solver)
        return
    update_kerr_electric_curls(solver)


def update_general_nonlinear_coefficients(solver):
    for component_name, decay_dynamic, curl_dynamic, eps, external, sigma_static, chi2, chi3, tpa in (
        (
            "Ex",
            solver.cex_decay_dynamic,
            solver.cex_curl_dynamic,
            solver.eps_Ex,
            solver.cex_decay_external,
            solver.sigma_e_Ex,
            solver.nonlinear_chi2_Ex,
            solver.kerr_chi3_Ex,
            solver.tpa_sigma_Ex,
        ),
        (
            "Ey",
            solver.cey_decay_dynamic,
            solver.cey_curl_dynamic,
            solver.eps_Ey,
            solver.cey_decay_external,
            solver.sigma_e_Ey,
            solver.nonlinear_chi2_Ey,
            solver.kerr_chi3_Ey,
            solver.tpa_sigma_Ey,
        ),
        (
            "Ez",
            solver.cez_decay_dynamic,
            solver.cez_curl_dynamic,
            solver.eps_Ez,
            solver.cez_decay_external,
            solver.sigma_e_Ez,
            solver.nonlinear_chi2_Ez,
            solver.kerr_chi3_Ez,
            solver.tpa_sigma_Ez,
        ),
    ):
        solver.fdtd_module.updateNonlinearElectricCoefficients3D(
            DynamicDecay=decay_dynamic,
            DynamicCurl=curl_dynamic,
            Ex=solver.Ex,
            Ey=solver.Ey,
            Ez=solver.Ez,
            LinearPermittivity=eps,
            ExternalDecay=external,
            SigmaStatic=sigma_static,
            Chi2=chi2,
            Chi3=chi3,
            TpaSigma=tpa,
            component=component_name,
            dt=solver.dt,
            eps0=solver.eps0,
        ).launchRaw(blockSize=solver.kernel_block_size, gridSize=solver._field_launch_shapes[component_name])


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
