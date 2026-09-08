"""Material-aware, prepare-time bounds for the explicit Yee/ADE update."""
from __future__ import annotations

import math

import torch


def _minimum(value):
    return float(torch.as_tensor(value).detach().amin())


def _maximum(value):
    return float(torch.as_tensor(value).detach().abs().amax())


def _tensor_lower_bound(diagonal, offdiagonal):
    if not any(bool(torch.any(value != 0)) for value in offdiagonal.values()):
        return min(_minimum(value) for value in diagonal.values())
    xx, yy, zz = (diagonal[axis] for axis in "xyz")
    xy, xz, yz = (offdiagonal[pair] for pair in ("xy", "xz", "yz"))
    matrix = torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), dim=-1).reshape(*xx.shape, 3, 3)
    return _minimum(torch.linalg.eigvalsh(matrix.double())[..., 0])


def _pole_rate(entries, background, kind):
    rate = 0.0
    passive = True
    for entry in entries:
        pole = entry["pole"]
        weight = _maximum(entry["weight"])
        if weight == 0:
            continue
        if kind == "drude":
            rate += 2 * math.pi * _maximum(pole.plasma_frequency) * math.sqrt(weight / background)
        else:
            strength = _maximum(pole.delta_eps) * weight / background
            passive &= _minimum(pole.delta_eps) >= 0
            if kind == "debye":
                # Explicit feedback of the trapezoidal polarization into E/H
                # still has a strength-dependent stability restriction.
                rate += (1.0 + strength) / float(pole.tau)
            else:
                rate += 2 * math.pi * float(pole.resonance_frequency) * math.sqrt(1.0 + strength)
    return rate, passive


def compile_fdtd_time_step(solver, material_model):
    """Freeze dt before binding any material, boundary, or source coefficients.

    Sum curl and local oscillator norm bounds instead of independently clipping
    each subsystem's dt. All reductions happen during preparation, on the material
    device; automatic selection is numerical configuration, not an autograd input.
    """
    epsilon = min(_minimum(getattr(solver, "eps_" + name)) / solver.eps0 for name in ("Ex", "Ey", "Ez"))
    permeability = min(_minimum(getattr(solver, "mu_" + name)) / solver.mu0 for name in ("Hx", "Hy", "Hz"))
    offdiagonal = material_model.get("eps_offdiag_components", {})
    if offdiagonal:
        epsilon = min(epsilon, _tensor_lower_bound(material_model["eps_components"], offdiagonal))
    modulation = _maximum(torch.hypot(material_model["modulation_cos"], material_model["modulation_sin"]))
    if modulation:
        from .materials import average_node_to_component

        for name in ("Ex", "Ey", "Ez"):
            cosine = average_node_to_component(solver, material_model["modulation_cos"], name)
            sine = average_node_to_component(solver, material_model["modulation_sin"], name)
            relative = getattr(solver, "eps_" + name) / solver.eps0
            epsilon = min(epsilon, _minimum(relative * (1 - torch.hypot(cosine, sine))))
    # Declared design bounds keep density perturbations on one discrete problem,
    # even when the realized density does not occupy the fastest material.
    for region in getattr(solver.scene, "material_regions", ()):
        epsilon = min(epsilon, min(region.eps_bounds) * (1 - modulation))
        permeability = min(permeability, *region.mu_bounds)
    nonlinear_floor = epsilon
    chi2 = _maximum(material_model["chi2"])
    negative_chi3 = max(0.0, -_minimum(material_model["kerr_chi3"]))
    solver._nonlinear_peak = None
    solver._nonlinear_field_limit = None
    if chi2 or negative_chi3:
        amplitude = getattr(solver, "_max_electric_field", None)
        if amplitude is None:
            # Limit the possible negative increment to half the background.
            # The envelope is checked on device over every nonlinear update.
            budget = epsilon / (1 + modulation)
            amplitude = budget / (chi2 + math.sqrt(chi2**2 + 6 * negative_chi3 * budget))
        nonlinear_floor -= (chi2 * amplitude + 3 * negative_chi3 * amplitude**2) * (1 + modulation)
        solver._nonlinear_field_limit = float(amplitude)
        solver._nonlinear_peak = solver.eps_Ex.new_zeros(())
    epsilon = nonlinear_floor
    if not math.isfinite(epsilon * permeability) or min(epsilon, permeability) <= 0:
        raise ValueError("FDTD requires positive finite high-frequency material bounds throughout the update.")

    spacings = [min(float(getattr(solver.scene, f"d{axis}_primal64").min()),
                    float(getattr(solver.scene, f"d{axis}_dual64").min())) for axis in "xyz"]
    curl_rate = 2 * solver.c * math.sqrt(sum(1 / spacing**2 for spacing in spacings) / (epsilon * permeability))
    material_rate = 0.0
    passive = True
    for prefix, background in (("", epsilon), ("mu_", permeability)):
        for kind in ("debye", "drude", "lorentz"):
            rate, family_passive = _pole_rate(material_model.get(prefix + kind + "_poles", ()), background, kind)
            material_rate += rate
            passive &= family_passive
    bound = 2.0 / (curl_rate + material_rate)
    safety = 0.9
    allowed = safety * bound
    requested = getattr(solver, "_requested_dt", None)
    if requested is not None:
        if not math.isfinite(requested) or requested <= 0 or requested > allowed:
            raise ValueError(f"TimeConfig.dt={requested!r} exceeds the safe FDTD time-step limit {allowed:.12g} s or is not positive finite.")
        dt = float(requested)
        limiting = "explicit"
    else:
        material_resolution = 2 * math.pi / (30 * material_rate) if material_rate else math.inf
        dt = min(float(solver.dt), allowed, material_resolution)
        limiting = "material_cfl" if dt == allowed else "frequency_resolution"
    solver.dt = dt
    solver._maxwell_cfl_limit = 2.0 / curl_rate
    solver._time_step_metadata = {
        "dt": dt, "stability_limit": bound, "safety_factor": safety,
        "epsilon_lower_bound": epsilon, "mu_lower_bound": permeability,
        "curl_rate_bound": curl_rate, "material_rate_bound": material_rate,
        "passive_material": passive, "limiting_factor": limiting,
        "gradient_convention": "fixed_dt",
        "max_electric_field": solver._nonlinear_field_limit,
    }


def record_nonlinear_field_envelope(solver):
    peak = getattr(solver, "_nonlinear_peak", None)
    if peak is not None:
        for name in ("Ex", "Ey", "Ez"):
            torch.maximum(peak, getattr(solver, name).abs().amax(), out=peak)


def validate_nonlinear_field_envelope(solver):
    peak = getattr(solver, "_nonlinear_peak", None)
    if peak is not None:
        observed = float(peak)
        if not math.isfinite(observed) or observed > solver._nonlinear_field_limit:
            raise RuntimeError(
                "The nonlinear field exceeded the time-step stability envelope "
                f"({observed:.8g} > {solver._nonlinear_field_limit:.8g} V/m). "
                "Set TimeConfig.max_electric_field to a valid larger bound and rerun; "
                "the selected dt is only certified within that envelope."
            )
