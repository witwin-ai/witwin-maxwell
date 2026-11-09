"""Material-aware, prepare-time bounds for the explicit Yee/ADE update."""
from __future__ import annotations

import math

import torch


def _minimum(value):
    result = float(torch.as_tensor(value).detach().amin())
    if not math.isfinite(result):
        raise ValueError("FDTD material bounds must be finite.")
    return result


def _maximum(value):
    result = float(torch.as_tensor(value).detach().abs().amax())
    if not math.isfinite(result):
        raise ValueError("FDTD material bounds must be finite.")
    return result


def _tensor_lower_bound(diagonal, offdiagonal):
    if not any(bool(torch.any(value != 0)) for value in offdiagonal.values()):
        return min(_minimum(value) for value in diagonal.values())
    xx, yy, zz = (diagonal[axis].double() for axis in "xyz")
    xy, xz, yz = (offdiagonal[pair].double() for pair in ("xy", "xz", "yz"))
    # For SPD A, lambda_min(A) >= 1 / ||A^-1||_infinity. Cofactors
    # evaluate this bound with linear storage, avoiding a batched eigensolver's
    # potentially quadratic workspace for millions of independent 3x3 cells.
    cxx, cyy, czz = yy * zz - yz.square(), xx * zz - xz.square(), xx * yy - xy.square()
    cxy, cxz, cyz = xz * yz - xy * zz, xy * yz - xz * yy, xy * xz - xx * yz
    determinant = xx * cxx + xy * cxy + xz * cxz
    roundoff = 64 * torch.finfo(torch.float64).eps
    determinant = determinant - roundoff * (abs(xx * cxx) + abs(xy * cxy) + abs(xz * cxz))
    row_norm = torch.maximum(torch.maximum(abs(cxx) + abs(cxy) + abs(cxz),
        abs(cxy) + abs(cyy) + abs(cyz)), abs(cxz) + abs(cyz) + abs(czz))
    positive = (xx > 0) & (czz > 0) & (determinant > 0)
    lower = determinant / (row_norm * (1 + roundoff))
    return _minimum(torch.where(positive, lower, torch.zeros_like(lower)))


def _pole_rate(entries, background, kind):
    rate = 0.0
    passive = True
    for entry in entries:
        pole = entry["pole"]
        weight = max(map(_maximum, entry["edge_weights"].values())) if "edge_weights" in entry else _maximum(entry["weight"])
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
    pole_nodes = material_model.get("pole_tensor_nodes", {})
    offdiagonal = pole_nodes.get("offdiag", material_model.get("eps_offdiag_components", {}))
    if offdiagonal:
        epsilon = min(epsilon, _tensor_lower_bound(pole_nodes.get("diagonal", material_model["eps_components"]), offdiagonal))
    modulation = _maximum(torch.hypot(material_model["modulation_cos"], material_model["modulation_sin"]))
    if modulation:
        from .materials import _modulation_component_quadratures

        for name in ("Ex", "Ey", "Ez"):
            cosine, sine = _modulation_component_quadratures(material_model, name)
            modulation = max(modulation, _maximum(torch.hypot(cosine, sine)))
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
    solver._nonlinear_envelope_indices = None
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
        if not negative_chi3:
            # chi2 depends only on the component at the occupied nonlinear edge.
            # A strong field in a separate linear source cell cannot lower eps.
            solver._nonlinear_envelope_indices = {
                name: torch.nonzero(getattr(solver, "nonlinear_chi2_" + name).flatten() != 0,
                                    as_tuple=True)[0]
                for name in ("Ex", "Ey", "Ez")
            }
    epsilon = nonlinear_floor
    if not math.isfinite(epsilon * permeability) or min(epsilon, permeability) <= 0:
        raise ValueError("FDTD requires positive finite high-frequency material bounds throughout the update.")

    spacings = [min(float(getattr(solver.scene, f"d{axis}_primal64").min()),
                    float(getattr(solver.scene, f"d{axis}_dual64").min())) for axis in "xyz"]
    curl_rate = 2 * solver.c * math.sqrt(sum(1 / spacing**2 for spacing in spacings) / (epsilon * permeability))
    # A spatial full-tensor conduction update applies the assembled inverse to
    # the edge current explicitly. Include its feedback in the joint norm bound;
    # the local conductive shift alone is not a global implicit inverse.
    conductivity_rate = 0.0
    if getattr(solver, "full_aniso_enabled", False):
        conductivity_rate = max(_maximum(getattr(solver, "sigma_e_" + name))
                                for name in ("Ex", "Ey", "Ez")) / (solver.eps0 * epsilon)
    material_rate = conductivity_rate
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
        frequency = getattr(solver, "_requested_source_frequency", 0.0)
        source_resolution = 1 / (30 * frequency) if frequency else math.inf
        dt = min(float(solver.dt), allowed, material_resolution, source_resolution)
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
        indices = getattr(solver, "_nonlinear_envelope_indices", None)
        for name in ("Ex", "Ey", "Ez"):
            values = getattr(solver, name)
            if indices is not None:
                if not indices[name].numel():
                    continue
                values = values.flatten().index_select(0, indices[name])
            torch.maximum(peak, values.abs().amax(), out=peak)


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
