"""Convert a maxwell Scene to a Tidy3D Simulation.

Usage::

    td_sim = scene.to_tidy3d(frequencies=frequencies)
    td_result = tidy3d.web.run(td_sim)

Requires ``tidy3d`` as an optional dependency.

**Unit convention**: maxwell uses metres; Tidy3D uses micrometres (um).
All spatial quantities are multiplied by ``length_scale`` (default 1e6)
during conversion.  Frequencies stay in Hz.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
from witwin.core.material import VACUUM_PERMITTIVITY
from ..fdtd.excitation.spatial import resolve_injection_axis, soft_plane_wave_coordinate

if TYPE_CHECKING:
    import tidy3d

# Default length conversion factor: metres to micrometres.
_M_TO_UM = 1e6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_tidy3d():
    """Import and return tidy3d, raising a clear error if missing."""
    try:
        import tidy3d as td
        return td
    except ImportError:
        raise ImportError(
            "tidy3d is required for Scene.to_tidy3d(). "
            "Install it with: pip install tidy3d"
        ) from None


def _scale3(vec, s):
    """Scale a 3-element vector by *s*, returning a Python-float tuple."""
    return (float(vec[0]) * s, float(vec[1]) * s, float(vec[2]) * s)


def _domain_to_center_size(domain, s):
    """Convert Domain bounds to Tidy3D (center, size) in Tidy3D units."""
    bounds = domain.bounds
    center = tuple((lo + hi) / 2.0 * s for lo, hi in bounds)
    size = tuple((hi - lo) * s for lo, hi in bounds)
    return center, size


def _polarization_to_component(polarization: tuple[float, float, float]) -> str:
    """Map a unit polarization vector to 'Ex', 'Ey', or 'Ez'."""
    px, py, pz = polarization
    mapping = {(1, 0, 0): "Ex", (0, 1, 0): "Ey", (0, 0, 1): "Ez"}
    key = (int(round(px)), int(round(py)), int(round(pz)))
    if key in mapping:
        return mapping[key]
    raise ValueError(
        f"Cannot map polarization {polarization} to a single Tidy3D component. "
        "Only axis-aligned unit vectors are supported."
    )


def _axis_name_to_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


# ---------------------------------------------------------------------------
# Source-time conversion (no length scaling - purely temporal / frequency)
# ---------------------------------------------------------------------------

def _convert_source_time(source_time, td):
    """Convert maxwell source_time to a Tidy3D SourceTime."""
    from ..sources import CW, GaussianPulse, RickerWavelet

    if source_time is None:
        raise ValueError("source_time must be set for Tidy3D export.")

    if isinstance(source_time, CW):
        return td.ContinuousWave(
            freq0=source_time.frequency,
            amplitude=source_time.amplitude,
            phase=source_time.phase,
        )

    if isinstance(source_time, GaussianPulse):
        # Maxwell's pulsed source keeps the carrier delay inside the sampled
        # waveform, while Tidy3D's GaussianPulse encodes the carrier spectrum
        # phase separately from the envelope offset. Fold the carrier delay
        # into the exported phase so single-frequency benchmark fields compare
        # against the same source spectrum at the target frequency.
        offset = source_time.delay / source_time.sigma_t
        phase = source_time.phase + 2.0 * math.pi * source_time.frequency * source_time.delay
        return td.GaussianPulse(
            freq0=source_time.frequency,
            fwidth=source_time.fwidth,
            amplitude=source_time.amplitude,
            phase=phase,
            offset=offset,
        )

    if isinstance(source_time, RickerWavelet):
        return td.GaussianPulse(
            freq0=source_time.frequency,
            fwidth=source_time.frequency,
            amplitude=source_time.amplitude,
        )

    raise TypeError(f"Unsupported source_time type: {type(source_time).__name__}")


# ---------------------------------------------------------------------------
# Material conversion (no length scaling - dimensionless or frequency-based)
# ---------------------------------------------------------------------------

def _conductivity_pole(sigma_e: float):
    """Zero-frequency PoleResidue pole encoding a static electric conductivity.

    Under the e^{-i*omega*t} convention a passive conductor adds a loss term
    +i*sigma/(omega*eps0) to the relative permittivity. In Tidy3D's PoleResidue
    model ``eps(w) = eps_inf - sum_i [ c_i/(iw + a_i) + c_i*/(iw + a_i*) ]`` a real
    residue ``c`` placed at the pole ``a = 0`` contributes ``+2i*Re(c)/w``, so
    ``Re(c) = sigma/(2*eps0)`` reproduces the conductivity term exactly. The residue
    lives in the (dimensionless) permittivity domain with ``w`` in rad/s, so it is
    independent of the export length scale. ``eps0`` is the SI vacuum permittivity,
    matching ``Material.relative_permittivity``.
    """
    return (complex(0.0, 0.0), complex(float(sigma_e) / (2.0 * VACUUM_PERMITTIVITY), 0.0))


def _dispersive_medium(material, td, nonlinear_spec=None):
    """Convert a dispersive Maxwell Material (ignoring sigma_e) to a Tidy3D medium.

    An optional ``nonlinear_spec`` is forwarded to the constructed medium so a
    material that is both dispersive and Kerr/TPA nonlinear exports as a single
    dispersive+nonlinear Tidy3D medium.
    """
    has_debye = bool(material.debye_poles)
    has_drude = bool(material.drude_poles)
    has_lorentz = bool(material.lorentz_poles)
    extra = {} if nonlinear_spec is None else {"nonlinear_spec": nonlinear_spec}

    if has_drude and not has_debye and not has_lorentz:
        coeffs = [
            (pole.plasma_frequency, pole.gamma)
            for pole in material.drude_poles
        ]
        return td.Drude(eps_inf=material.eps_r, coeffs=coeffs, **extra)

    if has_lorentz and not has_debye and not has_drude:
        coeffs = [
            (pole.delta_eps, pole.resonance_frequency, pole.gamma)
            for pole in material.lorentz_poles
        ]
        return td.Lorentz(eps_inf=material.eps_r, coeffs=coeffs, **extra)

    if has_debye and not has_drude and not has_lorentz:
        coeffs = [
            (pole.delta_eps, pole.tau)
            for pole in material.debye_poles
        ]
        return td.Debye(eps_inf=material.eps_r, coeffs=coeffs, **extra)

    # Mixed pole types to PoleResidue
    poles = []
    omega_factor = 2.0 * math.pi
    for p in material.drude_poles:
        wp = p.plasma_frequency * omega_factor
        g = p.gamma * omega_factor
        a = complex(0, -g / 2)
        c = complex(0, -wp * wp / (2.0 * g)) if g > 0 else complex(-wp * wp / 2, 0)
        poles.append((a, c))
    for p in material.lorentz_poles:
        w0 = p.resonance_frequency * omega_factor
        g = p.gamma * omega_factor
        disc = g * g / 4.0 - w0 * w0
        if disc >= 0:
            sq = math.sqrt(disc)
            a1 = complex(0, -g / 2 + sq)
            a2 = complex(0, -g / 2 - sq)
        else:
            sq = math.sqrt(-disc)
            a1 = complex(sq, -g / 2)
            a2 = complex(-sq, -g / 2)
        c_val = p.delta_eps * w0 * w0 / (2.0 * (a1 - a2)) if abs(a1 - a2) > 0 else 0
        poles.append((a1, complex(c_val)))
        poles.append((a2, complex(-c_val)))
    for p in material.debye_poles:
        tau = p.tau
        a = complex(0, -1.0 / tau)
        c = complex(0, p.delta_eps / tau)
        poles.append((a, c))

    extra = {} if nonlinear_spec is None else {"nonlinear_spec": nonlinear_spec}
    return td.PoleResidue(eps_inf=material.eps_r, poles=poles, **extra)


def _axis_isotropic_material(material, axis_index: int):
    """Project an axis-aligned anisotropic Material onto one principal axis.

    Diagonal (``DiagonalTensor3``) anisotropy is three independent isotropic media,
    one per lab-frame axis. Electric dispersion enters each axis isotropically
    (``chi(omega) * I``), so the per-axis medium keeps the material's homogeneous
    pole set and only its background permittivity/conductivity change per axis.
    Returning a fresh isotropic ``Material`` lets ``_convert_material`` reuse the
    full isotropic pole/conductivity conversion (and its e^{-iwt} sign and
    length-scale conventions) for each Tidy3D ``AnisotropicMedium`` component.
    """
    from ..media import DiagonalTensor3, Material

    eps_tensor = material.epsilon_tensor
    if isinstance(eps_tensor, DiagonalTensor3):
        eps_axis = eps_tensor.as_tuple()[axis_index]
    else:
        eps_axis = float(material.eps_r)

    sigma_tensor = material.sigma_e_tensor
    if isinstance(sigma_tensor, DiagonalTensor3):
        sigma_axis = sigma_tensor.as_tuple()[axis_index]
    else:
        sigma_axis = float(material.sigma_e)

    return Material(
        eps_r=eps_axis,
        sigma_e=sigma_axis,
        debye_poles=material.debye_poles,
        drude_poles=material.drude_poles,
        lorentz_poles=material.lorentz_poles,
    )


def _fully_anisotropic_medium(material, td, length_scale: float):
    """Convert a full off-diagonal ``Tensor3x3`` permittivity to a Tidy3D medium.

    Tidy3D's ``FullyAnisotropicMedium`` carries a symmetric-positive-definite 3x3
    permittivity tensor plus a conductivity tensor that must share its principal
    axes; the maxwell tensor is validated SPD at construction, so it maps across
    row-for-row. The conductivity is diagonal in the lab frame (scalar ``sigma_e``
    or a ``DiagonalTensor3``) and scaled by the metre->Tidy3D length factor exactly
    as the isotropic path, so ``eps'' = sigma/(w*eps0)`` matches physically. When the
    diagonal conductivity does not share axes with an off-diagonal permittivity,
    Tidy3D's own commutation validator rejects the medium.
    """
    if material.is_dispersive:
        raise NotImplementedError(
            "Tidy3D's FullyAnisotropicMedium is strictly non-dispersive, so a full "
            "off-diagonal permittivity tensor combined with dispersive poles has no Tidy3D "
            "equivalent; supply an axis-aligned DiagonalTensor3 (which exports as a per-axis "
            "dispersive AnisotropicMedium) or drop the dispersion."
        )
    from ..media import DiagonalTensor3

    rows = material.epsilon_tensor.rows
    permittivity = [[float(rows[i][j]) for j in range(3)] for i in range(3)]

    sigma_tensor = material.sigma_e_tensor
    if isinstance(sigma_tensor, DiagonalTensor3):
        sigma_diag = sigma_tensor.as_tuple()
    else:
        sigma = float(material.sigma_e)
        sigma_diag = (sigma, sigma, sigma)
    conductivity = [
        [sigma_diag[i] / length_scale if i == j else 0.0 for j in range(3)]
        for i in range(3)
    ]
    return td.FullyAnisotropicMedium(permittivity=permittivity, conductivity=conductivity)


def _anisotropic_medium(material, td, length_scale: float):
    """Convert an anisotropic Material to a Tidy3D anisotropic medium.

    Axis-aligned ``DiagonalTensor3`` permittivity/conductivity exports as a
    ``td.AnisotropicMedium`` of three per-axis isotropic media (dispersion allowed);
    a full off-diagonal ``Tensor3x3`` exports as a non-dispersive
    ``td.FullyAnisotropicMedium``. A magnetic ``mu_tensor`` has no Tidy3D counterpart.
    """
    if material.mu_tensor is not None:
        raise NotImplementedError(
            "Tidy3D has no anisotropic magnetic medium: its AnisotropicMedium and "
            "FullyAnisotropicMedium describe the electric permittivity/conductivity tensor "
            "with mu_r = 1, so a mu_tensor (anisotropic permeability) has no Tidy3D equivalent."
        )
    if material.has_full_epsilon_tensor:
        return _fully_anisotropic_medium(material, td, length_scale)

    components = [
        _convert_material(_axis_isotropic_material(material, axis), td, length_scale)
        for axis in range(3)
    ]
    return td.AnisotropicMedium(xx=components[0], yy=components[1], zz=components[2])


def _nonlinear_spec(material, td, length_scale: float):
    """Build a Tidy3D ``NonlinearSpec`` for a Material's nonlinear channels, or None.

    maxwell and Tidy3D share the instantaneous Kerr form ``P_NL = eps0*chi3*|E|^2*E``
    and the two-photon-absorption model, so the chi3 (Kerr) and TPA channels map
    directly onto ``td.NonlinearSusceptibility`` and ``td.TwoPhotonAbsorption``.

    Unit conventions (Tidy3D is a micrometre solver with ``E`` in [V/um]):

    * ``chi3`` is [m^2/V^2] in maxwell and [um^2/V^2] in Tidy3D. The dimensionless
      fractional-permittivity correction ``chi3*|E|^2`` must be preserved, and a
      physical field of ``E`` [V/m] equals ``E/length_scale`` [V/um], so
      ``chi3_um = chi3_SI * length_scale**2``. (Cross-checked via Tidy3D's own
      ``n2 = 3/(4 n0^2 eps0 c0) chi3`` relation, which then yields the same physical
      ``n2`` scaled to [um^2/W].)
    * TPA ``beta`` is [m/W] in maxwell and [um/W] in Tidy3D, so
      ``beta_um = beta_SI * length_scale``. ``n0`` (the linear index used in the
      intensity conversion) defaults to ``sqrt(eps_r)`` in maxwell and is forwarded
      explicitly; otherwise Tidy3D would infer it from the source frequencies.

    Second-order (chi2) susceptibility has no public Tidy3D equivalent (Tidy3D's
    public nonlinear API is the chi3/Kerr/TPA family only) and is rejected.
    """
    from ..media import TwoPhotonAbsorption

    if not material.is_nonlinear:
        return None

    if material.nonlinear_chi2 != 0.0:
        raise NotImplementedError(
            "Tidy3D has no second-order (chi2 / second-harmonic-generation) nonlinear "
            "medium; its public nonlinear API is the chi3 / Kerr / two-photon-absorption "
            "family only. Validate chi2 against an FDTD analytic reference instead."
        )

    models = []

    chi3 = material.nonlinear_chi3
    if chi3 != 0.0:
        models.append(td.NonlinearSusceptibility(chi3=chi3 * length_scale ** 2))

    tpa_specs = [s for s in material.nonlinearity if isinstance(s, TwoPhotonAbsorption)]
    if tpa_specs:
        base_index = float(math.sqrt(float(material.eps_r)))
        n0_values = {
            float(spec.n0) if spec.n0 is not None else base_index for spec in tpa_specs
        }
        if len(n0_values) != 1:
            raise ValueError(
                "Tidy3D represents two-photon absorption with a single model carrying one "
                f"linear index n0, but the material combines TPA terms with differing n0 "
                f"({sorted(n0_values)}); merge them or give them a common n0."
            )
        n0 = n0_values.pop()
        beta = sum(float(spec.beta) for spec in tpa_specs)
        models.append(td.TwoPhotonAbsorption(beta=beta * length_scale, n0=n0))

    return td.NonlinearSpec(models=models)


def _modulated_medium(material, modulation, td, length_scale: float):
    """Convert a time-modulated Material to a Tidy3D ``Medium`` with a ``ModulationSpec``.

    maxwell's modulation makes the static permittivity harmonic,

    ``eps(x, t) = eps_static(x) * (1 + amplitude(x) * cos(2*pi*f*t + phase(x)))``,

    so the *absolute* permittivity deviation is
    ``delta_eps(x, t) = eps_static * amplitude * cos(2*pi*f*t + phase)``.

    Tidy3D adds a separable space-time modulation to the non-dispersive part
    (eps_inf / conductivity) of a medium,

    ``delta_eps(r, t) = Re[amp_time(t) * amp_space(r)]``, with
    ``amp_time(t) = A_t * e^{i*phi_t - 2*pi*i*f*t}`` and
    ``amp_space(r) = A_s(r) * e^{i*phi_s(r)}``.

    Putting all magnitude/phase in the space part (``A_t = 1``, ``phi_t = 0``) gives
    ``delta_eps = A_s * cos(2*pi*f*t - phi_s)``. Matching maxwell term-for-term:

    * ``A_s = eps_static * amplitude`` (Tidy3D's amplitude is an *absolute* permittivity
      deviation, not the dimensionless depth), and
    * ``phi_s = -phase`` (the sign flips because Tidy3D's e^{-i*omega*t} time factor is
      the conjugate of maxwell's ``+phase`` convention; cosine parity turns
      ``cos(2*pi*f*t + phi_s)`` back into ``cos(2*pi*f*t + phase)``).

    maxwell caps the modulation depth at ``< 0.5``, so ``A_s < 0.5*eps_static`` and the
    modulated permittivity stays strictly positive, satisfying Tidy3D's own
    non-negative-permittivity modulation validator.
    """
    # maxwell forbids a modulated Material from carrying anisotropy or a static
    # electric conductivity at construction, so a modulated export is always an
    # isotropic Medium whose only modulated quantity is eps_inf. Dispersion and the
    # instantaneous nonlinear channels, however, are folded through the SAME per-step
    # modulation factor as the eps_inf background in maxwell's runtime, whereas Tidy3D's
    # ModulationSpec modulates ONLY the non-dispersive part and leaves the pole /
    # nonlinear polarization currents unmodulated. The two models therefore disagree,
    # so a modulated + dispersive / + nonlinear material has no equivalent Tidy3D medium.
    if material.is_dispersive or material.is_nonlinear:
        raise NotImplementedError(
            "Tidy3D's ModulationSpec modulates only the non-dispersive permittivity "
            "(eps_inf) and conductivity, but maxwell folds the same per-step modulation "
            "factor through the dispersive polarization current and the instantaneous "
            "Kerr/chi2/TPA coefficient, so a time-modulated Material that also carries "
            "dispersion or a nonlinearity has no equivalent Tidy3D construct. Validate the "
            "modulated dispersive/nonlinear cell against an FDTD analytic reference instead."
        )

    if not isinstance(modulation.amplitude, float) or not isinstance(modulation.phase, float):
        raise NotImplementedError(
            "Tidy3D represents a spatially-varying modulation profile as a SpaceModulation "
            "carrying a SpatialDataArray on absolute simulation coordinates, but maxwell's "
            "per-cell modulation amplitude/phase tensors are defined relative to the owning "
            "structure's Box and are not available at material-conversion time. Export a "
            "scalar-amplitude, scalar-phase ModulationSpec instead."
        )

    eps_static = float(material.eps_r)
    space_modulation = td.SpaceModulation(
        amplitude=eps_static * modulation.amplitude,
        phase=-modulation.phase,
    )
    time_modulation = td.ContinuousWaveTimeModulation(
        freq0=modulation.frequency,
        amplitude=1.0,
        phase=0.0,
    )
    modulation_spec = td.ModulationSpec(
        permittivity=td.SpaceTimeModulation(
            space_modulation=space_modulation,
            time_modulation=time_modulation,
        )
    )
    return td.Medium(permittivity=eps_static, modulation_spec=modulation_spec)


def _sheet_surface_medium(medium, td, length_scale: float):
    """Build the Tidy3D surface-conductivity medium reproducing a ``Medium2D`` sheet.

    A maxwell ``Medium2D`` carries the complex sheet conductivity ``sigma_s(omega)``
    [S] (siemens; a *surface* conductance ``J_s = sigma_s * E_t``, so it is
    unit-system independent and is NOT length-scaled). Tidy3D models a 2D sheet as
    ``td.Medium2D(ss, tt)`` whose per-tangential media report the sheet conductivity
    through ``sigma_model(omega) = (eps_inf - eps(omega)) * i * omega * eps0``. This
    returns the isotropic ``ss == tt`` medium whose ``sigma_model`` equals
    ``medium.sheet_conductivity`` at every frequency:

    * a purely static sheet (only ``sigma_s``) is ``td.Medium(conductivity=sigma_s)``
      because ``Medium.sigma_model`` returns the conductivity verbatim (``eps0``
      cancels), so the sheet conductance is passed through unscaled;
    * a single Drude sheet term ``weight/(rate - i*omega)`` (the graphene intraband
      Kubo channel) is a ``td.Drude`` with ``plasma_frequency = sqrt(weight/eps0)``
      and ``gamma = rate``, since ``td``'s Drude conductivity evaluates to
      ``eps0*omega_p^2/(gamma_ang - i*omega)``;
    * any richer combination (static + Drude + Lorentz, e.g. graphene with the fitted
      interband Lorentz sheet terms) folds into a single ``td.PoleResidue`` by summing
      each channel's pole-residue contribution (the static conductance maps to the
      ``a = 0`` pole ``c = sigma_s/(2*eps0)``, and the Drude/Lorentz sheet terms reuse
      Tidy3D's own ``Drude``/``Lorentz`` -> ``pole_residue`` conversion).

    ``eps0`` is Tidy3D's micrometre-unit vacuum permittivity ``eps0_SI / length_scale``.
    It cancels in ``sigma_model`` for the static channel and sets the Drude/Lorentz
    residue magnitudes for the dispersive ones, so the reported sheet conductance stays
    in physical siemens.
    """
    eps0 = VACUUM_PERMITTIVITY / length_scale
    sigma_s = float(medium.sigma_s)
    drude_terms = tuple(medium.sheet_pole_terms())
    lorentz_terms = tuple(medium.sheet_lorentz_terms())

    if not drude_terms and not lorentz_terms:
        kwargs = {"permittivity": 1.0}
        if sigma_s != 0.0:
            kwargs["conductivity"] = sigma_s
        return td.Medium(**kwargs)

    if sigma_s == 0.0 and len(drude_terms) == 1 and not lorentz_terms:
        weight, rate = drude_terms[0]
        plasma_frequency = math.sqrt(weight / eps0) / (2.0 * math.pi)
        gamma = rate / (2.0 * math.pi)
        return td.Drude(eps_inf=1.0, coeffs=((plasma_frequency, gamma),))

    poles: list = []
    if sigma_s != 0.0:
        poles.append((complex(0.0, 0.0), complex(sigma_s / (2.0 * eps0), 0.0)))
    for weight, rate in drude_terms:
        plasma_frequency = math.sqrt(weight / eps0) / (2.0 * math.pi)
        gamma = rate / (2.0 * math.pi)
        poles.extend(
            td.Drude(eps_inf=1.0, coeffs=((plasma_frequency, gamma),)).pole_residue.poles
        )
    for strength, omega_0, gamma in lorentz_terms:
        # td.Lorentz eps uses eps_inf + de*f0^2/(f0^2 - 2i*f*delta - f^2), so its
        # angular damping is 2*(2*pi*delta); matching the sheet term's angular gamma
        # gives delta = gamma / (4*pi), and eps0*de = strength gives de = strength/eps0.
        delta_eps = strength / eps0
        resonance = omega_0 / (2.0 * math.pi)
        damping = gamma / (4.0 * math.pi)
        poles.extend(
            td.Lorentz(eps_inf=1.0, coeffs=((delta_eps, resonance, damping),)).pole_residue.poles
        )
    return td.PoleResidue(eps_inf=1.0, poles=tuple(poles))


def _medium2d(material, td, length_scale: float):
    """Convert a maxwell ``Medium2D`` (including ``Graphene``) to a Tidy3D ``Medium2D``.

    The sheet is isotropic in its tangential plane, so both Tidy3D tangential surface
    media ``ss`` and ``tt`` are the same surface-conductivity medium reproducing
    ``material.sheet_conductivity(omega)``. ``Graphene`` exports through the same path:
    its intraband Drude sheet term and any fitted interband Lorentz sheet terms are
    carried by ``_sheet_surface_medium``.
    """
    surface = _sheet_surface_medium(material, td, length_scale)
    return td.Medium2D(ss=surface, tt=surface)


def _lossy_metal_medium(material, td, length_scale: float, frequencies):
    """Convert a maxwell ``LossyMetalMedium`` to a Tidy3D ``LossyMetalMedium``.

    Both model a good conductor through a surface-impedance (Leontovich) boundary
    condition. The maxwell bulk ``conductivity`` [S/m] maps to Tidy3D's ``conductivity``
    [S/um] by the metre->Tidy3D length scale, exactly as the volumetric ``sigma_e``
    path, so the exported Leontovich surface impedance
    ``Z_s(omega) = (1 - i) * sqrt(omega*mu0/(2*sigma))`` [ohm] matches maxwell's
    ``surface_impedance`` (ohms are unit-system independent). Tidy3D vector-fits
    ``Z_s(omega)`` over a required ``frequency_range``, so the export frequencies must be
    supplied; a single operating frequency (the narrowband SIBC case) is widened into a
    non-degenerate fit band around it.
    """
    if not frequencies:
        raise ValueError(
            "LossyMetalMedium export needs the operating frequencies: Tidy3D fits the "
            "surface impedance Z_s(omega) over a frequency_range, so pass frequencies=... "
            "to Scene.to_tidy3d()."
        )
    freqs = tuple(float(frequency) for frequency in frequencies)
    f_min, f_max = min(freqs), max(freqs)
    if f_min == f_max:
        f_min, f_max = 0.5 * f_min, 2.0 * f_max
    return td.LossyMetalMedium(
        conductivity=float(material.conductivity) / length_scale,
        frequency_range=(f_min, f_max),
    )


def _convert_material(material, td, length_scale: float = _M_TO_UM, frequencies=None):
    """Convert a Maxwell Material to a Tidy3D medium."""
    if getattr(material, "is_pec", False):
        # A PEC marker has no finite permittivity (eps_r stays at its 1.0
        # default). Without this early branch it would fall through to the
        # non-dispersive td.Medium(permittivity=1.0) path and silently export
        # as vacuum. Tidy3D models a perfect electric conductor with the
        # dedicated PECMedium (eps_model -> -inf), not a finite dielectric.
        return td.PECMedium()
    if getattr(material, "is_medium2d", False):
        return _medium2d(material, td, length_scale)
    if getattr(material, "is_lossy_metal", False):
        return _lossy_metal_medium(material, td, length_scale, frequencies)
    if getattr(material, "has_custom_poles", False):
        raise NotImplementedError(
            "Tidy3D export for spatially-varying custom dispersive poles is not implemented yet."
        )
    if getattr(material, "perturbation", None) is not None:
        raise NotImplementedError("Tidy3D export for PerturbationMedium is not implemented yet.")
    if material.is_magnetic_dispersive:
        raise NotImplementedError("Tidy3D export for magnetic dispersive Material is not implemented yet.")
    if not math.isclose(float(material.mu_r), 1.0, rel_tol=0.0, abs_tol=1.0e-12) or float(
        getattr(material, "sigma_m", 0.0)
    ) != 0.0:
        raise NotImplementedError(
            "Tidy3D export currently assumes mu_r = 1 and no static magnetic conductivity "
            "(sigma_m = 0); magnetically-lossy media have no Tidy3D equivalent and are not implemented yet."
        )

    modulation = getattr(material, "modulation", None)
    if modulation is not None:
        return _modulated_medium(material, modulation, td, length_scale)

    if material.is_anisotropic:
        return _anisotropic_medium(material, td, length_scale)

    nonlinear_spec = _nonlinear_spec(material, td, length_scale)
    sigma_e = float(getattr(material, "sigma_e", 0.0))

    if not material.is_dispersive:
        kwargs = {"permittivity": material.eps_r}
        if sigma_e != 0.0:
            # Tidy3D works in micrometres, so its Medium.conductivity is [S/um].
            # Convert the SI conductivity [S/m] by the metre->Tidy3D length scale so
            # the exported eps'' = sigma/(w*eps0) matches the physical value.
            kwargs["conductivity"] = sigma_e / length_scale
        if nonlinear_spec is not None:
            kwargs["nonlinear_spec"] = nonlinear_spec
        return td.Medium(**kwargs)

    if sigma_e == 0.0:
        return _dispersive_medium(material, td, nonlinear_spec)

    # Static electric conductivity combined with a dispersive pole model. Tidy3D's
    # specialized Drude/Lorentz/Debye media cannot carry a conductivity, so fold the
    # dispersion and the conductivity into a single PoleResidue: convert the
    # dispersive medium to its equivalent pole-residue form and append the
    # zero-frequency conductivity pole. Any Kerr/TPA nonlinearity rides along on the
    # same PoleResidue.
    base = _dispersive_medium(material, td).pole_residue
    poleres_kwargs = dict(
        eps_inf=base.eps_inf,
        poles=tuple(base.poles) + (_conductivity_pole(sigma_e),),
    )
    if nonlinear_spec is not None:
        poleres_kwargs["nonlinear_spec"] = nonlinear_spec
    return td.PoleResidue(**poleres_kwargs)


# ---------------------------------------------------------------------------
# Geometry conversion (all lengths scaled)
# ---------------------------------------------------------------------------

def _convert_geometry(geometry, td, s):
    """Convert maxwell geometry to a Tidy3D geometry (lengths x *s*)."""
    kind = geometry.kind

    if kind == "box":
        return td.Box(center=_scale3(geometry.position, s), size=_scale3(geometry.size, s))

    if kind == "sphere":
        return td.Sphere(center=_scale3(geometry.position, s), radius=float(geometry.radius) * s)

    if kind == "cylinder":
        axis_idx = _axis_name_to_index(geometry.axis)
        return td.Cylinder(
            center=_scale3(geometry.position, s),
            radius=float(geometry.radius) * s,
            length=float(geometry.height) * s,
            axis=axis_idx,
        )

    if kind == "cone":
        axis_idx = _axis_name_to_index(geometry.axis)
        sidewall_angle = math.atan2(float(geometry.radius), float(geometry.height))
        return td.Cylinder(
            center=_scale3(geometry.position, s),
            radius=float(geometry.radius) * s,
            length=float(geometry.height) * s,
            axis=axis_idx,
            sidewall_angle=sidewall_angle,
        )

    if kind == "ellipsoid":
        rx, ry, rz = float(geometry.radii[0]), float(geometry.radii[1]), float(geometry.radii[2])
        if abs(rx - ry) < 1e-12 and abs(ry - rz) < 1e-12:
            return td.Sphere(center=_scale3(geometry.position, s), radius=rx * s)
        raise NotImplementedError(
            "Tidy3D has no native Ellipsoid geometry. "
            "Consider using a Box approximation or a GDS-based PolySlab."
        )

    raise NotImplementedError(
        f"Geometry type '{kind}' has no Tidy3D mapping yet. "
        f"Supported: box, sphere, cylinder, cone."
    )


# ---------------------------------------------------------------------------
# Structure conversion
# ---------------------------------------------------------------------------

def _convert_structure(structure, td, s, frequencies=None):
    """Convert maxwell Structure to a Tidy3D Structure."""
    td_geometry = _convert_geometry(structure.geometry, td, s)
    td_material = _convert_material(structure.material, td, s, frequencies=frequencies)
    return td.Structure(geometry=td_geometry, medium=td_material)


# ---------------------------------------------------------------------------
# Source conversion
# ---------------------------------------------------------------------------

def _convert_source(source, scene, td, s):
    """Convert a maxwell source to a Tidy3D source (lengths x *s*)."""
    from ..sources import PointDipole, PlaneWave, GaussianBeam
    domain_bounds = scene.domain.bounds

    if isinstance(source, PointDipole):
        component = _polarization_to_component(source.polarization)
        td_source_time = _convert_source_time(source.source_time, td)
        return td.PointDipole(
            center=_scale3(source.position, s),
            source_time=td_source_time,
            polarization=component,
            name=source.name or "point_dipole",
        )

    if isinstance(source, PlaneWave):
        td_source_time = _convert_source_time(source.source_time, td)
        direction = source.direction

        injection_axis = resolve_injection_axis(direction, source.injection_axis)
        axis_idx = _axis_name_to_index(injection_axis)
        inject_dir = "+" if direction[axis_idx] > 0 else "-"

        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [td.inf, td.inf, td.inf]
        center[axis_idx] = soft_plane_wave_coordinate(scene, injection_axis, float(direction[axis_idx])) * s
        size[axis_idx] = 0.0

        pol_angle, angle_theta, angle_phi = _direction_to_angles(direction, axis_idx)

        return td.PlaneWave(
            center=tuple(center),
            size=tuple(size),
            source_time=td_source_time,
            direction=inject_dir,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            name=source.name or "plane_wave",
        )

    if isinstance(source, GaussianBeam):
        td_source_time = _convert_source_time(source.source_time, td)
        direction = source.direction

        abs_dir = [abs(d) for d in direction]
        dominant_axis = int(np.argmax(abs_dir))
        inject_dir = "+" if direction[dominant_axis] > 0 else "-"

        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[dominant_axis] = source.focus[dominant_axis] * s
        size[dominant_axis] = 0.0

        pol_angle, angle_theta, angle_phi = _direction_to_angles(direction, dominant_axis)

        return td.GaussianBeam(
            center=tuple(center),
            size=tuple(size),
            source_time=td_source_time,
            direction=inject_dir,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            waist_radius=source.beam_waist * s,
            name=source.name or "gaussian_beam",
        )

    raise NotImplementedError(
        f"Source type '{type(source).__name__}' has no Tidy3D mapping yet."
    )


def _direction_to_angles(direction, dominant_axis):
    """Compute Tidy3D (pol_angle, angle_theta, angle_phi) from direction vector."""
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-15:
        return 0.0, 0.0, 0.0
    dx, dy, dz = dx / norm, dy / norm, dz / norm

    tol = 1e-6
    if dominant_axis == 0 and abs(dy) < tol and abs(dz) < tol:
        return 0.0, 0.0, 0.0
    if dominant_axis == 1 and abs(dx) < tol and abs(dz) < tol:
        return 0.0, 0.0, 0.0
    if dominant_axis == 2 and abs(dx) < tol and abs(dy) < tol:
        return 0.0, 0.0, 0.0

    if dominant_axis == 2:
        angle_theta = math.acos(abs(dz))
        angle_phi = math.atan2(dy, dx)
    elif dominant_axis == 0:
        angle_theta = math.acos(abs(dx))
        angle_phi = math.atan2(dz, dy)
    else:
        angle_theta = math.acos(abs(dy))
        angle_phi = math.atan2(dx, dz)

    return 0.0, angle_theta, angle_phi


# ---------------------------------------------------------------------------
# Monitor conversion
# ---------------------------------------------------------------------------

def _convert_monitor(monitor, domain_bounds, frequencies, td, s):
    """Convert a maxwell monitor to a Tidy3D monitor (lengths x *s*)."""
    from ..monitors import FinitePlaneMonitor, ModeMonitor, PointMonitor, PlaneMonitor

    monitor_frequencies = (
        monitor.frequencies
        if hasattr(monitor, "frequencies") and monitor.frequencies
        else frequencies
    )
    if not monitor_frequencies:
        raise ValueError(
            f"Monitor '{monitor.name}' has no frequencies and none were passed to to_tidy3d()."
        )

    if isinstance(monitor, PointMonitor):
        return td.FieldMonitor(
            center=_scale3(monitor.position, s),
            size=(0.0, 0.0, 0.0),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, ModeMonitor):
        axis_idx = _axis_name_to_index(monitor.axis)
        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[axis_idx] = monitor.plane_position * s
        size[axis_idx] = 0.0
        return td.FieldMonitor(
            center=tuple(center),
            size=tuple(size),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, FinitePlaneMonitor):
        center = _scale3(monitor.position, s)
        size = _scale3(monitor.size, s)

        if monitor.compute_flux:
            return td.FluxMonitor(
                center=center,
                size=size,
                freqs=list(monitor_frequencies),
                name=monitor.name,
                normal_dir=monitor.normal_direction,
            )

        return td.FieldMonitor(
            center=center,
            size=size,
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, PlaneMonitor):
        axis_idx = _axis_name_to_index(monitor.axis)
        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[axis_idx] = monitor.position * s
        size[axis_idx] = 0.0

        if monitor.compute_flux:
            return td.FluxMonitor(
                center=tuple(center),
                size=tuple(size),
                freqs=list(monitor_frequencies),
                name=monitor.name,
                normal_dir=monitor.normal_direction,
            )

        return td.FieldMonitor(
            center=tuple(center),
            size=tuple(size),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    raise NotImplementedError(
        f"Monitor type '{type(monitor).__name__}' has no Tidy3D mapping yet."
    )


# ---------------------------------------------------------------------------
# Boundary conversion (no length scaling)
# ---------------------------------------------------------------------------

def _convert_boundary(boundary, td):
    """Convert maxwell BoundarySpec to Tidy3D BoundarySpec."""
    kind = boundary.kind
    if boundary.bloch_wavevector == "auto":
        raise ValueError(
            "Automatic Bloch wavevectors require Simulation.prepare() and cannot be exported to Tidy3D unresolved."
        )

    if kind == "pml":
        pml = td.PML(num_layers=boundary.num_layers)
        return td.BoundarySpec.all_sides(boundary=pml)

    if kind == "periodic":
        periodic = td.Periodic()
        return td.BoundarySpec.all_sides(boundary=periodic)

    if kind == "pec":
        pec = td.PECBoundary()
        return td.BoundarySpec.all_sides(boundary=pec)

    if kind == "pmc":
        pmc = td.PMCBoundary()
        return td.BoundarySpec.all_sides(boundary=pmc)

    if kind == "bloch":
        bloch_x = td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[0])
        bloch_y = td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[1])
        bloch_z = td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[2])
        return td.BoundarySpec(
            x=td.Boundary(plus=bloch_x, minus=bloch_x),
            y=td.Boundary(plus=bloch_y, minus=bloch_y),
            z=td.Boundary(plus=bloch_z, minus=bloch_z),
        )

    if kind == "none":
        pec = td.PECBoundary()
        return td.BoundarySpec.all_sides(boundary=pec)

    if kind == "mixed":
        axis_index = {"x": 0, "y": 1, "z": 2}

        def convert_face(face_kind, axis):
            if face_kind == "pml":
                return td.PML(num_layers=boundary.num_layers)
            if face_kind == "periodic":
                return td.Periodic()
            if face_kind == "pec":
                return td.PECBoundary()
            if face_kind == "pmc":
                return td.PMCBoundary()
            if face_kind == "bloch":
                return td.BlochBoundary(bloch_vec=boundary.bloch_wavevector[axis_index[axis]])
            if face_kind == "none":
                return td.PECBoundary()
            raise ValueError(f"Unsupported boundary kind: {face_kind}")

        return td.BoundarySpec(
            x=td.Boundary(
                minus=convert_face(boundary.face_kind("x", "low"), "x"),
                plus=convert_face(boundary.face_kind("x", "high"), "x"),
            ),
            y=td.Boundary(
                minus=convert_face(boundary.face_kind("y", "low"), "y"),
                plus=convert_face(boundary.face_kind("y", "high"), "y"),
            ),
            z=td.Boundary(
                minus=convert_face(boundary.face_kind("z", "low"), "z"),
                plus=convert_face(boundary.face_kind("z", "high"), "z"),
            ),
        )

    raise ValueError(f"Unsupported boundary kind: {kind}")


# ---------------------------------------------------------------------------
# Grid conversion
# ---------------------------------------------------------------------------

def _convert_grid(grid, td, s):
    """Convert maxwell GridSpec to Tidy3D GridSpec (dl x *s*)."""
    if grid.is_custom or grid.is_auto:
        raise NotImplementedError(
            "Tidy3D export does not support nonuniform (GridSpec.custom / "
            "GridSpec.auto) grids; use a uniform GridSpec."
        )
    if grid.dx == grid.dy == grid.dz:
        return td.GridSpec.uniform(dl=grid.dx * s)
    return td.GridSpec(
        grid_x=td.UniformGrid(dl=grid.dx * s),
        grid_y=td.UniformGrid(dl=grid.dy * s),
        grid_z=td.UniformGrid(dl=grid.dz * s),
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def scene_to_tidy3d(
    scene,
    *,
    frequencies: float | Sequence[float] | None = None,
    run_time: float | None = None,
    length_scale: float = _M_TO_UM,
    **kwargs,
) -> tidy3d.Simulation:
    """Convert a maxwell ``Scene`` to a ``tidy3d.Simulation``.

    Parameters
    ----------
    scene : Scene
        The maxwell scene to convert.
    frequencies : float or sequence of float, optional
        Monitoring / source frequencies in Hz.  Required for monitors.
    run_time : float, optional
        Simulation run time in seconds.  If not given, an estimate is made
        from the domain size (10 light-crossing times).
    length_scale : float
        Factor to convert maxwell lengths (metres) to Tidy3D lengths.
        Default ``1e6`` (metres tomicrometres, the Tidy3D convention).
    **kwargs
        Extra keyword arguments forwarded to ``tidy3d.Simulation``.

    Returns
    -------
    tidy3d.Simulation
    """
    td = _ensure_tidy3d()
    s = length_scale  # shorthand

    # -- frequencies -----------------------------------------------------------
    if isinstance(frequencies, (int, float)):
        frequencies = (float(frequencies),)
    elif frequencies is not None:
        frequencies = tuple(float(frequency) for frequency in frequencies)

    # -- domain ----------------------------------------------------------------
    if scene.domain is None:
        raise ValueError("Scene must have a Domain for Tidy3D export.")
    center, size = _domain_to_center_size(scene.domain, s)

    # -- run_time estimate -----------------------------------------------------
    if run_time is None:
        c0 = 299_792_458.0
        max_extent_m = max(
            b[1] - b[0] for b in scene.domain.bounds
        )
        run_time = 10.0 * max_extent_m / c0

    # -- grid ------------------------------------------------------------------
    td_grid = None
    if scene.grid is not None:
        td_grid = _convert_grid(scene.grid, td, s)

    # -- boundary --------------------------------------------------------------
    td_boundary = td.BoundarySpec.all_sides(boundary=td.PML())
    if scene.boundary is not None:
        td_boundary = _convert_boundary(scene.boundary, td)

    # -- structures ------------------------------------------------------------
    td_structures = []
    for structure in (scene.structures or []):
        td_structures.append(_convert_structure(structure, td, s, frequencies=frequencies))

    # -- sources ---------------------------------------------------------------
    domain_bounds = scene.domain.bounds
    td_sources = []
    sources = scene.resolved_sources() if hasattr(scene, "resolved_sources") else (scene.sources or [])
    for source in sources:
        td_sources.append(_convert_source(source, scene, td, s))

    # -- monitors --------------------------------------------------------------
    td_monitors = []
    monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else (scene.monitors or [])
    for monitor in monitors:
        td_monitors.append(_convert_monitor(monitor, domain_bounds, frequencies, td, s))

    # -- symmetry --------------------------------------------------------------
    td_symmetry = (0, 0, 0)
    if scene.symmetry is not None:
        # Tidy3D encodes symmetry about the domain center only; the folded face
        # (low/high) has no Tidy3D counterpart and is dropped in the export.
        sym_map = {"PEC": -1, "PMC": 1}
        td_symmetry = tuple(
            0 if entry is None else sym_map.get(entry[0], 0)
            for entry in scene.symmetry
        )

    # -- build simulation ------------------------------------------------------
    sim_kwargs = dict(
        center=center,
        size=size,
        run_time=run_time,
        structures=td_structures,
        sources=td_sources,
        monitors=td_monitors,
        boundary_spec=td_boundary,
        symmetry=td_symmetry,
    )
    if td_grid is not None:
        sim_kwargs["grid_spec"] = td_grid
    sim_kwargs.update(kwargs)

    return td.Simulation(**sim_kwargs)
