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
import torch
from witwin.core import BooleanGeometry
from witwin.core.material import VACUUM_PERMITTIVITY
from ..fdtd.excitation.spatial import resolve_injection_axis, soft_plane_wave_coordinate

if TYPE_CHECKING:
    import tidy3d

# Default length conversion factor: metres to micrometres.
_M_TO_UM = 1e6
_C0 = 299_792_458.0
# Tidy3D's TFSF source injects 1 W/um^2 by definition, corresponding
# to this incident electric-field amplitude in V/um. Maxwell's TFSF
# source amplitude is instead the physical incident E field in V/m.
_TIDY3D_TFSF_UNIT_FIELD = math.sqrt(2.0 / (_C0 * VACUUM_PERMITTIVITY))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def low_frequency_smoothing_to_tidy3d(spec, monitors):
    """Map a LowFrequencySmoothingSpec to the reference spec for the named mode monitors."""
    td = _ensure_tidy3d()
    return td.LowFrequencySmoothingSpec(
        monitors=tuple(monitors),
        min_sampling_time=spec.min_sampling_time,
        max_sampling_time=spec.max_sampling_time,
        order=spec.order,
        max_deviation=spec.max_deviation,
    )


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


def _mode_polarization_fraction(normal_axis: str, polarization) -> str:
    """Map a requested tangential field axis to Tidy3D's modal polarization fraction."""
    tangential_axes = tuple(axis for axis in "xyz" if axis != normal_axis)
    polarization_axis = "xyz"[int(np.argmax(np.abs(polarization)))]
    if polarization_axis not in tangential_axes:
        raise ValueError("Mode polarization must be tangential to the source or monitor plane.")
    return "TE_fraction" if polarization_axis == tangential_axes[0] else "TM_fraction"


def _mode_sort_spec(td, normal_axis: str, polarization):
    """Prefer the requested field axis using Tidy3D's current mode-sorting API."""
    return td.ModeSortSpec(
        filter_key=_mode_polarization_fraction(normal_axis, polarization),
        filter_reference=0.5,
        filter_order="over",
    )


def _mode_candidate_count(mode_index: int) -> int:
    """Request a padded two-polarization window through the selected order."""
    return 2 * (int(mode_index) + 4)


# ---------------------------------------------------------------------------
# Source-time conversion (no length scaling - purely temporal / frequency)
# ---------------------------------------------------------------------------

def _convert_source_time(source_time, td, *, amplitude_scale: float = 1.0):
    """Convert maxwell source_time to a Tidy3D SourceTime."""
    from ..sources import CW, BroadbandPulse, GaussianPulse, RickerWavelet

    if source_time is None:
        raise ValueError("source_time must be set for Tidy3D export.")

    if isinstance(source_time, CW):
        return td.ContinuousWave(
            freq0=source_time.frequency,
            fwidth=0.1 * source_time.frequency,
            amplitude=source_time.amplitude * amplitude_scale,
            phase=source_time.phase,
        )

    if isinstance(source_time, GaussianPulse):
        offset = source_time.delay / source_time.sigma_t
        # Maxwell defines a real Gaussian-envelope cosine centered at ``delay``.
        # Tidy3D uses Re[i exp(i*phase) exp(-i*omega*t)] when DC removal is off,
        # so compensate both its Fourier sign convention and the delayed carrier.
        phase = math.remainder(
            2.0 * math.pi * source_time.frequency * source_time.delay
            - source_time.phase
            - 0.5 * math.pi,
            2.0 * math.pi,
        )
        return td.GaussianPulse(
            freq0=source_time.frequency,
            fwidth=source_time.fwidth,
            amplitude=source_time.amplitude * amplitude_scale,
            phase=phase,
            offset=offset,
            remove_dc_component=False,
        )

    if isinstance(source_time, BroadbandPulse):
        # Tidy3D's BroadbandPulse waveform is computed by its licensed extension;
        # export the shared declarative contract (band, in-band minimum relative
        # amplitude, amplitude, phase, offset in units of 1/(2*pi*bandwidth)).
        # Spectrally normalized results compare directly; raw time traces are
        # implementation-defined on each side.
        return td.BroadbandPulse(
            freq_range=tuple(source_time.freq_range),
            minimum_amplitude=source_time.minimum_amplitude,
            amplitude=source_time.amplitude * amplitude_scale,
            phase=source_time.phase,
            offset=source_time.offset,
        )

    from ..sources import BasebandGaussianPulse, BasebandRectangularPulse

    if isinstance(source_time, BasebandGaussianPulse):
        return td.BasebandGaussianPulse(
            twidth=source_time.twidth,
            offset=source_time.offset,
            amplitude=source_time.amplitude * amplitude_scale,
        )

    if isinstance(source_time, BasebandRectangularPulse):
        return td.BasebandRectangularPulse(
            rise_time=source_time.rise_time,
            twidth=source_time.twidth,
            offset=source_time.offset,
            amplitude=source_time.amplitude * amplitude_scale,
        )

    if isinstance(source_time, RickerWavelet):
        return td.GaussianPulse(
            freq0=source_time.frequency,
            fwidth=source_time.frequency,
            amplitude=source_time.amplitude * amplitude_scale,
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
            # Tidy3D writes the Lorentz denominator as
            # f0^2 - f^2 - 2j*f*delta, while Maxwell stores gamma in
            # f0^2 - f^2 - j*f*gamma after converting angular frequency to Hz.
            (pole.delta_eps, pole.resonance_frequency, 0.5 * pole.gamma)
            for pole in material.lorentz_poles
        ]
        return td.Lorentz(eps_inf=material.eps_r, coeffs=coeffs, **extra)

    if has_debye and not has_drude and not has_lorentz:
        coeffs = [
            # Tidy3D uses 1 - j*f*tau, whereas Maxwell's DebyePole uses
            # 1 - j*omega*tau. Scale tau so both describe the same response.
            (pole.delta_eps, 2.0 * math.pi * pole.tau)
            for pole in material.debye_poles
        ]
        return td.Debye(eps_inf=material.eps_r, coeffs=coeffs, **extra)

    # Mixed pole types to PoleResidue. Let Tidy3D lower each specialized
    # medium into its own PoleResidue representation so its sign and damping
    # conventions remain exactly consistent with eps_model().
    poles = []
    for p in material.drude_poles:
        medium = td.Drude(
            eps_inf=1.0,
            coeffs=((p.plasma_frequency, p.gamma),),
        )
        poles.extend(medium.pole_residue.poles)
    for p in material.lorentz_poles:
        medium = td.Lorentz(
            eps_inf=1.0,
            coeffs=((p.delta_eps, p.resonance_frequency, 0.5 * p.gamma),),
            allow_gain=bool(p.allow_gain),
        )
        poles.extend(medium.pole_residue.poles)
    for p in material.debye_poles:
        medium = td.Debye(
            eps_inf=1.0,
            coeffs=((p.delta_eps, 2.0 * math.pi * p.tau),),
        )
        poles.extend(medium.pole_residue.poles)

    extra = {} if nonlinear_spec is None else {"nonlinear_spec": nonlinear_spec}
    return td.PoleResidue(eps_inf=material.eps_r, poles=tuple(poles), **extra)


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
    from ..media import Tensor3x3
    if isinstance(material.sigma_e_tensor, Tensor3x3):
        raise NotImplementedError(
            'Reference-backend export of full sigma_e_tensor is not implemented; '
            'its off-diagonal conductivity must be preserved.'
        )
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

    if getattr(material,'has_tensor_nonlinearity',False):
        raise NotImplementedError('Tensor nonlinear susceptibility export is not supported')

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
    roughness = getattr(material, "roughness", None)
    if roughness is None:
        return td.LossyMetalMedium(
            conductivity=float(material.conductivity) / length_scale,
            frequency_range=(f_min, f_max),
        )
    return td.LossyMetalMedium(
        conductivity=float(material.conductivity) / length_scale,
        frequency_range=(f_min, f_max),
        roughness=_surface_roughness(roughness, td, length_scale),
    )


def _surface_roughness(roughness, td, length_scale: float):
    """Convert a maxwell surface-roughness model to the matching Tidy3D roughness.

    Both sides apply the same causal complex correction to the smooth surface
    impedance; roughness lengths [m] are scaled to Tidy3D micrometres. Tidy3D's Huray
    ``coeffs`` are ``(surface_ratio, sphere_radius)`` pairs. The real Groisse factor has
    no Tidy3D counterpart and is rejected.
    """
    from ..roughness import HammerstadRoughness, HurayRoughness

    if isinstance(roughness, HammerstadRoughness):
        return td.HammerstadSurfaceRoughness(
            rq=float(roughness.rms_height) * length_scale,
            roughness_factor=float(roughness.roughness_factor),
        )
    if isinstance(roughness, HurayRoughness):
        return td.HuraySurfaceRoughness(
            relative_area=float(roughness.relative_area),
            coeffs=((float(roughness.surface_ratio), float(roughness.sphere_radius) * length_scale),),
        )
    raise ValueError(
        f"{type(roughness).__name__} has no Tidy3D LossyMetalMedium equivalent: Tidy3D "
        "accepts only the causal Hammerstad and Huray roughness models."
    )


def _uniform_grid_value(grid) -> float | None:
    """Single value of a per-cell parameter grid if spatially uniform, else None.

    Custom dispersive poles and ``PerturbationMedium`` carry a 3D torch tensor of
    per-cell oscillator strength / perturbation amplitude defined relative to the
    owning structure's ``Box``. A spatially-uniform grid lowers to a homogeneous
    Tidy3D medium; a varying grid would need a ``CustomMedium`` / ``CustomPoleResidue``
    on a ``SpatialDataArray`` in absolute simulation coordinates, which the box-relative
    grid cannot supply at material-conversion time.
    """
    import torch

    with torch.no_grad():
        lo = float(grid.min())
        hi = float(grid.max())
    return lo if hi == lo else None


def _homogenize_custom_poles(material):
    """Lower a Material whose dispersive poles are per-cell custom poles to scalar poles.

    Each ``CustomPole`` whose strength grid is spatially uniform maps to its scalar
    ``reference_pole()`` (the peak equals the uniform value, so the lowering is exact);
    a spatially-varying custom pole has no homogeneous Tidy3D equivalent and raises. The
    reconstructed Material keeps every other channel (background permittivity/permeability,
    conductivity, tensors, nonlinearity, modulation) so it re-enters ``_convert_material``
    through the ordinary pole path.
    """
    from ..media import CustomDrudePole, CustomPole, Material

    def resolve(pole):
        if not isinstance(pole, CustomPole):
            return pole
        grid = pole.plasma_frequency if isinstance(pole, CustomDrudePole) else pole.delta_eps
        if _uniform_grid_value(grid) is None:
            raise NotImplementedError(
                "Tidy3D export of a spatially-varying custom dispersive pole would need a "
                "CustomPoleResidue on a SpatialDataArray in absolute simulation coordinates, but "
                "maxwell's per-cell pole-strength grid is defined relative to the owning structure's "
                "Box and cannot be resolved into Tidy3D coordinates at material-conversion time. "
                "Export a spatially-uniform custom pole (it lowers to the equivalent homogeneous "
                "pole) or validate the per-cell profile against an FDTD/analytic reference."
            )
        return pole.reference_pole()

    return Material(
        eps_r=material.eps_r,
        mu_r=material.mu_r,
        sigma_e=material.sigma_e,
        sigma_m=material.sigma_m,
        debye_poles=tuple(resolve(p) for p in material.debye_poles),
        drude_poles=tuple(resolve(p) for p in material.drude_poles),
        lorentz_poles=tuple(resolve(p) for p in material.lorentz_poles),
        mu_debye_poles=tuple(resolve(p) for p in material.mu_debye_poles),
        mu_drude_poles=tuple(resolve(p) for p in material.mu_drude_poles),
        mu_lorentz_poles=tuple(resolve(p) for p in material.mu_lorentz_poles),
        epsilon_tensor=material.epsilon_tensor,
        mu_tensor=material.mu_tensor,
        sigma_e_tensor=material.sigma_e_tensor,
        kerr_chi3=material.kerr_chi3,
        nonlinearity=material.nonlinearity,
        modulation=material.modulation,
    )


def _homogenize_perturbation(material):
    """Lower a ``PerturbationMedium`` with a spatially-uniform perturbation to a plain Material.

    ``eps(x) = eps_base + eps_sensitivity * perturbation(x)`` shifts the background
    permittivity (``eps_inf``) only. A spatially-uniform perturbation is a constant shift
    ``eps_sensitivity * value``, applied to the scalar ``eps_r`` or, for a diagonal
    ``DiagonalTensor3`` base, to each principal axis; the base's poles / conductivity /
    nonlinearity / modulation ride through unchanged so the reconstructed Material exports
    exactly like its dispersive/anisotropic base at the shifted background. A spatially-
    varying perturbation has no homogeneous Tidy3D equivalent and raises.
    """
    from ..media import DiagonalTensor3, Material

    value = _uniform_grid_value(material.perturbation)
    if value is None:
        raise NotImplementedError(
            "Tidy3D export of a spatially-varying PerturbationMedium would need a CustomMedium on a "
            "SpatialDataArray in absolute simulation coordinates, but its perturbation field is "
            "defined relative to the owning structure's Box and cannot be resolved into Tidy3D "
            "coordinates at material-conversion time. Export a spatially-uniform perturbation (it "
            "lowers to a homogeneous permittivity shift eps_base + eps_sensitivity*value) or validate "
            "the perturbed profile against an FDTD/analytic reference."
        )

    shift = float(material.eps_sensitivity) * value
    base_tensor = material.epsilon_tensor
    if isinstance(base_tensor, DiagonalTensor3):
        eps_tensor = DiagonalTensor3(base_tensor.xx + shift, base_tensor.yy + shift, base_tensor.zz + shift)
        eps_r = material.eps_r
    else:
        eps_tensor = None
        eps_r = float(material.eps_r) + shift

    return Material(
        eps_r=eps_r,
        mu_r=material.mu_r,
        sigma_e=material.sigma_e,
        sigma_m=material.sigma_m,
        debye_poles=material.debye_poles,
        drude_poles=material.drude_poles,
        lorentz_poles=material.lorentz_poles,
        mu_debye_poles=material.mu_debye_poles,
        mu_drude_poles=material.mu_drude_poles,
        mu_lorentz_poles=material.mu_lorentz_poles,
        epsilon_tensor=eps_tensor,
        mu_tensor=material.mu_tensor,
        sigma_e_tensor=material.sigma_e_tensor,
        kerr_chi3=material.kerr_chi3,
        nonlinearity=material.nonlinearity,
        modulation=material.modulation,
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
    if getattr(material, "is_pmc", False):
        # The magnetic dual of the PEC branch: without it a PMC marker (eps_r = 1)
        # would silently export as vacuum.
        return td.PMCMedium()
    if getattr(material, "is_medium2d", False):
        return _medium2d(material, td, length_scale)
    if getattr(material, "is_lossy_metal", False):
        return _lossy_metal_medium(material, td, length_scale, frequencies)
    if getattr(material, "is_surface_impedance", False):
        # A generic SurfaceImpedanceMedium carries its physics in a broadband causal
        # rational surface-impedance model, not in a finite bulk permittivity (eps_r
        # stays at its 1.0 default, sigma_e = 0). Without this branch it would fall
        # through to the non-dispersive td.Medium(permittivity=1.0) path and silently
        # export as vacuum, dropping the entire surface response -- the exact trap the
        # PEC branch documents. The good-conductor LossyMetalMedium is the only surface
        # with a designed reference-backend mapping (handled above); the generic
        # rational surface has no reference-backend surface-impedance construct wired
        # yet, so fail closed rather than export a physically wrong vacuum structure.
        raise NotImplementedError(
            "Reference-backend export of a generic SurfaceImpedanceMedium is not yet "
            "designed: its physics is a broadband causal rational tangential surface "
            "impedance E_t = Z_s(omega) (n x H), which has no finite bulk-permittivity "
            "equivalent and no reference-backend surface-impedance construct mapped for "
            "it. The narrowband good-conductor LossyMetalMedium exports through the "
            "dedicated lossy-metal surface path; a rational model has no such mapping "
            "until the surface-impedance adapter is designed. Resolve the metal "
            "volumetrically with Material(sigma_e=...), use a LossyMetalMedium for a "
            "good conductor, or use Material.pec() for a lossless conductor."
        )
    if getattr(material, "is_gyromagnetic", False):
        # A gyromagnetic ferrite carries its physics in a non-reciprocal complex 3x3
        # Polder permeability tensor. Because its scalar mu_r stays at mu_infinity
        # (= 1 by default) and it exposes no permeability pole, it is neither caught
        # by the magnetic-dispersive guard nor the mu_r != 1 guard below, and would
        # fall through to the plain td.Medium(permittivity=eps_r) path -- silently
        # exporting as a reciprocal isotropic dielectric and dropping the entire
        # gyrotropy (the same silent-vacuum trap the PEC branch documents). Tidy3D
        # has no non-reciprocal / gyromagnetic medium construct, so fail closed.
        raise NotImplementedError(
            "Tidy3D export for a gyromagnetic ferrite has no equivalent: its permeability is a "
            "non-reciprocal complex 3x3 Polder tensor, and Tidy3D has no gyromagnetic (magnetic, "
            "non-reciprocal) material model -- its FDTD solver fixes mu_r = 1 and exposes no "
            "permeability tensor. Validate the ferrite against the analytic LLG reference oracle "
            "(tests/materials/ferrite/test_ferrite_reference.py) instead."
        )
    if getattr(material, "perturbation", None) is not None:
        # Resolve the background shift first: the lowered Material may still carry
        # (custom) poles, which the recursion then handles through the pole path.
        return _convert_material(_homogenize_perturbation(material), td, length_scale, frequencies=frequencies)
    if getattr(material, "has_custom_poles", False):
        return _convert_material(_homogenize_custom_poles(material), td, length_scale, frequencies=frequencies)
    if material.is_magnetic_dispersive:
        raise NotImplementedError(
            "Tidy3D export for magnetic dispersive Material has no equivalent: Tidy3D has no "
            "magnetic-material model. Its FDTD solver fixes the relative permeability at mu_r = 1 "
            "and exposes no permeability pole (Debye/Drude/Lorentz) construct, so a frequency-"
            "dispersive mu(omega) cannot be represented."
        )
    if not math.isclose(float(material.mu_r), 1.0, rel_tol=0.0, abs_tol=1.0e-12) or float(
        getattr(material, "sigma_m", 0.0)
    ) != 0.0:
        raise NotImplementedError(
            "Tidy3D export currently assumes mu_r = 1 and no static magnetic conductivity "
            "(sigma_m = 0): Tidy3D has no magnetic-material model. Its FDTD solver fixes the "
            "relative permeability at 1 and models no magnetic loss, so a magnetic (mu_r != 1) "
            "or magnetically-lossy (sigma_m != 0) medium has no Tidy3D equivalent."
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

def _triangle_mesh(geometry, td, s):
    """Tessellate a maxwell SDF primitive into a Tidy3D ``TriangleMesh``.

    Every ``witwin.core`` primitive exposes ``to_mesh() -> (vertices, faces)`` with the
    structure's position/rotation already baked into the (metre) vertices, so the mesh is
    in absolute world coordinates. Scaling the vertices by the metre->Tidy3D length factor
    (leaving the integer face table untouched) gives the exact same watertight surface
    Tidy3D voxelizes onto its Yee grid, so a primitive that has no analytic Tidy3D
    counterpart (torus, pyramid, prism, hollow box, general ellipsoid) still round-trips
    geometrically rather than being rejected.
    """
    import torch  # local import, matching the lazy-torch convention in this module

    vertices, faces = geometry.to_mesh()
    vertices_np = vertices.detach().cpu().to(torch.float64).numpy() * float(s)
    faces_np = faces.detach().cpu().to(torch.int64).numpy()
    return td.TriangleMesh.from_vertices_faces(vertices_np, faces_np)


def _poly_slab(geometry, td, s):
    """Convert an untransformed simple ``PolySlab`` to a Tidy3D ``PolySlab``.

    Tidy3D's ``PolySlab`` carries the transverse polygon plus the axial ``slab_bounds`` in
    absolute simulation coordinates and applies no position/rotation, so the faithful
    primitive mapping is exact only when the maxwell slab is un-rotated, centred at the
    origin, and vertical (``sidewall_angle == 0``). The polygon vertices and slab bounds
    scale by the metre->Tidy3D length factor; ``reference_plane`` carries across verbatim.
    A tapered, rotated, or offset slab is baked into a ``TriangleMesh`` instead (its
    ``to_mesh`` already applies the taper and transform), so no geometry is silently lost.
    """
    import torch  # local import, matching the lazy-torch convention in this module

    rotation = getattr(geometry, "rotation", None)
    if rotation is None:
        is_identity_rotation = True
    else:
        # GeometryBase stores rotation as a unit quaternion; the identity is (1, 0, 0, 0).
        quat = [float(v) for v in rotation.detach().cpu().tolist()]
        is_identity_rotation = abs(quat[0] - 1.0) < 1e-12 and all(abs(c) < 1e-12 for c in quat[1:])
    position = [float(v) for v in geometry.position.detach().cpu().tolist()]
    sidewall = float(geometry.sidewall_angle.detach().cpu().item())
    is_simple = (
        is_identity_rotation
        and all(abs(component) < 1e-12 for component in position)
        and abs(sidewall) < 1e-12
    )
    if not is_simple:
        return _triangle_mesh(geometry, td, s)

    vertices = geometry.vertices.detach().cpu().to(torch.float64).numpy() * float(s)
    lo, hi = geometry.bounds.detach().cpu().to(torch.float64).tolist()
    return td.PolySlab(
        vertices=[(float(u), float(v)) for u, v in vertices],
        slab_bounds=(lo * float(s), hi * float(s)),
        axis=_axis_name_to_index(geometry.axis),
        reference_plane=geometry.reference_plane,
    )


def _boolean_geometry(geometry, td, s):
    """Convert a core boolean geometry to Tidy3D's boolean constructs.

    A sharp ``Union`` maps to ``GeometryGroup``, ``Intersection`` to chained
    ``ClipOperation("intersection")`` and ``Difference`` to
    ``ClipOperation("difference")`` against the group of its tools; children
    convert recursively, so analytic children stay analytic. A non-identity node
    frame wraps the result in ``Transformed`` with the node rotation and the
    length-scaled translation. Tidy3D booleans are sharp, so a smooth (blended)
    node exports its own closed ``to_mesh()`` surface as a ``TriangleMesh``.
    """
    import torch  # local import, matching the lazy-torch convention in this module

    if geometry.smooth:
        return _triangle_mesh(geometry, td, s)

    children = [_convert_geometry(child, td, s) for child in geometry.geometries]

    def group(items):
        return items[0] if len(items) == 1 else td.GeometryGroup(geometries=tuple(items))

    if geometry.kind == "union":
        converted = group(children)
    elif geometry.kind == "intersection":
        converted = children[0]
        for child in children[1:]:
            converted = td.ClipOperation(operation="intersection", geometry_a=converted, geometry_b=child)
    elif geometry.kind == "difference":
        converted = td.ClipOperation(operation="difference", geometry_a=children[0], geometry_b=group(children[1:]))
    else:
        return _triangle_mesh(geometry, td, s)

    rotation = np.asarray(geometry._rotation_matrix_np(), dtype=np.float64)
    translation = geometry.position.detach().cpu().to(dtype=torch.float64).numpy() * float(s)
    if np.allclose(rotation, np.eye(3), rtol=0.0, atol=1e-12) and np.allclose(translation, 0.0, rtol=0.0, atol=0.0):
        return converted
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return td.Transformed(geometry=converted, transform=transform)


def _planar_pec_rectangle(geometry, td=None, s=1.0):
    """Zero-thickness reference Box of an axis-aligned rectangular PlanarPEC, or None."""
    vertices = geometry.vertices.detach().cpu().double().numpy()
    if vertices.shape[0] != 4:
        return None
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    corners = {(float(x), float(y)) for x, y in vertices}
    if corners != {(low[0], low[1]), (high[0], low[1]), (high[0], high[1]), (low[0], high[1])}:
        return None
    if td is None:
        return True
    normal = "xyz".index(geometry.normal_axis)
    transverse = [axis for axis in range(3) if axis != normal]
    center = [0.0, 0.0, 0.0]
    size = [0.0, 0.0, 0.0]
    center[normal] = geometry.coordinate * s
    for index, axis in enumerate(transverse):
        center[axis] = 0.5 * (low[index] + high[index]) * s
        size[axis] = (high[index] - low[index]) * s
    return td.Box(center=tuple(center), size=tuple(size))


def _pec_frame_record(structure):
    metadata = getattr(structure, "metadata", None) or {}
    return metadata.get("pec_frame") if "pec_frame" in tuple(getattr(structure, "tags", ()) or ()) else None


def _rigid_transform(geometry, s):
    """4x4 local-to-world transform of a rotated primitive, or None for the identity rotation."""
    rotation = np.asarray(geometry._rotation_matrix_np(), dtype=np.float64)
    if np.allclose(rotation, np.eye(3), rtol=0.0, atol=1e-12):
        return None
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = geometry.position.detach().cpu().to(dtype=torch.float64).numpy() * float(s)
    return transform


def _convert_geometry(geometry, td, s):
    """Convert maxwell geometry to a Tidy3D geometry (lengths x *s*)."""
    kind = geometry.kind

    if isinstance(geometry, BooleanGeometry):
        return _boolean_geometry(geometry, td, s)

    if kind == "box":
        transform = _rigid_transform(geometry, s)
        if transform is not None:
            # Rotated boxes keep their exact analytic faces in the local frame.
            return td.Transformed(geometry=td.Box(center=(0.0, 0.0, 0.0), size=_scale3(geometry.size, s)),
                                  transform=transform)
        return td.Box(center=_scale3(geometry.position, s), size=_scale3(geometry.size, s))

    if kind == "sphere":
        return td.Sphere(center=_scale3(geometry.position, s), radius=float(geometry.radius) * s)

    if kind == "cylinder":
        axis_idx = _axis_name_to_index(geometry.axis)
        transform = _rigid_transform(geometry, s)
        if transform is not None:
            local = td.Cylinder(center=(0.0, 0.0, 0.0), radius=float(geometry.radius) * s,
                                length=float(geometry.height) * s, axis=axis_idx)
            return td.Transformed(geometry=local, transform=transform)
        return td.Cylinder(
            center=_scale3(geometry.position, s),
            radius=float(geometry.radius) * s,
            length=float(geometry.height) * s,
            axis=axis_idx,
        )

    if kind == "cone":
        rotation = getattr(geometry, "rotation", None)
        if rotation is not None:
            quat = [float(v) for v in rotation.detach().cpu().tolist()]
            is_identity_rotation = abs(quat[0] - 1.0) < 1e-12 and all(
                abs(component) < 1e-12 for component in quat[1:]
            )
            if not is_identity_rotation:
                return _triangle_mesh(geometry, td, s)

        axis_idx = _axis_name_to_index(geometry.axis)
        height = float(geometry.height)
        center = [float(v) for v in geometry.position.detach().cpu().tolist()]
        center[axis_idx] += 0.5 * height
        return td.Cylinder(
            center=tuple(coordinate * s for coordinate in center),
            radius=float(geometry.radius) * s,
            length=height * s,
            axis=axis_idx,
            sidewall_angle=-math.atan2(float(geometry.radius), height),
            reference_plane="top",
        )

    if kind == "ellipsoid":
        rx, ry, rz = float(geometry.radii[0]), float(geometry.radii[1]), float(geometry.radii[2])
        if abs(rx - ry) < 1e-12 and abs(ry - rz) < 1e-12:
            # An isotropic ellipsoid is exactly a sphere; use Tidy3D's analytic
            # Sphere rather than a tessellated approximation.
            return td.Sphere(center=_scale3(geometry.position, s), radius=rx * s)
        return _triangle_mesh(geometry, td, s)

    if kind == "poly_slab":
        return _poly_slab(geometry, td, s)

    if kind == "planar_pec" and not geometry.holes and _planar_pec_rectangle(geometry) is not None:
        return _planar_pec_rectangle(geometry, td, s)

    if hasattr(geometry, "to_mesh"):
        # Any remaining primitive with no analytic Tidy3D equivalent (torus, pyramid,
        # prism, hollow box) tessellates to a TriangleMesh from its own surface mesh.
        return _triangle_mesh(geometry, td, s)

    raise NotImplementedError(
        f"Geometry type '{kind}' exposes no surface mesh, so it has no Tidy3D geometry mapping. "
        f"Analytic primitives (box, sphere, cylinder, cone) and any mesh-able primitive are supported."
    )


# ---------------------------------------------------------------------------
# Structure conversion
# ---------------------------------------------------------------------------

def _convert_structure(structure, td, s, frequencies=None):
    """Convert maxwell Structure to a Tidy3D Structure."""
    td_geometry = _convert_geometry(structure.geometry, td, s)
    td_material = _convert_material(structure.material, td, s, frequencies=frequencies)
    return td.Structure(geometry=td_geometry, medium=td_material)


def _convert_material_region(region, td, s):
    """Lower a uniform density region to one homogeneous Tidy3D structure."""
    import torch

    if getattr(region.geometry, "kind", None) != "box":
        raise NotImplementedError("Tidy3D export supports MaterialRegion with Box geometry only.")
    if region.filter_radius is not None:
        raise NotImplementedError(
            "Tidy3D export cannot preserve a filtered MaterialRegion density field."
        )

    density = region.density.detach().cpu().to(torch.float64)
    density_min = float(density.min())
    density_max = float(density.max())
    if not np.isclose(density_min, density_max, rtol=0.0, atol=1.0e-12):
        raise NotImplementedError(
            "Tidy3D export currently supports only spatially uniform MaterialRegion density."
        )

    lower, upper = region.bounds
    normalized = float(np.clip((density_min - lower) / (upper - lower), 0.0, 1.0))
    if region.projection_beta is not None:
        beta = float(region.projection_beta)
        normalized = float(
            (np.tanh(0.5 * beta) + np.tanh(beta * (normalized - 0.5)))
            / (2.0 * np.tanh(0.5 * beta))
        )
    eps_lo, eps_hi = region.eps_bounds
    mu_lo, mu_hi = region.mu_bounds
    eps_r = eps_lo + normalized * (eps_hi - eps_lo)
    mu_r = mu_lo + normalized * (mu_hi - mu_lo)
    if not np.isclose(mu_r, 1.0):
        raise NotImplementedError(
            "Tidy3D export does not support MaterialRegion with relative permeability other than 1."
        )

    return td.Structure(
        geometry=_convert_geometry(region.geometry, td, s),
        medium=td.Medium(permittivity=float(eps_r)),
    )


# ---------------------------------------------------------------------------
# Source conversion
# ---------------------------------------------------------------------------

def _tfsf_source(source, scene, td, s):
    """Convert a total-field/scattered-field-injected plane wave to a Tidy3D ``TFSF``.

    Tidy3D models a TFSF injection as a dedicated ``td.TFSF`` volume source whose box is the
    total-field region, ``injection_axis`` is the propagation axis, and ``direction`` its sign;
    the incident wave enters the low face of that axis (``+``) or the high face (``-``). maxwell
    carries the same information on the ``TFSF`` injection object: a ``box`` mode gives the full
    (x, y, z) total-field bounds, while a ``slab`` mode bounds only the propagation axis and is
    infinite in the two transverse directions (a 1D total-field layer). The plane-wave angles reuse
    the same ``_direction_to_angles`` mapping as the soft plane wave, so a TFSF and a soft plane
    wave of identical direction inject the same incident field.
    """
    injection = source.injection
    direction = source.direction
    injection_axis = resolve_injection_axis(direction, source.injection_axis)
    axis_idx = _axis_name_to_index(injection_axis)
    # Physical peak E gives normal power density |E|^2*cos(theta)/(2*eta0).
    # The reference unit TFSF amplitude gives 1 W/um^2 along the injection
    # axis, so equating powers requires multiplying by sqrt(|cos(theta)|),
    # rather than dividing. At normal incidence this factor is exactly one.
    td_source_time = _convert_source_time(
        source.source_time,
        td,
        amplitude_scale=math.sqrt(abs(direction[axis_idx])) / (s * _TIDY3D_TFSF_UNIT_FIELD),
    )
    inject_dir = "+" if direction[axis_idx] > 0 else "-"
    pol_angle, angle_theta, angle_phi = _direction_to_angles(
        direction, axis_idx, source.polarization
    )

    if injection.mode == "slab":
        lo, hi = injection.axis_bounds
        center = list((b_lo + b_hi) / 2.0 * s for b_lo, b_hi in scene.domain.bounds)
        size = [td.inf, td.inf, td.inf]
        center[axis_idx] = 0.5 * (lo + hi) * s
        size[axis_idx] = (hi - lo) * s
    else:
        bounds = injection.bounds
        center = [0.5 * (lo + hi) * s for lo, hi in bounds]
        size = [(hi - lo) * s for lo, hi in bounds]

    return td.TFSF(
        center=tuple(center),
        size=tuple(size),
        source_time=td_source_time,
        direction=inject_dir,
        injection_axis=axis_idx,
        pol_angle=pol_angle,
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        name=source.name or "tfsf",
    )


def _custom_source_dataset(dataset, td, freq, s, component_map):
    """Build a Tidy3D field/current dataset from a maxwell rectilinear dataset.

    Tidy3D custom field/current sources carry a ``FieldDataset`` of ``ScalarFieldDataArray``
    components on coordinates ``(x, y, z, f)`` that are **relative to the source center** and a
    single frequency in the source band. maxwell stores absolute-metre coordinates and a
    time-domain spatial profile, so the center is the midpoint of each coordinate span, the
    exported coordinates are ``(coord - center) * length_scale`` (um, relative), the frequency
    axis is the single injection frequency, and ``component_map`` renames maxwell components
    (identity for field sources; ``J -> E`` / ``M -> H`` for current sources, matching Tidy3D's
    convention that the current dataset reuses ``E``/``H`` slots for ``J``/``M``). Returns the
    Tidy3D dataset together with the source center and size in Tidy3D units.
    """
    import numpy as _np

    coords_m = dataset.coords
    center_m = tuple(0.5 * (float(axis[0]) + float(axis[-1])) for axis in coords_m)
    size_um = tuple((float(axis[-1]) - float(axis[0])) * s for axis in coords_m)
    rel_coords = tuple(
        _np.asarray([(float(v) - center_m[i]) * s for v in coords_m[i]], dtype=_np.float64)
        for i in range(3)
    )
    freqs = _np.asarray([float(freq)], dtype=_np.float64)
    td_components = {}
    for source_name, values in dataset.components.items():
        td_name = component_map[source_name]
        data = _np.asarray(values, dtype=_np.complex128)[..., _np.newaxis]
        td_components[td_name] = td.ScalarFieldDataArray(
            data, coords=dict(x=rel_coords[0], y=rel_coords[1], z=rel_coords[2], f=freqs)
        )
    return td.FieldDataset(**td_components), tuple(c * s for c in center_m), size_um


def _custom_source_frequency(source, frequencies):
    """Single injection frequency for a custom current/field source export.

    Tidy3D requires the custom dataset to carry exactly one frequency inside the source band. It
    is taken from the attached ``source_time`` (its characteristic frequency), falling back to the
    first export frequency when the source has no explicit waveform.
    """
    source_time = source.source_time
    if source_time is not None:
        return float(source_time.frequency)
    if frequencies:
        return float(frequencies[0])
    raise ValueError(
        f"Custom source '{source.name}' export needs an injection frequency: attach a source_time "
        "or pass frequencies=... to Scene.to_tidy3d()."
    )


def _convert_source(source, scene, td, s, frequencies=None):
    """Convert a maxwell source to a Tidy3D source (lengths x *s*)."""
    from ..sources import (
        AstigmaticGaussianBeam,
        CustomCurrentSource,
        CustomFieldSource,
        GaussianBeam,
        ModeSource,
        PlaneWave,
        PointDipole,
        TFSF,
        UniformCurrentSource,
    )
    domain_bounds = scene.domain.bounds

    if isinstance(source, PointDipole):
        component = _polarization_to_component(source.polarization)
        # Maxwell's SI point-source amplitude is an electric current moment in A*m;
        # Tidy3D's micrometre coordinate system uses A*um.
        td_source_time = _convert_source_time(source.source_time, td, amplitude_scale=s)
        return td.PointDipole(
            center=_scale3(source.position, s),
            source_time=td_source_time,
            polarization=component,
            name=source.name or "point_dipole",
        )

    if isinstance(source, PlaneWave):
        if isinstance(source.injection, TFSF):
            return _tfsf_source(source, scene, td, s)
        td_source_time = _convert_source_time(source.source_time, td)
        direction = source.direction

        injection_axis = resolve_injection_axis(direction, source.injection_axis)
        axis_idx = _axis_name_to_index(injection_axis)
        inject_dir = "+" if direction[axis_idx] > 0 else "-"

        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [td.inf, td.inf, td.inf]
        center[axis_idx] = soft_plane_wave_coordinate(scene, injection_axis, float(direction[axis_idx])) * s
        size[axis_idx] = 0.0

        pol_angle, angle_theta, angle_phi = _direction_to_angles(
            direction, axis_idx, source.polarization
        )

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

    if isinstance(source, (GaussianBeam, AstigmaticGaussianBeam)):
        if isinstance(source.injection, TFSF):
            return _tfsf_source(source, scene, td, s)
        td_source_time = _convert_source_time(source.source_time, td)
        direction = source.direction

        abs_dir = [abs(d) for d in direction]
        dominant_axis = int(np.argmax(abs_dir))
        inject_dir = "+" if direction[dominant_axis] > 0 else "-"

        source_plane = soft_plane_wave_coordinate(
            scene, "xyz"[dominant_axis], float(direction[dominant_axis])
        )
        source_axis_parameter = (
            source_plane - float(source.focus[dominant_axis])
        ) / float(direction[dominant_axis])
        center_m = [
            float(source.focus[index]) + source_axis_parameter * float(direction[index])
            for index in range(3)
        ]
        center = [value * s for value in center_m]
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        size[dominant_axis] = 0.0

        pol_angle, angle_theta, angle_phi = _direction_to_angles(
            direction, dominant_axis, source.polarization
        )

        if isinstance(source, AstigmaticGaussianBeam):
            return td.AstigmaticGaussianBeam(
                center=tuple(center),
                size=tuple(size),
                source_time=td_source_time,
                direction=inject_dir,
                pol_angle=pol_angle,
                angle_theta=angle_theta,
                angle_phi=angle_phi,
                waist_sizes=(source.beam_waist[0] * s, source.beam_waist[1] * s),
                waist_distances=(
                    (source_axis_parameter - source.focus_u) * s,
                    (source_axis_parameter - source.focus_v) * s,
                ),
                name=source.name or "astigmatic_gaussian_beam",
            )

        return td.GaussianBeam(
            center=tuple(center),
            size=tuple(size),
            source_time=td_source_time,
            direction=inject_dir,
            pol_angle=pol_angle,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            waist_radius=source.beam_waist * s,
            waist_distance=source_axis_parameter * s,
            name=source.name or "gaussian_beam",
        )

    if isinstance(source, ModeSource):
        td_source_time = _convert_source_time(source.source_time, td)
        axis_idx = _axis_name_to_index(source.normal_axis)
        center = list(_scale3(source.position, s))
        size = list(_scale3(source.size, s))
        size[axis_idx] = 0.0
        return td.ModeSource(
            center=tuple(center),
            size=tuple(size),
            source_time=td_source_time,
            direction=source.direction,
            mode_spec=td.ModeSpec(
                num_modes=_mode_candidate_count(source.mode_index),
                sort_spec=_mode_sort_spec(td, source.normal_axis, source.polarization),
            ),
            mode_index=int(source.mode_index),
            name=source.name or "mode_source",
        )

    if isinstance(source, UniformCurrentSource):
        # Current density is A/m^2 in Maxwell and A/um^2 in Tidy3D.
        td_source_time = _convert_source_time(source.source_time, td, amplitude_scale=1.0 / s**2)
        return td.UniformCurrentSource(
            center=_scale3(source.center, s),
            size=_scale3(source.size, s),
            source_time=td_source_time,
            polarization=_polarization_to_component(source.polarization),
            name=source.name or "uniform_current",
        )

    if isinstance(source, CustomFieldSource):
        # Dataset E/H fields are SI V/m and A/m; Tidy3D stores them per micrometre.
        td_source_time = _convert_source_time(source.source_time, td, amplitude_scale=1.0 / s)
        freq = _custom_source_frequency(source, frequencies)
        identity = {name: name for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        field_dataset, center, size = _custom_source_dataset(
            source.field_dataset, td, freq, s, identity
        )
        return td.CustomFieldSource(
            center=center,
            size=size,
            source_time=td_source_time,
            field_dataset=field_dataset,
            name=source.name or "custom_field",
        )

    if isinstance(source, CustomCurrentSource):
        # Electric and magnetic volume-current densities both acquire two inverse-length factors.
        td_source_time = _convert_source_time(source.source_time, td, amplitude_scale=1.0 / s**2)
        freq = _custom_source_frequency(source, frequencies)
        current_map = {"Jx": "Ex", "Jy": "Ey", "Jz": "Ez", "Mx": "Hx", "My": "Hy", "Mz": "Hz"}
        current_dataset, center, size = _custom_source_dataset(
            source.current_dataset, td, freq, s, current_map
        )
        return td.CustomCurrentSource(
            center=center,
            size=size,
            source_time=td_source_time,
            current_dataset=current_dataset,
            name=source.name or "custom_current",
        )

    raise NotImplementedError(
        f"Source type '{type(source).__name__}' has no Tidy3D source mapping."
    )


# ---------------------------------------------------------------------------
# RF port excitation mapping (WavePort / LumpedPort -> reference constructs)
# ---------------------------------------------------------------------------
#
# Ports declare an aperture (WavePort) or a two-terminal feed (LumpedPort) but no
# ``source_time``: the internal FDTD runtime supplies the drive through its own
# port-excitation machinery, and a scattering matrix is assembled by driving one
# port at a time. A single exported reference simulation therefore adopts one
# documented drive convention (see ``_convert_ports_for_reference``):
#
#   * The FIRST declared WavePort is the driven port. It maps to the reference
#     solver's modal launch (``ModeSource``) of its fundamental (TEM/lowest)
#     mode; every WavePort -- driven and receiving alike -- additionally maps to
#     a receiving ``ModeMonitor`` at its aperture so S-parameters are read from
#     the directional mode amplitudes (``amp_+`` / ``amp_-``).
#   * If a scene carries no WavePort, the FIRST declared LumpedPort is the driven
#     port. A lumped delta-gap feed maps to its equivalent current injection: a
#     ``UniformCurrentSource`` (electric current filament) spanning the feed gap,
#     oriented along the voltage-path axis from the negative to the positive
#     terminal. Amplitude is a unit reference drive; input impedance and radiation
#     pattern are amplitude-independent ratios so the normalization cancels.
#
# A full N-port scattering reference requires one export per driven port; the
# single-export convention drives port index 0. ``ModePort`` is expanded upstream
# by ``Scene.resolved_sources`` and is intentionally NOT handled here.


def _port_source_time(td, frequencies):
    """Broadband Gaussian drive covering the requested frequency band."""
    freqs = [float(freq) for freq in frequencies]
    if not freqs:
        raise ValueError("Port export requires at least one frequency for the drive.")
    fmin, fmax = min(freqs), max(freqs)
    freq0 = 0.5 * (fmin + fmax)
    fwidth = max(0.5 * (fmax - fmin), 0.2 * freq0)
    return td.GaussianPulse(freq0=freq0, fwidth=fwidth)


def _wave_port_center_size(port, s):
    axis_idx = _axis_name_to_index(port.normal_axis)
    center = list(_scale3(port.position, s))
    size = list(_scale3(port.size, s))
    size[axis_idx] = 0.0
    return tuple(center), tuple(size)


def _wave_port_mode_spec(td):
    """Fundamental-mode window for a TEM / lowest-order aperture launch."""
    return td.ModeSpec(num_modes=1)


def _wave_port_mode_source(port, td, s, source_time):
    center, size = _wave_port_center_size(port, s)
    return td.ModeSource(
        center=center,
        size=size,
        source_time=source_time,
        direction=port.direction,
        mode_spec=_wave_port_mode_spec(td),
        mode_index=0,
        name=f"{port.name}::drive",
    )


def _wave_port_mode_monitor(port, td, s, frequencies):
    center, size = _wave_port_center_size(port, s)
    return td.ModeMonitor(
        center=center,
        size=size,
        freqs=list(frequencies),
        mode_spec=_wave_port_mode_spec(td),
        name=port.name,
    )


def _resolve_lumped_terminals(port, scene):
    """Return (negative_xyz, positive_xyz) in metres for a LumpedPort feed."""
    binding = getattr(port, "wire_binding", None)
    if binding is not None:
        wires = {wire.name: wire for wire in getattr(scene, "thin_wires", ())}

        def _node(ref):
            wire = wires.get(ref.wire)
            if wire is None:
                raise ValueError(
                    f"LumpedPort {port.name!r} wire_binding references unknown wire "
                    f"{ref.wire!r}."
                )
            points = wire.points
            if not (0 <= ref.point < len(points)):
                raise ValueError(
                    f"LumpedPort {port.name!r} wire node index {ref.point} is out of "
                    f"range for wire {ref.wire!r}."
                )
            return tuple(float(value) for value in points[ref.point])

        return _node(binding.negative), _node(binding.positive)

    # A coordinate-bound LumpedPort always carries explicit terminals (enforced
    # by its constructor), so the negative/positive coordinates are present here.
    negative = tuple(float(value) for value in port.negative)
    positive = tuple(float(value) for value in port.positive)
    return negative, positive


def _lumped_port_current_source(port, scene, td, s, source_time):
    negative, positive = _resolve_lumped_terminals(port, scene)
    deltas = [positive[axis] - negative[axis] for axis in range(3)]
    axis_idx = int(np.argmax([abs(value) for value in deltas]))
    gap = deltas[axis_idx]
    if abs(gap) < 1e-15:
        raise ValueError(
            f"LumpedPort {port.name!r} terminals are coincident; cannot orient the "
            "equivalent current injection."
        )
    center = [0.5 * (negative[axis] + positive[axis]) * s for axis in range(3)]
    # Filamentary electric-current source spanning the feed gap along the voltage
    # axis; zero transverse extent keeps it a delta-gap-class injection.
    size = [0.0, 0.0, 0.0]
    size[axis_idx] = abs(gap) * s
    component = "xyz"[axis_idx]
    return td.UniformCurrentSource(
        center=tuple(center),
        size=tuple(size),
        source_time=source_time,
        polarization=f"E{component}",
        # Interpret the unit amplitude as a total feed current (A) across the
        # zero-area filament rather than a current density (A/m**2), which is
        # ill-defined for a delta-gap injection.
        current_amplitude_definition="total",
        name=f"{port.name}::drive",
    )


def _coaxial_ports(scene):
    from ..ports import CoaxialLumpedPort

    return [port for port in getattr(scene, "ports", ()) or () if isinstance(port, CoaxialLumpedPort)]


def coaxial_lumped_port_to_tidy3d(port, *, length_scale: float = _M_TO_UM):
    """Map a :class:`CoaxialLumpedPort` to the reference terminal-modeler coaxial port."""
    td = _ensure_tidy3d()
    from tidy3d.plugins.smatrix import CoaxialLumpedPort as ReferenceCoaxialLumpedPort

    impedance = port.reference_impedance
    if torch.is_tensor(impedance):
        impedance = complex(impedance.detach().cpu().item())
    return ReferenceCoaxialLumpedPort(
        center=_scale3(port.center, length_scale),
        outer_diameter=port.outer_diameter * length_scale,
        inner_diameter=port.inner_diameter * length_scale,
        normal_axis=_axis_name_to_index(port.normal_axis),
        direction=port.direction,
        name=port.name,
        impedance=impedance,
    )


def _plane_vertices(points, axis_index, length_scale):
    tangential = [index for index in range(3) if index != axis_index]
    return [[float(point[index]) * length_scale for index in tangential] for point in points]


def voltage_integral_to_tidy3d(integral, *, axis: str, length_scale: float = _M_TO_UM):
    """Map a :class:`CustomVoltageIntegral` to the reference 2D voltage spec.

    The reference evaluates ``V = phi(last) - phi(first)``; witwin integrates
    ``E . dl`` from the first point, so the vertex order is reversed.
    Composite voltage integrals have no reference equivalent and are rejected.
    """
    td = _ensure_tidy3d()
    from ..path_integrals import CustomVoltageIntegral

    if not isinstance(integral, CustomVoltageIntegral):
        raise ValueError("Only CustomVoltageIntegral has a reference-adapter voltage equivalent.")
    axis_index = _axis_name_to_index(axis)
    points = tuple(reversed(integral.path.points))
    return td.Custom2DVoltageIntegralSpec(
        axis=axis_index,
        position=float(points[0][axis_index]) * length_scale,
        vertices=_plane_vertices(points, axis_index, length_scale),
    )


def _loop_spec(loop, axis_index, orientation, length_scale, td):
    import numpy as np

    vertices = np.asarray(_plane_vertices(loop.vertices(), axis_index, 1.0), dtype=np.float64)
    area = 0.5 * float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1]))
    if (area > 0) != (orientation > 0):
        vertices = vertices[::-1]
    vertices = np.vstack((vertices, vertices[:1])) * length_scale
    return td.Custom2DCurrentIntegralSpec(
        axis=axis_index,
        position=float(loop.vertices()[0][axis_index]) * length_scale,
        vertices=vertices.tolist(),
    )


def current_integral_to_tidy3d(integral, *, axis: str, direction: str, length_scale: float = _M_TO_UM):
    """Map a custom or +/-1 composite current integral to the reference specs.

    Loops are emitted right-handed about the propagation direction (witwin's
    convention). A composite maps to a summed composite spec when every
    coefficient is +1 or -1 (a -1 term reverses its loop).
    """
    td = _ensure_tidy3d()
    from ..path_integrals import CompositeCurrentIntegral, CustomCurrentIntegral

    axis_index = _axis_name_to_index(axis)
    handed = -1 if axis_index == 1 else 1
    orientation = handed * (1 if direction == "+" else -1)
    if isinstance(integral, CustomCurrentIntegral):
        return _loop_spec(integral.loop, axis_index, orientation, length_scale, td)
    if not isinstance(integral, CompositeCurrentIntegral):
        raise ValueError("current integral has no reference-adapter equivalent.")
    specs = []
    for coefficient, term in integral.terms:
        if not isinstance(term, CustomCurrentIntegral) or coefficient not in (1.0, -1.0):
            raise ValueError("Only flat composites with +/-1 coefficients map to the reference composite current.")
        specs.append(_loop_spec(term.loop, axis_index, orientation * int(coefficient), length_scale, td))
    return td.CompositeCurrentIntegralSpec(path_specs=tuple(specs), sum_spec="sum")


def terminal_wave_port_to_tidy3d(port, *, length_scale: float = _M_TO_UM):
    """Map a :class:`TerminalWavePort` to the reference terminal-driven wave port.

    Each terminal becomes one custom impedance spec with its voltage path and
    optional current loop. ``reference_impedance='characteristic'`` maps to
    ``'Z0'`` and a uniform diagonal to a scalar; per-terminal or coupled
    references, modal-basis ports and composite voltages have no reference
    equivalent and are rejected rather than approximated.
    """
    td = _ensure_tidy3d()
    from tidy3d.plugins.smatrix import TerminalWavePort as ReferenceTerminalWavePort
    from ..path_integrals import AxisSegment, CustomVoltageIntegral

    if port.basis != "terminal":
        raise ValueError("A modal-basis TerminalWavePort has no reference-adapter equivalent.")
    reference = port.reference_impedance
    if reference == "characteristic":
        impedance = "Z0"
    else:
        values = [[complex(item) for item in row] for row in reference]
        count = len(values)
        diagonal = [values[index][index] for index in range(count)]
        if any(values[row][column] != 0 for row in range(count) for column in range(count) if row != column):
            raise ValueError("A coupled reference-impedance matrix has no reference-adapter equivalent.")
        if any(value != diagonal[0] for value in diagonal):
            raise ValueError("Per-terminal reference impedances need reference terminal labels; use one value.")
        impedance = diagonal[0].real if diagonal[0].imag == 0 else diagonal[0]
    voltages = port.voltage_integrals
    if voltages is None:
        record = port._default_voltage_path
        if record is None:
            raise ValueError("TerminalWavePort default voltage path is unresolved; add it to a Scene first.")
        start = list(port.position)
        end = list(port.position)
        for point, key in ((start, "start"), (end, "end")):
            point["xyz".index(record["path_axis"])] = record[key]
            point["xyz".index(record["offset_axis"])] = record["offset"]
        voltages = (CustomVoltageIntegral(AxisSegment(tuple(start), tuple(end))),)
    specs = []
    for index, voltage in enumerate(voltages):
        current = None if port.current_integrals is None else current_integral_to_tidy3d(
            port.current_integrals[index], axis=port.normal_axis, direction=port.direction, length_scale=length_scale)
        specs.append(td.CustomImpedanceSpec(
            voltage_spec=voltage_integral_to_tidy3d(voltage, axis=port.normal_axis, length_scale=length_scale),
            current_spec=current,
        ))
    return ReferenceTerminalWavePort(
        center=_scale3(port.position, length_scale),
        size=_scale3(port.size, length_scale),
        name=port.name,
        direction=port.direction,
        reference_impedance=impedance,
        terminal_specs=tuple(specs),
    )


def _coaxial_port_source(port, td, s, source_time):
    return coaxial_lumped_port_to_tidy3d(port, length_scale=s).to_source(source_time)


def _coaxial_port_monitors(port, td, s, frequencies):
    reference = coaxial_lumped_port_to_tidy3d(port, length_scale=s)
    return [reference.to_voltage_monitor(list(frequencies)), reference.to_current_monitor(list(frequencies))]


def _convert_coaxial_lumped_elements(scene, td, s):
    """Reference annular resistors: every coaxial port load plus standalone coaxial resistors."""
    from ..lumped import CoaxialLumpedResistor

    elements = [coaxial_lumped_port_to_tidy3d(port, length_scale=s).to_load() for port in _coaxial_ports(scene)]
    for element in getattr(scene, "lumped_elements", ()) or ():
        if isinstance(element, CoaxialLumpedResistor):
            elements.append(td.CoaxialLumpedResistor(
                center=_scale3(element.center, s),
                outer_diameter=element.outer_diameter * s,
                inner_diameter=element.inner_diameter * s,
                normal_axis=_axis_name_to_index(element.normal_axis),
                resistance=element.resistance,
                name=element.name,
            ))
    return elements


def _pec_frame_lengths(scene):
    lengths = {}
    for structure in getattr(scene, "structures", ()) or ():
        record = _pec_frame_record(structure)
        if record is not None:
            lengths[record["port"]] = int(record["length"])
    return lengths


def _convert_ports_for_reference(scene, td, s, frequencies):
    """Map RF ports to (sources, monitors) for a runnable reference export."""
    from ..ports import LumpedPort, WavePort

    # WavePort and LumpedPort carry drive-mappable geometry directly. TerminalPort
    # needs structure resolution and stays out of the adapter-drive scope for now;
    # a TerminalPort-only scene keeps the existing sources=0 runnable-gate fail-close.
    ports = list(getattr(scene, "ports", ()) or ())
    wave_ports = [port for port in ports if isinstance(port, WavePort)]
    lumped_ports = [port for port in ports if isinstance(port, LumpedPort)]
    if not wave_ports and not lumped_ports and _coaxial_ports(scene):
        if frequencies is None:
            raise ValueError(
                "Exporting a port-driven Scene requires frequencies for the reference drive."
            )
        coaxial = _coaxial_ports(scene)
        return (
            [_coaxial_port_source(coaxial[0], td, s, _port_source_time(td, frequencies))],
            [monitor for port in coaxial for monitor in _coaxial_port_monitors(port, td, s, frequencies)],
        )
    if not wave_ports and not lumped_ports:
        return [], []
    if frequencies is None:
        raise ValueError(
            "Exporting a port-driven Scene requires frequencies for the reference drive."
        )

    source_time = _port_source_time(td, frequencies)
    sources = []
    monitors = []

    # Receive apertures for every wave port.
    for port in wave_ports:
        monitors.append(_wave_port_mode_monitor(port, td, s, frequencies))

    # Single documented drive convention: wave ports take precedence.
    if wave_ports:
        sources.append(_wave_port_mode_source(wave_ports[0], td, s, source_time))
        frame = _pec_frame_lengths(scene).get(wave_ports[0].name)
        if frame is not None:
            sources[-1] = sources[-1].updated_copy(frame=td.PECFrame(length=frame))
    else:
        sources.append(
            _lumped_port_current_source(lumped_ports[0], scene, td, s, source_time)
        )
    return sources, monitors


def _direction_to_angles(direction, dominant_axis, polarization=None):
    """Compute Tidy3D source angles from a propagation and polarization vector.

    Tidy3D defines its angles in a local frame whose normal is the injection
    axis. Its ``direction='-'`` option reverses the complete propagation vector,
    while ``pol_angle`` rotates the electric field from the local P basis toward
    the S basis. Expressing both Maxwell vectors in that frame preserves
    arbitrary transverse polarization for every injection axis and sign.
    """
    direction_vector = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction_vector))
    if norm < 1e-15:
        return 0.0, 0.0, 0.0
    direction_vector /= norm

    tangential_axes = [axis for axis in range(3) if axis != dominant_axis]
    direction_sign = 1.0 if direction_vector[dominant_axis] >= 0.0 else -1.0
    local_direction = np.array(
        (
            direction_vector[tangential_axes[0]] / direction_sign,
            direction_vector[tangential_axes[1]] / direction_sign,
            direction_vector[dominant_axis] / direction_sign,
        ),
        dtype=np.float64,
    )
    angle_theta = math.acos(float(np.clip(local_direction[2], -1.0, 1.0)))
    if math.sin(angle_theta) < 1e-12:
        angle_phi = 0.0
    else:
        angle_phi = math.atan2(float(local_direction[1]), float(local_direction[0]))

    if polarization is None:
        return 0.0, angle_theta, angle_phi

    cos_theta = math.cos(angle_theta)
    sin_theta = math.sin(angle_theta)
    cos_phi = math.cos(angle_phi)
    sin_phi = math.sin(angle_phi)
    p_basis_local = np.array(
        (cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta),
        dtype=np.float64,
    )
    s_basis_local = np.array((-sin_phi, cos_phi, 0.0), dtype=np.float64)

    def _to_global(local_vector):
        global_vector = np.empty(3, dtype=np.float64)
        global_vector[dominant_axis] = local_vector[2]
        global_vector[tangential_axes[0]] = local_vector[0]
        global_vector[tangential_axes[1]] = local_vector[1]
        return global_vector

    polarization_vector = np.asarray(polarization, dtype=np.float64)
    polarization_norm = float(np.linalg.norm(polarization_vector))
    if polarization_norm < 1e-15:
        return 0.0, angle_theta, angle_phi
    polarization_vector /= polarization_norm
    p_basis = _to_global(p_basis_local)
    s_basis = _to_global(s_basis_local)
    pol_angle = math.atan2(
        float(np.dot(polarization_vector, s_basis)),
        float(np.dot(polarization_vector, p_basis)),
    )
    return pol_angle, angle_theta, angle_phi


# ---------------------------------------------------------------------------
# Monitor conversion
# ---------------------------------------------------------------------------

def _plane_center_size(axis_idx, plane_position, domain_bounds, s):
    """(center, size) of a full-domain plane at *plane_position* along *axis_idx*."""
    center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
    size = [(hi - lo) * s for lo, hi in domain_bounds]
    center[axis_idx] = float(plane_position) * s
    size[axis_idx] = 0.0
    return center, size


def _convert_monitor(monitor, domain_bounds, frequencies, td, s, *, time_step=None, prepared_scene=None):
    """Convert a maxwell monitor to a Tidy3D monitor (lengths x *s*)."""
    from ..monitors import (
        DiffractionMonitor,
        FieldTimeMonitor,
        FinitePlaneMonitor,
        FluxTimeMonitor,
        ModeMonitor,
        PermittivityMonitor,
        PlaneMonitor,
        PointMonitor,
    )

    plane_position = None
    if prepared_scene is not None and (
        getattr(monitor, "compute_flux", False)
        or getattr(monitor, "closure_planes", None) is not None
        or isinstance(monitor, FluxTimeMonitor)
    ):
        from ..fdtd.observers import _flux_yee_plane_position

        requested = monitor.position if isinstance(monitor, (PlaneMonitor, FluxTimeMonitor)) else monitor.plane_position
        # Use the runtime resolver, including its tie-breaking and grid precision.
        plane_position = _flux_yee_plane_position(prepared_scene, monitor.axis, requested)

    # Time-domain monitors sample the running field, not a frequency spectrum, so they
    # are converted before the frequency requirement below.
    if isinstance(monitor, FieldTimeMonitor):
        time_scale = 1.0 if time_step is None else float(time_step)
        return td.FieldTimeMonitor(
            center=_scale3(monitor.position, s),
            size=_scale3(monitor.size, s),
            start=monitor.start * time_scale,
            stop=None if monitor.stop is None else monitor.stop * time_scale,
            interval=monitor.interval,
            fields=list(monitor.components),
            name=monitor.name,
        )

    if isinstance(monitor, FluxTimeMonitor):
        time_scale = 1.0 if time_step is None else float(time_step)
        axis_idx = _axis_name_to_index(monitor.axis)
        center, size = _plane_center_size(axis_idx, monitor.position if plane_position is None else plane_position, domain_bounds, s)
        return td.FluxTimeMonitor(
            center=tuple(center),
            size=tuple(size),
            start=monitor.start * time_scale,
            stop=None if monitor.stop is None else monitor.stop * time_scale,
            interval=monitor.interval,
            normal_dir=monitor.normal_direction,
            name=monitor.name,
        )

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
            fields=list(monitor.fields),
            name=monitor.name,
            colocate=True,
        )

    if isinstance(monitor, DiffractionMonitor):
        # Tidy3D decomposes the plane wave into diffraction orders over the whole periodic
        # cell, so the transverse extent must be td.inf; the maxwell monitor's finite period
        # is the Bloch cell carried by the simulation boundaries, not the monitor size.
        axis_idx = _axis_name_to_index(monitor.axis)
        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [td.inf, td.inf, td.inf]
        center[axis_idx] = float(monitor.plane_position if plane_position is None else plane_position) * s
        size[axis_idx] = 0.0
        return td.DiffractionMonitor(
            center=tuple(center),
            size=tuple(size),
            freqs=list(monitor_frequencies),
            normal_dir=monitor.normal_direction,
            name=monitor.name,
        )

    if isinstance(monitor, PermittivityMonitor):
        return td.PermittivityMonitor(
            center=_scale3(monitor.position, s),
            size=_scale3(monitor.size, s),
            freqs=list(monitor_frequencies),
            name=monitor.name,
        )

    if isinstance(monitor, ModeMonitor):
        # A modal monitor projects the field onto waveguide eigenmodes, which Tidy3D
        # models with a dedicated ModeMonitor + ModeSpec, not a raw FieldMonitor. The mode
        # solver must resolve at least mode_index + 1 modes to expose the requested order.
        axis_idx = _axis_name_to_index(monitor.normal_axis)
        center = list(_scale3(monitor.position, s))
        if plane_position is not None:
            center[axis_idx] = plane_position * s
        size = list(_scale3(monitor.size, s))
        size[axis_idx] = 0.0
        return td.ModeMonitor(
            center=tuple(center),
            size=tuple(size),
            freqs=list(monitor_frequencies),
            mode_spec=td.ModeSpec(
                num_modes=_mode_candidate_count(monitor.mode_index),
                sort_spec=_mode_sort_spec(td, monitor.normal_axis, monitor.polarization),
            ),
            name=monitor.name,
        )

    if isinstance(monitor, FinitePlaneMonitor):
        center = list(_scale3(monitor.position, s))
        if plane_position is not None:
            center[_axis_name_to_index(monitor.axis)] = plane_position * s
        center = tuple(center)
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
            fields=list(monitor.fields),
            name=monitor.name,
            colocate=True,
        )

    if isinstance(monitor, PlaneMonitor):
        axis_idx = _axis_name_to_index(monitor.axis)
        center = list((lo + hi) / 2.0 * s for lo, hi in domain_bounds)
        size = [(hi - lo) * s for lo, hi in domain_bounds]
        center[axis_idx] = (monitor.position if plane_position is None else plane_position) * s
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
            fields=list(monitor.fields),
            name=monitor.name,
            colocate=True,
        )

    raise NotImplementedError(
        f"Monitor type '{type(monitor).__name__}' has no Tidy3D monitor mapping."
    )


_TD_FACE_LABELS = {
    "x_neg": "x-", "x_pos": "x+", "y_neg": "y-", "y_pos": "y+", "z_neg": "z-", "z_pos": "z+",
}


def _projection_surface_kwargs(monitor, frequencies, s):
    monitor_frequencies = monitor.frequencies or frequencies
    if not monitor_frequencies:
        raise ValueError(
            f"Monitor '{monitor.name}' has no frequencies and none were passed to to_tidy3d()."
        )
    kwargs = dict(
        center=_scale3(monitor.position, s),
        size=_scale3(monitor.size, s),
        freqs=list(monitor_frequencies),
        name=monitor.name,
        custom_origin=_scale3(monitor.origin, s),
    )
    if monitor.surface_kind == "planar":
        kwargs["normal_dir"] = monitor.normal_direction
        if monitor.window_size is not None:
            kwargs["window_size"] = tuple(monitor.window_size)
    elif monitor.exclude_surfaces:
        kwargs["exclude_surfaces"] = tuple(_TD_FACE_LABELS[label] for label in monitor.exclude_surfaces)
    return kwargs


def _convert_in_simulation_monitor(monitor, frequencies, td, s):
    """Export a maxwell in-simulation monitor to its Tidy3D counterpart."""
    from ..monitors import (
        AngleProjectionMonitor,
        AstigmaticGaussianOverlapMonitor,
        CartesianProjectionMonitor,
        GaussianOverlapMonitor,
        KSpaceProjectionMonitor,
    )

    if isinstance(monitor, AngleProjectionMonitor):
        return td.FieldProjectionAngleMonitor(
            **_projection_surface_kwargs(monitor, frequencies, s),
            theta=list(monitor.theta),
            phi=list(monitor.phi),
            proj_distance=monitor.radius * s,
            far_field_approx=monitor.far_field_approx,
        )
    if isinstance(monitor, CartesianProjectionMonitor):
        return td.FieldProjectionCartesianMonitor(
            **_projection_surface_kwargs(monitor, frequencies, s),
            proj_axis=_axis_name_to_index(monitor.axis),
            proj_distance=monitor.distance * s,
            x=[value * s for value in monitor.x],
            y=[value * s for value in monitor.y],
            far_field_approx=monitor.far_field_approx,
        )
    if isinstance(monitor, KSpaceProjectionMonitor):
        planar_normal = monitor.surface_kind == "planar" and monitor.faces[0].axis == monitor.axis
        expected = monitor.normal_direction if planar_normal else "+"
        if monitor.hemisphere != expected:
            raise NotImplementedError(
                f"KSpaceProjectionMonitor '{monitor.name}' projects onto the {monitor.hemisphere!r} hemisphere; "
                "the Tidy3D k-space monitor projects onto the hemisphere of its normal direction only."
            )
        return td.FieldProjectionKSpaceMonitor(
            **_projection_surface_kwargs(monitor, frequencies, s),
            proj_axis=_axis_name_to_index(monitor.axis),
            proj_distance=monitor.radius * s,
            ux=list(monitor.ux),
            uy=list(monitor.uy),
        )
    if isinstance(monitor, (GaussianOverlapMonitor, AstigmaticGaussianOverlapMonitor)):
        monitor_frequencies = monitor.frequencies or frequencies
        if not monitor_frequencies:
            raise ValueError(
                f"Monitor '{monitor.name}' has no frequencies and none were passed to to_tidy3d()."
            )
        axis_index = _axis_name_to_index(monitor.axis)
        if monitor.normal_direction != "+":
            raise NotImplementedError(
                f"Gaussian-overlap monitor '{monitor.name}' declares a beam travelling toward -{monitor.axis}; "
                "this adapter does not map a -normal beam onto the Tidy3D overlap monitor's backward ('-') "
                "amplitude, so export a +normal beam."
            )
        pol_angle, angle_theta, angle_phi = _direction_to_angles(
            monitor.direction, axis_index, monitor.polarization
        )
        waist_distance = (
            float(monitor.plane_position) - float(monitor.focus[axis_index])
        ) / float(monitor.direction[axis_index])
        common = dict(
            center=_scale3(monitor.position, s),
            size=_scale3(monitor.size, s),
            freqs=list(monitor_frequencies),
            name=monitor.name,
            angle_theta=angle_theta,
            angle_phi=angle_phi,
            pol_angle=pol_angle,
        )
        if isinstance(monitor, AstigmaticGaussianOverlapMonitor):
            return td.AstigmaticGaussianOverlapMonitor(
                **common,
                waist_sizes=(monitor.beam_waist[0] * s, monitor.beam_waist[1] * s),
                waist_distances=(
                    (waist_distance - monitor.focus_u) * s,
                    (waist_distance - monitor.focus_v) * s,
                ),
            )
        return td.GaussianOverlapMonitor(
            **common,
            waist_radius=monitor.beam_waist * s,
            waist_distance=waist_distance * s,
        )
    raise TypeError(f"{type(monitor).__name__} is not an in-simulation projection or overlap monitor.")


def _convert_surface_field_time_monitor(monitor, td, s, *, time_step):
    """Export a surface field-time monitor as one Tidy3D field-time monitor per face.

    Tidy3D's own surface field-time monitor samples PEC/lossy-metal structure
    surfaces inside its box, which is a different observable; the declared
    faces are recorded equivalently by planar field-time monitors.
    """
    time_scale = 1.0 if time_step is None else float(time_step)
    return [
        td.FieldTimeMonitor(
            center=_scale3(face_position, s),
            size=_scale3(face_size, s),
            start=monitor.start * time_scale,
            stop=None if monitor.stop is None else monitor.stop * time_scale,
            interval=monitor.interval,
            fields=list(monitor.components),
            name=f"{monitor.name}_{label}",
        )
        for label, _, _, face_position, face_size in monitor.face_specs
    ]


# ---------------------------------------------------------------------------
# Boundary conversion (no length scaling)
# ---------------------------------------------------------------------------

def _convert_mur_abc(mur_abc, td):
    """First-order MurABC -> ABCBoundary with an explicit propagation permittivity."""
    if mur_abc.order != 1:
        raise ValueError(
            "Second-order Mur boundaries have no Tidy3D equivalent; Tidy3D ABCBoundary is first order."
        )
    # maxwell's default one-way speed is vacuum; Tidy3D's None would infer the
    # permittivity from the boundary medium instead, so always pass it.
    permittivity = 1.0 if mur_abc.permittivity is None else float(mur_abc.permittivity)
    return td.ABCBoundary(permittivity=permittivity)


def _convert_mode_abc(mode_abc, td, *, normal_axis, center, size):
    """ModeABC -> ModeABCBoundary on the given (already scaled) cross-section."""
    from ..sources import _resolve_mode_source_polarization_axis, polarization_vector

    polarization_axis = _resolve_mode_source_polarization_axis(normal_axis, size, mode_abc.polarization)
    if mode_abc.is_broadband:
        # The reference backend fits its own pole-residue index model over the band;
        # maxwell's Higdon order has no one-to-one counterpart and is not exported.
        freq_spec = td.BroadbandModeABCSpec(
            frequency_range=tuple(float(value) for value in mode_abc.frequency_range),
        )
    else:
        freq_spec = None if mode_abc.frequency is None else float(mode_abc.frequency)
    return td.ModeABCBoundary(
        mode_spec=td.ModeSpec(
            num_modes=_mode_candidate_count(mode_abc.mode_index),
            sort_spec=_mode_sort_spec(td, normal_axis, polarization_vector(f"E{polarization_axis}")),
        ),
        mode_index=int(mode_abc.mode_index),
        freq_spec=freq_spec,
        plane=td.Box(center=tuple(center), size=tuple(size)),
    )


def _convert_mode_abc_face(boundary, td, domain_bounds, axis, side, length_scale):
    if domain_bounds is None:
        raise ValueError("ModeABC boundary export requires the physical domain bounds.")
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    center = []
    size = []
    for index, (low, high) in enumerate(domain_bounds):
        if index == axis_index:
            center.append((float(low) if side == "low" else float(high)) * length_scale)
            size.append(0.0)
        else:
            center.append(0.5 * (float(low) + float(high)) * length_scale)
            size.append((float(high) - float(low)) * length_scale)
    return _convert_mode_abc(boundary.mode_abc, td, normal_axis=axis, center=center, size=size)


def _convert_internal_absorber(absorber, td, s):
    from ..absorbers import ModeABC

    center = tuple(float(value) * s for value in absorber.position)
    size = tuple(float(value) * s for value in absorber.size)
    if isinstance(absorber.boundary, ModeABC):
        spec = _convert_mode_abc(absorber.boundary, td, normal_axis=absorber.normal_axis, center=center, size=size)
    else:
        spec = _convert_mur_abc(absorber.boundary, td)
    return td.InternalAbsorber(
        center=center,
        size=size,
        direction=absorber.direction,
        grid_shift=int(absorber.grid_shift),
        boundary_spec=spec,
    )


def _convert_boundary(boundary, td, domain_bounds=None, length_scale=_M_TO_UM):
    """Convert maxwell BoundarySpec to Tidy3D BoundarySpec."""
    kind = boundary.kind
    if boundary.bloch_wavevector == "auto":
        raise ValueError(
            "Automatic Bloch wavevectors require Simulation.prepare() and cannot be exported to Tidy3D unresolved."
        )

    def normalized_bloch_component(axis: str) -> float:
        if domain_bounds is None:
            raise ValueError("Bloch boundary export requires the physical domain bounds.")
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        lower, upper = domain_bounds[axis_index]
        period = float(upper) - float(lower)
        return float(boundary.bloch_wavevector[axis_index]) * period / (2.0 * np.pi)

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
        bloch_x = td.BlochBoundary(bloch_vec=normalized_bloch_component("x"))
        bloch_y = td.BlochBoundary(bloch_vec=normalized_bloch_component("y"))
        bloch_z = td.BlochBoundary(bloch_vec=normalized_bloch_component("z"))
        return td.BoundarySpec(
            x=td.Boundary(plus=bloch_x, minus=bloch_x),
            y=td.Boundary(plus=bloch_y, minus=bloch_y),
            z=td.Boundary(plus=bloch_z, minus=bloch_z),
        )

    if kind == "none":
        pec = td.PECBoundary()
        return td.BoundarySpec.all_sides(boundary=pec)

    if kind in ("mixed", "mur", "mode_abc"):
        def convert_face(face_kind, axis, side):
            if face_kind == "pml":
                return td.PML(num_layers=boundary.num_layers)
            if face_kind == "periodic":
                return td.Periodic()
            if face_kind == "pec":
                return td.PECBoundary()
            if face_kind == "pmc":
                return td.PMCBoundary()
            if face_kind == "bloch":
                return td.BlochBoundary(bloch_vec=normalized_bloch_component(axis))
            if face_kind == "none":
                return td.PECBoundary()
            if face_kind == "mur":
                return _convert_mur_abc(boundary.mur_abc, td)
            if face_kind == "mode_abc":
                return _convert_mode_abc_face(boundary, td, domain_bounds, axis, side, length_scale)
            raise ValueError(f"Unsupported boundary kind: {face_kind}")

        return td.BoundarySpec(
            x=td.Boundary(
                minus=convert_face(boundary.face_kind("x", "low"), "x", "low"),
                plus=convert_face(boundary.face_kind("x", "high"), "x", "high"),
            ),
            y=td.Boundary(
                minus=convert_face(boundary.face_kind("y", "low"), "y", "low"),
                plus=convert_face(boundary.face_kind("y", "high"), "y", "high"),
            ),
            z=td.Boundary(
                minus=convert_face(boundary.face_kind("z", "low"), "z", "low"),
                plus=convert_face(boundary.face_kind("z", "high"), "z", "high"),
            ),
        )

    raise ValueError(f"Unsupported boundary kind: {kind}")


# ---------------------------------------------------------------------------
# Grid conversion
# ---------------------------------------------------------------------------

def _axis_grid_1d(coords, spacing, td, s):
    """One Tidy3D 1D grid for a single axis (lengths x *s*).

    A nonuniform axis carries explicit node coordinates and maps to
    ``td.CustomGridBoundaries`` (grid *boundary* coordinates in Tidy3D units); a
    uniform axis maps to ``td.UniformGrid`` at the scaled step. maxwell's custom
    node arrays already span the domain exactly (validated at prepare time), so the
    scaled coordinates coincide with the exported simulation bounds and Tidy3D
    discretizes the interior on the identical Yee boundaries.
    """
    if coords is not None:
        return td.CustomGridBoundaries(coords=np.asarray(coords, dtype=np.float64) * s)
    return td.UniformGrid(dl=float(spacing) * s)


def _convert_grid(scene, td, s):
    """Convert a maxwell ``GridSpec`` to a Tidy3D ``GridSpec`` (lengths x *s*).

    Uniform and per-axis-anisotropic grids export as ``td.UniformGrid`` steps. A
    nonuniform grid exports as per-axis ``td.CustomGridBoundaries`` carrying the exact
    Yee node coordinates, so the Tidy3D simulation discretizes on the *same* grid
    maxwell uses and the two solvers compare cell-for-cell instead of across two
    independent meshers:

    * ``GridSpec.custom`` forwards its node arrays directly.
    * ``GridSpec.auto`` is first resolved through maxwell's own mesher
      (``resolve_auto_grid``), which honours the index-aware step targets,
      ``override_structures`` and ``layer_refinement`` (including corner, gap and
      layer-bounds refinement, the counterpart of Tidy3D ``LayerRefinementSpec`` /
      ``GridRefinement`` / ``CornerFinderSpec``) that a lossy parameter map to
      Tidy3D's ``AutoGrid`` would drop or re-mesh differently. Exporting the resolved nodes reproduces the
      maxwell mesh exactly rather than letting Tidy3D's AutoGrid pick a different one.
    """
    grid = scene.grid

    if grid.is_auto:
        from ..fdtd.meshing import resolve_auto_grid

        x_nodes, y_nodes, z_nodes = resolve_auto_grid(scene)
        return td.GridSpec(
            grid_x=_axis_grid_1d(x_nodes, None, td, s),
            grid_y=_axis_grid_1d(y_nodes, None, td, s),
            grid_z=_axis_grid_1d(z_nodes, None, td, s),
        )

    if grid.is_custom:
        return td.GridSpec(
            grid_x=_axis_grid_1d(grid.x_coords, grid.dx, td, s),
            grid_y=_axis_grid_1d(grid.y_coords, grid.dy, td, s),
            grid_z=_axis_grid_1d(grid.z_coords, grid.dz, td, s),
        )

    if grid.dx == grid.dy == grid.dz:
        return td.GridSpec.uniform(dl=grid.dx * s)
    return td.GridSpec(
        grid_x=td.UniformGrid(dl=grid.dx * s),
        grid_y=td.UniformGrid(dl=grid.dy * s),
        grid_z=td.UniformGrid(dl=grid.dz * s),
    )


def _scene_time_step(scene, courant: float) -> float:
    """Return the physical FDTD step represented by Tidy3D's Courant factor."""
    from ..scene import prepare_scene

    prepared = prepare_scene(scene.clone(device="cpu"))
    spacings = (
        float(prepared.dx_primal64.min()),
        float(prepared.dy_primal64.min()),
        float(prepared.dz_primal64.min()),
    )
    dt_cfl = 1.0 / (_C0 * math.sqrt(sum(1.0 / spacing**2 for spacing in spacings)))
    return float(courant) * dt_cfl


def _validate_tidy3d_symmetry_equivalence(scene) -> None:
    """Reject face symmetry that is not invariant under moving the plane to center."""
    from ..sources import PlaneWave

    for axis_index, entry in enumerate(scene.symmetry):
        if entry is None:
            continue
        lower, upper = scene.domain.bounds[axis_index]
        span = float(upper) - float(lower)
        for structure in scene.structures:
            geometry = structure.geometry
            if getattr(geometry, "kind", None) != "box":
                raise NotImplementedError(
                    "Tidy3D symmetry is center-based; Maxwell face symmetry can only be "
                    "exported when the scene is translation-invariant along the symmetry axis."
                )
            if float(geometry.size[axis_index]) < span - 1.0e-12:
                raise NotImplementedError(
                    "Tidy3D symmetry is center-based; Maxwell face symmetry can only be "
                    "exported when every structure spans the symmetry axis."
                )
        for region in scene.material_regions:
            geometry = region.geometry
            if getattr(geometry, "kind", None) != "box" or float(geometry.size[axis_index]) < span - 1.0e-12:
                raise NotImplementedError(
                    "Tidy3D symmetry is center-based; Maxwell MaterialRegion must span the "
                    "symmetry axis for an equivalent export."
                )
        for source in scene.resolved_sources():
            if not isinstance(source, PlaneWave) or abs(float(source.direction[axis_index])) > 1.0e-12:
                raise NotImplementedError(
                    "Tidy3D symmetry is center-based; Maxwell face symmetry with a localized "
                    "or symmetry-axis-varying source has no equivalent direct export."
                )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def _cw_steady_state_run_time(scene, run_time: float) -> float:
    """Dilute CW ramp/propagation history without changing Fourier normalization."""
    from ..sources import CW

    # Time-modulated scenes keep the bounded observation window: no single steady state,
    # and long periodic-cell runs expose late-time growth (see the benchmark assessment).
    if any(getattr(structure.material, "modulation", None) is not None for structure in scene.structures):
        return run_time
    carriers = [
        source.source_time
        for source in scene.resolved_sources()
        if isinstance(getattr(source, "source_time", None), CW)
    ]
    if not carriers:
        return run_time
    # ContinuousWave defaults to offset=5 and fwidth=0.1*frequency. A thousand
    # diagonal crossings makes one vacuum transit's power deficit about 0.2%.
    ramp_offset = max(5.0 / (2.0 * math.pi * 0.1 * carrier.frequency) for carrier in carriers)
    crossing_time = math.sqrt(sum((hi - lo) ** 2 for lo, hi in scene.domain.bounds)) / _C0
    return max(run_time, ramp_offset + 1000.0 * crossing_time)


def scene_to_tidy3d(
    scene,
    *,
    frequencies: float | Sequence[float] | None = None,
    run_time: float | None = None,
    length_scale: float = _M_TO_UM,
    cw_steady_state: bool = False,
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
    cw_steady_state : bool
        Extend CW-driven runs to at least 1000 vacuum domain-diagonal crossing
        times plus the source ramp offset, reducing finite-history phasor bias.
        Time-modulated materials and pulse-only runs retain the requested duration.
        Default False permits explicit short runs.
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
    if cw_steady_state:
        run_time = _cw_steady_state_run_time(scene, run_time)

    # -- grid ------------------------------------------------------------------
    td_grid = None
    if scene.grid is not None:
        td_grid = _convert_grid(scene, td, s)

    # -- boundary --------------------------------------------------------------
    td_boundary = td.BoundarySpec.all_sides(boundary=td.PML())
    if scene.boundary is not None:
        td_boundary = _convert_boundary(scene.boundary, td, scene.domain.bounds, length_scale=s)

    # -- structures ------------------------------------------------------------
    td_structures = []
    from ..ports import WavePort as _WavePort

    driven_wave = next((port.name for port in (scene.ports or []) if isinstance(port, _WavePort)), None)
    for structure in (scene.structures or []):
        frame = _pec_frame_record(structure)
        if frame is not None and frame["port"] == driven_wave:
            # Exported as the reference mode source's own PECFrame.
            continue
        td_structures.append(_convert_structure(structure, td, s, frequencies=frequencies))
    for region in (scene.material_regions or []):
        td_structures.append(_convert_material_region(region, td, s))

    # -- sources ---------------------------------------------------------------
    domain_bounds = scene.domain.bounds
    td_sources = []
    sources = scene.resolved_sources() if hasattr(scene, "resolved_sources") else (scene.sources or [])
    for source in sources:
        td_sources.append(_convert_source(source, scene, td, s, frequencies=frequencies))

    # RF ports (WavePort / LumpedPort) carry no ``source_time`` of their own; map
    # their excitation to reference sources + receive monitors under the single
    # documented drive convention (see ``_convert_ports_for_reference``).
    port_sources, port_monitors = _convert_ports_for_reference(scene, td, s, frequencies)
    td_sources.extend(port_sources)

    # -- monitors --------------------------------------------------------------
    td_monitors = []
    monitors = scene.resolved_monitors() if hasattr(scene, "resolved_monitors") else (scene.monitors or [])
    from ..monitors import (
        GAUSSIAN_OVERLAP_MONITOR_TYPES,
        PROJECTION_MONITOR_TYPES,
        AuxFieldTimeMonitor,
        FieldTimeMonitor,
        FluxTimeMonitor,
        SurfaceFieldTimeMonitor,
    )

    surface_owners = {
        monitor.name: monitor
        for monitor in (scene.monitors or [])
        if isinstance(monitor, (*PROJECTION_MONITOR_TYPES, *GAUSSIAN_OVERLAP_MONITOR_TYPES))
    }
    exported_owners = set()
    time_step = None
    prepared_scene = None
    if any(isinstance(monitor, (FieldTimeMonitor, FluxTimeMonitor, SurfaceFieldTimeMonitor)) for monitor in monitors):
        time_step = _scene_time_step(scene, kwargs.get("courant", 0.99))
    for monitor in monitors:
        owner_name = getattr(monitor, "surface_name", None)
        if owner_name in surface_owners:
            # Faces recorded for a projection / Gaussian-overlap monitor export
            # as that single reference monitor.
            if owner_name not in exported_owners:
                exported_owners.add(owner_name)
                td_monitors.append(_convert_in_simulation_monitor(surface_owners[owner_name], frequencies, td, s))
            continue
        if isinstance(monitor, SurfaceFieldTimeMonitor):
            td_monitors.extend(_convert_surface_field_time_monitor(monitor, td, s, time_step=time_step))
            continue
        if isinstance(monitor, AuxFieldTimeMonitor):
            raise NotImplementedError(
                f"AuxFieldTimeMonitor '{monitor.name}' records dispersive ADE polarization state; the "
                "Tidy3D auxiliary-field monitor only stores free-carrier densities, so there is no equivalent."
            )
        if prepared_scene is None and (
            getattr(monitor, "compute_flux", False)
            or getattr(monitor, "closure_planes", None) is not None
            or isinstance(monitor, FluxTimeMonitor)
        ):
            from ..scene import prepare_scene

            prepared_scene = prepare_scene(scene.clone(device="cpu"), lazy_meshgrid=True)
        td_monitors.append(
            _convert_monitor(
                monitor,
                domain_bounds,
                frequencies,
                td,
                s,
                time_step=time_step,
                prepared_scene=prepared_scene,
            )
        )
    td_monitors.extend(port_monitors)

    # -- symmetry --------------------------------------------------------------
    td_symmetry = (0, 0, 0)
    if scene.symmetry is not None:
        _validate_tidy3d_symmetry_equivalence(scene)
        # Tidy3D encodes symmetry about the domain center only; the folded face
        # (low/high) has no Tidy3D counterpart and is dropped in the export.
        sym_map = {"PEC": -1, "PMC": 1}
        td_symmetry = tuple(
            0 if entry is None else sym_map.get(entry[0], 0)
            for entry in scene.symmetry
        )

    # -- build simulation ------------------------------------------------------
    coaxial_elements = _convert_coaxial_lumped_elements(scene, td, s)
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
    internal_absorbers = tuple(getattr(scene, "internal_absorbers", ()) or ())
    if internal_absorbers:
        sim_kwargs["internal_absorbers"] = tuple(
            _convert_internal_absorber(absorber, td, s) for absorber in internal_absorbers
        )
    if coaxial_elements:
        sim_kwargs["lumped_elements"] = coaxial_elements
    sim_kwargs.update(kwargs)

    return td.Simulation(**sim_kwargs)
