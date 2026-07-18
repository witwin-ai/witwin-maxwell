"""Torch reference model and frozen acceptance budget for the surface-impedance boundary.

This module is intentionally independent of the production stepping runtime. It
defines the numerical contract used to validate the generalized surface-impedance
subsystem (the causal rational state-space tangential surface impedance/admittance
that the narrowband good-conductor Leontovich prototype is being converged into),
and it is deliberately not a CPU fallback: nothing in the solver may import it.

It provides, for Phase 0:

* :class:`SurfaceAcceptanceBudget` -- the pre-registered acceptance thresholds,
  pinned by ``tests/fdtd/surface_impedance/test_surface_impedance_reference.py``
  (``test_surface_acceptance_budget_is_frozen``). Every threshold carries its prior
  value, the measured evidence, and the technical reason once measured; a change
  must retain those and update the frozen test (the same discipline as
  ``witwin/maxwell/array.py`` and ``witwin/maxwell/fdtd/thin_wire_reference.py``).
* Analytic Fresnel/Leontovich complex reflection for a surface impedance, including
  oblique TE/TM incidence -- the exact oracle behind the ``analytic_*`` gates and the
  reproduction of the incumbent narrowband ``|Gamma|`` gate.
* The discrete surface power form with the edge/corner unique-owner assembly rule,
  proven nonnegative and free of double counting on a hand-built two-face corner
  case -- the de-risking of the edge/corner double-counting risk before any
  multi-face kernel exists.

The binding to the acceptance gates and the incumbent prototype lives in
``tests/fdtd/surface_impedance/test_surface_impedance_reference.py``. Keep that
binding alive: without a consumer this reference silently stops constraining
anything.

Convention: the repository uses ``s = -i * 2 * pi * f`` / ``e^{-i * omega * t}``,
so the good-conductor surface impedance is ``Z_s = (1 - i) * sqrt(omega * mu / (2 * sigma))``
(matching ``witwin/maxwell/media.py::LossyMetalMedium.surface_impedance``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import torch


EPSILON_0 = 8.8541878128e-12
MU_0 = 1.25663706212e-6
ETA_0 = math.sqrt(MU_0 / EPSILON_0)


@dataclass(frozen=True)
class SurfaceAcceptanceBudget:
    """Pre-registered Phase 0 acceptance limits for the surface-impedance subsystem.

    The thresholds are frozen at Phase 0 (contract freeze) and cover every phase of
    the plan, including gates a later phase enforces (gradient, no-regression). A
    change to any field must retain the previous value, the measured evidence, and
    the technical reason in a comment, and update
    ``test_surface_acceptance_budget_is_frozen``.
    """

    # --- fit quality (plan default: max complex error <= 1e-3 or a stricter user tol) ---
    # Equals the shared fitter's own RationalFitConfig.relative_tolerance default
    # (witwin/maxwell/rational.py), so a fit that passes the fitter passes the budget
    # without introducing a second, looser standard. Measured: a good-conductor
    # (sigma=5.8e7) Z_s over 1-40 GHz fits to a worst-case relative error of 3.9e-5
    # at order 8 and 2.3e-6 at order 10, well inside this bound.
    fit_max_complex_error: float = 1.0e-3
    # Certified passive means margin >= 0 with no slack, mirroring the embedded-network
    # compile exit gate (compiler/networks.py "passive and certified"). Passivity is a
    # hard exit gate, never a warning.
    fit_min_passivity_margin: float = 0.0
    # Matches the shared network passivity default (rational.py / networks.py).
    passivity_tolerance: float = 1.0e-9
    # Bilinear (trapezoidal) discretization keeps every discrete pole strictly inside
    # the unit circle; StateSpaceNetwork.discretize enforces |z| < 1 - stability_margin
    # with stability_margin = 1e-7. Measured worst-case |z| for the order-8/10
    # good-conductor fits is 0.99996 / 0.99997.
    discrete_pole_radius_max: float = 1.0

    # --- reference / analytic agreement (plan: complex R or loss rel err <= 2%, phase <= 3 deg) ---
    analytic_reflection_relative_error: float = 2.0e-2
    analytic_phase_error_deg: float = 3.0
    # Narrowband degenerate reproduction of the EXISTING prototype: this is the
    # incumbent published gate in tests/validation/physics/test_lossy_metal_sibc.py
    # (5%). The generic model must not regress the prototype accuracy; freezing at the
    # existing number prevents a silent loosening during migration. DO NOT loosen.
    narrowband_reproduction_relative_error: float = 5.0e-2

    # --- power balance / energy (plan: residual <= 1%, local dissipation nonnegative) ---
    # Input power = reflected + transmitted/radiated + surface/material loss. The
    # analytic surface-loss form and the reflection form close to solver precision
    # (measured residual < 1e-12 for the good conductor at normal and oblique
    # incidence), so 1% is a generous engineering bound for the discrete runtime.
    power_residual: float = 1.0e-2
    # Per-cell time-averaged surface dissipation must be nonnegative (hard, no negative).
    # This is the physics falsification for the passive branch: a non-passive fit or an
    # opposite-sign surface resistance injects energy and grows without bound.
    min_local_surface_dissipation: float = 0.0

    # --- convergence (plan: >= 3 levels of grid / dt / fit-order refinement) ---
    convergence_levels: int = 3

    # --- gradient (plan: parameter gradient rel err < 2%) [Phase 4, frozen now] ---
    gradient_relative_error: float = 2.0e-2
    # Avoid dividing a relative error by a near-zero reference gradient (the same
    # div-by-tiny guard used by witwin/maxwell/array.py's AcceptanceBudget).
    gradient_absolute_floor: float = 1.0e-8

    # --- performance / parity (plan) ---
    # < 1% overhead when no surface is present, enforced by the zero-launch / zero-state
    # contract (a scene with no surface-impedance structure allocates no surface state
    # and issues no surface kernel launches). Multi-GPU value/loss/gradient parity
    # inherits plan 02's frozen distributed budget verbatim; that budget is referenced,
    # not re-frozen here.
    no_surface_impedance_runtime_regression: float = 1.0e-2


SURFACE_ACCEPTANCE_BUDGET = SurfaceAcceptanceBudget()


def _as_frequency_tensor(frequencies) -> torch.Tensor:
    tensor = (
        frequencies
        if isinstance(frequencies, torch.Tensor)
        else torch.as_tensor(frequencies, dtype=torch.float64)
    )
    if tensor.is_complex() or not tensor.dtype.is_floating_point:
        raise TypeError("frequencies must be real floating point.")
    tensor = tensor.to(dtype=torch.float64)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("frequencies must have non-empty shape [F].")
    if not bool(torch.all(torch.isfinite(tensor))) or not bool(torch.all(tensor > 0.0)):
        raise ValueError("frequencies must be finite and strictly positive.")
    return tensor


def good_conductor_surface_impedance(
    conductivity: float,
    frequencies,
    *,
    permeability: float = MU_0,
) -> torch.Tensor:
    """Leontovich good-conductor surface impedance ``Z_s(omega)`` in ohms.

    ``Z_s = (1 - i) * sqrt(omega * mu / (2 * sigma))`` under the repository
    ``e^{-i omega t}`` convention (matches ``LossyMetalMedium.surface_impedance``).
    Returns a ``complex128`` tensor with the shape of ``frequencies``.
    """

    sigma = float(conductivity)
    mu = float(permeability)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("conductivity must be finite and positive.")
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError("permeability must be finite and positive.")
    omega = 2.0 * math.pi * _as_frequency_tensor(frequencies)
    magnitude = torch.sqrt(omega * mu / (2.0 * sigma))
    return torch.complex(magnitude, -magnitude)


def transverse_wave_impedance(
    *,
    eta: float = ETA_0,
    angle: float = 0.0,
    polarization: Literal["normal", "te", "tm"] = "normal",
) -> float:
    """Transverse wave impedance of the incident medium seen at the surface.

    * ``normal`` / ``angle == 0``: ``eta``.
    * ``te`` (E perpendicular to the plane of incidence): ``eta / cos(theta)``.
    * ``tm`` (H perpendicular to the plane of incidence): ``eta * cos(theta)``.

    Both oblique forms reduce to ``eta`` at normal incidence.
    """

    value = float(eta)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("eta must be finite and positive.")
    theta = float(angle)
    if not math.isfinite(theta) or not (-0.5 * math.pi < theta < 0.5 * math.pi):
        raise ValueError("angle must be a propagating incidence angle in (-pi/2, pi/2).")
    cos_theta = math.cos(theta)
    kind = polarization.lower()
    if kind == "normal":
        return value
    if kind == "te":
        return value / cos_theta
    if kind == "tm":
        return value * cos_theta
    raise ValueError("polarization must be 'normal', 'te', or 'tm'.")


def leontovich_reflection(
    surface_impedance: complex | torch.Tensor,
    *,
    eta: float = ETA_0,
    angle: float = 0.0,
    polarization: Literal["normal", "te", "tm"] = "normal",
) -> torch.Tensor:
    """Complex reflection coefficient for a surface impedance load.

    ``Gamma = (Z_s - Z_t) / (Z_s + Z_t)`` with ``Z_t`` the transverse wave impedance
    of the incident medium for the requested polarization/angle. For a passive
    surface (``Re(Z_s) >= 0``) this obeys ``|Gamma| <= 1``. Returns ``complex128``.
    """

    z_s = torch.as_tensor(surface_impedance).to(dtype=torch.complex128)
    z_t = transverse_wave_impedance(eta=eta, angle=angle, polarization=polarization)
    return (z_s - z_t) / (z_s + z_t)


def power_balance_residual(
    surface_impedance: complex | torch.Tensor,
    *,
    eta: float = ETA_0,
    angle: float = 0.0,
    polarization: Literal["normal", "te", "tm"] = "normal",
) -> torch.Tensor:
    """Relative residual between the two independent absorbed-power expressions.

    For a plane wave on a surface impedance load, the absorbed fraction computed
    from the reflection coefficient, ``1 - |Gamma|^2``, must equal the absorbed
    fraction computed from the surface-loss form ``Re(Z_s) |H_t|^2 / (2 S_inc)``,
    where the total tangential field on the load is fixed by ``Gamma``. This returns
    ``|fraction_reflection - fraction_surface| / max(fraction_reflection, floor)``
    and is an exact identity in continuous theory (residual at solver precision),
    so it is the analytic anchor for the discrete ``power_residual`` gate.
    """

    z_s = torch.as_tensor(surface_impedance).to(dtype=torch.complex128)
    z_t = transverse_wave_impedance(eta=eta, angle=angle, polarization=polarization)
    gamma = (z_s - z_t) / (z_s + z_t)
    # Normalize the incident tangential H to unity. Total tangential H on the load is
    # (1 - Gamma) for a load referenced to the transverse impedance z_t; the incident
    # power density normal to the surface is 0.5 * z_t * |H_inc|^2.
    absorbed_surface = 0.5 * z_s.real * (1.0 - gamma).abs().square()
    incident = 0.5 * z_t
    fraction_surface = absorbed_surface / incident
    fraction_reflection = 1.0 - gamma.abs().square()
    denominator = torch.clamp(fraction_reflection.abs(), min=1.0e-12)
    return (fraction_reflection - fraction_surface).abs() / denominator


@dataclass(frozen=True)
class SurfaceEdgeContribution:
    """One face's claim on a tangential surface degree of freedom (a Yee edge).

    ``face_id`` is the global id of the face that contributes this edge, ``edge_id``
    the global id of the shared tangential-E edge (adjacent faces meeting at a corner
    share an ``edge_id``), ``surface_resistance`` the real part of the local surface
    impedance ``Re(Z_s) >= 0``, ``dual_area`` the positive area weight of the edge's
    dual face, and ``tangential_field`` the tangential magnetic field magnitude
    ``|H_t|`` sampled on the edge.
    """

    face_id: int
    edge_id: int
    surface_resistance: float
    dual_area: float
    tangential_field: float


def _validate_contributions(
    contributions: Iterable[SurfaceEdgeContribution],
) -> list[SurfaceEdgeContribution]:
    items = list(contributions)
    if not items:
        raise ValueError("surface dissipation requires at least one edge contribution.")
    for item in items:
        if not isinstance(item, SurfaceEdgeContribution):
            raise TypeError("contributions must be SurfaceEdgeContribution instances.")
        if item.surface_resistance < 0.0:
            # min_local_surface_dissipation is a hard nonnegativity gate: a negative
            # surface resistance is a non-passive surface that injects energy.
            raise ValueError(
                "surface_resistance must be nonnegative; a negative surface resistance "
                "is a non-passive surface that injects energy and grows without bound."
            )
        if not math.isfinite(item.dual_area) or item.dual_area <= 0.0:
            raise ValueError("dual_area must be finite and positive.")
        if not math.isfinite(item.tangential_field):
            raise ValueError("tangential_field must be finite.")
    return items


def _edge_dissipation(contribution: SurfaceEdgeContribution) -> float:
    return (
        0.5
        * float(contribution.surface_resistance)
        * float(contribution.dual_area)
        * float(contribution.tangential_field) ** 2
    )


def assemble_surface_dissipation(
    contributions: Iterable[SurfaceEdgeContribution],
) -> tuple[float, dict[int, float]]:
    """Total time-averaged surface dissipation under the unique-owner rule.

    Adjacent faces sharing a Yee edge would otherwise double-apply (or, in a
    last-write-wins kernel, silently drop) the tangential contribution. This assigns
    each shared ``edge_id`` a single deterministic owner -- the minimum global
    ``face_id`` among the faces claiming it, matching the plan-07 minimum-global-edge
    owner discipline -- and counts the edge exactly once using the owner's local
    values. Returns ``(total, per_edge)`` with ``per_edge`` keyed by ``edge_id``. Each
    per-edge value is nonnegative because ``Re(Z_s) >= 0`` is enforced.
    """

    items = _validate_contributions(contributions)
    owner: dict[int, SurfaceEdgeContribution] = {}
    for item in items:
        current = owner.get(item.edge_id)
        if current is None or item.face_id < current.face_id:
            owner[item.edge_id] = item
    per_edge = {edge_id: _edge_dissipation(item) for edge_id, item in owner.items()}
    total = float(sum(per_edge.values()))
    return total, per_edge


def naive_double_counted_dissipation(
    contributions: Iterable[SurfaceEdgeContribution],
) -> float:
    """Order-dependent assembly that sums every (face, edge) contribution.

    This is the failure mode the unique-owner rule prevents: shared edges are counted
    once per adjoining face. It exists only so tests can falsify that the owner rule
    is actually active (it must not equal :func:`assemble_surface_dissipation` when
    any edge is shared).
    """

    items = _validate_contributions(contributions)
    return float(sum(_edge_dissipation(item) for item in items))


__all__ = [
    "EPSILON_0",
    "ETA_0",
    "MU_0",
    "SURFACE_ACCEPTANCE_BUDGET",
    "SurfaceAcceptanceBudget",
    "SurfaceEdgeContribution",
    "assemble_surface_dissipation",
    "good_conductor_surface_impedance",
    "leontovich_reflection",
    "naive_double_counted_dissipation",
    "power_balance_residual",
    "transverse_wave_impedance",
]
