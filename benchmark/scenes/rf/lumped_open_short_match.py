"""Coaxial one-port open/short/match (SOL) calibration bench.

Rebuild (audit S1 follow-up). The retired bench placed two lumped ports two cells
apart in a tiny PML box: the feed saw near-total reflection INDEPENDENT of the
load (identical Gamma for matched / open / short) because the feed port was not
coupled to the load through any transmission structure. This rebuild puts the
three canonical short-open-load standards at the end of the PROVEN air coax line
(same cross-section as ``coax_thru``), so the launched TEM wave actually
propagates to the load plane and reflects according to the termination.

A single ``WavePort`` TEM feed at the left launches the coax TEM mode; the line
runs to a de-embedded load reference plane at :data:`LOAD_PLANE`, terminated by
one of three physical standards:

* ``matched`` -- the coax runs on THROUGH the right computational PML (as in
  ``coax_thru``), a reflectionless termination that presents the line's own
  characteristic impedance ``Z0 = eta0/(2*pi) ln(b/a)`` (~83 ohm). This is the
  matched ``R = Z0`` standard (Gamma -> 0); the absorber, not a lumped resistor,
  supplies the Z0 match, exactly as the proven coax bench does.
* ``short`` -- a solid PEC plug (radius ``OUTER_RADIUS``) fills the coax from the
  load plane rightward, shorting the inner conductor to the shield at the plane
  (E = 0 at the plug face -> Gamma = -1).
* ``open`` -- the inner conductor is truncated at the load plane; the shield
  continues through the PML. Below the outer circular-guide's first
  azimuthally-symmetric cutoff (TM01, :data:`TM01_CUTOFF_HZ` ~ 0.717 GHz for
  ``b = OUTER_RADIUS``) the beyond-plane region is evanescent, so the open end
  reflects (Gamma -> +1). The fat coax has a substantial open-end fringe
  capacitance that shifts the effective open reference plane outward; this is
  measured and documented rather than hidden (see the acceptance doc).

Operating band: the default sweep sits BELOW the TM01 cutoff so the ``open``
standard does not leak into a propagating circular-guide mode. The TEM incident
is azimuthally symmetric and couples only to TM0n; the TE11 branch
(:data:`TE11_CUTOFF_HZ`) is antisymmetric and is not excited, so only the TM01
cutoff constrains the band.

Phase-reference convention (documented gate contract): the ``short`` standard
defines the -1 reference at the load plane (a standard one-port SOL primitive).
Referenced to that plane the ``open`` lands in the +1 class (Re(Gamma) > 0),
offset from an ideal open only by the measured fringe capacitance, and the
``matched`` load sits at |Gamma| ~ 0. The three responses are mutually
distinguishable -- which is precisely what the retired bench failed to achieve.
"""

from __future__ import annotations

import math

import witwin.maxwell as mw

from benchmark.scenes.rf.coax_thru import (
    DOMAIN_TRANSVERSE,
    INNER_RADIUS,
    OUTER_RADIUS,
    SHIELD_OUTER,
    _HollowCylinder,
    _coax_port,
    analytic_z0,
)

C0 = 299792458.0

DOMAIN_X = 0.12
PML_LAYERS = 12          # thicker than coax_thru's 6: the sub-GHz band needs an
                         # electrically-thick absorber for the matched |Gamma| <= -20 dB gate.
FEED_X = -0.08           # WavePort TEM feed reference plane.
LOAD_PLANE = 0.06        # de-embedded load reference plane (short face / open rod end).

# Circular-guide cutoffs of the outer shield (radius b = OUTER_RADIUS). Below TM01
# the open-end beyond-region is evanescent (clean open); TE11 is antisymmetric and
# is not excited by the symmetric TEM incident.
TM01_CUTOFF_HZ = 2.405 * C0 / (2.0 * math.pi * OUTER_RADIUS)
TE11_CUTOFF_HZ = 1.841 * C0 / (2.0 * math.pi * OUTER_RADIUS)

STANDARDS = ("matched", "short", "open")


def line_length(dx: float) -> float:
    """Conductor length that runs THROUGH the PML to the padded grid edges.

    Mirrors ``coax_thru.line_length`` but for this bench's ``PML_LAYERS`` so the
    shield (and, for the matched/short standards, the inner rod) terminate at the
    padded grid edge instead of ending in an open stub at the PML interface.
    """
    return 2.0 * (DOMAIN_X + PML_LAYERS * dx) + 8.0 * dx


def default_frequencies() -> tuple[float, ...]:
    """Sweep band, safely below the TM01 cutoff so the open does not leak."""
    return (0.45e9, 0.475e9, 0.50e9)


def analytic_gamma(standard: str) -> complex:
    """Ideal reflection coefficient of each standard at the load plane."""
    if standard == "matched":
        return 0.0 + 0.0j
    if standard == "short":
        return -1.0 + 0.0j
    if standard == "open":
        return 1.0 + 0.0j
    raise ValueError(f"Unknown standard {standard!r}; expected one of {STANDARDS}.")


def coax_sol_scene(
    standard: str = "matched",
    *,
    dx: float = 0.01,
    device: str = "cuda",
) -> mw.Scene:
    """Build the coax one-port terminated by the requested SOL standard."""

    if standard not in STANDARDS:
        raise ValueError(f"standard must be one of {STANDARDS}, got {standard!r}.")

    length = line_length(dx)
    left_edge = -0.5 * length

    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-DOMAIN_X, DOMAIN_X),
                (-DOMAIN_TRANSVERSE, DOMAIN_TRANSVERSE),
                (-DOMAIN_TRANSVERSE, DOMAIN_TRANSVERSE),
            )
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        ports=(_coax_port("feed", FEED_X, "+", dx=dx),),
        device=device,
    )

    # Outer shield runs the full padded length through both PMLs for every standard.
    scene.add_structure(
        _HollowCylinder(
            position=(0.0, 0.0, 0.0),
            inner_radius=OUTER_RADIUS,
            outer_radius=SHIELD_OUTER,
            length=length,
        ).with_material(mw.Material.pec(), name="shield")
    )

    if standard == "open":
        # Inner rod stops at the load plane; beyond it the shield is a below-cutoff
        # circular guide (evanescent) -> open-circuit reflection.
        rod_length = LOAD_PLANE - left_edge
        rod_center = 0.5 * (left_edge + LOAD_PLANE)
        scene.add_structure(
            mw.Cylinder(
                position=(rod_center, 0.0, 0.0),
                radius=INNER_RADIUS,
                height=rod_length,
                axis="x",
            ).with_material(mw.Material.pec(), name="inner_conductor")
        )
    else:
        # matched / short: inner rod runs the full padded length.
        scene.add_structure(
            mw.Cylinder(
                position=(0.0, 0.0, 0.0),
                radius=INNER_RADIUS,
                height=length,
                axis="x",
            ).with_material(mw.Material.pec(), name="inner_conductor")
        )

    if standard == "short":
        # Solid PEC plug shorts inner->shield at the load plane and fills to the edge.
        plug_length = 0.5 * length - LOAD_PLANE
        plug_center = 0.5 * (LOAD_PLANE + 0.5 * length)
        scene.add_structure(
            mw.Cylinder(
                position=(plug_center, 0.0, 0.0),
                radius=OUTER_RADIUS,
                height=plug_length,
                axis="x",
            ).with_material(mw.Material.pec(), name="short_plug")
        )

    return scene


# Nominal Z0 (documented matched-standard target impedance).
ANALYTIC_Z0 = analytic_z0()
