"""Center-fed half-wave dipole antenna scene (FDTD, lumped wire-gap feed).

Two collinear PEC thin-wire arms along ``z`` are joined by a node-bound
:class:`LumpedPort` feed, enclosed by a :class:`ClosedSurfaceMonitor` box for the
near-field-to-far-field (NF2FF) transform. A real FDTD ``Scene -> Simulation ->
Result`` run driven at the feed yields, through :meth:`Result.antenna` (no
monkeypatch), the engineering pattern/directivity/gain plus the feed
:class:`PortData` from which the input impedance and the accepted/radiated power
balance are read.

Physics references (thin center-fed half-wave dipole):

* radiation resistance class ``R ~ 73 Ohm`` (thin-wire), reactance small at the
  natural resonance (``L ~ 0.47*lambda``);
* E-plane power pattern ``~ sin^2(theta)`` (the canonical dipole doughnut);
* directivity ``~ 1.64`` (linear) ``= 2.15 dBi``.

The dipole is sized to ``L = lambda/2`` at ``design_frequency``. The physical
half-wave length does NOT coincide with the FDTD delta-gap resonance: the wire
feed carries a large positive (inductive) reactance offset, so the input
*reactance* is feed-dominated and the electrical resonance (``X = 0``) sits above
the physical half-wave frequency. The input *resistance* (radiation resistance)
still sweeps through the thin-dipole ``73 Ohm`` class within the band, and the
radiated pattern, directivity, and accepted-vs-radiated power balance reproduce
the analytic dipole. This is documented rather than hidden (see the acceptance
record); the reactance offset is a known thin-wire delta-gap feed characteristic.

The builder returns geometry, feed port, and NF2FF surface only. Excitation, run
length, and spectral sampling are chosen by the caller (benchmark runner or test)
so one scene serves both a single design-frequency evaluation and a swept
impedance characterization.
"""

from __future__ import annotations

import math

import torch

import witwin.maxwell as mw

C0 = 299792458.0
ETA0 = 376.730313668

# Thin-wire radius stays inside the accepted a/delta_perp <= 0.2 validity band of
# the thin-wire model (a = 0.16 * dx at the default tier).
WIRE_RADIUS_FRACTION = 0.16
ARM_SEGMENTS = 20
# NF2FF box and PML padding, expressed in cells so the surround scales with dx.
BOX_MARGIN_CELLS = 16
DOMAIN_MARGIN_CELLS = 20
PML_LAYERS = 8


def design_arm_half_length(design_frequency: float) -> float:
    """Quarter-wavelength arm half-length (total dipole = lambda/2)."""

    return 0.25 * C0 / float(design_frequency)


def analytic_directivity_dbi() -> float:
    """Ideal thin half-wave dipole directivity."""

    return 10.0 * math.log10(1.643)


def analytic_radiation_resistance() -> float:
    """Ideal thin half-wave dipole radiation resistance (Ohm)."""

    return 73.1


def default_frequencies(design_frequency: float) -> tuple[float, ...]:
    """Impedance/pattern sweep band centred a little above the physical resonance.

    The band brackets the frequency where the radiation resistance crosses the
    thin-dipole ``73 Ohm`` class in this FDTD model (above the physical half-wave
    frequency because of the feed reactance offset).
    """

    base = float(design_frequency)
    return tuple(base * scale for scale in (0.80, 0.90, 1.00, 1.07, 1.13, 1.20))


def half_wave_dipole_scene(
    *,
    design_frequency: float = 3.0e9,
    frequencies: tuple[float, ...] | None = None,
    dx: float = 1.25e-3,
    device: str = "cuda",
) -> mw.Scene:
    """Build a center-fed half-wave dipole with a wire-gap feed and NF2FF box."""

    arm_half = design_arm_half_length(design_frequency)
    resolved_frequencies = (
        default_frequencies(design_frequency)
        if frequencies is None
        else tuple(float(freq) for freq in frequencies)
    )
    radius = WIRE_RADIUS_FRACTION * dx
    gap = dx

    negative_nodes = torch.linspace(-arm_half, -0.5 * gap, ARM_SEGMENTS + 1)
    positive_nodes = torch.linspace(0.5 * gap, arm_half, ARM_SEGMENTS + 1)
    negative_arm = mw.ThinWire(
        "negative_arm",
        tuple((0.0, 0.0, float(value)) for value in negative_nodes),
        radius,
        mw.WireConductor.pec(),
        snap="continuous",
    )
    positive_arm = mw.ThinWire(
        "positive_arm",
        tuple((0.0, 0.0, float(value)) for value in positive_nodes),
        radius,
        mw.WireConductor.pec(),
        snap="continuous",
    )
    feed = mw.LumpedPort(
        "feed",
        wire_binding=mw.WirePortBinding.nodes(
            negative=mw.WireNodeRef("negative_arm", ARM_SEGMENTS),
            positive=mw.WireNodeRef("positive_arm", 0),
        ),
        reference_impedance=analytic_radiation_resistance(),
    )

    box_half = arm_half + BOX_MARGIN_CELLS * dx
    surface = mw.ClosedSurfaceMonitor.box(
        "radiation",
        position=(0.0, 0.0, 0.0),
        size=(2.0 * box_half, 2.0 * box_half, 2.0 * box_half),
        frequencies=resolved_frequencies,
    )

    domain_half = arm_half + DOMAIN_MARGIN_CELLS * dx
    return mw.Scene(
        domain=mw.Domain(bounds=((-domain_half, domain_half),) * 3),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        thin_wires=(negative_arm, positive_arm),
        ports=(feed,),
        monitors=(surface,),
        device=device,
    )
