"""Lossy-wall rectangular-waveguide TE10 attenuation bench (surface-impedance).

A hollow air guide whose four PEC walls are replaced by a good-conductor
surface-impedance boundary (``LossyMetalMedium``). A real FDTD two-port sweep
launches the TE10 mode; the transmitted ``|S21|`` decays as ``exp(-alpha L)`` with
the guided length. Running two guides of different length and forming the
two-line ratio

    alpha [Np/m] = ln(|S21_short| / |S21_long|) / (L_long - L_short)

cancels the (identical) port-junction launch/receive loss, leaving the conductor
attenuation of the walls. It is compared against the analytic TE10 conductor
attenuation (Pozar, *Microwave Engineering*, eq. 3.96)::

    alpha_c = R_s / (a^3 b beta k eta0) * (2 b pi^2 + a^3 k^2)   [Np/m]

with ``R_s = sqrt(omega mu0 / (2 sigma))`` the wall surface resistance,
``k = omega / c``, ``beta = sqrt(k^2 - (pi/a)^2)`` the TE10 phase constant, ``a``
the broad-wall (y) width and ``b`` the narrow-wall (z) height of the guide
interior. The surface-impedance runtime evaluates ``R_s`` at the source frequency
(the narrowband good-conductor order-0 model), so the analytic ``R_s`` and the
runtime surface resistance are the same physical quantity and the bench measures
whether the SIBC wall reproduces the textbook skin-effect loss.

Termination follows the ``rf/rectangular_waveguide`` precedent: the walls run
``2*(DOMAIN_X + num_layers*dx)`` through the PML to the padded grid edges so the
launched TE10 wave terminates at both ends rather than reflecting off an open
guide stub. The four walls share ONE ``LossyMetalMedium`` instance so their
shared corner edges resolve through the deterministic surface owner rank instead
of the conflicting-owner guard (two different surface materials on one interface
plane fail closed by design).
"""

from __future__ import annotations

import math

import witwin.maxwell as mw

C0 = 299792458.0
ETA0 = 376.730313668
MU0 = 4.0e-7 * math.pi

# Guide interior cross-section: a (broad wall, y) x b (narrow wall, z).
GUIDE_A = 0.10
GUIDE_B = 0.05
# Good-conductor wall conductivity [S/m]. Deliberately in the well-characterised
# SIBC good-conductor regime (the resistive-Leontovich stability domain the rest
# of the SIBC suite exercises); low enough that the wall loss is a large, cleanly
# measurable |S21| decay yet |Z_s| << eta0 (a genuine good conductor).
WALL_CONDUCTIVITY = 25.0
DESIGN_FREQUENCY = 2.4e9

# Two-line lengths (port reference-plane separation along x).
SHORT_LENGTH = 0.60
LONG_LENGTH = 0.90

# Transverse domain: the walls sit flush against the y/z domain bounds so their
# outer faces back onto the PML (never illuminated); only the guide-interior faces
# carry the surface impedance.
DOMAIN_Y = 0.075
DOMAIN_Z = 0.05
WALL_THICKNESS = 0.025

# PML physical thickness held fixed in metres across grid tiers (rectangular_waveguide
# precedent: a fixed layer count would shrink the absorber as dx falls).
PML_THICKNESS_M = 0.04


def cutoff_frequency() -> float:
    return C0 / (2.0 * GUIDE_A)


def _pml_layers(dx: float) -> int:
    return max(int(round(PML_THICKNESS_M / dx)), 6)


def analytic_alpha_te10(frequency: float, *, sigma: float = WALL_CONDUCTIVITY,
                        a: float = GUIDE_A, b: float = GUIDE_B) -> float:
    """Analytic TE10 conductor attenuation [Np/m] (Pozar eq. 3.96)."""
    omega = 2.0 * math.pi * frequency
    k = omega / C0
    beta = math.sqrt(k * k - (math.pi / a) ** 2)
    r_s = math.sqrt(omega * MU0 / (2.0 * sigma))
    return r_s / (a**3 * b * beta * k * ETA0) * (2.0 * b * math.pi**2 + a**3 * k * k)


def wall_length(dx: float, guide_len: float) -> float:
    """Wall length that runs THROUGH the PML to the padded grid edges.

    The prepared grid appends ``num_layers*dx`` beyond the declared ``+-DOMAIN_X``;
    the walls must span the full padded extent plus a margin so the launched TE10
    wave terminates at the absorber rather than an open guide end.
    """
    domain_x = 0.5 * guide_len + 0.03
    return 2.0 * (domain_x + _pml_layers(dx) * dx) + 8.0 * dx


def _wave_port(name: str, x: float, direction: str) -> mw.WavePort:
    return mw.WavePort(
        name,
        position=(x, 0.0, 0.0),
        size=(0.0, GUIDE_A, GUIDE_B),
        direction=direction,
        reference_plane=x,
        modes=(mw.WaveModeSpec("te", polarization="Ez"),),
    )


def lossy_waveguide_scene(
    *,
    guide_len: float = SHORT_LENGTH,
    dx: float = 0.0025,
    sigma: float = WALL_CONDUCTIVITY,
    pec_walls: bool = False,
    device: str = "cuda",
) -> mw.Scene:
    """Build a two-port lossy-wall (or PEC-wall) rectangular waveguide of ``guide_len``."""

    domain_x = 0.5 * guide_len + 0.03
    half = 0.5 * guide_len
    wl = wall_length(dx, guide_len)
    metal = mw.Material.pec() if pec_walls else mw.LossyMetalMedium(conductivity=sigma)
    yb, zb, tw = 0.5 * GUIDE_A, 0.5 * GUIDE_B, WALL_THICKNESS
    # All four walls carry the SAME material instance so shared corner edges resolve
    # by owner rank (not the different-material conflicting-owner rejection).
    walls = (
        ((0.0, yb + 0.5 * tw, 0.0), (wl, tw, 2.0 * DOMAIN_Z), "wall_y_high"),
        ((0.0, -yb - 0.5 * tw, 0.0), (wl, tw, 2.0 * DOMAIN_Z), "wall_y_low"),
        ((0.0, 0.0, zb + 0.5 * tw), (wl, 2.0 * yb, tw), "wall_z_high"),
        ((0.0, 0.0, -zb - 0.5 * tw), (wl, 2.0 * yb, tw), "wall_z_low"),
    )
    structures = tuple(
        mw.Box(position=position, size=size).with_material(metal, name=name)
        for position, size, name in walls
    )
    ports = (_wave_port("left", -half, "+"), _wave_port("right", half, "-"))
    return mw.Scene(
        domain=mw.Domain(bounds=((-domain_x, domain_x), (-DOMAIN_Y, DOMAIN_Y), (-DOMAIN_Z, DOMAIN_Z))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=_pml_layers(dx)),
        ports=ports,
        structures=structures,
        device=device,
    )


# In-band design frequencies (all single-mode TE10: fc < f < 2 fc). Each is run as a
# single-frequency two-port sweep so the SIBC surface resistance is set exactly at the
# analytic-comparison frequency (the narrowband model evaluates R_s at the source
# frequency).
def design_frequencies() -> tuple[float, ...]:
    return (2.0e9, 2.4e9, 2.8e9)


# External-reference cross-check geometry. A single TE10 ModeSource launches into the
# lossy guide; two forward ModeMonitors ``ref_in`` / ``ref_out`` a known distance apart
# both DOWNSTREAM of the source give the attenuation from the forward-mode decay in ONE
# run: ``alpha_ref = ln(|amp_in| / |amp_out|) / REF_LENGTH``. This is normalization-
# independent (both amplitudes come from the same monitor family) and, unlike the FDTD
# two-line bench, needs a single export -- so one authorized cloud job cross-checks alpha
# directly. Mirrors ``rectangular_waveguide_reference_scene``.
REF_GUIDE_LENGTH = 1.0
REF_SOURCE_X = -0.40
REF_IN_X = -0.30
REF_OUT_X = 0.30
REF_LENGTH = REF_OUT_X - REF_IN_X


def lossy_waveguide_reference_scene(
    *,
    dx: float = 0.0025,
    sigma: float = WALL_CONDUCTIVITY,
    frequencies: tuple[float, ...] | None = None,
    device: str = "cpu",
) -> mw.Scene:
    """TE10 ModeSource-driven lossy guide for the external-reference attenuation cross-check.

    Identical good-conductor surface-impedance walls as the two-port bench, but excited
    by a single TE10 ``ModeSource`` (which the interoperability adapter exports as a
    genuine reference modal launch) with forward ``ModeMonitor`` planes at ``REF_IN_X`` /
    ``REF_OUT_X``. The reference solver reports the forward mode amplitude at each plane;
    the log-magnitude decay over ``REF_LENGTH`` is the conductor attenuation alpha.
    """
    freqs = tuple(frequencies) if frequencies is not None else design_frequencies()
    center = 0.5 * (freqs[0] + freqs[-1])
    fwidth = max(freqs[-1] - freqs[0], 0.3 * center)

    wl = wall_length(dx, REF_GUIDE_LENGTH)
    domain_x = 0.5 * REF_GUIDE_LENGTH + 0.03
    yb, zb, tw = 0.5 * GUIDE_A, 0.5 * GUIDE_B, WALL_THICKNESS
    metal = mw.LossyMetalMedium(conductivity=sigma)
    walls = (
        ((0.0, yb + 0.5 * tw, 0.0), (wl, tw, 2.0 * DOMAIN_Z), "wall_y_high"),
        ((0.0, -yb - 0.5 * tw, 0.0), (wl, tw, 2.0 * DOMAIN_Z), "wall_y_low"),
        ((0.0, 0.0, zb + 0.5 * tw), (wl, 2.0 * yb, tw), "wall_z_high"),
        ((0.0, 0.0, -zb - 0.5 * tw), (wl, 2.0 * yb, tw), "wall_z_low"),
    )
    structures = tuple(
        mw.Box(position=position, size=size).with_material(metal, name=name)
        for position, size, name in walls
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-domain_x, domain_x), (-DOMAIN_Y, DOMAIN_Y), (-DOMAIN_Z, DOMAIN_Z))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=_pml_layers(dx)),
        structures=structures,
        device=device,
    )
    scene.add_source(
        mw.ModeSource(
            position=(REF_SOURCE_X, 0.0, 0.0),
            size=(0.0, GUIDE_A, GUIDE_B),
            mode_index=0,
            direction="+",
            polarization="Ez",
            source_time=mw.GaussianPulse(frequency=center, fwidth=fwidth),
            name="te10_source",
        )
    )
    for name, x in (("ref_in", REF_IN_X), ("ref_out", REF_OUT_X)):
        scene.add_monitor(
            mw.ModeMonitor(
                name,
                position=(x, 0.0, 0.0),
                size=(0.0, GUIDE_A, GUIDE_B),
                mode_index=0,
                direction="+",
                polarization="Ez",
                frequencies=freqs,
            )
        )
    return scene
