"""Hollow rectangular waveguide two-port scene (TE10).

Two ``WavePort`` mode ports terminate a PEC-walled air guide. A real FDTD sweep
launches the TE10 mode at each port; the extracted network S-matrix, modal
propagation constant, and characteristic impedance are compared against the
exact analytic waveguide references::

    fc     = c / (2 a)                       (TE10 cutoff)
    beta   = sqrt(k0^2 - (pi/a)^2)           (propagation constant)
    Z_TE10 = eta0 * k0 / beta                (wave impedance)

The guide interior width ``a`` is the y-extent between the PEC side walls and
``b`` is the z-extent between the top/bottom walls. The default geometry keeps
the operating frequency above cutoff so the TE10 mode is propagating.

Termination (audit S1 round-4, EXECUTED): the FDTD grid appends the PML nodes
OUTSIDE the declared domain bounds, so the walls must run ``2*(DOMAIN_X +
num_layers*dx)`` to cross the PML to the padded grid edges; ``2*DOMAIN_X`` (rounds
2/3) ended them at the PML interface (verified against the prepared x-grid: the
grid spans ``+-0.8`` while ``DOMAIN_X = 0.6``). NOTE (round-4 correction): unlike
the coax, terminating the waveguide does NOT by itself fix the passive-port
illumination -- the ~0.4 a_passive/a_driven floor persists with the walls run
through the PML (executed: 0.419 -> 0.396). That floor is a symptom of a
CONTAMINATED TE10 mode template (a checkerboard-aliased eigenvector injected in
place of sin(pi y/a)); the real waveguide defect is in the mode selector
(witwin/maxwell/fdtd/excitation/modes.py), not the termination.
"""

from __future__ import annotations

import witwin.maxwell as mw

# Interior cross-section (between the inner PEC wall faces): a (y) x b (z).
GUIDE_A = 0.60
GUIDE_B = 0.30
GUIDE_LENGTH = 0.60  # port-to-port separation along x
DOMAIN_X = 0.6       # x half-extent (declared bounds; the grid pads the PML beyond this)
# Hold the PML physical thickness FIXED IN METERS across grid tiers so the guide
# termination geometry is identical at every dx (audit S1: with a fixed layer
# count the PML thickness shrank as dx fell, moving the open guide end out of the
# absorber and injecting a dx-dependent standing-wave ripple). num_layers scales
# with 1/dx: 0.2 m -> 4 / 8 / 10 layers at dx = 0.05 / 0.025 / 0.02.
PML_THICKNESS_M = 0.2


def _pml_layers(dx: float) -> int:
    return max(int(round(PML_THICKNESS_M / dx)), 4)


def wall_length(dx: float) -> float:
    """Wall length that runs THROUGH the PML to the padded grid edges (F4).

    The prepared grid appends ``num_layers*dx`` beyond the declared ``+-DOMAIN_X``,
    so the walls must span ``2*(DOMAIN_X + num_layers*dx)`` plus a margin to reach
    the outermost grid nodes; verified against the prepared PEC occupancy.
    """
    return 2.0 * (DOMAIN_X + _pml_layers(dx) * dx) + 8.0 * dx


# Nominal wall length at the default tier for callers that import a constant.
WALL_LENGTH = wall_length(0.05)


def _wave_port(name: str, x: float, direction: str) -> mw.WavePort:
    return mw.WavePort(
        name,
        position=(x, 0.0, 0.0),
        size=(0.0, GUIDE_A, GUIDE_B),
        direction=direction,
        reference_plane=x,
        modes=(mw.WaveModeSpec("te", polarization="Ez"),),
    )


def rectangular_waveguide_scene(
    *,
    dx: float = 0.05,
    short_far_port: bool = False,
    device: str = "cuda",
) -> mw.Scene:
    """Build a two-port (or shorted one-port) hollow rectangular waveguide."""

    half = 0.5 * GUIDE_LENGTH
    ports = [_wave_port("left", -half, "+")]
    structures = []
    # Walls run THROUGH the PML to the padded grid edges (F4); see wall_length.
    wl = wall_length(dx)
    walls = (
        ((0.0, 0.40, 0.0), (wl, 0.20, 0.70), "wall_y_high"),
        ((0.0, -0.40, 0.0), (wl, 0.20, 0.70), "wall_y_low"),
        ((0.0, 0.0, 0.25), (wl, 0.60, 0.20), "wall_z_high"),
        ((0.0, 0.0, -0.25), (wl, 0.60, 0.20), "wall_z_low"),
    )
    for position, size, name in walls:
        structures.append(
            mw.Box(position=position, size=size).with_material(
                mw.Material.pec(), name=name
            )
        )
    if short_far_port:
        structures.append(
            mw.Box(position=(half, 0.0, 0.0), size=(dx, GUIDE_A, GUIDE_B)).with_material(
                mw.Material.pec(), name="short_termination"
            )
        )
    else:
        ports.append(_wave_port("right", half, "-"))

    return mw.Scene(
        domain=mw.Domain(bounds=((-DOMAIN_X, DOMAIN_X), (-0.5, 0.5), (-0.35, 0.35))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=_pml_layers(dx)),
        ports=tuple(ports),
        structures=tuple(structures),
        device=device,
    )
