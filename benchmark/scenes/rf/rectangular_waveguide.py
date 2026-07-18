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
"""

from __future__ import annotations

import witwin.maxwell as mw

# Interior cross-section (between the inner PEC wall faces): a (y) x b (z).
GUIDE_A = 0.60
GUIDE_B = 0.30
GUIDE_LENGTH = 0.60  # port-to-port separation along x


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
    walls = (
        ((0.0, 0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_high"),
        ((0.0, -0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_low"),
        ((0.0, 0.0, 0.25), (1.0, 0.60, 0.20), "wall_z_high"),
        ((0.0, 0.0, -0.25), (1.0, 0.60, 0.20), "wall_z_low"),
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
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.5, 0.5), (-0.35, 0.35))),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=4),
        ports=tuple(ports),
        structures=tuple(structures),
        device=device,
    )
