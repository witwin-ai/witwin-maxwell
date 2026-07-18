"""Microstrip line two-port scene (quasi-TEM).

A metal strip over a grounded dielectric substrate is driven through two
``WavePort`` quasi-TEM mode ports. The extracted characteristic impedance and
effective permittivity are compared against the Hammerstad-Jensen closed-form
microstrip references. Because microstrip is quasi-TEM (fringing fields split
between substrate and air), the analytic reference itself carries a few-percent
model uncertainty; the benchmark records the measured value against Hammerstad
with that modelling gap stated rather than asserting a tighter bound than the
quasi-static formula supports.
"""

from __future__ import annotations

import math

import witwin.maxwell as mw

C0 = 299792458.0

SUBSTRATE_EPS = 4.4
SUBSTRATE_H = 0.020   # substrate thickness (z)
STRIP_W = 0.030       # strip width (y)
LINE_LENGTH = 0.30    # port-to-port separation (x)
GROUND_W = 0.20


def analytic_microstrip(eps_r: float = SUBSTRATE_EPS, w: float = STRIP_W, h: float = SUBSTRATE_H):
    """Hammerstad-Jensen quasi-static eps_eff and Z0 (returns dict)."""

    u = w / h
    if u >= 1.0:
        eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 / u) ** -0.5
        z0 = (120.0 * math.pi / math.sqrt(eps_eff)) / (
            u + 1.393 + 0.667 * math.log(u + 1.444)
        )
    else:
        eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (
            (1.0 + 12.0 / u) ** -0.5 + 0.04 * (1.0 - u) ** 2
        )
        z0 = 60.0 / math.sqrt(eps_eff) * math.log(8.0 / u + u / 4.0)
    return {"eps_eff": eps_eff, "z0": z0}


def _strip_port(name: str, x: float, direction: str, *, dx: float) -> mw.WavePort:
    contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
    # Quasi-TEM voltage path from strip (top of substrate) down to ground.
    mode = mw.WaveModeSpec(
        "tem",
        polarization="Ez",
        voltage_path=((x, 0.0, SUBSTRATE_H), (x, 0.0, 0.0)),
        current_contour=mw.Box(
            position=(contour_x, 0.0, SUBSTRATE_H),
            size=(0.0, STRIP_W + 0.02, 0.02),
        ),
    )
    return mw.WavePort(
        name,
        position=(x, 0.0, 0.03),
        size=(0.0, 0.16, 0.08),
        direction=direction,
        reference_plane=x,
        modes=(mode,),
    )


def microstrip_two_port_scene(*, dx: float = 0.005, device: str = "cuda") -> mw.Scene:
    """Build a two-port grounded-substrate microstrip line."""

    half = 0.5 * LINE_LENGTH
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-half - 0.03, half + 0.03), (-0.10, 0.10), (-0.03, 0.10))
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(
            _strip_port("in", -half, "+", dx=dx),
            _strip_port("out", half, "-", dx=dx),
        ),
        device=device,
    )
    # Ground plane at z=0.
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, -0.5 * dx), size=(LINE_LENGTH, GROUND_W, dx)).with_material(
            mw.Material.pec(), name="ground"
        )
    )
    # Dielectric substrate between ground and strip.
    scene.add_structure(
        mw.Box(
            position=(0.0, 0.0, 0.5 * SUBSTRATE_H),
            size=(LINE_LENGTH, GROUND_W, SUBSTRATE_H),
        ).with_material(mw.Material(eps_r=SUBSTRATE_EPS), name="substrate")
    )
    # Signal strip on top of the substrate.
    scene.add_structure(
        mw.Box(
            position=(0.0, 0.0, SUBSTRATE_H + 0.5 * dx),
            size=(LINE_LENGTH, STRIP_W, dx),
        ).with_material(mw.Material.pec(), name="strip")
    )
    return scene
