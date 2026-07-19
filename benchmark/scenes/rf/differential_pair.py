"""Coupled-line differential pair scene (mixed-mode, four-port).

Two parallel microstrip conductors over a common grounded substrate form a
coupled transmission line. Four ``WavePort`` quasi-TEM ports (two per line)
allow extraction of the 4-port single-ended S matrix, which converts to the
mixed-mode (differential/common) representation. The analytic reference is the
weak-coupling coupled-line model (even/odd-mode impedances); mixed-mode
conversion and mode-conversion terms (Sdc / Scd) are the validation targets.

BLOCKED (audit S1). As with microstrip, the contour-snap ``ValueError`` fires
first and masks the mode solve; underneath it the coupled microstrip cross-section
is inhomogeneous (substrate + air), so all four ``WaveModeSpec('tem')`` ports hit
the same categorical TEM-inapplicability
(``NotImplementedError`` at ``modes.py:1943-1946``). A hybrid vector mode solve on
the coupled cross-section is required before any 4-port / mixed-mode extraction.
The validation runner records this scene as ``blocked`` with
``reference: pending-generation``.
"""

from __future__ import annotations

import witwin.maxwell as mw

SUBSTRATE_EPS = 4.4
SUBSTRATE_H = 0.020
STRIP_W = 0.024
STRIP_GAP = 0.016       # edge-to-edge separation between the two strips (y)
LINE_LENGTH = 0.30
GROUND_W = 0.24

_STRIP_OFFSET = 0.5 * (STRIP_W + STRIP_GAP)  # strip center |y|


def _strip_port(name: str, x: float, y: float, direction: str, *, dx: float) -> mw.WavePort:
    contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
    mode = mw.WaveModeSpec(
        "tem",
        polarization="Ez",
        voltage_path=((x, y, SUBSTRATE_H), (x, y, 0.0)),
        current_contour=mw.Box(
            position=(contour_x, y, SUBSTRATE_H),
            size=(0.0, STRIP_W + 0.01, 0.02),
        ),
    )
    return mw.WavePort(
        name,
        position=(x, y, 0.03),
        size=(0.0, STRIP_W + STRIP_GAP, 0.08),
        direction=direction,
        reference_plane=x,
        modes=(mode,),
    )


def differential_pair_scene(*, dx: float = 0.005, device: str = "cuda") -> mw.Scene:
    """Build a four-port coupled microstrip differential pair."""

    half = 0.5 * LINE_LENGTH
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-half - 0.03, half + 0.03), (-0.14, 0.14), (-0.03, 0.10))
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(
            _strip_port("p1", -half, +_STRIP_OFFSET, "+", dx=dx),
            _strip_port("p2", -half, -_STRIP_OFFSET, "+", dx=dx),
            _strip_port("p3", half, +_STRIP_OFFSET, "-", dx=dx),
            _strip_port("p4", half, -_STRIP_OFFSET, "-", dx=dx),
        ),
        device=device,
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, -0.5 * dx), size=(LINE_LENGTH, GROUND_W, dx)).with_material(
            mw.Material.pec(), name="ground"
        )
    )
    scene.add_structure(
        mw.Box(
            position=(0.0, 0.0, 0.5 * SUBSTRATE_H),
            size=(LINE_LENGTH, GROUND_W, SUBSTRATE_H),
        ).with_material(mw.Material(eps_r=SUBSTRATE_EPS), name="substrate")
    )
    for sign, name in ((+1.0, "strip_pos"), (-1.0, "strip_neg")):
        scene.add_structure(
            mw.Box(
                position=(0.0, sign * _STRIP_OFFSET, SUBSTRATE_H + 0.5 * dx),
                size=(LINE_LENGTH, STRIP_W, dx),
            ).with_material(mw.Material.pec(), name=name)
        )
    return scene
