"""Coupled-line differential pair scene (mixed-mode, four-port).

Two parallel microstrip conductors over a common grounded substrate form a
coupled transmission line. Four ``WavePort`` quasi-TEM ports (two per line) allow
extraction of the 4-port single-ended S matrix, which converts to the mixed-mode
(differential/common) representation. The analytic reference is the weak-coupling
coupled-line model (even/odd-mode impedances); mixed-mode conversion and
mode-conversion terms (Sdc / Scd) are the validation targets.

Wave-level bench (F2b): each single-ended port aperture spans one strip plus the
grounded reference, so its inhomogeneous quasi-TEM mode is solved by the
quasi-static electrostatic line-mode engine (the same path that unblocked
``microstrip_two_port``). Two structural changes unblock the extraction, following
the ``coax_thru`` precedent: (1) the four measurement ports sit near the origin
(``+-PORT_X``) so the current-contour planes stay on the Yee half-grid in single
precision, and (2) the ground/substrate/strips run THROUGH the PML to the padded
grid edges so the launched waves terminate. All transverse dimensions are integer
grid cells so the strips, gap and contours land on Yee nodes/half-nodes.
"""

from __future__ import annotations

import numpy as np

import witwin.maxwell as mw

SUBSTRATE_EPS = 4.4
SUBSTRATE_H = 0.020      # 4 cells at dx=0.005
STRIP_W = 0.025         # 5 cells
STRIP_GAP = 0.015       # edge-to-edge separation between the two strips (y), 3 cells
GROUND_W = 0.24
DOMAIN_X = 0.12
PORT_X = 0.02
PML_LAYERS = 6

_STRIP_OFFSET = 0.5 * (STRIP_W + STRIP_GAP)  # strip center |y| = 0.02 (a node)


def line_length(dx: float) -> float:
    """Conductor length that runs THROUGH the PML to the padded grid edges."""
    return 2.0 * (DOMAIN_X + PML_LAYERS * dx) + 8.0 * dx


LINE_LENGTH = line_length(0.005)


def _strip_port(name: str, x: float, y: float, direction: str, *, dx: float) -> mw.WavePort:
    contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
    # Odd half-cell extents keep the contour edges on the Yee half-grid (the contour
    # centre y, z sit on nodes). The loop encloses one strip (not the ground, not the
    # neighbouring strip).
    contour_hy = 3.5 * dx   # >= STRIP_W/2, stays clear of the neighbour strip
    contour_hz = 1.5 * dx
    mode = mw.WaveModeSpec(
        "tem",
        polarization="Ez",
        voltage_path=((x, y, SUBSTRATE_H), (x, y, 0.0)),
        current_contour=mw.Box(
            position=(contour_x, y, SUBSTRATE_H),
            size=(0.0, 2.0 * contour_hy, 2.0 * contour_hz),
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

    length = line_length(dx)
    # Integer-cell node arrays (arange * dx) so the strips, gap, ports and contours land
    # on exact Yee nodes/half-nodes; GridSpec.uniform overshoots the y-span by one cell
    # (its float ceil), shifting the effective spacing off dx.
    nx = round(DOMAIN_X / dx)
    ny = round(0.14 / dx)
    z_lo_cells = round(-0.03 / dx)
    z_hi_cells = round(0.10 / dx)
    x_nodes = np.arange(-nx, nx + 1) * dx
    y_nodes = np.arange(-ny, ny + 1) * dx
    z_nodes = np.arange(z_lo_cells, z_hi_cells + 1) * dx
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (float(x_nodes[0]), float(x_nodes[-1])),
                (float(y_nodes[0]), float(y_nodes[-1])),
                (float(z_nodes[0]), float(z_nodes[-1])),
            )
        ),
        grid=mw.GridSpec.custom(x_nodes, y_nodes, z_nodes),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        ports=(
            _strip_port("p1", -PORT_X, +_STRIP_OFFSET, "+", dx=dx),
            _strip_port("p2", -PORT_X, -_STRIP_OFFSET, "+", dx=dx),
            _strip_port("p3", PORT_X, +_STRIP_OFFSET, "-", dx=dx),
            _strip_port("p4", PORT_X, -_STRIP_OFFSET, "-", dx=dx),
        ),
        device=device,
    )
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, -0.5 * dx), size=(length, GROUND_W, dx)).with_material(
            mw.Material.pec(), name="ground"
        )
    )
    scene.add_structure(
        mw.Box(
            position=(0.0, 0.0, 0.5 * SUBSTRATE_H),
            size=(length, GROUND_W, SUBSTRATE_H),
        ).with_material(mw.Material(eps_r=SUBSTRATE_EPS), name="substrate")
    )
    for sign, name in ((+1.0, "strip_pos"), (-1.0, "strip_neg")):
        scene.add_structure(
            mw.Box(
                position=(0.0, sign * _STRIP_OFFSET, SUBSTRATE_H + 0.5 * dx),
                size=(length, STRIP_W, dx),
            ).with_material(mw.Material.pec(), name=name)
        )
    return scene
