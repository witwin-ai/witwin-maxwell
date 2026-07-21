"""Microstrip line two-port scene (quasi-TEM).

A metal strip over a grounded dielectric substrate. The Hammerstad-Jensen
closed-form ``Z0`` / ``eps_eff`` are the analytic references.

Wave-level bench (F2b): the quasi-TEM mode on the inhomogeneous (substrate + air)
cross-section is now solved by the quasi-static electrostatic line-mode engine
(``eps_eff = C/C0``; see ``witwin/maxwell/fdtd/excitation/modes.py``), so
``WaveModeSpec('tem')`` is no longer categorically blocked here. Two structural
changes unblock the two-port FDTD extraction, both following the ``coax_thru``
precedent:

1. The two measurement ports sit NEAR THE ORIGIN (``+-PORT_X``), not at the line
   ends. Large port coordinates truncated the current-contour plane below the
   half-grid snap tolerance in single precision (a ``+-0.15`` contour rounds by
   ~6.6e-9 > the ~5e-9 grid tolerance); a small ``+-PORT_X`` keeps the contour on
   the Yee half-grid at every tier.
2. The ground plane, substrate and strip run THROUGH the computational PML to the
   padded grid edges (``line_length``), so the launched quasi-TEM wave is absorbed
   instead of reflecting off an open line end into the passive port.

The network S is assembled by solving ``B = S*A`` across the drive columns and is
gated (coax precedent) on extraction conditioning ``cond(A)`` plus post-solve
passivity; ``beta`` from ``arg(S21)/L`` is compared against the analytic
Hammerstad-Jensen quasi-TEM phase constant ``beta = k0 sqrt(eps_eff)``.
"""

from __future__ import annotations

import math

import numpy as np

import witwin.maxwell as mw

C0 = 299792458.0

SUBSTRATE_EPS = 4.4
SUBSTRATE_H = 0.020   # substrate thickness (z), 4 cells at dx=0.005
STRIP_W = 0.030       # strip width (y), 6 cells at dx=0.005
GROUND_W = 0.20       # ground/substrate transverse width (y)
DOMAIN_X = 0.12       # half-length of the declared x-domain (line runs through the PML)
PORT_X = 0.02         # measurement ports at +-PORT_X, near the origin
PML_LAYERS = 6


def line_length(dx: float) -> float:
    """Conductor length that runs THROUGH the PML to the padded grid edges.

    The prepared grid appends the PML nodes OUTSIDE the declared ``+-DOMAIN_X``
    (``scene._build_axis_grid64`` extends the bounds by ``PML_LAYERS*dx`` per side).
    To terminate the microstrip the ground/substrate/strip must span the full padded
    extent; a few-cell margin guarantees they reach the outermost grid nodes at every
    tier (verify against the prepared PEC occupancy, not this constant).
    """
    return 2.0 * (DOMAIN_X + PML_LAYERS * dx) + 8.0 * dx


# Nominal length at the default tier for callers that import a constant.
LINE_LENGTH = line_length(0.005)


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
    # The current contour is a rectangular loop around the signal strip. Its edges
    # must land on the Yee transverse half-grid; with the contour centred on a node
    # (y=0, z=SUBSTRATE_H) an ODD half-cell extent (2*(k+0.5)*dx) puts every edge on
    # a half-grid line at any tier. The loop encloses only the strip (not the ground).
    contour_hy = 4.5 * dx   # >= STRIP_W/2, encloses the strip in y
    contour_hz = 1.5 * dx   # brackets the strip just above the substrate
    # Quasi-TEM voltage path from strip (top of substrate) down to ground.
    mode = mw.WaveModeSpec(
        "tem",
        polarization="Ez",
        voltage_path=((x, 0.0, SUBSTRATE_H), (x, 0.0, 0.0)),
        current_contour=mw.Box(
            position=(contour_x, 0.0, SUBSTRATE_H),
            size=(0.0, 2.0 * contour_hy, 2.0 * contour_hz),
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

    length = line_length(dx)
    # Integer-cell node arrays (arange * dx) so every conductor face, port plane and
    # contour lands on an exact Yee node/half-node. GridSpec.uniform cannot guarantee
    # this: its cell count is a float ceil that overshoots by one cell for spans whose
    # length/dx is a hair above an integer, shifting the effective spacing off dx.
    nx = round(DOMAIN_X / dx)
    ny = round(0.10 / dx)
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
            _strip_port("in", -PORT_X, "+", dx=dx),
            _strip_port("out", PORT_X, "-", dx=dx),
        ),
        device=device,
    )
    # Ground plane at z=0.
    scene.add_structure(
        mw.Box(position=(0.0, 0.0, -0.5 * dx), size=(length, GROUND_W, dx)).with_material(
            mw.Material.pec(), name="ground"
        )
    )
    # Dielectric substrate between ground and strip.
    scene.add_structure(
        mw.Box(
            position=(0.0, 0.0, 0.5 * SUBSTRATE_H),
            size=(length, GROUND_W, SUBSTRATE_H),
        ).with_material(mw.Material(eps_r=SUBSTRATE_EPS), name="substrate")
    )
    # Signal strip on top of the substrate.
    scene.add_structure(
        mw.Box(
            position=(0.0, 0.0, SUBSTRATE_H + 0.5 * dx),
            size=(length, STRIP_W, dx),
        ).with_material(mw.Material.pec(), name="strip")
    )
    return scene
