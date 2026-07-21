"""Probe-fed rectangular microstrip patch antenna on a finite grounded slab.

A rectangular PEC patch sits on a finite dielectric slab above a finite PEC
ground plane; a vertical lumped probe feeds it at an inset point on the resonant
(``x``) dimension. The whole finite structure is enclosed by a
:class:`ClosedSurfaceMonitor` box that lies in the homogeneous air exterior, so
the near-field-to-far-field transform (and therefore :meth:`Result.antenna`) is
well posed -- an infinite substrate that runs into the PML would leave no
homogeneous Huygens surface.

Grid discipline: the substrate height, patch, ground, feed inset, and domain are
all defined in integer grid cells and the axis coordinates are built as
``arange(...) * dx`` (via :func:`GridSpec.custom`) so the ground plane (``z=0``),
the patch underside (``z=h``), and the feed terminals land EXACTLY on Yee nodes.
``GridSpec.uniform`` cannot guarantee this for arbitrary metric bounds because its
cell count is a float ``ceil`` that overshoots by one cell for lengths whose
``length/dx`` is a hair above an integer in double precision.

References (cavity model, dominant ``TM010`` mode):

    eps_eff = (eps_r+1)/2 + (eps_r-1)/2 * (1 + 12 h / W)^(-1/2)
    dL      = 0.412 h (eps_eff+0.3)(W/h+0.264) / ((eps_eff-0.258)(W/h+0.8))
    f_r     = c / (2 (L + 2 dL) sqrt(eps_eff))

Feed (F2b): a galvanic PEC probe via now carries the feed current from the patch
underside down to a single-cell lumped-port gap above the ground plane (the retired
design drove the whole substrate gap with no metal probe and coupled only
capacitively). The via cut the feed reactance ~5x, but the patch still does NOT
resonate at feasible resolution: a wide-band sweep shows ``Re(Zin) < 4`` Ohm with no
resonance peak across 2-8 GHz and a purely capacitive reactance -- the lumped gap is
still capacitively shorted by the adjacent PEC, so the TM010 cavity mode is not
galvanically excited, and the radiation is a broadside-null monopole pattern from the
via. The cavity resonance (~3.39 GHz) also sits below the historically driven band,
and the finite ground extends only ~0.07 lambda beyond the patch. A matched broadside
``TM010`` (``D >= 5`` dBi, ``|Gamma|`` dip at resonance) needs a wire-bound clean-gap
probe feed, a larger finite ground, and an on-resonance drive -- a multi-run antenna
co-design deferred with the strict-xfail gate kept fail-closed (see
``docs/assessments/f2-rf-trio-acceptance-2026-07-21.md``). The builder's contract here
remains a valid radiating structure that exercises the real ``Result.antenna``
pipeline.
"""

from __future__ import annotations

import math

import numpy as np

import witwin.maxwell as mw

C0 = 299792458.0

# Integer-cell geometry (multiplied by dx to land on exact Yee nodes).
SUBSTRATE_HEIGHT_CELLS = 3
PATCH_LENGTH_CELLS = 28          # resonant dimension, x
PATCH_WIDTH_CELLS = 34           # non-resonant dimension, y
GROUND_MARGIN_CELLS = 6          # finite ground/substrate extension beyond patch
FEED_INSET_CELLS = 8             # feed offset from patch centre toward the -x edge
DOMAIN_HALF_X_CELLS = 35
DOMAIN_HALF_Y_CELLS = 40
DOMAIN_Z_LO_CELLS = -20
DOMAIN_Z_HI_CELLS = 33
PML_LAYERS = 8
DEFAULT_PERMITTIVITY = 2.2


def effective_permittivity(eps_r: float, height: float, width: float) -> float:
    return 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) / math.sqrt(1.0 + 12.0 * height / width)


def cavity_resonance(
    *, eps_r: float, height: float, length: float, width: float
) -> float:
    """Dominant ``TM010`` resonance with the edge-fringing length extension."""

    eps_eff = effective_permittivity(eps_r, height, width)
    delta_l = (
        0.412
        * height
        * (eps_eff + 0.3)
        * (width / height + 0.264)
        / ((eps_eff - 0.258) * (width / height + 0.8))
    )
    return C0 / (2.0 * (length + 2.0 * delta_l) * math.sqrt(eps_eff))


def default_frequencies(design_frequency: float) -> tuple[float, ...]:
    base = float(design_frequency)
    return tuple(base * scale for scale in (0.55, 0.68, 0.81, 0.94, 1.06, 1.19, 1.32, 1.45))


def patch_antenna_scene(
    *,
    dx: float = 1.0e-3,
    eps_r: float = DEFAULT_PERMITTIVITY,
    frequencies: tuple[float, ...] | None = None,
    device: str = "cuda",
) -> mw.Scene:
    """Build a probe-fed rectangular patch on a finite grounded dielectric slab."""

    height = SUBSTRATE_HEIGHT_CELLS * dx
    length = PATCH_LENGTH_CELLS * dx
    width = PATCH_WIDTH_CELLS * dx
    footprint_half_x = (0.5 * PATCH_LENGTH_CELLS + GROUND_MARGIN_CELLS) * dx
    footprint_half_y = (0.5 * PATCH_WIDTH_CELLS + GROUND_MARGIN_CELLS) * dx
    feed_x = -FEED_INSET_CELLS * dx

    resonance = cavity_resonance(
        eps_r=eps_r, height=height, length=length, width=width
    )
    resolved_frequencies = (
        default_frequencies(resonance)
        if frequencies is None
        else tuple(float(freq) for freq in frequencies)
    )

    x_nodes = np.arange(-DOMAIN_HALF_X_CELLS, DOMAIN_HALF_X_CELLS + 1) * dx
    y_nodes = np.arange(-DOMAIN_HALF_Y_CELLS, DOMAIN_HALF_Y_CELLS + 1) * dx
    z_nodes = np.arange(DOMAIN_Z_LO_CELLS, DOMAIN_Z_HI_CELLS + 1) * dx
    grid = mw.GridSpec.custom(x_nodes, y_nodes, z_nodes)
    domain = mw.Domain(
        bounds=(
            (float(x_nodes[0]), float(x_nodes[-1])),
            (float(y_nodes[0]), float(y_nodes[-1])),
            (float(z_nodes[0]), float(z_nodes[-1])),
        )
    )

    substrate = mw.Box(
        position=(0.0, 0.0, 0.5 * height),
        size=(2.0 * footprint_half_x, 2.0 * footprint_half_y, height),
    ).with_material(mw.Material(eps_r=eps_r), name="substrate")
    ground = mw.Box(
        position=(0.0, 0.0, -0.5 * dx),
        size=(2.0 * footprint_half_x, 2.0 * footprint_half_y, dx),
    ).with_material(mw.Material.pec(), name="ground")
    patch = mw.Box(
        position=(0.0, 0.0, height + 0.5 * dx),
        size=(length, width, dx),
    ).with_material(mw.Material.pec(), name="patch")
    # Galvanic coaxial-probe feed: a PEC via carries the feed current from the patch
    # underside (z = height) down to one cell above the ground plane, and the LumpedPort
    # drives the single-cell gap between the ground (z = 0) and the via base (z = dx).
    # Driving the full substrate gap with no metal probe (the retired design) couples only
    # capacitively -- Re(Zin) stays ~0.5 Ohm and the cavity mode is never excited. The via
    # gives the galvanic coupling that lets the patch resonate and present its radiation
    # resistance at the feed.
    probe = mw.Box(
        position=(feed_x, 0.0, 0.5 * (dx + height)),
        size=(dx, dx, height - dx),
    ).with_material(mw.Material.pec(), name="probe")
    feed = mw.LumpedPort(
        "feed",
        positive=(feed_x, 0.0, dx),
        negative=(feed_x, 0.0, 0.0),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(feed_x, 0.0, 0.5 * dx), size=(dx, dx, 0.0)
        ),
        reference_impedance=50.0,
    )

    box_half_x = (0.5 * PATCH_LENGTH_CELLS + GROUND_MARGIN_CELLS + 6) * dx
    box_half_y = (0.5 * PATCH_WIDTH_CELLS + GROUND_MARGIN_CELLS + 7) * dx
    box_z_lo = (DOMAIN_Z_LO_CELLS + 10) * dx
    box_z_hi = (DOMAIN_Z_HI_CELLS - 10) * dx
    surface = mw.ClosedSurfaceMonitor.box(
        "radiation",
        position=(0.0, 0.0, 0.5 * (box_z_lo + box_z_hi)),
        size=(2.0 * box_half_x, 2.0 * box_half_y, box_z_hi - box_z_lo),
        frequencies=resolved_frequencies,
    )

    scene = mw.Scene(
        domain=domain,
        grid=grid,
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        device=device,
    )
    for structure in (substrate, ground, patch, probe):
        scene.add_structure(structure)
    scene.add_port(feed)
    scene.add_monitor(surface)
    return scene
