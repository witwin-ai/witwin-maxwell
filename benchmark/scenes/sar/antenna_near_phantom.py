"""Center-fed half-wave dipole radiating into a nearby tissue block.

An end-to-end exposure scene: a PEC dipole with a lumped wire-gap feed radiates
into a homogeneous tissue block placed a fixed distance broadside of the antenna.
Driving the feed and normalizing to a 1 W accepted port power turns the run into
the full SAR chain -- accepted power -> point SAR -> 1 g / 10 g peaks ->
save/load -- on a compact grid.

The dipole arms are PEC boxes with a coordinate-defined :class:`LumpedPort` gap
rather than thin wires. NOTE (recorded design blocker): the current FDTD build
does not support a driven port alongside a conductive (lossy) background. Both
the thin-wire runtime ("Thin-wire FDTD does not yet support a conductive
background electric update") and the lumped-port runtime ("Lumped FDTD coupling
in conductive media requires a conductance-aware port update coefficient") fail
closed on a lossy medium, and a tissue phantom is conductive by construction.
Preparing this scene therefore raises ``NotImplementedError`` today. The scene
and its fail-closed behaviour are shipped and gated as the documented blocker;
the accepted-power -> SAR -> 1 g/10 g -> save/load chain is validated on
synthetic ports and on the plane-wave phantom scenes. Unblocking requires a
conductance-aware lumped-port update coefficient (out of this stage's scope).

Redistributable canonical geometry only (PEC boxes plus a homogeneous tissue
box); tissue values are the published-class numbers in ``_tissue.py``. The grid
is deliberately compact: this scene exercises the end-to-end pipeline and its
determinism, not a converged absolute dosimetry number (convergence is a
separate grid study).
"""

from __future__ import annotations

import witwin.maxwell as mw

from benchmark.models import ScenarioDefinition
from benchmark.scenes.sar._tissue import tissue_material

C0 = 299792458.0

DESIGN_FREQUENCY = 3.0e9
FREQUENCIES = (DESIGN_FREQUENCY,)

PML_LAYERS = 8

# Tissue block geometry: broadside (+x) of the dipole, a fixed physical gap from
# the dipole axis.
BLOCK_GAP = 0.010
BLOCK_SIZE = (0.020, 0.030, 0.030)


def dipole_length() -> float:
    """Half-wavelength dipole total length at the design frequency."""

    return 0.5 * C0 / DESIGN_FREQUENCY


def block_center_x() -> float:
    return BLOCK_GAP + 0.5 * BLOCK_SIZE[0]


def build_scene(*, dx: float = 0.0025, device: str = "cuda") -> mw.Scene:
    total = dipole_length()
    gap = 2.0 * dx
    arm_length = 0.5 * (total - gap)
    arm_width = 2.0 * dx
    arm_center_z = 0.5 * gap + 0.5 * arm_length

    feed = mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.5 * gap),
        negative=(0.0, 0.0, -0.5 * gap),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, 0.0), size=(arm_width, arm_width, 0.0)
        ),
        reference_impedance=73.0,
    )
    arms = tuple(
        mw.Structure(
            name=f"arm_{sign}",
            geometry=mw.Box(
                position=(0.0, 0.0, sign * arm_center_z),
                size=(arm_width, arm_width, arm_length),
            ),
            material=mw.Material.pec(),
        )
        for sign in (-1.0, 1.0)
    )

    cx = block_center_x()
    block = mw.Structure(
        geometry=mw.Box(position=(cx, 0.0, 0.0), size=BLOCK_SIZE),
        material=tissue_material("phantom"),
        name="phantom",
    )

    domain_half_x = cx + 0.5 * BLOCK_SIZE[0] + 16 * dx
    domain_half_z = 0.5 * total + 16 * dx
    domain_half_y = max(0.5 * BLOCK_SIZE[1], arm_width) + 16 * dx
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-16 * dx, domain_half_x),
                (-domain_half_y, domain_half_y),
                (-domain_half_z, domain_half_z),
            )
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        ports=(feed,),
        structures=(*arms, block),
        device=device,
    )
    scene.add_monitor(
        mw.PowerLossMonitor(
            "loss",
            position=(cx, 0.0, 0.0),
            size=BLOCK_SIZE,
            frequencies=FREQUENCIES,
            channels=("conduction",),
        )
    )
    return scene


SCENARIO = ScenarioDefinition(
    name="sar_antenna_near_phantom",
    description="Half-wave dipole radiating into a nearby tissue block (end-to-end SAR chain)",
    builder=build_scene,
    frequencies=FREQUENCIES,
    display_monitor="loss",
    display_component="Ex",
)
