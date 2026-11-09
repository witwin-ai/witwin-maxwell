"""Center-fed half-wave dipole radiating into a nearby tissue block.

An end-to-end exposure scene: a PEC dipole with a lumped wire-gap feed radiates
into a homogeneous tissue block placed a fixed distance broadside of the antenna.
Driving the feed and normalizing to a 1 W accepted port power turns the run into
the full SAR chain -- accepted power -> point SAR -> 1 g / 10 g peaks ->
save/load -- on a compact grid.

The dipole arms are PEC boxes with a coordinate-defined :class:`LumpedPort` gap.
The conductive-material port update supports this functional workflow. Explicit
node coordinates preserve the physical domain and feed endpoints; the native
magnetic contour encloses the arms on Yee half-grid planes.

Redistributable canonical geometry only (PEC boxes plus a homogeneous tissue
box); tissue values are the published-class numbers in ``_tissue.py``. The grid
is deliberately compact: this scene exercises the end-to-end pipeline and its
determinism, not a converged absolute dosimetry number (convergence is a
separate grid study).
"""

from __future__ import annotations

import math
import numpy as np
import torch
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


def _axis_nodes(low,high,dx):
    """Keep the physical endpoints and exact integer-multiple feed grid."""
    coordinates = np.arange(math.ceil(low/dx),math.floor(high/dx)+1,dtype=np.float64)*dx
    roundoff = 16*np.finfo(np.float64).eps*max(abs(low),abs(high),dx)
    return np.concatenate(([low],coordinates[(coordinates > low+roundoff) & (coordinates < high-roundoff)],[high]))


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
            position=torch.tensor((0.0,0.0,-.5*dx),dtype=torch.float64),
            size=torch.tensor((3*dx,3*dx,0.0),dtype=torch.float64),
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
    bounds = ((-16*dx,domain_half_x),(-domain_half_y,domain_half_y),(-domain_half_z,domain_half_z))
    scene = mw.Scene(
        domain=mw.Domain(bounds=bounds),
        grid=mw.GridSpec.custom(*(_axis_nodes(low,high,dx) for low,high in bounds)),
        boundary=mw.BoundarySpec.pml(num_layers=PML_LAYERS),
        ports=(feed,),
        structures=(*arms, block),
        device=device,
        metadata={"qualification":"functional exposure fixture; absolute dosimetry convergence and reference validation pending",
            "tissue_values":"unchanged canonical 900 MHz catalogue values used as fixed benchmark coefficients at 3 GHz"},
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
