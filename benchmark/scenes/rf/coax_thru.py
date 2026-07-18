"""Coaxial line two-port thru scene (TEM).

An air-filled round coaxial line (inner PEC rod, outer PEC shield) is driven
through two ``WavePort`` TEM mode ports. A real FDTD sweep launches the TEM mode
at each port; the extracted characteristic impedance and propagation constant are
compared against the exact analytic coax references::

    Z0   = eta0 / (2*pi) * ln(b / a)     (TEM characteristic impedance)
    beta = k0                            (TEM phase constant, air-filled)

with ``a`` the inner-conductor radius and ``b`` the shield inner radius. The
outer shield is a hollow PEC cylinder built from a signed-distance geometry (no
hollow-cylinder primitive exists in the public geometry set). The current-contour
half-width is snapped deterministically to the Yee transverse half-grid for the
requested ``dx`` (see :func:`snap_contour_half`, B5) so the scene builds across
refinement tiers instead of raising a snapping error.

Honest-exit note (audit S1): a genuine FDTD two-port sweep of this bench reflects
almost all incident power (|S11| ~ 1, gross non-passivity) because the TEM
WavePort does not launch/absorb a clean matched TEM wave on the round coax at
benchmark resolution, and the mirror-symmetric geometry makes reciprocity
trivial. The validation runner therefore records this scene as a wave-level FAIL
with the measured numbers; the modal-eigensolve ``Z0`` is kept only as supporting
evidence, never as the exit gate.
"""

from __future__ import annotations

import math

import torch
from witwin.core import GeometryBase

import witwin.maxwell as mw

ETA0 = 376.730313668

INNER_RADIUS = 0.04
OUTER_RADIUS = 0.16
SHIELD_OUTER = 0.20
LINE_LENGTH = 0.06
DOMAIN_TRANSVERSE = 0.205  # matches the proven float32-safe coax cross-section
CONTOUR_HALF = 0.10125     # 0.2025 / 2 -- proven float32-safe half-grid at dx=0.0025
PORT_X = 0.02
DOMAIN_X = 0.05


def analytic_z0() -> float:
    return ETA0 / (2.0 * math.pi) * math.log(OUTER_RADIUS / INNER_RADIUS)


class _HollowCylinder(GeometryBase):
    """Axial PEC tube (shield) between ``inner_radius`` and ``outer_radius``."""

    kind = "coax_shield"

    def __init__(self, *, position, inner_radius, outer_radius, length, device=None):
        super().__init__(position=position, device=device)
        self.inner_radius = float(inner_radius)
        self.outer_radius = float(outer_radius)
        self.length = float(length)

    def signed_distance(self, xx, yy, zz):
        dx, dy, dz = self._local_coords(xx, yy, zz)
        radial = torch.sqrt(dy.square() + dz.square())
        axial = torch.abs(dx) - 0.5 * self.length
        outer = torch.maximum(radial - self.outer_radius, axial)
        inner = torch.maximum(radial - self.inner_radius, axial)
        return torch.maximum(outer, -inner)

    def to_mesh(self, segments=32):
        return mw.Cylinder(
            position=self.position,
            radius=self.outer_radius,
            height=self.length,
            axis="x",
            device=self.device,
        ).to_mesh(segments=segments)


def snap_contour_half(dx: float, *, target: float = CONTOUR_HALF) -> tuple[float, float]:
    """Deterministic half-grid snap of the current-contour half-width (B5).

    The contour box boundary at +-half must land on the Yee transverse half-grid
    (node + dx/2), where the node grid is ``y_n = -DOMAIN_TRANSVERSE + n*dx``.
    Returns the snapped half-width and the snap distance so refinement tiers can
    be built deterministically (grid-commensurate across dx) and the snap is
    recorded rather than crashing the contour builder.
    """
    lo = -DOMAIN_TRANSVERSE
    # half-grid coordinate: lo + (n + 0.5)*dx ; solve for the n nearest to +target
    n = round((target - lo) / dx - 0.5)
    snapped_edge = lo + (n + 0.5) * dx
    half = abs(snapped_edge)
    return half, abs(half - target)


def _coax_port(name: str, x: float, direction: str, *, dx: float) -> mw.WavePort:
    # The current contour sits on the magnetic half-grid one half-cell into the
    # guide from the reference plane, in the propagation direction.
    contour_x = x + (0.5 * dx if direction == "+" else -0.5 * dx)
    contour_half, _snap = snap_contour_half(dx)
    mode = mw.WaveModeSpec(
        "tem",
        polarization="Ey",
        voltage_path=((x, INNER_RADIUS, 0.0), (x, OUTER_RADIUS, 0.0)),
        current_contour=mw.Box(
            position=(contour_x, 0.0, 0.0),
            size=(0.0, 2.0 * contour_half, 2.0 * contour_half),
        ),
    )
    return mw.WavePort(
        name,
        position=(x, 0.0, 0.0),
        size=(0.0, 0.40, 0.40),
        direction=direction,
        reference_plane=x,
        modes=(mode,),
    )


def coax_thru_scene(*, dx: float = 0.0025, device: str = "cuda") -> mw.Scene:
    """Build a two-port air-filled coaxial thru line."""

    scene = mw.Scene(
        domain=mw.Domain(
            bounds=(
                (-DOMAIN_X, DOMAIN_X),
                (-DOMAIN_TRANSVERSE, DOMAIN_TRANSVERSE),
                (-DOMAIN_TRANSVERSE, DOMAIN_TRANSVERSE),
            )
        ),
        grid=mw.GridSpec.uniform(dx),
        boundary=mw.BoundarySpec.pml(num_layers=6),
        ports=(
            _coax_port("left", -PORT_X, "+", dx=dx),
            _coax_port("right", PORT_X, "-", dx=dx),
        ),
        device=device,
    )
    scene.add_structure(
        mw.Cylinder(
            position=(0.0, 0.0, 0.0),
            radius=INNER_RADIUS,
            height=LINE_LENGTH,
            axis="x",
        ).with_material(mw.Material.pec(), name="inner_conductor")
    )
    scene.add_structure(
        _HollowCylinder(
            position=(0.0, 0.0, 0.0),
            inner_radius=OUTER_RADIUS,
            outer_radius=SHIELD_OUTER,
            length=LINE_LENGTH,
        ).with_material(mw.Material.pec(), name="outer_conductor")
    )
    return scene
