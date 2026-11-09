"""Finite-thickness sheet impedance and deterministic device RWG loss assembly."""

from dataclasses import dataclass
import math

import torch

from ..topology import INTERIOR


MU0 = 4e-7 * math.pi


def sheet_impedance(frequencies, conductivity, thickness, roughness=None):
    """Two-face total-current sheet impedance [ohm], frequencies in Hz.

    For exp(-i omega t), gamma=(1-i)/delta and Z=gamma*coth(gamma*t/2)/(2*sigma).
    The thick limit is half the single-face surface impedance. The small-x
    series for x*coth(x) retains the DC limit without cancellation; decaying
    exponentials avoid overflow for thick metal.
    """
    omega = 2 * math.pi * frequencies
    delta = torch.sqrt(2 / (omega * MU0 * conductivity))
    x = (1 - 1j) * thickness / (2 * delta)
    x2 = x * x
    series = (1 + x2 / 3 - x2 * x2 / 45 + 2 * x2 * x2 * x2 / 945) / (conductivity * thickness)
    decay = torch.exp(-2 * x)
    regular = x * (1 + decay) / (-torch.expm1(-2 * x)) / (conductivity * thickness)
    impedance = torch.where(x.abs() < .01, series, regular)
    if roughness is not None:
        impedance = impedance * roughness.impedance_factor(delta)
    return impedance


def sheet_material_payload(material):
    sigma, thickness, roughness = material
    return {"sigma": sigma, "thickness": thickness,
            "roughness": None if roughness is None else {"model": type(roughness).__name__, **vars(roughness)}}


@dataclass
class SheetLoss:
    blocks: torch.Tensor
    columns: torch.Tensor
    supports: torch.Tensor
    impedance: torch.Tensor

    def add_to(self, z, frequency_index):
        """Add Z_sheet*G in O(N) with six fixed-order writes and no float atomics."""
        rows = torch.arange(z.shape[0], device=z.device)
        values = self.blocks * self.impedance[frequency_index, self.supports, None]
        # E_inc + E_scattered = Z_sheet J and Z=-<f,E_scattered>, hence
        # b - Z*x = Z_sheet*G*x, or (Z + Z_sheet*G)*x = b.
        # One entry per row in each write: two support faces, three local slots.
        for side in range(2):
            for slot in range(3):
                z[rows, self.columns[:, side, slot]] += values[:, side, slot]


def prepare_sheet_loss(layout, rectangles, materials, frequencies, *, vias=(), via_materials=()):
    """Assign centroid materials and cache exact local Gram rows on the device.

    Rectangles may overlap only when their material and thickness agree.
    Bridge-only faces have no owning physical rectangle and remain ideal/lossless.
    """
    for i, (a, material) in enumerate(zip(rectangles, materials)):
        for b, other in zip(rectangles[:i], materials[:i]):
            if (a[4] == b[4] and max(a[0], b[0]) < min(a[1], b[1])
                    and max(a[2], b[2]) < min(a[3], b[3]) and material != other):
                raise ValueError("Overlapping rectangles on one plane require identical materials and thicknesses")
    if not any(sigma is not None for sigma, _, _ in (*materials, *via_materials)):
        return None
    mesh, basis = layout.mesh, layout.basis
    centres = mesh.centroids
    impedance = torch.zeros((len(frequencies), len(mesh.faces)), device=mesh.device, dtype=torch.complex128)
    for rectangle, (sigma, thickness, roughness) in zip(rectangles, materials):
        if sigma is None:
            continue
        x0, x1, y0, y1, z = rectangle
        owned = ((centres[:, 0] > x0) & (centres[:, 0] < x1) & (centres[:, 1] > y0)
                 & (centres[:, 1] < y1) & (centres[:, 2] == z))
        value = sheet_impedance(frequencies, sigma, thickness, roughness)
        impedance[:, owned] = value[:, None]
    for via, (sigma, _, roughness) in zip(vias, via_materials):
        if sigma is None:
            continue
        x0,x1,y0,y1,z0,z1 = via
        owned = ((layout.triangle_planes < 0) & (centres[:, 0] >= x0) & (centres[:, 0] <= x1)
                 & (centres[:, 1] >= y0) & (centres[:, 1] <= y1)
                 & (centres[:, 2] > z0) & (centres[:, 2] < z1))
        # A tube wall is one bulk-metal face, unlike a two-face thin sheet.
        delta = torch.sqrt(2 / (2*math.pi*frequencies*MU0*sigma))
        value = (1-1j)/(sigma*delta)
        if roughness is not None:
            value = value*roughness.impedance_factor(delta)
        impedance[:, owned] = value[:, None]
    supports = torch.stack((basis.plus_faces, basis.minus_faces), -1)
    slots = (mesh.topology.incidence_slots[mesh.topology.kind == INTERIOR]
             if layout.support_slots is None else layout.support_slots)
    blocks = basis.mass_blocks()[supports.clamp_min(0), slots]
    blocks = torch.where((supports >= 0)[..., None], blocks, 0.0)
    return SheetLoss(blocks, basis.face_edges[supports.clamp_min(0)], supports, impedance)
