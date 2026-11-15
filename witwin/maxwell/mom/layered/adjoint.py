"""Native adjoint gradients of layered MPIE network admittances (local loss parameters).

Network extraction already computes the forward currents X = Z^-1 V of every
port (``solve_prepared_network``). For any parameter theta of Z,

    dY/dtheta = -X^T (dZ/dtheta) X,

so Y, Z, S, L and Q gradients need no extra solve. The parameters covered here
enter only the local loss operator R = sum_f Zs_f(theta) G_f: conductivity,
roughness and (sheet model) thickness of PlanarPEC metal, via conductivity and
roughness (the via wall resistance). Reference impedances and port loads act
on Result.network through the PyTorch-native NetworkData algebra (renormalize,
terminate); a trainable LumpedPort.reference_impedance is still refused by the
shared Simulation admission for non-FDTD methods. The per-face surface impedance is a PyTorch-native
function of the Scene tensors; the contraction over the mesh is the native
``loss_adjoint`` kernel (cuda/adjoint.cu). With gradients off nothing here runs.
"""

import copy
import dataclasses
import math

import torch

from ...geometry import PlanarPEC
from ...media import LossyMetalMedium
from ...network import NetworkData
from .build import build_extension
from .loss import MU0, sheet_impedance
from ...errors import UnsupportedFeatureError


def _trainable(value):
    return isinstance(value, torch.Tensor) and value.requires_grad


def _fixed(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else value


def fixed_roughness(roughness):
    """Float-valued copy of a roughness descriptor; the same object when already fixed."""
    if roughness is None or not any(_trainable(v) for v in vars(roughness).values()):
        return roughness
    copied = copy.copy(roughness)
    for name, value in vars(roughness).items():
        object.__setattr__(copied, name, _fixed(value))
    return copied


def fixed_record(material, thickness):
    """Forward (sigma, thickness, roughness) record with detached float values."""
    if isinstance(material, LossyMetalMedium):
        return (_fixed(material.conductivity), _fixed(thickness), fixed_roughness(material.roughness))
    return (None, _fixed(thickness), None)


@dataclasses.dataclass
class LossAdjoint:
    """Fixed-order face/group incidence of the local loss operator for ``loss_adjoint``.

    rows/row_offsets: CSR face -> local Gram row (2*edge+side), ascending row.
    faces/face_offsets: CSR parameter group -> owned faces, ascending face.
    Both come from stable integer sorts, so every reduction order is fixed.
    """

    blocks: torch.Tensor
    columns: torch.Tensor
    rows: torch.Tensor
    row_offsets: torch.Tensor
    faces: torch.Tensor
    face_offsets: torch.Tensor

    @classmethod
    def build(cls, loss, face_groups, groups):
        faces_count = len(face_groups)
        supports = loss.supports.reshape(-1)
        # Ideal (one-face/grounded) sides carry support -1 and sort first.
        # Bin 0 holds the -1 entries, so cumulative counts are the CSR offsets:
        # face f owns sorted positions [offsets[f], offsets[f+1]).
        rows = torch.sort(supports, stable=True).indices
        row_offsets = torch.cumsum(torch.bincount(supports + 1, minlength=faces_count + 1), 0)
        # Group CSR over faces; ideal faces have group -1 and are skipped.
        faces = torch.sort(face_groups, stable=True).indices
        face_offsets = torch.cumsum(torch.bincount(face_groups + 1, minlength=groups + 1), 0)
        return cls(loss.blocks.contiguous(), loss.columns.contiguous(), rows.contiguous(),
                   row_offsets.contiguous(), faces.contiguous(), face_offsets.contiguous())


class _LossAdmittance(torch.autograd.Function):
    """Y(Zs_groups) with the forward values of the unchanged network solve."""

    @staticmethod
    def forward(ctx, impedance, admittance, currents, adjoint):
        ctx.currents, ctx.adjoint = currents, adjoint
        return admittance.clone()

    @staticmethod
    def backward(ctx, grad):
        a = ctx.adjoint
        grad_impedance = torch.ops.witwin_maxwell_mom_layered.loss_adjoint(
            ctx.currents, grad.to(torch.complex128).contiguous(), a.blocks, a.columns,
            a.rows, a.row_offsets, a.faces, a.face_offsets)
        return grad_impedance, None, None, None


def network_admittance(impedance, admittance, currents, adjoint):
    """Attach the native VJP: ``impedance`` [K,G] differentiable group Zs, ``admittance`` [K,P,P]."""
    build_extension()
    return _LossAdmittance.apply(impedance, admittance, currents.contiguous(), adjoint)


def group_impedance(frequencies, groups, conductor_model):
    """[K, G] complex128 surface impedance of every parameter group, PyTorch-native.

    The expressions are those of prepare_sheet_loss (two-face finite-thickness
    sheets, single-face bulk via walls) and of the native closed_impedance
    (single-face Leontovich on every exposed body face), each times the public
    roughness factor K(delta). Gradients flow to every tensor they contain.
    """
    columns = []
    for kind, material, thickness in groups:
        sigma = material.conductivity
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=frequencies.device, dtype=torch.float64)
        roughness = material.roughness
        if kind == "sheet" and conductor_model == "sheet":
            if isinstance(thickness, torch.Tensor):
                thickness = thickness.to(device=frequencies.device, dtype=torch.float64)
            columns.append(sheet_impedance(frequencies, sigma, thickness, roughness))
            continue
        delta = torch.sqrt(2 / (2 * math.pi * frequencies * MU0 * sigma))
        value = (1 - 1j) / (sigma * delta)
        if roughness is not None:
            value = value * roughness.impedance_factor(delta)
        columns.append(value.to(torch.complex128))
    return torch.stack(columns, 1)


class LayeredGradients:
    """Gradient request of one prepared layered network (None when nothing is trainable)."""

    def __init__(self, sources, conductor_model):
        self.conductor_model = conductor_model
        sheets, vias = sources
        keys, self.groups, record_groups = {}, [], []
        for kind, records in (("sheet", sheets), ("via", vias)):
            for structure in records:
                material = structure.material
                if not isinstance(material, LossyMetalMedium):
                    record_groups.append(-1)
                    continue
                thickness = structure.geometry.physical_thickness if kind == "sheet" else 0.0
                key = (kind, id(material), id(thickness) if isinstance(thickness, torch.Tensor) else thickness)
                if key not in keys:
                    keys[key] = len(self.groups)
                    self.groups.append((kind, material, thickness))
                record_groups.append(keys[key])
        self.record_groups = record_groups

    def face_groups(self, prepared):
        """[F] parameter group of every face's impedance; -1 for ideal faces (exact integers)."""
        owners = prepared.loss.owners
        record_groups = torch.tensor(self.record_groups, device=owners.device, dtype=torch.int64)
        # Closed thick-terminal port blocks own the faces after the records (network._thick_terminal_blocks): ideal.
        ideal = (owners < 0) | (owners >= len(record_groups))
        return torch.where(ideal, -1, record_groups[owners.clamp(0, len(record_groups) - 1)]).contiguous()

    def loss_adjoint(self, prepared):
        """(LossAdjoint, differentiable [K,G] group impedance) of a prepared lossy network."""
        adjoint = LossAdjoint.build(prepared.loss, self.face_groups(prepared), len(self.groups))
        return adjoint, group_impedance(prepared.table.frequencies, self.groups, self.conductor_model)

    def attach(self, prepared, network, currents):
        """Return NetworkData whose Y carries autograd graphs to the Scene tensors.

        Y is recovered from the forward S (one S->Y->S round trip, so the
        gradient-enabled S agrees with the forward S to rounding, not bitwise).
        """
        adjoint, impedance = self.loss_adjoint(prepared)
        y = network_admittance(impedance, network.to_y(), currents, adjoint)
        return NetworkData.from_y(frequencies=network.frequencies, y=y, z0=network.z0,
                                  port_names=network.port_names, metadata=network.metadata)


def layered_gradients(simulation, sources):
    """Admit supported trainable inputs; refuse every other trainable tensor explicitly."""
    scene, options = simulation.scene, simulation.config.layered
    supported = False
    for structure in scene.structures:
        if not getattr(structure, "enabled", True):
            continue
        geometry, material = structure.geometry, structure.material
        for name, value in vars(geometry).items():
            if not _trainable(value):
                continue
            if not (isinstance(geometry, PlanarPEC) and name == "physical_thickness"):
                raise UnsupportedFeatureError(
                    f"Layered MoM geometry/material derivatives are not supported for "
                    f"{type(geometry).__name__}.{name} (geometry gradient)",
                    feature="guard:6ab3bc08", method="mom.layered")
            if options.conductor_model == "closed":
                raise UnsupportedFeatureError(
                    "Closed-conductor thickness moves the extruded body faces: it is a geometry derivative, "
                    "not a local loss parameter, and is not supported",
                    feature="guard:aaa4731a", method="mom.layered")
            # An ideal PEC sheet has no loss: its thickness gradient is exactly zero.
            supported |= isinstance(material, LossyMetalMedium)
        if isinstance(material, LossyMetalMedium):
            values = [material.conductivity, *vars(material.roughness).values()] if material.roughness else [
                material.conductivity]
            supported |= any(_trainable(v) for v in values)
    for layer in getattr(scene.background, "layers", ()):
        if any(_trainable(v) for v in vars(layer).values()):
            raise UnsupportedFeatureError("Layered MoM process-stack gradients are not supported",
                                          feature="guard:3a0d8a20", method="mom.layered")
    if not supported or not torch.is_grad_enabled():
        return None
    if options.conductor_model == "surface_admittance":
        raise UnsupportedFeatureError("conductor_model='surface_admittance' does not propagate gradients: "
                                  "its per-body admittance blocks have no loss adjoint",
                                      feature="guard:cce99a3c", method="mom.layered")
    if options.sweep is not None:
        raise UnsupportedFeatureError("Adaptive layered sweeps do not propagate gradients; use explicit frequencies",
                                      feature="guard:7785057d", method="mom.layered")
    # conductor_model="auto" selects "sheet" for gradient runs (runtime.select_conductor_model).
    return LayeredGradients(sources, "sheet" if options.conductor_model == "auto" else options.conductor_model.value)
