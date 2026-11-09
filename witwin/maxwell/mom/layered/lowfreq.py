"""Device loop/tree decomposition and frequency-scaled layered MPIE.

P0 is unit-integral on each face; D is the signed integrated edge flux.
Fundamental cycles span ker(D): each chord plus its unique forest path is
solenoidal, and its chord coefficient is a unique pivot. Tree incidence has
rank F-C, so these N-(F-C) cycles and F-C singleton trees are complete.

For exp(-i*w*t), S=[Lambda, (w/w0)T], A=S.T Z S/w:
 A_LL=-i*mu0*V_LL,
 A_LT=-i*mu0*(w/w0)*V_LT,
 A_TT=i*D_T.T P D_T/(eps0*w0**2)-i*mu0*(w/w0)**2*V_TT.
Loss adds S.T R S/w. Diagonal congruence balances physical units and loss.
The scalar potential NEVER enters a loop block. Transforms use sparse gathers.

At DC the PEC blocks decouple into magnetostatic loops and electrostatic
charge. Unit-voltage loop current is proportional to 1/w (finite w*I),
Z_loop=-i*w*L; a drive proportional to w has finite loop current. A capacitive
port has I proportional to w and Z=+i/(w*C). Loss tends to the DC Gram
resistance. Literal zero-frequency voltage-driven PEC loops are singular;
the admitted lower frequency is 1 Hz, with DC defined by these limits.
"""

from dataclasses import dataclass

import torch

from .assembly import assemble_potentials
from .build import build_extension


MIN_FREQUENCY = 1.0
C0 = 299792458.0


@dataclass
class LoopTree:
    transform: torch.Tensor
    order: torch.Tensor
    parents: torch.Tensor
    components: int
    loops: int
    q: torch.Tensor
    r: torch.Tensor
    charge: torch.Tensor
    omega0: float
    sparse_q: list[torch.Tensor]
    sparse_charge: list[torch.Tensor]

    @property
    def trees(self):
        return self.transform.shape[1] - self.loops

    @property
    def lam(self):
        return self.transform[:, :self.loops]

    @property
    def tree(self):
        return self.transform[:, self.loops:]


def build_loop_tree(layout, *, length_scale):
    """Deterministic CUDA forest/cycles; one small setup count transfer."""
    build_extension()
    op = torch.ops.witwin_maxwell_mom_layered
    basis = layout.basis
    minus, edges, signs = basis.minus_faces, basis.face_edges, basis.face_signs
    if layout.grounded:
        minus = torch.where(minus < 0, len(layout.mesh.faces), minus).contiguous()
        edges = torch.cat((edges, torch.zeros_like(edges[:1])))
        signs = torch.cat((signs, torch.zeros_like(signs[:1])))
    order, parents, counts = op.forest(basis.plus_faces, minus, edges.contiguous(), signs.contiguous())
    components, loops = counts.tolist()
    transform = op.cycles(basis.plus_faces, minus, basis.edge_lengths.contiguous(),
                          order, parents, loops)
    # Column normalization retains sparse fundamental cycles and singleton
    # trees. Dense orthogonalization would turn each frequency into N^3 work.
    norms = torch.linalg.vector_norm(transform, dim=0)
    q = (transform / norms).contiguous()
    r = torch.diag(norms)
    charge = op.divergence(q[:, loops:].contiguous(), basis.face_edges.contiguous(),
                           basis.face_signs.contiguous(), basis.edge_lengths.contiguous())
    sparse_q = op.sparse_columns(q)
    sparse_charge = op.sparse_columns(charge)
    return LoopTree(transform, order, parents, components, loops,
                    q.to(torch.complex128), r.to(torch.complex128), charge.to(torch.complex128),
                    C0 / length_scale, sparse_q, sparse_charge)


def scale_potentials(prepared, vector, scalar, *, frequency_start=0):
    """Transform one chunk with batch-size-independent vendor GEMM dispatch."""
    lt = prepared.loop_tree
    op = torch.ops.witwin_maxwell_mom_layered
    count = vector.shape[0]
    frequencies = prepared.table.frequencies[frequency_start:frequency_start + count]
    magnetic = op.sparse_product(op.sparse_product(vector, lt.sparse_q, True), lt.sparse_q, False)
    charge = op.sparse_product(op.sparse_product(scalar, lt.sparse_charge, True),
                               lt.sparse_charge, False)
    loss = torch.empty(0, device=vector.device, dtype=vector.dtype)
    if prepared.loss is not None:
        sheet = prepared.loss
        resistance = op.loss_matrix(
            sheet.blocks.contiguous(), sheet.columns.contiguous(), sheet.supports.contiguous(),
            sheet.impedance[frequency_start:frequency_start + count].contiguous())
        loss = op.sparse_product(op.sparse_product(resistance, lt.sparse_q, True), lt.sparse_q, False)
    # Integer cycle work preserves topological zero loop excitation at DC.
    work = op.port_work(lt.transform, prepared.layout.basis.edge_lengths.contiguous(),
                        prepared.excitation.contiguous())
    rhs = torch.linalg.solve_triangular(lt.r.T, work, upper=False).contiguous()
    return op.scaled(magnetic, charge, loss, rhs, frequencies, lt.loops, lt.omega0)


def scaled_system(prepared, *, frequency_start=0, frequency_stop=None):
    """Return equilibrated blocks/RHS/recovery for the requested frequency chunk."""
    vector, scalar = assemble_potentials(
        prepared.layout, prepared.table, frequency_index=-1,
        frequency_start=frequency_start, frequency_stop=frequency_stop,
        far_dtype=torch.float32, _network_accuracy=True)
    return scale_potentials(prepared, vector, scalar, frequency_start=frequency_start)
