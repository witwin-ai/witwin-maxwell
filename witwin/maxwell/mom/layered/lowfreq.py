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

from .assembly import assemble_system
from .build import build_extension


MIN_FREQUENCY = 1.0
C0 = 299792458.0


@dataclass
class LoopTree:
    """Sparse loop/tree basis; only O(N) and O(nnz) tensors persist.

    ``sparse_transform`` holds the raw fundamental columns (+-1/l), ``sparse_q``
    the same pattern divided by ``norms``, ``sparse_q_rows`` the CSC of Q^T for
    current recovery and ``sparse_charge`` the face charge D*Q_tree.
    """
    order: torch.Tensor
    parents: torch.Tensor
    components: int
    loops: int
    norms: torch.Tensor
    omega0: float
    sparse_transform: list[torch.Tensor]
    sparse_q: list[torch.Tensor]
    sparse_q_rows: list[torch.Tensor]
    sparse_charge: list[torch.Tensor]

    @property
    def trees(self):
        return len(self.norms) - self.loops


def _transpose_columns(columns, rows):
    """CSC of A^T from the CSC of A; setup-only integer sort with unique keys."""
    offsets, indices, values = columns
    count = len(offsets) - 1
    owners = torch.repeat_interleave(torch.arange(count, device=offsets.device), offsets.diff())
    permutation = torch.argsort(indices * count + owners)
    counts = torch.bincount(indices, minlength=rows)
    return [torch.cat((counts.new_zeros(1), torch.cumsum(counts, 0))),
            owners[permutation].contiguous(), values[permutation].contiguous()]


def build_loop_tree(layout, *, length_scale, dense_setup=True):
    """Deterministic CUDA forest/cycles; one small setup count transfer.

    ``dense_setup=False`` (the matrix-free backend) never forms the dense
    cycle matrix: the cycle walk emits sparse columns directly and the column
    norms are fixed-order fp64 sums of squares (not the vendor dense norm).
    """
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
    if not dense_setup:
        return _sparse_loop_tree(layout, basis, minus, order, parents, components, loops, length_scale)
    # The dense [N,N] fp64 cycle matrix exists only during setup: the vendor
    # column norm keeps the established normalization, then only sparse
    # columns persist. Column normalization retains sparse fundamental cycles
    # and singleton trees; dense orthogonalization would be N^3 per frequency.
    transform = op.cycles(basis.plus_faces, minus, basis.edge_lengths.contiguous(), order, parents, loops)
    norms = torch.linalg.vector_norm(transform, dim=0)
    raw = op.sparse_columns(transform)
    del transform
    owners = torch.repeat_interleave(torch.arange(len(norms), device=norms.device), raw[0].diff())
    sparse_q = [raw[0], raw[1], (raw[2] / norms[owners]).contiguous()]
    sparse_charge = op.tree_charge(sparse_q, torch.stack((basis.plus_faces, basis.minus_faces), -1).contiguous(),
                                   basis.face_edges.contiguous(), basis.face_signs.contiguous(),
                                   basis.edge_lengths.contiguous(), loops)
    return LoopTree(order, parents, components, loops, norms, C0 / length_scale, raw, sparse_q,
                    _transpose_columns(sparse_q, len(norms)), sparse_charge)


def _sparse_loop_tree(layout, basis, minus, order, parents, components, loops, length_scale):
    op = torch.ops.witwin_maxwell_mom_layered
    n = len(order)
    columns, rows, values = op.cycle_entries(basis.plus_faces, minus, basis.edge_lengths.contiguous(), order,
                                             parents, loops)
    # Ascending edge order within each column, as the dense column packing.
    permutation = torch.sort(columns * n + rows).indices
    columns, rows, values = columns[permutation], rows[permutation], values[permutation].contiguous()
    offsets = torch.cat((columns.new_zeros(1), torch.cumsum(torch.bincount(columns, minlength=n), 0)))
    norms = op.segment_sum(values.square().to(torch.complex128), offsets).real.sqrt()
    raw = [offsets, rows.contiguous(), values]
    sparse_q = [offsets, raw[1], (values / norms[columns]).contiguous()]
    sparse_charge = op.tree_charge(sparse_q, torch.stack((basis.plus_faces, basis.minus_faces), -1).contiguous(),
                                   basis.face_edges.contiguous(), basis.face_signs.contiguous(),
                                   basis.edge_lengths.contiguous(), loops)
    return LoopTree(order, parents, components, loops, norms, C0 / length_scale, raw, sparse_q,
                    _transpose_columns(sparse_q, n), sparse_charge)


def port_drive(prepared):
    """Normalized loop/tree port drive [N,P] for the delta-gap excitation.

    Integer cycle work preserves topological zero loop excitation at DC.
    """
    lt = prepared.loop_tree
    work = torch.ops.witwin_maxwell_mom_layered.sparse_port_work(
        lt.sparse_q, prepared.layout.basis.edge_lengths.contiguous(), prepared.excitation.contiguous())
    return (work / lt.norms[:, None]).contiguous()


def assemble_chunk(prepared, *, frequency_start=0, frequency_stop=None):
    """Allocate one chunk's storage and gather V (basis positions) and packed P.

    Storage is one complex128 [K,N,N] system plus one complex64 workspace per
    frequency (packed P, then the LU); see operator.dense_frequency_bytes.
    """
    from .operator import DenseLayeredSystem
    lt, table = prepared.loop_tree, prepared.table
    frequencies = table.frequencies[frequency_start:frequency_stop]
    system = DenseLayeredSystem.allocate(len(frequencies), len(lt.norms), len(prepared.layout.mesh.faces),
                                         frequencies, lt.loops)
    assemble_system(prepared.layout, table, system.matrix, system.workspace, lt.order,
                    frequency_start=frequency_start, frequency_stop=frequency_stop,
                    far_dtype=torch.float32, _network_accuracy=True)
    return system


def transform_chunk(prepared, system, *, frequency_start=0, admittance_blocks=None):
    """Fold loss, apply the sparse congruence and equilibrate, all in place."""
    lt, op = prepared.loop_tree, torch.ops.witwin_maxwell_mom_layered
    count = len(system.frequencies)
    if prepared.loss is not None:
        sheet = prepared.loss
        op.loss_fold(system.matrix, sheet.blocks.contiguous(), sheet.columns.contiguous(),
                     sheet.supports.contiguous(),
                     sheet.impedance[frequency_start:frequency_start + count].contiguous(), lt.order,
                     system.frequencies)
    if prepared.admittance is not None:
        # Dense per-body surface-admittance blocks, in the same Z/(-i w mu0) units.
        prepared.admittance.fold(system, lt.order, admittance_blocks)
    op.congruence(system.matrix, lt.sparse_q, lt.order, lt.loops)
    system.rhs, system.recovery = op.scale_system(system.matrix, system.workspace, lt.sparse_charge,
                                                  port_drive(prepared), system.frequencies, lt.loops, lt.omega0)
    return system


def scaled_system(prepared, *, frequency_start=0, frequency_stop=None):
    """Assemble the equilibrated loop/tree system of one frequency chunk in place."""
    # Surface-admittance blocks first: their per-body interior work sets are
    # released before the dense chunk storage exists.
    blocks = None if prepared.admittance is None else prepared.admittance.blocks(
        prepared.table.frequencies[frequency_start:frequency_stop].tolist())
    system = assemble_chunk(prepared, frequency_start=frequency_start, frequency_stop=frequency_stop)
    return transform_chunk(prepared, system, frequency_start=frequency_start, admittance_blocks=blocks)
