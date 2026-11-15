"""Dense layered MPIE backend: in-place chunk system and surface operator protocol.

A frequency chunk owns exactly two O(N^2) device buffers per frequency:

* ``matrix`` [K,N,N] complex128. The tile gather writes V[m,n] at the loop/tree
  basis position (order[m], order[n]); the local loss is folded in; the sparse
  congruence Q^T(.)Q and the diagonal equilibration run in place. It ends as
  the equilibrated system A and is the only complex128 matrix (refinement
  residuals use it through the vendor GEMM).
* ``workspace`` [K,W] complex64, W=max(N^2, F(F+1)) (even). During assembly
  its first F(F+1)/2 complex128 slots hold the packed lower face potential P,
  which the equilibration reads on the fly for the tree-tree charge block.
  Then the same storage becomes the column-major complex64 LU, factored in
  place.

Hence ``dense_frequency_bytes`` = 16 N^2 + 8 W plus O(F) tile and O(N P) vector
terms; for F <= N this is at most 24 N^2 + O(N P) bytes.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ..operator_protocol import IterativeState
from .assembly import NETWORK_RULE_POINTS


# Scaled and standalone solves use the same normwise backward-error gate.
BACKWARD = 64 * 2.0**-53
ITERATIONS = 10
SOURCE_TILE = 64


def workspace_elements(unknowns, faces):
    """complex64 elements per frequency: the LU or packed complex128 P, even."""
    elements = max(unknowns * unknowns, faces * (faces + 1))
    return elements + elements % 2


def dense_frequency_bytes(unknowns, faces, ports):
    """Device bytes each frequency adds to a dense chunk.

    16 N^2 system, 8 W workspace, the [F, min(64,F), 10] complex128 face-pair
    tile and 256 N P bytes for RHS, drive, solution, residual and correction
    temporaries of the refinement and recovery.
    """
    return (16 * unknowns * unknowns + 8 * workspace_elements(unknowns, faces)
            + 160 * faces * min(SOURCE_TILE, faces) + 256 * unknowns * ports)


def dense_chunk_bytes(faces, wall_nodes):
    """Frequency-independent assembly scratch of one chunk.

    Quadrature-point offsets [F, R, 2] fp64, with vertical walls cached z
    weights [F, R, 2K+2] fp64 for the K-node wall table (R rule rows), and
    512 B/face for per-face geometry arrays and table views of the call.
    """
    return 8 * faces * NETWORK_RULE_POINTS * (2 + (2 * wall_nodes + 2 if wall_nodes else 0)) + 512 * faces


@dataclass
class DenseLayeredSystem:
    """One frequency chunk of the equilibrated loop/tree system and its LU."""

    matrix: torch.Tensor
    workspace: torch.Tensor | None
    frequencies: torch.Tensor
    loops: int
    rhs: torch.Tensor | None = None
    recovery: torch.Tensor | None = None
    pivots: torch.Tensor | None = None
    info: torch.Tensor | None = None

    @classmethod
    def allocate(cls, count, unknowns, faces, frequencies, loops):
        device = frequencies.device
        matrix = torch.zeros((count, unknowns, unknowns), device=device, dtype=torch.complex128)
        workspace = torch.empty((count, workspace_elements(unknowns, faces)), device=device,
                                dtype=torch.complex64)
        return cls(matrix, workspace, frequencies, loops)

    @classmethod
    def from_matrix(cls, matrix, rhs):
        """Wrap an explicit equilibrated [K,N,N] complex128 system (no copy)."""
        count, unknowns = matrix.shape[:2]
        workspace = torch.empty((count, workspace_elements(unknowns, 0)), device=matrix.device,
                                dtype=torch.complex64)
        frequencies = torch.zeros(count, device=matrix.device, dtype=torch.float64)
        return cls(matrix, workspace, frequencies, unknowns, rhs)

    @property
    def lu(self):
        count, unknowns = self.matrix.shape[:2]
        return self.workspace.as_strided((count, unknowns, unknowns), (self.workspace.stride(0), 1, unknowns))

    def factor(self):
        """complex64 LU in the workspace; packed P is consumed before this."""
        self.pivots, self.info = torch.ops.witwin_maxwell_mom_layered.factor_system(self.matrix, self.lu)

    def solve(self):
        """complex64 LU + complex128 refinement, escalating failed frequencies only."""
        solution, status = refine(self.matrix, self.lu, self.pivots, self.info, self.rhs)
        factor_dtype = ["complex64"] * len(status)
        failed = [i for i, (step, _) in enumerate(status) if step < 0]
        if failed:
            # The complex64 factors are no longer needed: release them first.
            self.workspace = None
            for index in failed:
                corrected, final = escalate(self.matrix[index:index + 1], self.rhs[index:index + 1])
                if final[0][0] < 0:
                    raise RuntimeError(f"Scaled dense refinement did not reach backward error {BACKWARD:.3g}")
                solution[index:index + 1] = corrected
                factor_dtype[index], status[index] = "complex128", final[0]
        return solution, (tuple(factor_dtype), tuple(int(s[0]) for s in status), tuple(s[1] for s in status))

    def operator(self, index):
        return LayeredDenseOperator(self, index)


def refine(matrix, lu, pivots, info, rhs):
    """Fixed device refinement budget, followed by one chunk convergence transfer."""
    op = torch.ops.witwin_maxwell_mom_layered
    dtype = lu.dtype
    solution = op.lu_solve(lu, pivots, rhs.to(dtype)).to(torch.complex128).contiguous()
    steps = torch.full((matrix.shape[0],), -1, device=matrix.device, dtype=torch.int64)
    matrix_norm = op.frequency_norm(matrix, True)
    rhs_norm = op.frequency_norm(rhs, False)
    # All convergence decisions and first-passing-step latches stay on device.
    # Freeze each converged frequency so chunk composition cannot change X.
    for step in range(ITERATIONS + 1):
        residual = op.frequency_residual(matrix, solution, rhs)
        error, active = op.refinement_check(
            op.frequency_norm(residual, False), matrix_norm,
            op.frequency_norm(solution, False), rhs_norm, info, steps, step, BACKWARD)
        if step < ITERATIONS:
            correction = op.lu_solve(lu, pivots, residual.to(dtype)).to(torch.complex128).contiguous()
            op.refinement_update(solution, correction, active)
    status = torch.stack((steps.to(torch.float64), error), -1).tolist()
    return solution, status


def escalate(matrix, rhs):
    """complex128 in-place LU for one frequency: 16 N^2 extra bytes, admitted explicitly."""
    from .network import available_work_memory
    unknowns, ports = matrix.shape[1], rhs.shape[2]
    needed = 16 * unknowns * unknowns + 256 * unknowns * ports
    available = available_work_memory(matrix.device)
    if needed > available:
        raise RuntimeError(f"complex128 refinement escalation needs {needed / 2**30:.2f} GiB; "
                           f"{available / 2**30:.2f} GiB available")
    lu = torch.empty_strided((1, unknowns, unknowns), (unknowns * unknowns, 1, unknowns),
                             device=matrix.device, dtype=torch.complex128)
    pivots, info = torch.ops.witwin_maxwell_mom_layered.factor_system(matrix, lu)
    return refine(matrix, lu, pivots, info, rhs)


class LayeredDenseOperator:
    """SurfaceOperatorProtocol view of one frequency of a factored dense chunk.

    The space is the equilibrated loop/tree basis A = D S^T Z S D / w, with S
    the frequency-scaled loop/tree transform and D the diagonal congruence.
    Vectors use the last dimension; the residual metric is Euclidean in this
    space, so the Riesz inverse is the identity. ``solve`` is the backend's
    direct complex64 LU + complex128 refinement for arbitrary right-hand sides.
    """

    supports_direct = True

    def __init__(self, system, index):
        self.system, self.index = system, index
        self.matrix = system.matrix[index]
        self.shape = tuple(self.matrix.shape)
        self.device, self.dtype = self.matrix.device, self.matrix.dtype
        self.iterative_state = IterativeState()
        self._binding = (id(system.matrix), system.matrix.data_ptr(), id(system.workspace))
        self._apply_calls = self._adjoint_calls = 0
        self._scales = {}

    def refresh(self):
        system = self.system
        if (id(system.matrix), system.matrix.data_ptr(), id(system.workspace)) != self._binding:
            raise ValueError("Layered dense chunk storage changed after operator creation")

    def read_access(self):
        return nullcontext()

    def apply(self, x):
        self.refresh()
        self._apply_calls += 1
        return x @ self.matrix.mT

    def adjoint(self, y):
        self.refresh()
        self._adjoint_calls += 1
        # Conjugate only vectors: a conjugated matrix operand may be copied.
        return (y.conj() @ self.matrix).conj().resolve_conj()

    def riesz_inverse(self, residual):
        return residual

    def preconditioner_scales(self, kind):
        self.refresh()
        ones = torch.ones(self.shape[0], device=self.device, dtype=torch.float64)
        if kind == "none":
            return ones, ones, False
        cached = kind in self._scales
        if kind == "row_norm" and not cached:
            norms = torch.linalg.vector_norm(self.matrix, dim=1)
            self._scales[kind] = (torch.where(norms > 0, norms.reciprocal(), ones), ones)
        elif kind == "row_column_norm" and not cached:
            norms = torch.linalg.vector_norm(self.matrix, dim=0)
            column = torch.where(norms > 0, norms.reciprocal(), ones)
            balanced = torch.cat([torch.linalg.vector_norm(rows * column[None], dim=1)
                                  for rows in self.matrix.split(256)])
            self._scales[kind] = (torch.where(balanced > 0, balanced.reciprocal(), ones), column)
        elif kind not in ("row_norm", "row_column_norm"):
            raise ValueError("Layered dense operators support none, row_norm or row_column_norm scaling")
        return (*self._scales[kind], cached)

    def solve(self, rhs):
        """Direct solve A X = B for B [..., N] with the chunk factor and refinement."""
        self.refresh()
        system, i = self.system, self.index
        columns = rhs.reshape(-1, self.shape[0]).mT.unsqueeze(0).contiguous()
        solution, status = refine(system.matrix[i:i + 1], system.lu[i:i + 1], system.pivots[i:i + 1],
                                  system.info[i:i + 1], columns)
        if status[0][0] < 0:
            raise RuntimeError(f"Layered dense refinement did not reach backward error {BACKWARD:.3g}")
        return solution[0].mT.reshape(rhs.shape)

    def persistence_receipt(self):
        self.refresh()
        return {"version": "layered_dense_operator_v1", "operator_backend": "dense",
                "formulation_id": "mpie_layered_v1", "space": "equilibrated_loop_tree",
                "frequency_Hz": float(self.system.frequencies[self.index]), "unknowns": self.shape[0],
                "loops": self.system.loops, "factor_dtype": "complex64", "refinement": "complex128",
                "backward_error_gate": BACKWARD, "executable_snapshot": False}

    def operator_stats(self):
        self.refresh()
        count = self.system.matrix.shape[0]
        storage = self.system.matrix.nbytes + self.system.workspace.nbytes
        return {"operator_backend": "dense", "apply_calls": self._apply_calls,
                "adjoint_calls": self._adjoint_calls, "storage_bytes": storage // count,
                "storage_scope": "complex128 system plus complex64 LU workspace of this frequency",
                "workspace_bytes": self.system.workspace.nbytes // count}
