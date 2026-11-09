"""Precorrected-FFT single-layer operator for the open-surface PEC EFIE.

The dense S-1 operator is Z = i*omega*mu*(vector - div^T scalar_p0 div / k^2)
over all sheets. Here the same D = vector - div^T S div / k^2 is applied on
one exterior grid over every sheet (``PFFTRegion`` without grad G spectra),
and every edge pair closer than the precorrection radius gets the exact entry
from the S-1 face-pair rules (``single_layer_pair_integrals`` and
``adaptive_near_single_layer``, same classification and orders as the dense
assembly). Z is symmetric, so Z^H y = conj(Z conj(y)).

The residual metric is the RWG L2 Gram of the dense S-1 operator, applied
from local 3x3 face blocks and inverted by residual-checked PCG instead of a
dense factor. Right preconditioners (for example the Calderon one, whose
setup stays dense) bind through the same handle attributes as the dense
EFIE handle.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import math
import time

import torch

from ..constants import MU_0
from .carrier import IncidentCarrier
from .kernels import FREE_SPACE
from .operator import classify_cross_pairs, local_geometry
from .pfft_grid import PFFTProjection
from .pfft_operator import PFFTRegion
from .pfft_system import PFFTPrecorrection, body_pair_chunks
from .quadrature import classify_pairs
from .riesz import LocalRWGRiesz
from .single_layer import adaptive_near_single_layer, single_layer_pair_integrals

PFFT_EFIE_VERSION = 'pfft_precorrected_open_pec_efie_v1'


def edge_face_slots(basis):
    """Per interior edge: its (plus, minus) faces [E, 2] and the local slot in each face."""
    if bool((basis.minus_faces < 0).any()):
        raise ValueError('pFFT EFIE requires two-face RWG functions (no boundary half functions)')
    edges = torch.arange(basis.n_edges, device=basis.mesh.device)
    faces = torch.stack((basis.plus_faces, basis.minus_faces), 1)
    # Boundary slots carry an arbitrary index with zero sign; match on the signed slot only.
    match = (basis.face_edges[faces] == edges[:, None, None]) & (basis.face_signs[faces] != 0)
    if not bool((match.sum(-1) == 1).all()):
        raise ValueError('Every RWG edge must own exactly one signed slot on each support face')
    return faces, match.to(torch.int8).argmax(-1)


def _face_pair_blocks(test, source, rows, cols, k, integration, kernel):
    """S-1 vector [U, 3, 3] and scalar [U] blocks of face pairs with the dense classification."""
    if test is source:
        classes = classify_pairs(test.mesh, rows, cols)
    else:
        classes, distance = classify_cross_pairs(test, source, rows, cols)
        if not bool((distance > 0).all()):
            raise ValueError('Single-layer cross blocks require strictly separated surfaces')
    dtype = torch.complex128 if test.mesh.dtype == torch.float64 else torch.complex64
    vector = torch.empty((len(rows), 3, 3), device=rows.device, dtype=dtype)
    scalar = torch.empty(len(rows), device=rows.device, dtype=dtype)
    for category in range(5):
        selected = (classes == category).nonzero().flatten()
        batch = integration.far_pair_batch if category == 4 else integration.pair_batch
        for ids in selected.split(batch):
            if not len(ids):
                continue
            if category == 3:
                v, s, _ = adaptive_near_single_layer(test, source, rows[ids], cols[ids], k, integration, kernel=kernel)
            else:
                order = integration.order if category == 4 else integration.singular_order
                v, s = single_layer_pair_integrals(test, source, rows[ids], cols[ids], k, order, category,
                                                   kernel=kernel)
            vector[ids], scalar[ids] = v.to(dtype), s.to(dtype)
    return vector, scalar


def single_layer_edge_entries(bodies, rows, cols, k, integration, *, batch, kernel=FREE_SPACE):
    """Original D[m, n] = (vector - div^T S div / k^2)[m, n] of global edge pairs (complex128).

    ``bodies`` is a sequence of (basis, edge_offset). Each pair sums its four
    support face pairs; face pairs shared inside a chunk are integrated once.
    """
    out = torch.empty(len(rows), device=rows.device, dtype=torch.complex128)
    slots = [edge_face_slots(basis) for basis, _ in bodies]
    slopes = [local_geometry(basis)[2] for basis, _ in bodies]
    for ti, si, chunk in body_pair_chunks(bodies, rows, cols, batch):
        (test, test_offset), (source, source_offset) = bodies[ti], bodies[si]
        (row_faces, row_slots), (col_faces, col_slots) = slots[ti], slots[si]
        r, c = rows[chunk]-test_offset, cols[chunk]-source_offset
        rf, rs, cf, cs = row_faces[r], row_slots[r], col_faces[c], col_slots[c]
        width = len(source.mesh.faces)
        unique, inverse = torch.unique((rf[:, :, None]*width+cf[:, None, :]).reshape(-1), return_inverse=True)
        vector, scalar = _face_pair_blocks(test, source, unique//width, unique % width, k, integration, kernel)
        inverse = inverse.reshape(-1, 2, 2)
        local = vector[inverse, rs[:, :, None].long(), cs[:, None, :].long()]
        # RWG divergence on a support face is twice its signed slope.
        divergence = 4*slopes[ti][rf, rs.long()][:, :, None]*slopes[si][cf, cs.long()][:, None, :]
        out[chunk] = (local-divergence.to(local.dtype)*scalar[inverse]/k**2).sum((1, 2)).to(torch.complex128)
    return out


class SheetRWGRiesz(LocalRWGRiesz):
    """The dense S-1 residual metric (RWG L2 Gram over all sheets) from local face blocks.

    J-only coefficient space; the inverse is the residual-checked Jacobi PCG of
    ``LocalRWGRiesz`` with no impedance balancing.
    """

    def __init__(self, system, **kwargs):
        super().__init__(system, 1., **kwargs)
        self.shape = (system.total_edges,)*2
        self.diagonal = self.diagonal[system.total_edges:].clone()
        self._versions = tuple((id(t), t._version) for t in self.tensors())

    def apply(self, value):
        self._validate(value)
        result = torch.zeros_like(value)
        for start, count, indices, blocks in self.records:
            local = torch.einsum('tij,...tj->...ti', blocks, value[..., start:start+count][..., indices])
            scatter_indices = indices.flatten().expand(*value.shape[:-1], -1)
            result[..., start:start+count].scatter_add_(-1, scatter_indices, local.flatten(-2))
        return result

    def receipt(self):
        return {**super().receipt(), 'version': 'sheet_rwg_riesz_v1', 'metric': 'rwg_L2_dual_proxy'}


class PFFTOpenPECOperator:
    """Matrix-free EFIE handle: precorrected grid single layer times i*omega*mu (iterative only)."""

    current_components = ('J',)
    supports_direct = False

    def __init__(self, system, exterior, sources, config, *, kernel=FREE_SPACE):
        from .efie_open import OpenPECOperator, efie_identity
        from .operator_protocol import IterativeState
        if config.operator_backend != 'pfft':
            raise ValueError('pFFT EFIE handle requires operator_backend=pfft')
        if kernel.receipt() != FREE_SPACE.receipt():
            raise ValueError('The open-surface EFIE currently requires the free-space kernel')
        system.assert_current()
        torch.cuda.synchronize(system.device)
        start = time.perf_counter()
        self.system, self.exterior, self.kernel = system, exterior, kernel
        self.integration, self.sources, self.pfft = config.integration, tuple(sources), config.pfft
        bases = [body.basis for body in system.bodies]
        spec = config.pfft.grid_spec(exterior, bases)
        self.region = PFFTRegion(PFFTProjection.for_bases(bases, spec), exterior, double_layer=False)
        sheets = [(body.basis, body.edge_offset) for body in system.bodies]

        def entries(rows, cols):
            return single_layer_edge_entries(sheets, rows, cols, exterior.k, config.integration,
                                             batch=config.pfft.near_pair_batch, kernel=kernel), None
        self.precorrection = PFFTPrecorrection(self.region, sheets, exterior, config.integration,
                                               batch=config.pfft.near_pair_batch, entries=entries)
        self.factor = 1j*2*math.pi*exterior.frequency*MU_0*exterior.mu_r
        self.metric = SheetRWGRiesz(system)
        self.shape = (system.total_edges,)*2
        self.device, self.dtype = system.device, torch.complex128
        self.iterative_state = IterativeState()
        self._apply_calls = self._adjoint_calls = 0
        self.incident_carrier = IncidentCarrier.compile(self.sources, exterior)
        torch.cuda.synchronize(system.device)
        self.setup_seconds = time.perf_counter()-start
        self.metadata = {**efie_identity(config), 'version': PFFT_EFIE_VERSION, 'unknown_order': ['all_sheet_J'],
                         'body_count': len(system.bodies), 'residual_metric': 'rwg_L2_dual_proxy',
                         'trace_norm_certified': False, 'coordinate_units': 'm', 'row_units': 'V*m',
                         'integration': asdict(config.integration), 'physical_qualification': False,
                         'original_global_residual_certified': False}
        self._binding = self._state()
        self._build_rhs = OpenPECOperator.build_rhs

    def _state(self):
        return (id(self.system), id(self.metric), self.pfft,
                tuple(id(t) for t in (*self.precorrection.tensors(), self.region.green_hat)))

    def refresh(self):
        if self._state() != self._binding:
            raise ValueError('pFFT EFIE tensors or bindings changed')
        self.system.assert_current()
        self.metric.assert_current()

    @contextmanager
    def read_access(self):
        self.refresh()
        try:
            yield
        finally:
            self.refresh()

    def _vector(self, value):
        if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != self.shape[0]
                or value.dtype != self.dtype or value.device != self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('pFFT EFIE vectors require finite matching complex128 CUDA coefficients')

    def _action(self, value):
        flat = value.reshape(-1, self.shape[0])
        output = torch.empty_like(flat)
        for row, x in enumerate(flat):
            d = self.region.apply_single_layer(x).to(torch.complex128)
            output[row] = self.factor*(d+self.precorrection.apply(self.precorrection.delta_d, x))
        return output.reshape(value.shape)

    def apply(self, value):
        with self.read_access():
            self._vector(value)
            self._apply_calls += 1
            return self._action(value)

    def adjoint(self, value):
        """Z^H y = conj(Z conj(y)): the Galerkin single layer is symmetric."""
        with self.read_access():
            self._vector(value)
            self._adjoint_calls += 1
            return self._action(value.conj().resolve_conj()).conj().resolve_conj()

    def riesz_inverse(self, value):
        with self.read_access():
            return self.metric.inverse(value)

    def preconditioner_scales(self, kind):
        with self.read_access():
            if kind == 'none':
                one = torch.ones_like(self.metric.diagonal)
                return one, one, False
            if kind != 'rwg_diagonal':
                raise ValueError('pFFT EFIE supports none or rwg_diagonal scaling')
            scale = self.metric.diagonal.rsqrt()
            return scale, scale, False

    def build_rhs(self):
        """-<f_m, E_inc> on every sheet, exactly as the dense EFIE handle."""
        return self._build_rhs(self)

    def receipt(self):
        region = self.region.receipt()
        correction = self.precorrection.receipt()
        return {'version': PFFT_EFIE_VERSION, 'configuration': self.pfft.payload(), 'region': region,
                'precorrection': correction, 'metric': self.metric.receipt(),
                'retained_bytes': (region['kernel_bytes']+region['projection']['retained_bytes']
                                   + correction['retained_bytes']),
                'setup_seconds': self.setup_seconds, 'physical_qualification': False}

    def persistence_receipt(self):
        self.refresh()
        return {'version': PFFT_EFIE_VERSION, 'construction': self.receipt(), 'executable_snapshot': False}

    def operator_stats(self):
        self.refresh()
        return {**self.metadata, 'apply_calls': self._apply_calls, 'adjoint_calls': self._adjoint_calls,
                'construction': self.receipt()}


def pfft_open_pec_operator(system, exterior, sources, config):
    """Construct the pFFT EFIE handle (used by ``OpenPECRuntime`` dispatch)."""
    return PFFTOpenPECOperator(system, exterior, sources, config)
