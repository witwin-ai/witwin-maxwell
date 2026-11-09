"""Selected original Ewald weak blocks without a retained global matrix."""
import math
import torch
from .nearfield import ScalarSystemBlocks, _selected_edges, _local_indices
from .tensor_nearfield import GeneralizedSystemBlocks
from .physics import Medium
from .periodic import EwaldHelmholtzKernel
from .periodic_operator import validate_periodic_cell, _RegularKernel
from .operator import _pair_integrals
from ..constants import EPSILON_0, MU_0


class _EwaldBlockMixin:
    """Primary PMCHWT plus configured Ewald exterior regular contribution.

    Inherited exterior-only methods remain free-space primitives for periodic
    FMM near-image construction. Only full selected blocks are periodized.
    """
    def __init__(self, system, exterior, config, kernel, **regional_options):
        if not isinstance(kernel, EwaldHelmholtzKernel):
            raise TypeError('Original Ewald blocks require EwaldHelmholtzKernel')
        if not isinstance(exterior, Medium):
            raise TypeError('Ewald block exterior must be a sampled scalar medium')
        validate_periodic_cell(system, kernel.lattice)
        kernel._guard(complex(exterior.k))
        super().__init__(system, exterior, config, **regional_options)
        self.kernel = kernel
        self._kernel_binding = self._kernel_state()
        self.metadata.update(version=self._block_version, kernel_id=kernel.kernel_id,
            kernel_version=kernel.version,
            original_residual_scope='configured_ewald_quadrature_and_cutoffs', physical_qualification=False)

    def _kernel_state(self):
        return (id(self.kernel), id(self.kernel.lattice), *(getattr(self.kernel, name) for name in
            ('eta', 'real_shells', 'reciprocal_shells', 'quadrature_order', 'image_batch', 'query_batch',
             'pole_relative_tolerance', 'kernel_id', 'version')))

    def assert_current(self):
        super().assert_current()
        if self._kernel_state() != self._kernel_binding:
            raise ValueError('Original Ewald kernel binding changed')
        self.kernel.lattice.assert_current()

    def __call__(self, test_edges, source_edges):
        self.assert_current()
        return super().__call__(test_edges, source_edges) + self.regular_block(test_edges, source_edges)

    def regular_block(self, test_edges, source_edges):
        """Smooth exterior correction only, without cancellation against primary blocks."""
        self.assert_current()
        self._validate_selection(test_edges)
        self._validate_selection(source_edges)
        nr, nc = len(test_edges), len(source_edges)
        result = torch.zeros((2*nr, 2*nc), device=self.system.device, dtype=torch.complex128 if self.system.dtype == torch.float64 else torch.complex64)
        regular = _RegularKernel(self.kernel)
        omega = 2*math.pi*self.exterior.frequency
        for ti, selected_r in self._body_selection(test_edges):
            test = self.system.bodies[ti]
            re, rp, rf = _selected_edges(test.basis, test_edges[selected_r]-test.edge_offset)
            for si, selected_c in self._body_selection(source_edges):
                source = self.system.bodies[si]
                ce, cp, cf = _selected_edges(source.basis, source_edges[selected_c]-source.edge_offset)
                for start in range(0, len(rf)*len(cf), self.config.pair_batch):
                    ids = torch.arange(start, min(start+self.config.pair_batch, len(rf)*len(cf)), device=self.system.device)
                    rows, cols = rf[ids//len(cf)], cf[ids % len(cf)]
                    ri, rv = _local_indices(test.basis.face_edges[rows], re, rp)
                    ci, cv = _local_indices(source.basis.face_edges[cols], ce, cp)
                    keep = rv[:, :, None] & cv[:, None, :]
                    rr = selected_r[ri[:, :, None].expand(-1, -1, 3)[keep]]
                    cc = selected_c[ci[:, None, :].expand(-1, 3, -1)[keep]]
                    d, magnetic = _pair_integrals(test.basis, rows, cols, self.exterior,
                        self.config.order, self.config.order, False, kernel=regular,
                        source_basis=source.basis, zero_planar_self=False,
                        geometry=(self.geometry[ti], self.geometry[si]))
                    for i, j, value in ((0, 0, magnetic), (1, 1, magnetic),
                        (0, 1, 1j*omega*EPSILON_0*self.exterior.epsilon_r*d),
                        (1, 0, -1j*omega*MU_0*self.exterior.mu_r*d)):
                        result.index_put_((rr+i*nr, cc+j*nc), value[keep], accumulate=True)
        self.assert_current()
        return result

    def sampled_action(self, test_edges, value, *, source_batch=64):
        """Original action with each smooth source-face pair integrated once.

        Primary singular blocks retain bounded edge batches. Smooth Ewald
        blocks contract all three source-face currents before selected-row
        scatter, avoiding repeated face integrations across edge batches.
        Forward pair storage is bounded; autograd may retain all pair blocks.
        """
        result = self._sampled_action(test_edges, value, source_batch, super().__call__)
        total, nr = self.system.total_edges, len(test_edges)
        regular = _RegularKernel(self.kernel)
        omega = 2*math.pi*self.exterior.frequency
        for ti, selected_r in self._body_selection(test_edges):
            test = self.system.bodies[ti]
            re, rp, rf = _selected_edges(test.basis, test_edges[selected_r]-test.edge_offset)
            for si, source in enumerate(self.system.bodies):
                nf = len(source.mesh.faces)
                for start in range(0, len(rf)*nf, self.config.pair_batch):
                    ids = torch.arange(start, min(start+self.config.pair_batch, len(rf)*nf),
                                       device=self.system.device)
                    rows, cols = rf[ids//nf], ids % nf
                    ri, valid = _local_indices(test.basis.face_edges[rows], re, rp)
                    rr = selected_r[ri[valid]]
                    d, magnetic = _pair_integrals(test.basis, rows, cols, self.exterior,
                        self.config.order, self.config.order, False, kernel=regular,
                        source_basis=source.basis, zero_planar_self=False,
                        geometry=(self.geometry[ti], self.geometry[si]))
                    edges = source.basis.face_edges[cols]+source.edge_offset
                    j, m = value[..., edges], value[..., edges+total]
                    h = torch.einsum('plk,...pk->...pl', magnetic, j)
                    h = h+1j*omega*EPSILON_0*self.exterior.epsilon_r*torch.einsum('plk,...pk->...pl', d, m)
                    e = torch.einsum('plk,...pk->...pl', magnetic, m)
                    e = e-1j*omega*MU_0*self.exterior.mu_r*torch.einsum('plk,...pk->...pl', d, j)
                    result = result.index_add(-1, rr, h[..., valid])
                    result = result.index_add(-1, rr+nr, e[..., valid])
        self.assert_current()
        return result


class EwaldSystemBlocks(_EwaldBlockMixin, ScalarSystemBlocks):
    """Scalar original Ewald blocks, including periodic FMM reference blocks."""
    _block_version = 'ewald_original_system_blocks_v1'


class GeneralizedEwaldSystemBlocks(_EwaldBlockMixin, GeneralizedSystemBlocks):
    """Scalar Ewald exterior and body-local tensor/scalar primary interiors.

    This provider deliberately is not a ScalarSystemBlocks subtype: scalar FMM
    cannot consume tensor interior regions through its scalar admission check.
    Selected blocks and sampled residual actions share the exterior correction;
    exterior-only primitives remain free space for existing regional protocols.
    """
    _block_version = 'ewald_generalized_system_blocks_v1'
