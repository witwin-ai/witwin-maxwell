"""Body-local analytic PMCHWT lifting with stable shared-grid kernel differences."""
from __future__ import annotations

import math
import time

import torch

from .lifting import (CURRENT_KIND, FORMULATION_ID, KERNEL_DIFFERENCE_VERSION,
                      _anisotropic_touching_values, _incident_traces, _material_kernels,
                      _merge_near, _near_values, _product_values, _touching_split,
                      _touching_values, _validated_batch, difference_kernels, medium_differences)
from .operator import NEAR_FALLBACK_BATCH, _pair_budget, _touching_batch
from .pfft_grid import PFFTProjection, _inverse_vandermonde
from .quadrature import _classify_pairs, _pair_geometry, triangle_rule


class PFFTLifting:
    """One body's analytic sources and RWG tests on one existing region grid.

    Both media use this grid. The difference is sampled with expm1 and the
    gradient series before conversion to grid precision, including weak
    contrasts. No incident trace is represented by RWG coefficients.
    """

    def __init__(self, handle, region, edge_slice, grid_region):
        from .pfft_system import near_pairs
        basis, projection = handle.basis, region.projection
        self.basis, self.exterior, self.interior = basis, handle.exterior, handle.interior
        self.region, self.grid_region = region, grid_region
        self.dtype, self.device = region.dtype, region.device
        self.indices, self.weights = region.indices[edge_slice], region.weights[edge_slice]
        self.test_base = projection.base[edge_slice][basis.face_edges]
        self.face_weights = PFFTProjection._weights(basis, projection.centres[edge_slice],
            region.config.order, region.config.spacing, face_local=True)
        centres, radii = _pair_geometry(basis.mesh)
        order, h = region.config.order, region.config.spacing
        self.source_base = torch.round((centres-projection.origin)/h-order/2).long()
        self.source_centres = projection.origin+(self.source_base.to(centres.dtype)+order/2)*h
        nodes = self.source_base[:, None]+projection.local_offsets
        self.source_indices = (nodes[..., 0]*region.shape[1]+nodes[..., 1])*region.shape[2]+nodes[..., 2]
        # Include the region's support-expanded near zone and every dense
        # near-disjoint face pair; only genuinely far pairs are left on the grid.
        self.radius = max(region.config.near_radius+2*float(projection.support_radius[edge_slice].max()),
                          4*float(radii.max()))
        rows, cols = near_pairs(centres, self.radius)
        off = rows != cols
        rows, cols = torch.cat((rows, cols[off])), torch.cat((cols, rows[off]))
        classes = _classify_pairs(basis.mesh, rows, cols, centres, radii, 2.)
        self.counts = torch.bincount(classes, minlength=5).tolist()
        sort = torch.argsort(classes, stable=True)
        self.rows, self.cols = rows[sort], cols[sort]
        self.spectra = self._spectra()
        # The same zero-displacement convention and stable samples supply
        # direct stencil interactions for exact-minus-grid precorrection.
        reach = int((self.test_base[self.rows]-self.source_base[self.cols, None]).abs().max())+order
        axis = torch.arange(-reach, reach+1, device=self.device, dtype=torch.float64)
        offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
        samples = self._samples(offsets*h)
        (vm, sm), (vj, sj) = _material_kernels(samples[..., 0], samples[..., 4], self.exterior, self.interior)
        self.table = torch.stack((vm, sm, vj, sj, *samples[..., 5:].unbind(-1)), -1).reshape(-1, 7)
        self.reach, self.width = reach, 2*reach+1

    def _samples(self, delta):
        """(G_e, grad G_e, dG, grad dG); grid self-interactions are zero."""
        nonzero = (delta != 0).any(-1)
        # difference_kernels requires noncoincident nodes. Its value at the
        # grid origin is discarded, exactly as in the matvec's kernel sampler.
        safe = torch.where(nonzero[..., None], delta, delta.new_tensor([self.region.config.spacing, 0., 0.]))
        _, _, dk = medium_differences(self.exterior, self.interior)
        ge, grad_ge, dg, grad_dg = difference_kernels(safe, self.exterior, self.interior, delta_k=dk)
        samples = torch.cat((ge[..., None], grad_ge, dg[..., None], grad_dg), -1)
        return torch.where(nonzero[..., None], samples, 0.)

    def _spectra(self, *, slab_points=1 << 18):
        """Sample in float64 slabs, then transform all eight channels once."""
        axes = [torch.arange(p, device=self.device) for p in self.region.padded]
        axes = [torch.where(a <= p//2, a, a-p).to(torch.float64) for a, p in zip(axes, self.region.padded)]
        samples = torch.empty((8, *self.region.padded), device=self.device, dtype=self.dtype)
        step = max(1, slab_points//(self.region.padded[1]*self.region.padded[2]))
        for start in range(0, self.region.padded[0], step):
            delta = torch.stack(torch.meshgrid(axes[0][start:start+step], axes[1], axes[2], indexing='ij'), -1)
            values = self._samples(delta*self.region.config.spacing)
            samples[:, start:start+step] = values.movedim(-1, 0)
        return torch.fft.fftn(samples, dim=(-3, -2, -1))

    def _source_weights(self, handles):
        """Project weighted analytic J/div J/M/div M at integration.order nodes."""
        basis, spec = self.basis, self.region.config
        dtype = self.face_weights.dtype
        inverse, _ = _inverse_vandermonde(spec.order, dtype, self.device)
        powers = torch.arange(spec.order+1, device=self.device, dtype=dtype)
        result = torch.empty((len(handles), len(basis.mesh.faces), spec.stencil_points, 8),
                             device=self.device, dtype=torch.complex128)
        for faces in torch.arange(len(basis.mesh.faces), device=self.device).split(512):
            points, weights = triangle_rule(basis.mesh.triangles[faces], handles[0].integration.order)
            u = (points-self.source_centres[faces, None])/spec.spacing
            axes = [(u[..., d, None]**powers)@inverse.T for d in range(3)]
            interpolation = torch.einsum('fqa,fqb,fqc->fqabc', *axes).flatten(2)*weights[..., None]
            j, m, div_j, div_m = _incident_traces(handles, points, faces)
            values = torch.cat((j, div_j[..., None], m, div_m[..., None]), -1)
            result[:, faces] = torch.einsum('fqp,sfqd->sfpd', interpolation.to(values.dtype), values)
        return result

    def _gather(self, spectrum, channel):
        grid = self.region._inverse(spectrum).reshape(-1)
        return (grid[self.indices]*self.weights[..., channel]).sum(-1).to(torch.complex128)

    def _grid_rhs(self, sources):
        """Stable material combinations of shared-grid analytic convolutions."""
        (vm, sm), (vj, sj) = _material_kernels(self.spectra[0], self.spectra[4], self.exterior, self.interior)
        gradients = self.spectra[5:]
        omega = self.exterior.omega
        rhs = torch.zeros((len(sources), 2, self.basis.n_edges), device=self.device, dtype=torch.complex128)
        for carrier, source in enumerate(sources):
            spectra = []
            for channel in range(8):
                # Accumulate analytic quadrature in complex128 before the FFT
                # conversion, avoiding a carrier-batch-dependent scatter error.
                grid = torch.zeros(math.prod(self.region.shape), device=self.device, dtype=torch.complex128)
                grid.index_add_(0, self.source_indices.flatten(), source[..., channel].flatten())
                spectra.append(self.region._forward(grid.reshape(self.region.shape).to(self.dtype)))
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                curl_j = gradients[b]*spectra[c]-gradients[c]*spectra[b]
                curl_m = gradients[b]*spectra[4+c]-gradients[c]*spectra[4+b]
                rhs[carrier, 0] -= self._gather(curl_j+1j*omega*vm*spectra[4+a], a)
                rhs[carrier, 1] -= self._gather(curl_m-1j*omega*vj*spectra[a], a)
            rhs[carrier, 0] += 1j*omega*self._gather(sm*spectra[7], 3)
            rhs[carrier, 1] -= 1j*omega*self._gather(sj*spectra[3], 3)
        return rhs

    def _grid_pair_values(self, rows, cols, sources):
        """The same grid action restricted to paired test/source faces [S,B,2,3]."""
        local = self.region.projection.local_offsets
        relative = local[:, None]-local[None, :]
        shift = (self.test_base[rows]-self.source_base[cols, None]).reshape(-1, 3)
        offsets = shift[:, None, None]+relative[None]+self.reach
        flat = (offsets[..., 0]*self.width+offsets[..., 1])*self.width+offsets[..., 2]
        kernels = self.table[flat]
        # Test all seven kernels once, sharing these contractions across carriers.
        wr = self.face_weights[rows].reshape(-1, len(local), 4).to(torch.complex128)
        tested = (wr.transpose(1, 2)@kernels.flatten(2)).reshape(len(wr), 4, len(local), 7)
        curl = torch.stack((tested[:, 1, :, 6]-tested[:, 2, :, 5],
                            tested[:, 2, :, 4]-tested[:, 0, :, 6],
                            tested[:, 0, :, 5]-tested[:, 1, :, 4]), -1)
        source = sources[:, cols, None].expand(-1, -1, 3, -1, -1).reshape(len(sources), -1, len(local), 8)
        j, m = source[..., :3], source[..., 4:7]
        first = torch.einsum('bqc,sbqc->sb', curl, j)+1j*self.exterior.omega*(
            torch.einsum('bcq,sbqc->sb', tested[:, :3, :, 0], m)
            - torch.einsum('bq,sbq->sb', tested[:, 3, :, 1], source[..., 7]))
        second = torch.einsum('bqc,sbqc->sb', curl, m)-1j*self.exterior.omega*(
            torch.einsum('bcq,sbqc->sb', tested[:, :3, :, 2], j)
            - torch.einsum('bq,sbq->sb', tested[:, 3, :, 3], source[..., 3]))
        return -torch.stack((first, second), -1).reshape(len(sources), len(rows), 3, 2).transpose(2, 3)

    def build_rhs_batch(self, handles):
        """Grid far lifting plus original exact-minus-grid near face pairs."""
        handles = _validated_batch(handles)
        started = time.perf_counter()
        config = handles[0].integration
        sources = self._source_weights(handles)
        rhs = self._grid_rhs(sources)
        stats = [{'maximum_order': 0, 'pair_evaluations': 0, 'maximum_accepted_relative_difference': 0.}
                 for _ in handles]
        budget = _pair_budget(config)

        def scatter(rows, values):
            rhs.index_add_(2, self.basis.face_edges[rows].flatten(), values.permute(0, 2, 1, 3).flatten(2))

        for start in range(0, len(self.rows), 256):
            rows, cols = self.rows[start:start+256], self.cols[start:start+256]
            scatter(rows, -self._grid_pair_values(rows, cols, sources))
        begin = 0
        for category, count in enumerate(self.counts):
            rows, cols = self.rows[begin:begin+count], self.cols[begin:begin+count]
            begin += count
            if category < 3:
                mask = _touching_split(self.basis.mesh, rows, cols)
                slivers, regular = mask.nonzero().flatten(), (~mask).nonzero().flatten()
                for start in range(0, len(slivers), NEAR_FALLBACK_BATCH):
                    ids = slivers[start:start+NEAR_FALLBACK_BATCH]
                    scatter(rows[ids], _anisotropic_touching_values(handles, rows[ids], cols[ids], category, config, stats))
                batch = _touching_batch(config, category, len(handles))
                for start in range(0, len(regular), batch):
                    ids = regular[start:start+batch]
                    scatter(rows[ids], _touching_values(handles, rows[ids], cols[ids], category, config.singular_order))
            else:
                batch = max(config.pair_batch, config.far_pair_batch)
                for start in range(0, count, batch):
                    r, c = rows[start:start+batch], cols[start:start+batch]
                    if category == 3:
                        values, observed = _near_values(handles, r, c, config, budget)
                        for total, record in zip(stats, observed):
                            _merge_near(total, record)
                    else:
                        values = _product_values(handles, r, c, config.order, budget)
                    scatter(r, values)
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter()-started
        records = [{'version': 'pfft_shared_grid_analytic_lifted_rhs_v1',
                    'formulation_id': FORMULATION_ID, 'current_kind': CURRENT_KIND,
                    'incident_carrier_identity': h.incident_carrier.identity,
                    'difference_kernel_version': KERNEL_DIFFERENCE_VERSION,
                    'rhs_far_acceleration': 'pfft_shared_grid_stable_difference',
                    'rhs_lifting_grid': self.receipt(), 'rhs_near_face_pairs': len(self.rows),
                    'rhs_triangle_pair_counts': self.counts, 'rhs_near_quadrature': record,
                    'rhs_integration': vars(config), 'rhs_source_batch': len(handles), 'rhs_seconds': seconds,
                    'global_rhs_error_certified': False}
                   for h, record in zip(handles, stats)]
        return rhs.flatten(1), records

    def receipt(self):
        return {'grid_region': self.grid_region, 'spacing_m': self.region.config.spacing,
                'order': self.region.config.order, 'precision': self.region.config.precision,
                'grid_shape': list(self.region.shape), 'padded_shape': list(self.region.padded),
                'kernel_channels': ['G_e', 'grad_G_e_x', 'grad_G_e_y', 'grad_G_e_z',
                                    'dG', 'grad_dG_x', 'grad_dG_y', 'grad_dG_z'],
                'kernel_bytes': self.spectra.numel()*self.spectra.element_size(),
                'near_radius_m': self.radius, 'near_face_pairs': len(self.rows),
                'far_face_pairs': len(self.basis.mesh.faces)**2-len(self.rows)}
