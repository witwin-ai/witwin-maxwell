"""FFT-accelerated single-medium RWG operators on a uniform grid (pFFT/AIM).

For one homogeneous medium with wavenumber k this applies the two original
scalar channels used by every RWG surface formulation in this package:

    D[i, j] = int int (f_i . f_j - div f_i div f_j / k^2) G(r - r') dS dS'
    K[i, j] = int f_i(r) . int grad G(r - r') x f_j(r') dS' dS

``D`` is the single-layer block: alone it gives the EFIE operator
(Z = i omega mu D) for open or closed PEC surfaces; PMCHWT combines D and K of
each region. Sources are projected onto the grid (``PFFTProjection``),
convolved with G and its gradient sampled at grid offsets through zero-padded
FFTs, and interpolated back with the same weights. Grid interactions between
nearby bases are inaccurate by construction; ``grid_entries`` returns them so
they can be replaced by original near-field integrals (precorrection).
"""
from __future__ import annotations

import math
import time

import torch

from .pfft_grid import PFFTGridSpec, PFFTProjection


def _fast_size(minimum):
    """Smallest 2^a 3^b 5^c not below ``minimum`` (efficient FFT length)."""
    best = 1 << max(0, math.ceil(math.log2(max(1, minimum))))
    for five in (1, 5, 25, 125):
        for three in (1, 3, 9, 27, 81):
            value = five*three
            while value < minimum:
                value *= 2
            best = min(best, value)
    return best


def green_samples(offsets, spacing, k, *, gradient):
    """G and optionally grad G at displacement offsets*spacing; zero displacement gives zero."""
    displacement = offsets*spacing
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    safe = torch.where(distance > 0, distance, torch.ones_like(distance))
    value = torch.exp(1j*k*safe)/(4*math.pi*safe)
    value = torch.where(distance > 0, value, torch.zeros_like(value))
    if not gradient:
        return value, None
    factor = value*(1j*k-1/safe)/safe
    return value, factor[..., None]*displacement


class PFFTRegion:
    """D and K actions of the bases of one projection in one scalar medium on a padded FFT grid."""

    def __init__(self, projection, medium, *, double_layer=True):
        if not isinstance(projection, PFFTProjection):
            raise TypeError('PFFTRegion requires a PFFTProjection')
        config = projection.config
        torch.cuda.synchronize(projection.device)
        start = time.perf_counter()
        self.medium, self.config = medium, config
        self.k = complex(medium.k)
        self.projection = projection
        self.count = projection.count
        self.device = projection.device
        self.real_dtype = torch.float64 if config.precision == 'complex128' else torch.float32
        self.dtype = torch.complex128 if config.precision == 'complex128' else torch.complex64
        self.shape = self.projection.shape
        self.padded = tuple(_fast_size(2*n-1) for n in self.shape)
        self.indices = self.projection.indices
        self.weights = self.projection.weights.to(self.real_dtype)
        # A single-layer-only region (open-sheet EFIE) keeps no grad G spectra.
        self.double_layer = bool(double_layer)
        self.green_hat = self._kernel_spectrum(0)
        self.gradient_hat = tuple(self._kernel_spectrum(d) for d in (1, 2, 3)) if self.double_layer else ()
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter()-start

    def _kernel_spectrum(self, channel, *, slab_points=1 << 24):
        """FFT of G (channel 0) or one grad G component (1..3) sampled on the padded grid.

        Samples are evaluated in float64 by x-slabs of at most ``slab_points`` and
        stored directly in the grid precision, so the float64 workspace stays
        bounded on large grids; values are identical to a single evaluation.
        """
        axes = [torch.arange(p, device=self.device) for p in self.padded]
        axes = [torch.where(a <= p//2, a, a-p).to(torch.float64) for a, p in zip(axes, self.padded)]
        samples = torch.empty(self.padded, device=self.device, dtype=self.dtype)
        step = max(1, slab_points//(self.padded[1]*self.padded[2]))
        for start in range(0, self.padded[0], step):
            offsets = torch.stack(torch.meshgrid(axes[0][start:start+step], axes[1], axes[2], indexing='ij'), -1)
            value, gradient = green_samples(offsets, float(self.config.spacing), self.k, gradient=channel > 0)
            samples[start:start+step] = value if channel == 0 else gradient[..., channel-1]
            del offsets, value, gradient
        return torch.fft.fftn(samples)

    # -- grid transfer ---------------------------------------------------
    def _scatter(self, coefficients, channel):
        grid = torch.zeros(math.prod(self.shape), device=self.device, dtype=self.dtype)
        values = self.weights[..., channel]*coefficients[:, None]
        grid.index_add_(0, self.indices.flatten(), values.flatten())
        return grid.reshape(self.shape)

    def _gather(self, grid, channel):
        values = grid.reshape(-1)[self.indices]
        return (values*self.weights[..., channel]).sum(-1)

    def _forward(self, grid):
        return torch.fft.fftn(grid, s=self.padded)

    def _inverse(self, spectrum):
        result = torch.fft.ifftn(spectrum)
        return result[:self.shape[0], :self.shape[1], :self.shape[2]]

    @classmethod
    def from_basis(cls, basis, medium, config, *, double_layer=True):
        return cls(PFFTProjection(basis, config), medium, double_layer=double_layer)

    def _require_double_layer(self):
        if not self.double_layer:
            raise ValueError('This pFFT region was built for the single layer only (double_layer=False)')

    def _check(self, value):
        if (not isinstance(value, torch.Tensor) or value.shape != (self.count,)
                or value.device != self.device or not value.is_complex()):
            raise ValueError('pFFT actions require one complex CUDA coefficient vector per basis')
        return value.to(self.dtype)

    # -- actions -----------------------------------------------------------
    def _potential(self, x, channels):
        result = torch.zeros_like(x)
        for channel in channels:
            potential = self._inverse(self._forward(self._scatter(x, channel))*self.green_hat)
            result = result+self._gather(potential, channel)
        return result

    def apply_vector_potential(self, value):
        """Grid int int f_i . f_j G (the vector-potential part of the single layer)."""
        return self._potential(self._check(value), (0, 1, 2))

    def apply_scalar_potential(self, value):
        """Grid int int div f_i div f_j G, i.e. div^T S_p0 div with piecewise-constant divergence."""
        return self._potential(self._check(value), (3,))

    def apply_single_layer(self, value):
        """Grid D x = vector - div^T S div / k^2 (the single-layer L block); EFIE is i omega mu D."""
        x = self._check(value)
        return self._potential(x, (0, 1, 2))-self._potential(x, (3,))/self.k**2

    def apply_channels(self, value):
        """(D x, K x) sharing the four forward transforms of the projected sources."""
        self._require_double_layer()
        x = self._check(value)
        sources = [self._forward(self._scatter(x, channel)) for channel in range(4)]
        d = torch.zeros_like(x)
        for channel in range(3):
            d = d+self._gather(self._inverse(sources[channel]*self.green_hat), channel)
        d = d-self._gather(self._inverse(sources[3]*self.green_hat), 3)/self.k**2
        gx, gy, gz = self.gradient_hat
        fields = (gy*sources[2]-gz*sources[1], gz*sources[0]-gx*sources[2], gx*sources[1]-gy*sources[0])
        k = torch.zeros_like(x)
        for channel, spectrum in enumerate(fields):
            k = k+self._gather(self._inverse(spectrum), channel)
        return d, k

    def apply_double_layer(self, value):
        """Grid K x: f_i . (grad G * f_j)."""
        self._require_double_layer()
        x = self._check(value)
        sources = [self._forward(self._scatter(x, channel)) for channel in range(3)]
        gx, gy, gz = self.gradient_hat
        fields = (gy*sources[2]-gz*sources[1], gz*sources[0]-gx*sources[2], gx*sources[1]-gy*sources[0])
        result = torch.zeros_like(x)
        for channel, spectrum in enumerate(fields):
            result = result+self._gather(self._inverse(spectrum), channel)
        return result

    # -- near-zone grid entries for precorrection ---------------------------
    def _offset_table(self, reach):
        """G and grad G at integer offsets in [-reach, reach]^3 (cached real-space table)."""
        cached = getattr(self, '_table', None)
        if cached is not None and cached[0] >= reach:
            return cached
        axis = torch.arange(-reach, reach+1, device=self.device, dtype=torch.float64)
        offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
        value, gradient = green_samples(offsets, float(self.config.spacing), self.k, gradient=True)
        table = torch.cat((value[..., None], gradient), -1).reshape(-1, 4)
        self._table = (reach, table)
        return self._table

    def grid_entries(self, rows, cols, *, batch=512):
        """Grid-operator entries D_grid[rows, cols], K_grid[rows, cols] for paired bases.

        Every stencil-to-stencil offset is an integer vector, so G and grad G come
        from one cached table of integer offsets instead of fresh evaluations.
        A single-layer-only region returns ``None`` for K.
        """
        order = self.config.order
        base = self.projection.base
        local = self.projection.local_offsets
        relative = (local[:, None, :]-local[None, :, :])
        shift_all = base[rows]-base[cols]
        reach = int(shift_all.abs().max())+order if len(rows) else order
        reach, table = self._offset_table(reach)
        width = 2*reach+1
        d_out = torch.empty(len(rows), device=self.device, dtype=torch.complex128)
        k_out = torch.empty_like(d_out) if self.double_layer else None
        weights = self.projection.weights
        for start in range(0, len(rows), batch):
            r, c = rows[start:start+batch], cols[start:start+batch]
            offsets = shift_all[start:start+batch, None, None, :]+relative[None]+reach
            flat = (offsets[..., 0]*width+offsets[..., 1])*width+offsets[..., 2]
            values = table[flat]
            wr, wc = weights[r].to(torch.complex128), weights[c].to(torch.complex128)
            value, gradient = values[..., 0], values[..., 1:]
            d = sum(torch.einsum('bp,bpq,bq->b', wr[..., ch], value, wc[..., ch]) for ch in range(3))
            d = d-torch.einsum('bp,bpq,bq->b', wr[..., 3], value, wc[..., 3])/self.k**2
            d_out[start:start+batch] = d
            if k_out is None:
                continue
            kk = 0
            for cc, dd, ee in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                kk = kk+torch.einsum('bp,bpq,bq->b', wr[..., cc], gradient[..., dd], wc[..., ee])
                kk = kk-torch.einsum('bp,bpq,bq->b', wr[..., cc], gradient[..., ee], wc[..., dd])
            k_out[start:start+batch] = kk
        return d_out, k_out

    def receipt(self):
        grid = math.prod(self.padded)*torch.empty((), dtype=self.dtype).element_size()
        return {'version': 'pfft_single_medium_region_v1', 'k': [self.k.real, self.k.imag],
                'grid_shape': list(self.shape), 'padded_shape': list(self.padded),
                'padded_points': math.prod(self.padded), 'precision': self.config.precision,
                'kernel_bytes': (1+len(self.gradient_hat))*grid, 'work_grid_bytes': grid,
                'double_layer': self.double_layer,
                'projection': self.projection.receipt(), 'setup_seconds': self.setup_seconds}
