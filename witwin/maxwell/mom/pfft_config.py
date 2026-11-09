"""Versioned configuration of the precorrected-FFT (pFFT/AIM) MoM backend."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .pfft_grid import PFFTGridSpec

PFFT_CONFIG_VERSION = 'pfft_precorrected_config_v2'
SPACING_RULE_VERSION = 'pfft_mesh_tied_spacing_v1'

# Operator-action targets characterised on sphere meshes from lambda/15 to lambda/30
# (docs/plans/mom_large_scale_learned_operators.md, L-1c.2): target -> (stencil order,
# minimum grid points per medium wavelength, largest grid spacing / mean RWG edge).
ACCURACY_TABLE = ((1e-2, 3, 10., 1.5), (1e-3, 3, 15., 1.), (1e-4, 4, 15., 1.))
# Below a quarter of the mean edge the stencil no longer spans the RWG support: the
# action error grows again while the grid volume grows as (h/h_g)^3.
MINIMUM_SPACING_TO_EDGE = 0.25


@dataclass(frozen=True)
class PFFTConfig:
    """Grid density, stencil order and admission tolerances of the pFFT backend.

    Each region (the exterior over all bodies, and each body interior) gets a
    uniform grid with (order+1)^3 stencils. By default the spacing is tied to
    the mesh and the operator-action ``target_accuracy``: the coarsest of
    lambda/ppw and (ratio * mean RWG edge) from ``ACCURACY_TABLE``, never below
    a quarter of the mean edge. An explicit ``points_per_wavelength`` fixes
    lambda/ppw instead and fails closed when that is finer than a quarter of
    the mean edge. Grid interactions between bases closer than
    ``precorrection_factor`` grid spacings (default order+2) plus twice the
    largest RWG support radius are replaced by the original near-field
    integrals in complex128. ``grid_precision`` selects the FFT grid and kernel
    precision; projection weights, precorrection and every residual stay in
    float64/complex128.

    ``exterior_grids='clusters'`` replaces the single exterior grid by one grid
    per cluster of bodies (clusters are merged until every pair of bounding
    spheres has centre distance >= ``cluster_separation`` times the radius sum)
    with single-level plane-wave coupling between clusters, truncated for
    ``coupling_digits`` (default 1 - log10(target_accuracy)).
    """
    points_per_wavelength: float | None = None
    order: int | None = None
    target_accuracy: float = 1e-3
    grid_precision: str = 'complex64'
    precorrection_factor: float | None = None
    near_pair_batch: int = 1 << 18
    tree_leaf_size: int = 32
    solve_dual_tolerance: float = 1e-8
    original_sample_tolerance: float = 1e-8
    exterior_grids: str = 'single'
    cluster_separation: float = 2.
    coupling_digits: float | None = None

    def __post_init__(self):
        if self.points_per_wavelength is not None and (
                type(self.points_per_wavelength) not in (int, float) or not math.isfinite(self.points_per_wavelength)
                or self.points_per_wavelength < 4):
            raise ValueError('pFFT points_per_wavelength must be finite and at least 4')
        if self.order is not None and (type(self.order) is not int or not 1 <= self.order <= 6):
            raise ValueError('pFFT order must be an integer in 1..6')
        if (type(self.target_accuracy) not in (int, float) or not math.isfinite(self.target_accuracy)
                or self.target_accuracy < ACCURACY_TABLE[-1][0] or self.target_accuracy >= 1):
            raise ValueError(f'pFFT target_accuracy must lie in [{ACCURACY_TABLE[-1][0]:g}, 1); tighter '
                             'operator accuracy is not characterised')
        if self.exterior_grids not in ('single', 'clusters'):
            raise ValueError("pFFT exterior_grids must be 'single' or 'clusters'")
        if (type(self.cluster_separation) not in (int, float) or not math.isfinite(self.cluster_separation)
                or self.cluster_separation < 1.5):
            raise ValueError('pFFT cluster_separation must be finite and at least 1.5')
        if self.coupling_digits is not None and (type(self.coupling_digits) not in (int, float)
                                                 or not 1 <= self.coupling_digits <= 12):
            raise ValueError('pFFT coupling_digits must lie in [1, 12]')
        if self.grid_precision not in ('complex64', 'complex128'):
            raise ValueError('pFFT grid_precision must be complex64 or complex128')
        if self.precorrection_factor is not None and (type(self.precorrection_factor) not in (int, float)
                                                      or not math.isfinite(self.precorrection_factor)
                                                      or self.precorrection_factor < self.stencil_order):
            raise ValueError('pFFT precorrection_factor must be finite and at least the stencil order')
        for name in ('near_pair_batch', 'tree_leaf_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name+' must be a positive integer')
        for name in ('solve_dual_tolerance', 'original_sample_tolerance'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')

    def _accuracy_row(self):
        return next(row for row in ACCURACY_TABLE if self.target_accuracy >= row[0])

    @property
    def stencil_order(self):
        return self._accuracy_row()[1] if self.order is None else self.order

    @property
    def digits(self):
        return 1-math.log10(self.target_accuracy) if self.coupling_digits is None else self.coupling_digits

    def grid_spec(self, medium, bases):
        """Grid of one region from its medium wavelength and the mean RWG edge of ``bases``."""
        wavelength = 2*math.pi/complex(medium.k).real
        edge = float(sum(float(b.edge_lengths.sum()) for b in bases)/sum(b.n_edges for b in bases))
        _, _, table_ppw, ratio = self._accuracy_row()
        floor = MINIMUM_SPACING_TO_EDGE*edge
        if self.points_per_wavelength is not None:
            spacing, rule = wavelength/self.points_per_wavelength, 'explicit_points_per_wavelength'
            if spacing < floor:
                raise ValueError(f'pFFT grid spacing {spacing:.4g} m is finer than {MINIMUM_SPACING_TO_EDGE:g} '
                                 f'of the mean RWG edge {edge:.4g} m: the stencil no longer spans a basis '
                                 'support. Lower points_per_wavelength or leave it unset for the mesh-tied rule')
        else:
            candidates = {'wavelength': wavelength/table_ppw, 'mesh': ratio*edge}
            rule = min(candidates, key=candidates.get)
            spacing = candidates[rule]
            if spacing < floor:
                spacing, rule = floor, 'quarter_edge_floor'
        order = self.stencil_order
        factor = order+2 if self.precorrection_factor is None else self.precorrection_factor
        return PFFTGridSpec(spacing=spacing, order=order, precision=self.grid_precision,
                            precorrection_radius=factor*spacing, mean_edge=edge, spacing_rule=rule)

    def solve_config(self):
        from .direct_operator import DirectSolveConfig
        return DirectSolveConfig(self.solve_dual_tolerance, self.original_sample_tolerance)

    def payload(self):
        return {'version': PFFT_CONFIG_VERSION, 'spacing_rule': SPACING_RULE_VERSION, 'parameters': asdict(self)}

    @classmethod
    def restore(cls, payload):
        if (not isinstance(payload, dict) or set(payload) != {'version', 'spacing_rule', 'parameters'}
                or payload['version'] != PFFT_CONFIG_VERSION or payload['spacing_rule'] != SPACING_RULE_VERSION
                or not isinstance(payload['parameters'], dict)
                or set(payload['parameters']) != set(cls.__dataclass_fields__)):
            raise ValueError('Unsupported or incomplete pFFT configuration')
        return cls(**payload['parameters'])


def admit_pfft(config):
    """Reject every MoMConfig combination the pFFT backend does not implement."""
    import torch
    from .linalg import StoppingRule
    unsupported = {'periodic': config.periodic, 'harmonic_balance': config.harmonic_balance,
                   'dyadic_integration': config.dyadic_integration, 'linear_coupled': config.linear_coupled,
                   'hierarchical': config.hierarchical, 'h2': config.h2, 'direct_h2': config.direct_h2,
                   'fmm': config.fmm}
    present = [name for name, value in unsupported.items() if value is not None]
    if present:
        raise NotImplementedError('pFFT currently supports free-space scalar linear PMCHWT only; remove '
                                  + ', '.join(present))
    if config.dtype != torch.complex128:
        raise ValueError('pFFT requires complex128 currents (grid precision is set by PFFTConfig)')
    if not isinstance(config.stopping, StoppingRule) or config.stopping.preconditioner not in ('none', 'rwg_diagonal'):
        raise ValueError('pFFT requires none or rwg_diagonal scaling')
