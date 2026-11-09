"""PyTorch-native topology-optimization / inverse-design toolkit.

Every operator here is a plain differentiable torch function or module on a
regular design grid, intended to produce the ``density`` of a
``MaterialRegion`` inside a ``SceneModule``. The native FDTD/FDFD adjoint pulls
the objective gradient back to that density, and ordinary autograd carries it
through projection, filtering and symmetry to the raw parameters.
"""

from .constraints import (
    gray_level_penalty,
    length_scale_coefficient,
    minimum_gap_constraint,
    minimum_linewidth_constraint,
    solid_indicator,
    void_indicator,
)
from .filters import (
    DensityFilter,
    apply_filter,
    conic_filter,
    design_spacing,
    disk_filter,
    filter_kernel,
    gaussian_filter,
)
from .optimize import DesignLoop, OptimizationHistory, clip_to_bounds_
from .parameterization import DensityParameterization
from .projection import (
    BetaSchedule,
    RobustProjections,
    robust_projections,
    smoothed_projection,
    tanh_projection,
)
from .symmetry import diagonal_symmetrize, mirror_symmetrize, rotational_symmetrize, symmetrize

__all__ = [
    "BetaSchedule",
    "DensityFilter",
    "DensityParameterization",
    "DesignLoop",
    "OptimizationHistory",
    "RobustProjections",
    "apply_filter",
    "clip_to_bounds_",
    "conic_filter",
    "design_spacing",
    "diagonal_symmetrize",
    "disk_filter",
    "filter_kernel",
    "gaussian_filter",
    "gray_level_penalty",
    "length_scale_coefficient",
    "minimum_gap_constraint",
    "minimum_linewidth_constraint",
    "mirror_symmetrize",
    "robust_projections",
    "rotational_symmetrize",
    "smoothed_projection",
    "solid_indicator",
    "symmetrize",
    "tanh_projection",
    "void_indicator",
]
