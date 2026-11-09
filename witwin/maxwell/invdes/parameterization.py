"""Trainable density parameterization composing symmetry, filter and projection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .filters import DensityFilter, design_spacing
from .projection import RobustProjections, robust_projections, smoothed_projection, tanh_projection
from .symmetry import symmetrize


class DensityParameterization(torch.nn.Module):
    """``params -> symmetrize -> filter -> project`` density pipeline.

    Holds the raw design parameters as ``self.params`` (an ``nn.Parameter`` on
    the design grid, nominally in ``[0, 1]``) and returns the projected density
    that a ``SceneModule`` passes to ``MaterialRegion(density=...)``. Leave the
    region's own ``filter_radius`` / ``projection_beta`` unset when using this
    module so filtering and projection happen exactly once.

    ``size`` is the physical extent of the design Box (metres); the design-cell
    spacing is ``size / shape``. ``radius=None`` disables filtering.
    ``beta`` / ``eta`` are plain attributes so a continuation schedule can update
    them between iterations. ``smoothed=True`` selects the subpixel-smoothed
    projection, which stays differentiable at ``beta = math.inf``.
    """

    def __init__(
        self,
        shape: Sequence[int],
        size: Sequence[float],
        *,
        radius: float | None = None,
        filter_kind: str = "conic",
        padding="edge",
        pad_value: float = 0.0,
        beta: float = 1.0,
        eta: float = 0.5,
        smoothed: bool = False,
        mirror: Sequence[int] = (),
        rotation: int | None = None,
        rotation_dims: tuple[int, int] = (0, 1),
        diagonal: tuple[int, int] | None = None,
        init=0.5,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        shape = tuple(int(n) for n in shape)
        self.spacing = design_spacing(size, shape)
        if isinstance(init, torch.Tensor):
            if tuple(init.shape) != shape:
                raise ValueError(f"init tensor shape {tuple(init.shape)} does not match {shape}.")
            initial = init.detach().to(device=device, dtype=dtype).clone()
        else:
            initial = torch.full(shape, float(init), device=device, dtype=dtype)
        self.params = torch.nn.Parameter(initial)
        self.filter = None if radius is None else DensityFilter(
            radius, self.spacing, kind=filter_kind, padding=padding, pad_value=pad_value
        )
        self.beta = beta
        self.eta = float(eta)
        self.smoothed = bool(smoothed)
        self.mirror = tuple(int(d) for d in mirror)
        self.rotation = rotation
        self.rotation_dims = tuple(rotation_dims)
        self.diagonal = None if diagonal is None else tuple(diagonal)
        # Validate the symmetry against the grid once at construction.
        self.symmetrize(initial)

    def symmetrize(self, x: torch.Tensor) -> torch.Tensor:
        return symmetrize(
            x, mirror=self.mirror, rotation=self.rotation, rotation_dims=self.rotation_dims, diagonal=self.diagonal
        )

    def filtered(self) -> torch.Tensor:
        """Symmetrized and filtered parameters (the input of every projection)."""
        x = self.symmetrize(self.params)
        return x if self.filter is None else self.filter(x)

    def project(self, filtered: torch.Tensor, *, beta=None, eta=None) -> torch.Tensor:
        beta = self.beta if beta is None else beta
        eta = self.eta if eta is None else eta
        if self.smoothed:
            return smoothed_projection(filtered, beta, eta, spacing=self.spacing)
        return tanh_projection(filtered, beta, eta)

    def forward(self, *, beta=None, eta=None) -> torch.Tensor:
        return self.project(self.filtered(), beta=beta, eta=eta)

    def robust(self, delta: float = 0.1, *, beta=None) -> RobustProjections:
        """Eroded / nominal / dilated densities sharing one filtered field."""
        beta = self.beta if beta is None else beta
        filtered = self.filtered()
        if self.smoothed:
            return RobustProjections(
                eroded=self.project(filtered, beta=beta, eta=self.eta + delta),
                nominal=self.project(filtered, beta=beta),
                dilated=self.project(filtered, beta=beta, eta=self.eta - delta),
            )
        return robust_projections(filtered, beta, self.eta, delta)

    def extra_repr(self) -> str:
        beta = self.beta if isinstance(self.beta, torch.Tensor) or not math.isinf(float(self.beta)) else "inf"
        return f"shape={tuple(self.params.shape)}, beta={beta}, eta={self.eta}, smoothed={self.smoothed}"
