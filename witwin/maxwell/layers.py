"""Planar, nonmagnetic process-stack descriptors (SI units)."""

from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True)
class Layer:
    name: str
    thickness: float
    eps_r: complex = 1.0
    sigma: float = 0.0
    mu_r: float = 1.0

    def __post_init__(self):
        if self.mu_r != 1:
            raise ValueError("Layered Green functions require mu_r=1")


@dataclass(frozen=True)
class HalfSpace:
    eps_r: complex = 1.0
    sigma: float = 0.0
    mu_r: float = 1.0

    def __post_init__(self):
        if self.mu_r != 1:
            raise ValueError("Layered Green functions require mu_r=1")


@dataclass(frozen=True)
class LayerStack:
    """Layers ordered bottom to top, with the first finite layer starting at z=0."""

    layers: tuple[Layer, ...]
    bottom: HalfSpace | str
    top: HalfSpace | str

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))

    def interfaces(self) -> tuple[float, ...]:
        """Interface heights, including the lower and upper exterior interfaces."""
        return (0.0, *accumulate(layer.thickness for layer in self.layers))
