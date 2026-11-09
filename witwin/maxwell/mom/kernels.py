"""Versioned scalar Green kernels independent of surface discretization."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol, runtime_checkable

import torch


def cartesian_distance(displacement):
    """Euclidean length over a trailing xyz axis as one elementwise pass.

    A generic reduction over a three-element innermost axis is several times
    slower on CUDA than the explicit component sum it computes.
    """
    x, y, z = displacement.unbind(-1)
    return torch.sqrt(x*x+y*y+z*z)


@runtime_checkable
class HelmholtzKernel(Protocol):
    kernel_id: str
    version: str
    local_singularity: str

    def evaluate(self, displacement: torch.Tensor, k: complex) -> tuple[torch.Tensor, torch.Tensor]: ...
    def receipt(self) -> dict: ...


@dataclass(frozen=True)
class FreeSpaceHelmholtzKernel:
    """Outgoing exp(+ikr)/(4 pi r) for the exp(-i omega t) convention."""
    kernel_id: str = field(default='free_space_helmholtz', init=False)
    version: str = field(default='outgoing_helmholtz_v1', init=False)
    local_singularity: str = field(default='coulomb_1_over_4pi_r', init=False)

    def evaluate(self, displacement, k):
        distance = cartesian_distance(displacement)
        if bool((distance == 0).any()):
            raise ValueError('Coincident kernel nodes require singular quadrature.')
        value = torch.exp(1j*k*distance)/(4*math.pi*distance)
        gradient = (value*(1j*k-1/distance)/distance)[..., None]*displacement
        return value, gradient

    def receipt(self):
        return {'kernel_id': self.kernel_id, 'version': self.version,
                'local_singularity': self.local_singularity, 'phasor_convention': 'exp(-i omega t)'}


FREE_SPACE = FreeSpaceHelmholtzKernel()
