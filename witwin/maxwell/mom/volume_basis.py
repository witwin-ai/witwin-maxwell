"""Discontinuous cellwise Cartesian vector basis for volume polarization."""
import torch
from .volume_mesh import TetrahedralVolumeMesh


class CellVectorBasis:
    """Three physical vector components per cell; no hidden linear contrast."""
    def __init__(self, mesh):
        if not isinstance(mesh, TetrahedralVolumeMesh):
            raise TypeError('Cell vector basis requires a tetrahedral volume mesh')
        mesh.assert_current()
        self.mesh = mesh

    @property
    def n_dofs(self):
        self.mesh.assert_current()
        return 3*len(self.mesh.cells)

    @property
    def region_id(self):
        self.mesh.assert_current()
        return self.mesh.region_id

    def mass_apply(self, coefficients):
        self.mesh.assert_current()
        if (not isinstance(coefficients, torch.Tensor) or coefficients.device != self.mesh.vertices.device
                or coefficients.shape != (self.n_dofs,)
                or coefficients.dtype != (torch.complex128 if self.mesh.vertices.dtype == torch.float64 else torch.complex64)
                or not bool(torch.isfinite(coefficients).all())):
            raise ValueError('Volume coefficients require matching finite CUDA complex cell vectors')
        return (coefficients.reshape(-1, 3)*self.mesh.volumes[:, None]).flatten()
