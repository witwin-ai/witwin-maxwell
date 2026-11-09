"""Original finite-image periodic weak blocks for independent residual sampling."""
import torch
from .nearfield import ScalarSystemBlocks
from .mesh import RWGBasis, SurfaceMesh, SurfaceSystem
from .periodic import LatticeSpec
from .periodic_operator import validate_periodic_cell


def translated_system_provider(provider, translation):
    """Translate all source bodies, preserving canonical edge and Scene ordering."""
    provider.assert_current()
    if (not isinstance(translation, torch.Tensor) or translation.shape != (3,)
            or translation.device != provider.system.device or translation.dtype != provider.system.dtype
            or translation.requires_grad or not bool(torch.isfinite(translation).all())):
        raise ValueError('Source translation must match fixed finite system geometry')
    entries = [(body.structure_index, body.name,
        RWGBasis(SurfaceMesh(body.mesh.vertices+translation, body.mesh.faces, validate=False)), body.interior)
        for body in provider.system.bodies]
    return ScalarSystemBlocks(SurfaceSystem.create(entries), provider.exterior, provider.config)


class PeriodicImageSystemBlocks(ScalarSystemBlocks):
    """Original primary PMCHWT plus exterior-only translated images.

    This finite absorbing sum matches the declared FMM image cube. Sampled
    residuals do not certify omitted images or infinite-periodic physics.
    Nonzero source geometry is streamed one image at a time without a retained
    image matrix collection. Exterior-only inherited methods remain free-space
    primitives for the periodic FMM's own hierarchical image construction.
    """
    def __init__(self, system, exterior, config, lattice, *, shells=2):
        if not isinstance(lattice, LatticeSpec):
            raise TypeError('Periodic weak blocks require LatticeSpec')
        if type(shells) is not int or shells < 1:
            raise ValueError('Periodic weak block shells must be a positive integer')
        if exterior.k.imag <= 0:
            raise ValueError('Periodic image weak blocks require positive absorption')
        validate_periodic_cell(system, lattice)
        super().__init__(system, exterior, config)
        self.lattice, self.shells = lattice, shells
        self._periodic_binding = (id(lattice), shells)
        self.metadata.update(version='finite_periodic_image_system_blocks_v1', shells=shells,
            original_residual_scope='configured_finite_image_cube', infinite_periodic_certified=False)

    def assert_current(self):
        super().assert_current()
        if (id(self.lattice), self.shells) != self._periodic_binding:
            raise ValueError('Periodic original block configuration changed')
        self.lattice.assert_current()

    def __call__(self, test_edges, source_edges):
        self.assert_current()
        result = super().__call__(test_edges, source_edges)
        width, dimension = 2*self.shells+1, self.lattice.dimension
        for start in range(0, width**dimension, 32):
            flat = torch.arange(start, min(start+32, width**dimension), device=self.system.device)
            indices = torch.stack([(flat//width**power)%width for power in reversed(range(dimension))], -1)-self.shells
            for index in indices[(indices != 0).any(-1)]:
                source = translated_system_provider(self, self.lattice.translations(index))
                result = result+self.lattice.phase(index)*self.exterior_cross_block(source, test_edges, source_edges)
                del source
        self.assert_current()
        return result
