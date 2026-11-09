"""Selected global weak blocks for mixed scalar and tensor regional media."""
import torch
from .physics import Medium
from .tensor_medium import TensorMedium
from .dyadic_green import DyadicGreenKernel
from .operator import IntegrationConfig, PreparedRWGGeometry
from .nearfield import ScalarSystemBlocks, scalar_regional_block
from .anisotropic_operator import tensor_regional_block


class GeneralizedSystemBlocks:
    """Free-space exterior shared by all bodies; interiors remain body local.

    This provider is deliberately not a ScalarSystemBlocks subtype: scalar
    multipole kernels must not accept tensor regions through an isinstance check.
    Selected output blocks and spectral caches are retained costs, separate from
    bounded panel-pair workspaces. Material gradients remain unsupported.
    """
    _validate_selection = ScalarSystemBlocks._validate_selection
    _body_selection = ScalarSystemBlocks._body_selection
    _regional_selection = ScalarSystemBlocks._regional_selection
    _sampled_action = ScalarSystemBlocks._sampled_action
    sampled_action = ScalarSystemBlocks.sampled_action
    support_boxes = ScalarSystemBlocks.support_boxes
    geometry_tensors = ScalarSystemBlocks.geometry_tensors
    __call__ = ScalarSystemBlocks.__call__
    exterior_block = ScalarSystemBlocks.exterior_block

    def __init__(self, system, exterior, config=IntegrationConfig(), *, exterior_kernel=None,
                 interior_kernels=None):
        system.assert_current()
        if not isinstance(config, IntegrationConfig):
            raise TypeError('Generalized blocks require IntegrationConfig')
        media = (exterior, *(body.interior for body in system.bodies))
        if any(not isinstance(m, (Medium, TensorMedium)) for m in media):
            raise TypeError('Generalized blocks require sampled scalar or tensor media')
        if any(m.frequency != exterior.frequency for m in media):
            raise ValueError('Generalized regions must share one frequency')
        for medium in media:
            if isinstance(medium, TensorMedium):
                medium.assert_current()
                if (medium.epsilon_r.device != system.device or medium.epsilon_r.real.dtype != system.dtype):
                    raise ValueError('Generalized tensor media must match geometry device and precision')
                if medium.epsilon_r.requires_grad or medium.mu_r.requires_grad:
                    raise NotImplementedError('Generalized blocks require fixed material samples')
        kernels = {} if interior_kernels is None else dict(interior_kernels)
        tensor_ids = {body.region_id for body in system.bodies if isinstance(body.interior, TensorMedium)}
        if any(type(key) is not int or key not in tensor_ids for key in kernels):
            raise ValueError('Interior kernels must identify tensor body regions')
        def bind(medium, kernel):
            if not isinstance(medium, TensorMedium):
                if kernel is not None:
                    raise ValueError('Scalar regions cannot own dyadic kernels')
                return None
            kernel = DyadicGreenKernel(medium) if kernel is None else kernel
            if (not isinstance(kernel, DyadicGreenKernel) or kernel.medium is not medium
                    or kernel.mode_cache.medium is not medium
                    or kernel.mode_cache.max_bytes != kernel.cache_max_bytes):
                raise ValueError('Dyadic kernels must bind the exact regional tensor sample and cache policy')
            return kernel
        self.exterior_kernel = bind(exterior, exterior_kernel)
        self.interior_kernels = tuple(bind(body.interior, kernels.get(body.region_id)) for body in system.bodies)
        self.system, self.exterior, self.config = system, exterior, config
        self._bound = (system, exterior, config)
        self.geometry = tuple(PreparedRWGGeometry(body.basis) for body in system.bodies)
        self._geometry = self.geometry
        self.edge_stops = torch.tensor([b.edge_offset+b.basis.n_edges for b in system.bodies],
                                      device=system.device, dtype=torch.int64)
        self._edge_stops = (id(self.edge_stops), self.edge_stops._version)
        self._regional_binding = self._state()
        self.metadata = dict(version='generalized_system_blocks_v1', kernel_id='free_space_dyadic_green',
            block_requests=0, triangle_pair_counts=[0]*5, maximum_classification_pairs=0,
            maximum_block_bytes=0, physical_qualification=False)

    def _state(self):
        def kernel_state(kernel):
            return None if kernel is None else (id(kernel), id(kernel.medium), kernel.integration,
                kernel.cache_max_bytes, id(kernel.mode_cache), id(kernel.mode_cache.medium),
                kernel.mode_cache.max_bytes, kernel._isotropic)
        return (tuple(id(body.interior) for body in self.system.bodies),
                kernel_state(self.exterior_kernel), tuple(map(kernel_state, self.interior_kernels)))

    def assert_current(self):
        ScalarSystemBlocks.assert_current(self)
        if self._state() != self._regional_binding:
            raise ValueError('Generalized regional medium or kernel binding changed')
        for medium in (self.exterior, *(body.interior for body in self.system.bodies)):
            if isinstance(medium, TensorMedium):
                medium.assert_current()

    def input_tensors(self):
        values = list(ScalarSystemBlocks.input_tensors(self))
        for medium in (self.exterior, *(body.interior for body in self.system.bodies)):
            if isinstance(medium, TensorMedium):
                values.extend((medium.epsilon_r, medium.mu_r))
        return tuple({(t.device, t.untyped_storage().data_ptr()): t for t in values}.values())

    def exterior_cross_block(self, source_provider, test_edges, source_edges):
        if (not isinstance(source_provider, GeneralizedSystemBlocks)
                or source_provider.exterior is not self.exterior or source_provider.config is not self.config
                or source_provider.exterior_kernel is not self.exterior_kernel
                or source_provider.system.device != self.system.device or source_provider.system.dtype != self.system.dtype):
            raise ValueError('Generalized exterior blocks require shared medium, kernel, integration and geometry precision')
        return self._regional_selection(test_edges, source_edges, include_interiors=False,
                                        source_provider=source_provider)

    def _body_block(self, test, source, rows, cols, *, include_interiors, source_provider):
        regions = [(self.exterior, self.exterior_kernel)]
        if include_interiors and test is source:
            regions.append((test.interior, self.interior_kernels[test.region_id-1]))
        result = None
        counts, maximum = [0]*5, 0
        for medium, kernel in regions:
            if isinstance(medium, TensorMedium):
                block, receipt = tensor_regional_block(test.basis, source.basis, rows, cols,
                                                       medium, self.config, kernel=kernel)
            else:
                block, receipt = scalar_regional_block(test.basis, source.basis, rows, cols,
                    (medium,), self.config, geometry=(self.geometry[test.region_id-1],
                                                       source_provider.geometry[source.region_id-1]))
            result = block if result is None else result+block
            counts = [a+b for a, b in zip(counts, receipt['triangle_pair_counts'])]
            maximum = max(maximum, receipt['maximum_classification_pairs'])
        return result, dict(triangle_pair_counts=counts, maximum_classification_pairs=maximum)
