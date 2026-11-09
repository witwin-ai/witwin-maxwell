"""Global selected tensor-periodic blocks for hierarchical construction."""
from .tensor_nearfield import GeneralizedSystemBlocks
from .tensor_medium import TensorMedium
from .periodic_dyadic import REGULAR_DYADIC_KERNEL_TYPES,QuasiperiodicDyadicRegularKernel
from .periodic_operator import validate_periodic_cell
from .periodic_dyadic_operator import periodic_dyadic_system_block
from .operator import IntegrationConfig


class PeriodicDyadicSystemBlocks(GeneralizedSystemBlocks):
    """Complete global selected blocks; exterior-only primitives stay free space.

    All selected J entries precede all selected M entries. Sampled actions stream
    complete blocks over source edges. Output blocks, spectral caches and any
    autograd graph are retained costs; this is not a linear-complexity claim.
    """
    def __init__(self,system,reference,config=IntegrationConfig(),*,exterior_kernel,interior_kernels):
        if not isinstance(reference,REGULAR_DYADIC_KERNEL_TYPES):
            raise TypeError('Periodic dyadic blocks require a supported regular kernel')
        validate_periodic_cell(system,reference.lattice)
        reference._validate()
        tensor_ids={body.region_id for body in system.bodies if isinstance(body.interior,TensorMedium)}
        if type(interior_kernels) is not dict or set(interior_kernels)!=tensor_ids:
            raise ValueError('Periodic dyadic blocks require explicit kernels for all tensor interiors')
        super().__init__(system,reference.medium,config,exterior_kernel=exterior_kernel,
            interior_kernels=interior_kernels)
        self.reference=reference
        self._periodic_binding=self._periodic_state()
        self.metadata.update(version='periodic_dyadic_system_blocks_v1',kernel_id=reference.kernel_id,
            nonprimary_panel_pairs=0,maximum_nonprimary_panel_pairs=0,
            complete_operator=True,interiors_periodic=False,physical_qualification=False)

    def _periodic_state(self):
        kernel=self.reference
        policy=(kernel.identity if isinstance(kernel,QuasiperiodicDyadicRegularKernel)
                else (kernel.shells,kernel.image_batch,kernel.integration))
        return id(kernel),id(kernel.medium),id(kernel.lattice),policy

    def assert_current(self):
        super().assert_current()
        if self._periodic_state()!=self._periodic_binding:
            raise ValueError('Periodic dyadic selected kernel binding changed')
        self.reference._validate()

    def input_tensors(self):
        lattice=self.reference.lattice
        values=(*super().input_tensors(),lattice.vectors,lattice.bloch,lattice.reciprocal)
        if isinstance(self.reference,QuasiperiodicDyadicRegularKernel):
            values=(*values,self.reference.normal)
        return tuple({(value.device,value.untyped_storage().data_ptr()):value for value in values}.values())

    def _body_block(self,test,source,rows,cols,*,include_interiors,source_provider):
        if not include_interiors:
            return super()._body_block(test,source,rows,cols,include_interiors=False,
                source_provider=source_provider)
        if source_provider is not self:
            raise ValueError('Complete periodic blocks require their bound global source system')
        kernels={body.region_id:kernel for body,kernel in zip(self.system.bodies,self.interior_kernels)
                 if kernel is not None}
        block,receipt=periodic_dyadic_system_block(self.system,test.region_id-1,source.region_id-1,
            rows,cols,reference=self.reference,exterior_kernel=self.exterior_kernel,
            interior_kernels=kernels,config=self.config)
        primary=[receipt['exterior_primary']]
        if receipt['local_interior'] is not None:primary.append(receipt['local_interior'])
        regular=receipt['nonprimary_exterior']
        self.metadata['nonprimary_panel_pairs']+=regular['panel_pairs']
        self.metadata['maximum_nonprimary_panel_pairs']=max(self.metadata['maximum_nonprimary_panel_pairs'],
            regular['maximum_panel_pairs'])
        return block,dict(triangle_pair_counts=[sum(value['triangle_pair_counts'][i] for value in primary) for i in range(5)],
            maximum_classification_pairs=max(value['maximum_classification_pairs'] for value in primary))
