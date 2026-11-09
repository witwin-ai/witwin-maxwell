"""Internal regional cell-collocation H actions with original near integration.

Reference cross setup verifies all far entries; it is not a fast setup claim.
This regional volume action neither solves nor approximates surface coupling.
"""
import torch
from .volume_mesh import TetrahedralVolumeMesh
from .physics import Medium
from .tensor_medium import TensorMedium
from .dyadic_green import DyadicGreenKernel
from .vie_operator import cell_current_field_blocks
from .cluster import build_support_octree
from .hierarchical_blocks import HierarchicalBlockAction,HierarchicalBlockConfig


class VolumeCellBlocks:
    """Cell-centroid E or H response in component-major Cartesian ordering."""
    def __init__(self, mesh, medium, *, field='E', order=6, tensor_kernel=None,
                 radial_order=6, face_rule='triangle', kernel_batch=128):
        if not isinstance(mesh,TetrahedralVolumeMesh) or not isinstance(medium,(Medium,TensorMedium)):
            raise TypeError('Volume blocks require a native tetrahedral mesh and regional medium')
        tensor=isinstance(medium,TensorMedium)
        if tensor:
            if type(tensor_kernel) is not DyadicGreenKernel or tensor_kernel.medium is not medium:
                raise ValueError('Tensor volume blocks require the exact regional dyadic kernel')
            if (tensor_kernel.mode_cache.medium is not medium
                    or tensor_kernel.mode_cache.max_bytes!=tensor_kernel.cache_max_bytes):
                raise ValueError('Tensor volume cache must match its declared material and capacity')
            medium.assert_current()
            expected=torch.complex128 if mesh.vertices.dtype==torch.float64 else torch.complex64
            if medium.epsilon_r.device!=mesh.vertices.device or medium.epsilon_r.dtype!=expected:
                raise ValueError('Tensor volume material must match mesh device and precision')
            if (type(radial_order) is not int or radial_order<2 or type(kernel_batch) is not int
                    or kernel_batch<1 or face_rule not in ('triangle','projected_duffy','distance_scaled_duffy')):
                raise ValueError('Invalid tensor volume integration policy')
            from .tensor_volume import tensor_volume_static_kernels
            tensor_volume_static_kernels(medium)
        elif tensor_kernel is not None or (radial_order,face_rule,kernel_batch)!=(6,'triangle',128):
            raise ValueError('Scalar volume blocks cannot own tensor integration settings')
        self.tensor_kernel=tensor_kernel
        self.radial_order,self.face_rule,self.kernel_batch=radial_order,face_rule,kernel_batch
        if field not in ('E','H') or type(order) is not int or order < 2:
            raise ValueError('Volume blocks require E/H and integer order >= 2')
        mesh.assert_current()
        self.mesh,self.medium,self.field,self.order=mesh,medium,field,order
        self.points=mesh.vertices[mesh.cells].mean(1)
        tetrahedra=mesh.vertices[mesh.cells]
        self.lower,self.upper=tetrahedra.amin(1),tetrahedra.amax(1)
        self.device=mesh.vertices.device
        self.dtype=torch.complex128 if mesh.vertices.dtype==torch.float64 else torch.complex64
        self.size=len(mesh.cells)
        self._binding=self._identity()
        self._versions=tuple((id(t),t._version) for t in self.tensors())

    def _identity(self):
        kernel=self.tensor_kernel
        state=None if kernel is None else (id(kernel),id(kernel.medium),kernel.integration,
            kernel.cache_max_bytes,id(kernel.mode_cache),id(kernel.mode_cache.medium),
            kernel.mode_cache.max_bytes,kernel._isotropic)
        return (id(self.mesh),id(self.medium),self.field,self.order,self.size,self.device,self.dtype,
                self.radial_order,self.face_rule,self.kernel_batch,state)

    def tensors(self):
        material=() if self.tensor_kernel is None else (self.medium.epsilon_r,self.medium.mu_r)
        return (*self.mesh._tensors(),self.points,self.lower,self.upper,*material)

    def assert_current(self):
        if self._identity()!=self._binding:
            raise ValueError('Volume block binding changed')
        self.mesh.assert_current()
        if isinstance(self.medium,TensorMedium):self.medium.assert_current()
        if tuple((id(t),t._version) for t in self.tensors())!=self._versions:
            raise ValueError('Volume block geometry changed')

    def indices(self, ids):
        if (not isinstance(ids,torch.Tensor) or ids.device!=self.device or ids.dtype!=torch.int64
                or ids.ndim!=1 or bool(((ids<0)|(ids>=self.size)).any())):
            raise ValueError('Volume block cell indices must be matching CUDA int64 in range')

    def __call__(self, rows, columns):
        self.assert_current();self.indices(rows);self.indices(columns)
        if self.tensor_kernel is not None:
            from .tensor_volume import cell_tensor_current_blocks
            values,_=cell_tensor_current_blocks(self.mesh,self.points[rows],self.tensor_kernel,
                cell_indices=columns,order=self.order,radial_order=self.radial_order,
                face_rule=self.face_rule,kernel_batch=self.kernel_batch)
            block=values[:,:,:3,:3] if self.field=='E' else values[:,:,3:6,:3]
            return block.permute(2,0,3,1).reshape(3*len(rows),3*len(columns))
        e,h=cell_current_field_blocks(self.mesh,self.points[rows],self.medium,
            cell_indices=columns,order=self.order)
        block=e if self.field=='E' else h
        return block.permute(1,0,3,2).reshape(3*len(rows),3*len(columns))


class VolumeHConfig(HierarchicalBlockConfig):
    """Configuration for regional cell-volume H actions."""


class VolumeHierarchicalAction(HierarchicalBlockAction):
    """Regional cell-centroid E/H action with cell-volume L2 block scaling."""
    def __init__(self,provider,config=VolumeHConfig()):
        if not isinstance(provider,VolumeCellBlocks) or not isinstance(config,VolumeHConfig):
            raise TypeError('Volume H action requires VolumeCellBlocks and VolumeHConfig')
        provider.assert_current()
        tree=build_support_octree(provider.lower,provider.upper,leaf_size=config.leaf_size,partition_points=provider.points)
        super().__init__(provider,tree,tree,test_components=3,source_components=3,
            row_scale=provider.mesh.volumes.sqrt().repeat(3),column_scale=provider.mesh.volumes.rsqrt().repeat(3),config=config)
        self.receipt.update(version=('tensor_volume_h_reference_v1' if provider.tensor_kernel is not None else 'scalar_volume_h_reference_v1'),field=provider.field,order=provider.order,
            surface_coupling_included=False,coefficient_layout='component_major_cell',block_error_metric='cell_volume_weighted_L2_Frobenius')

        if provider.tensor_kernel is not None:
            self.receipt.update(tensor_integration=dict(radial_order=provider.radial_order,
                face_rule=provider.face_rule,kernel_batch=provider.kernel_batch,
                dyadic_integration=provider.tensor_kernel.integration.payload()),
                spectral_cache_capacity_bytes=provider.tensor_kernel.cache_max_bytes,
                spectral_cache_in_retained_bytes=False)

    @property
    def tree(self):return self.test_tree
