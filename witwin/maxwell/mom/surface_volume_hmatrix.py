"""Regional rectangular H maps from cell currents to weak RWG source rows."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import torch
from ..constants import EPSILON_0,MU_0
from .tensor_medium import TensorMedium
from .dyadic_green import DyadicGreenKernel
from .volume_mesh import BoundVolumeRegion
from .volume_response_matrix import build_volume_source_block
from .cluster import build_support_octree
from .hierarchical_blocks import HierarchicalBlockAction,HierarchicalBlockConfig


@dataclass(frozen=True)
class _SourceLayout:
    system: object
    device: object
    dtype: object
    shape: tuple
    region_id: int
    tensor_kernel: object = None

    @property
    def body_handles(self):
        return tuple(SimpleNamespace(tensor_kernel=self.tensor_kernel if b.region_id==self.region_id else None)
                     for b in self.system.bodies)

    def validate_tensor_state(self):
        medium=self.system.bodies[self.region_id-1].interior
        kernel=self.tensor_kernel
        if (type(kernel) is not DyadicGreenKernel or kernel.medium is not medium
                or kernel.mode_cache.medium is not medium
                or kernel.mode_cache.max_bytes!=kernel.cache_max_bytes):
            raise ValueError('Cross block tensor kernel material or cache binding changed')
        medium.assert_current()


class SurfaceVolumeBlocks:
    """Owned regional weak blocks, J/M rows and component-major cell columns."""
    def __init__(self,binding,*,order=3,inner_order=6,tensor_kernel=None,
                 tensor_face_rule='triangle',tensor_kernel_batch=128):
        if not isinstance(binding,BoundVolumeRegion):raise TypeError('Cross blocks require BoundVolumeRegion')
        binding.assert_current()
        if any(type(v) is not int or v<2 for v in (order,inner_order)):
            raise ValueError('Cross block integration orders must be integers >=2')
        self.binding,self.order,self.inner_order=binding,order,inner_order
        volume=binding.volume;system=binding.surface_system
        self.device=volume.vertices.device
        self.dtype=torch.complex128 if volume.vertices.dtype==torch.float64 else torch.complex64
        medium=system.bodies[volume.region_id-1].interior
        if isinstance(medium,TensorMedium):
            if (type(tensor_kernel) is not DyadicGreenKernel or tensor_kernel.medium is not medium
                    or medium.epsilon_r.device!=self.device or medium.epsilon_r.dtype!=self.dtype):
                raise ValueError('Cross tensor blocks require the exact matching regional kernel')
            if (tensor_face_rule not in ('triangle','projected_duffy','distance_scaled_duffy')
                    or type(tensor_kernel_batch) is not int or tensor_kernel_batch<1):
                raise ValueError('Invalid cross tensor integration policy')
            from .tensor_volume import tensor_volume_static_kernels
            tensor_volume_static_kernels(medium)
        elif tensor_kernel is not None or tensor_face_rule!='triangle' or tensor_kernel_batch!=128:
            raise ValueError('Scalar cross blocks cannot own tensor integration settings')
        self.tensor_face_rule,self.tensor_kernel_batch=tensor_face_rule,tensor_kernel_batch
        self.layout=_SourceLayout(system,self.device,self.dtype,(2*system.total_edges,)*2,
                                  volume.region_id,tensor_kernel)
        if tensor_kernel is not None:self.layout.validate_tensor_state()
        self.basis=system.bodies[volume.region_id-1].basis
        basis=self.basis
        self.row_size,self.column_size=basis.n_edges,len(volume.cells)
        edges=basis.face_edges.flatten()
        face_lower=basis.mesh.triangles.amin(1);face_upper=basis.mesh.triangles.amax(1)
        self.row_lower=volume.vertices.new_full((basis.n_edges,3),torch.inf)
        self.row_upper=volume.vertices.new_full((basis.n_edges,3),-torch.inf)
        self.row_lower.scatter_reduce_(0,edges[:,None].expand(-1,3),face_lower[:,None,:].expand(-1,3,-1).reshape(-1,3),reduce='amin',include_self=True)
        self.row_upper.scatter_reduce_(0,edges[:,None].expand(-1,3),face_upper[:,None,:].expand(-1,3,-1).reshape(-1,3),reduce='amax',include_self=True)
        tetrahedra=volume.vertices[volume.cells]
        self.column_lower,self.column_upper=tetrahedra.amin(1),tetrahedra.amax(1)
        self.column_points=tetrahedra.mean(1)
        diagonal=volume.vertices.new_zeros(basis.n_edges).scatter_add_(0,edges,basis.mass_blocks().diagonal(dim1=-2,dim2=-1).flatten())
        # This diagonal RWG norm is a compression proxy, not the full test Riesz norm.
        eta=math.sqrt(MU_0/EPSILON_0) if tensor_kernel is not None else abs(medium.eta)
        self.row_scale=torch.cat((eta*diagonal.rsqrt(),diagonal.rsqrt()))
        self.column_scale=volume.volumes.rsqrt().repeat(3)
        self._binding=self._identity()
        self._versions=tuple((id(t),t._version) for t in self.tensors())

    @property
    def tensor_kernel(self):return self.layout.tensor_kernel

    def tensor_policy(self):
        if self.tensor_kernel is None:return None
        return dict(face_rule=self.tensor_face_rule,kernel_batch=self.tensor_kernel_batch,
            dyadic_integration=self.tensor_kernel.integration.payload(),
            cache_max_bytes=self.tensor_kernel.cache_max_bytes)

    def _identity(self):
        k=self.tensor_kernel
        state=None if k is None else (id(k),id(k.medium),k.integration,k.cache_max_bytes,
            id(k.mode_cache),id(k.mode_cache.medium),k.mode_cache.max_bytes,k._isotropic)
        return (id(self.binding),id(self.layout),id(self.basis),self.order,self.inner_order,
            self.device,self.dtype,self.row_size,self.column_size,self.tensor_face_rule,
            self.tensor_kernel_batch,state)

    def tensors(self):
        yield from self.binding.volume._tensors()
        system=self.binding.surface_system
        for obj in (system,*(item for body in system.bodies for item in (body.basis,body.mesh))):
            for value in vars(obj).values():
                if isinstance(value,torch.Tensor):yield value
        for body in system.bodies:
            if isinstance(body.interior,TensorMedium):
                yield body.interior.epsilon_r;yield body.interior.mu_r
        yield self.binding.boundary_to_surface_vertices
        yield from (self.row_lower,self.row_upper,self.column_lower,self.column_upper,self.column_points,self.row_scale,self.column_scale)

    def assert_current(self):
        if self._identity()!=self._binding:
            raise ValueError('Cross block binding changed')
        self.binding.assert_current()
        if self.tensor_kernel is not None:self.layout.validate_tensor_state()
        if tuple((id(t),t._version) for t in self.tensors())!=self._versions:
            raise ValueError('Cross block geometry or basis changed')

    def __call__(self,rows,columns):
        self.assert_current()
        block,_=build_volume_source_block(self.layout,self.binding,rows,columns,order=self.order,inner_order=self.inner_order,
            tensor_face_rule=self.tensor_face_rule,tensor_kernel_batch=self.tensor_kernel_batch)
        return block.reshape(2*len(rows),len(columns),3).permute(0,2,1).reshape(2*len(rows),3*len(columns))


class SurfaceVolumeHConfig(HierarchicalBlockConfig):
    """Configuration for rectangular surface-volume H maps."""


class SurfaceVolumeHierarchicalAction(HierarchicalBlockAction):
    def __init__(self,provider,config=SurfaceVolumeHConfig()):
        if not isinstance(provider,SurfaceVolumeBlocks) or not isinstance(config,SurfaceVolumeHConfig):
            raise TypeError('Cross H actions require SurfaceVolumeBlocks and SurfaceVolumeHConfig')
        provider.assert_current()
        test=build_support_octree(provider.row_lower,provider.row_upper,leaf_size=config.leaf_size)
        source=build_support_octree(provider.column_lower,provider.column_upper,leaf_size=config.leaf_size,partition_points=provider.column_points)
        super().__init__(provider,test,source,test_components=2,source_components=3,
            row_scale=provider.row_scale,column_scale=provider.column_scale,config=config)
        self.receipt.update(version='regional_surface_volume_h_v1',region_id=provider.binding.volume.region_id,
            surface_testing_order=provider.order,volume_order=provider.inner_order,
            block_error_metric='impedance_diagonal_RWG_dual_and_cell_volume_L2_proxy',
            complete_surface_volume_response=False,coefficient_layout='J_then_M_rows_component_major_cell_columns')
        if provider.tensor_kernel is not None:
            self.receipt.update(tensor_integration=provider.tensor_policy(),
                impedance_proxy='vacuum_eta',spectral_cache_in_retained_bytes=False)
