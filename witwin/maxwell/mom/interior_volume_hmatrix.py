"""Regional rectangular H maps from RWG currents to interior cell fields."""
import math
import torch
from ..constants import EPSILON_0,MU_0
from .surface_volume_hmatrix import SurfaceVolumeBlocks
from .hierarchical_blocks import HierarchicalBlockAction, HierarchicalBlockConfig
from .cluster import build_support_octree
from .fields import selected_interior_field_blocks
from .runtime import DecodeHandle
from .operator import IntegrationConfig
from .physics import Medium, compile_medium
from ..media import Material


class InteriorVolumeBlocks:
    """Cell-centroid field rows and local J/M columns, both component-major.

    Full cell boxes conservatively enclose the collocation points. Compression
    uses volume L2 rows and inverse impedance-balanced diagonal RWG norms;
    this is a proxy rather than a full Riesz operator-error certificate.
    """
    def __init__(self, binding, *, field='E', order=4, tensor_kernel=None):
        if field not in ('E','H'):
            raise ValueError('Interior blocks require E or H')
        self.geometry=SurfaceVolumeBlocks(binding,order=order,inner_order=2,tensor_kernel=tensor_kernel)
        body=binding.surface_system.bodies[binding.volume.region_id-1]
        self.field,self.order=field,order
        self.device,self.dtype=self.geometry.device,self.geometry.dtype
        self.row_size,self.column_size=self.geometry.column_size,self.geometry.row_size
        self.row_lower,self.row_upper=self.geometry.column_lower,self.geometry.column_upper
        self.points=self.geometry.column_points
        self.column_lower,self.column_upper=self.geometry.row_lower,self.geometry.row_upper
        self.row_scale=binding.volume.volumes.sqrt().repeat(3)
        eta=math.sqrt(MU_0/EPSILON_0) if tensor_kernel is not None else abs(body.interior.eta)
        inverse_gram=self.geometry.row_scale[self.column_size:]
        self.column_scale=torch.cat((inverse_gram,eta*inverse_gram))
        self._binding=self._identity()
        self._versions=tuple((id(t),t._version) for t in self.tensors())

    def _identity(self):
        return (id(self.geometry),self.field,self.order,self.device,self.dtype,self.row_size,self.column_size)

    def tensors(self):
        yield from self.geometry.tensors()
        yield from (self.row_lower,self.row_upper,self.points,self.column_lower,self.column_upper,self.row_scale,self.column_scale)

    def assert_current(self):
        if self._identity()!=self._binding:
            raise ValueError('Interior block binding changed')
        self.geometry.assert_current()
        if tuple((id(t),t._version) for t in self.tensors())!=self._versions:
            raise ValueError('Interior block geometry or scaling changed')

    def __call__(self, rows, columns):
        self.assert_current()
        if (not isinstance(rows,torch.Tensor) or rows.device!=self.device
                or rows.dtype!=torch.int64 or rows.ndim!=1
                or bool(((rows<0)|(rows>=self.row_size)).any())):
            raise ValueError('Interior block rows require matching CUDA int64 indices in range')
        binding=self.geometry.binding
        body=binding.surface_system.bodies[binding.volume.region_id-1]
        handle=DecodeHandle(body.basis,compile_medium(Material(),body.interior.frequency),body.interior,(),
            IntegrationConfig(order=self.order,singular_order=self.order),{})
        if self.geometry.tensor_kernel is not None:handle.tensor_kernel=self.geometry.tensor_kernel
        values=selected_interior_field_blocks(handle,self.points[rows],columns)[self.field=='H']
        return values.permute(1,0,2).reshape(3*len(rows),2*len(columns))


class InteriorVolumeHConfig(HierarchicalBlockConfig):
    """Configuration for regional surface-to-volume field H maps."""


class InteriorVolumeHierarchicalAction(HierarchicalBlockAction):
    def __init__(self,provider,config=InteriorVolumeHConfig()):
        if not isinstance(provider,InteriorVolumeBlocks) or not isinstance(config,InteriorVolumeHConfig):
            raise TypeError('Interior H actions require InteriorVolumeBlocks and InteriorVolumeHConfig')
        provider.assert_current()
        test=build_support_octree(provider.row_lower,provider.row_upper,
            leaf_size=config.leaf_size,partition_points=provider.points)
        source=build_support_octree(provider.column_lower,provider.column_upper,leaf_size=config.leaf_size)
        super().__init__(provider,test,source,test_components=3,source_components=2,
            row_scale=provider.row_scale,column_scale=provider.column_scale,config=config)
        self.receipt.update(version='regional_interior_volume_h_v1',
            region_id=provider.geometry.binding.volume.region_id,field=provider.field,
            field_order=provider.order,complete_surface_volume_response=False,
            block_error_metric='cell_volume_L2_and_impedance_diagonal_RWG_proxy',
            coefficient_layout='component_major_cell_rows_J_then_M_columns')
        if provider.geometry.tensor_kernel is not None:
            self.receipt.update(tensor_integration=provider.geometry.tensor_policy(),
                impedance_proxy='vacuum_eta',spectral_cache_in_retained_bytes=False)
