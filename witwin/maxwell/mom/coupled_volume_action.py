"""Complete fixed regional induced-current response through compressed components."""
import torch
from .tensor_medium import TensorMedium
from .implicit_surface_solve import FixedSurfaceInverse
from .surface_volume_hmatrix import SurfaceVolumeHierarchicalAction
from .interior_volume_hmatrix import InteriorVolumeHierarchicalAction
from .volume_hmatrix import VolumeHierarchicalAction
from .volume_h2 import VolumeNestedH2Action
from .fmm_capacity import storage_bytes


class CoupledVolumeAction:
    """Apply D + F A^-1 B in binding/cell/xyz layout, with its conjugate adjoint.

    B is the owning weak volume source, A the global surface equation, F the
    owning interior surface field, and D the regional direct volume field.
    Interbody coupling occurs through A. This is the induced response only;
    callers add the independently solved incident background exactly once.
    Fixed-order component errors and Krylov errors remain additional. No full
    response matrix is assembled and no physical/error certificate is implied.
    """
    def __init__(self,inverse,sources,fields,direct):
        if type(inverse) is not FixedSurfaceInverse:
            raise TypeError('Coupled response requires a guarded fixed surface inverse')
        sources,fields,direct=tuple(sources),tuple(fields),tuple(direct)
        if not sources or len(sources)!=len(fields) or len(fields)!=len(direct):
            raise ValueError('Coupled response requires one source, field and direct action per region')
        inverse.assert_current()
        system=inverse.operator.provider.system
        used=set();indices=[];sizes=[];field=None
        for source,decode,volume in zip(sources,fields,direct):
            if (type(source) is not SurfaceVolumeHierarchicalAction
                    or type(decode) is not InteriorVolumeHierarchicalAction
                    or type(volume) not in (VolumeHierarchicalAction,VolumeNestedH2Action)):
                raise TypeError('Coupled response requires native compressed regional actions')
            for action in (source,decode,volume):action.assert_current()
            binding=source.provider.binding
            if binding.surface_system is not system or decode.provider.geometry.binding is not binding:
                raise ValueError('Coupled actions require exact owning-system and region bindings')
            region=binding.volume.region_id
            if region in used:raise ValueError('Duplicate coupled volume region')
            used.add(region);body=system.bodies[region-1]
            if volume.provider.mesh is not binding.volume or volume.provider.medium is not body.interior:
                raise ValueError('Direct volume action has a different geometry or medium')
            if isinstance(body.interior,TensorMedium):
                kernels=getattr(inverse.operator.provider,'interior_kernels',())
                kernel=kernels[region-1] if len(kernels)==len(system.bodies) else None
                if kernel is None or any(k is not kernel for k in (
                        source.provider.tensor_kernel,decode.provider.geometry.tensor_kernel,volume.provider.tensor_kernel)):
                    raise ValueError('Coupled tensor actions require the exact surface regional kernel')
            if volume.provider.field!=decode.provider.field or (field is not None and field!=decode.provider.field):
                raise ValueError('Coupled response requires a single common E or H field')
            field=decode.provider.field
            if (volume.provider.order!=decode.provider.order
                    or not torch.equal(volume.provider.points,decode.provider.points)):
                raise ValueError('Direct and surface field sampling/order must match')
            n=3*len(binding.volume.cells);m=2*body.basis.n_edges
            if source.shape!=(m,n) or decode.shape!=(n,m) or volume.shape!=(n,n):
                raise ValueError('Coupled component dimensions differ')
            if any((a.device,a.dtype)!=(inverse.device,inverse.dtype) for a in (source,decode,volume)):
                raise ValueError('Coupled component precision/device differ')
            ids=torch.arange(body.edge_offset,body.edge_offset+body.basis.n_edges,device=inverse.device)
            indices.append(torch.cat((ids,ids+system.total_edges)));sizes.append(n)
        self.inverse,self.sources,self.fields,self.direct=inverse,sources,fields,direct
        self.indices,self.sizes=tuple(indices),tuple(sizes)
        self.device,self.dtype=inverse.device,inverse.dtype
        self.shape=(sum(sizes),)*2;self.field=field
        self._binding=self._identity()
        self._versions=tuple((id(t),t._version) for t in self.indices)
        self.receipt={'version':'compressed_coupled_volume_response_v1','field':field,
            'regions':[s.provider.binding.volume.region_id for s in sources],
            'coefficient_layout':'binding_cell_xyz','incident_background_included':False,
            'linear_contrast_added':False,'physical_qualification':False,
            'original_equation_verified':False,'complete_field_error_verified':False,
            'component_referenced_bytes':storage_bytes(self.component_tensors()),
            'memory_scope':'cross/direct components and index maps; excludes surface inverse, solver workspace and backward',
            'aggregate_capacity_enforced':False}

    def _identity(self):
        return (id(self.inverse),tuple(map(id,self.sources)),tuple(map(id,self.fields)),
            tuple(map(id,self.direct)),self.sizes,self.device,self.dtype,self.shape,self.field)

    def component_tensors(self):
        yield from self.indices
        for action in (*self.sources,*self.fields,*self.direct):yield from action.tensors()

    def assert_current(self):
        if self._identity()!=self._binding or tuple((id(t),t._version) for t in self.indices)!=self._versions:
            raise ValueError('Coupled response binding or maps changed')
        self.inverse.assert_current()
        for action in (*self.sources,*self.fields,*self.direct):action.assert_current()

    @staticmethod
    def _component_major(value):
        return value.reshape(*value.shape[:-1],-1,3).transpose(-1,-2).flatten(-2)

    @staticmethod
    def _cell_major(value):
        return value.reshape(*value.shape[:-1],3,-1).transpose(-1,-2).flatten(-2)

    def _split(self,value):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.device!=self.device or value.dtype!=self.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError('Coupled response requires finite matching binding/cell/xyz coefficients')
        return tuple(self._component_major(v) for v in value.split(self.sizes,dim=-1))

    def _surface(self,local,adjoint):
        rhs=local[0].new_zeros(*local[0].shape[:-1],self.inverse.shape[0])
        for x,source,decode,ids in zip(local,self.sources,self.fields,self.indices):
            contribution=decode.adjoint(x) if adjoint else source.apply(x)
            rhs=rhs.index_add(-1,ids,contribution)
        return self.inverse.adjoint(rhs) if adjoint else self.inverse.apply(rhs)

    def _decode(self,local,surface,adjoint):
        result=[]
        for x,source,decode,direct,ids in zip(local,self.sources,self.fields,self.direct,self.indices):
            coupled=source.adjoint(surface[...,ids]) if adjoint else decode.apply(surface[...,ids])
            own=direct.adjoint(x) if adjoint else direct.apply(x)
            result.append(self._cell_major(coupled+own))
        return torch.cat(result,dim=-1)

    def _action(self,value,adjoint):
        local=self._split(value)
        return self._decode(local,self._surface(local,adjoint),adjoint)

    def surface_response(self,value):
        """Recover induced global scattered RWG coefficients without field decode."""
        return self._surface(self._split(value),False)

    def apply(self,value):return self._action(value,False)
    def adjoint(self,value):return self._action(value,True)
