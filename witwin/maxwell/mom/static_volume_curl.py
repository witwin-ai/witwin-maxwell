"""Bounded-block static volume curl actions at the owning cell centers."""
import torch
from .volume_mesh import TetrahedralVolumeMesh
from .vie_operator import cell_green_potentials,PreparedVolumeGreenGeometry


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,owner,adjoint):
        ctx.owner,ctx.adjoint=owner,adjoint
        return owner._product(value,adjoint)

    @staticmethod
    def backward(ctx,gradient):
        ctx.owner.assert_current()
        return _Product.apply(gradient,ctx.owner,not ctx.adjoint),None,None


class StaticVolumeCurlAction:
    """C j = sum_cells integral(grad g) cross j, with g=1/(4 pi r).

    All body-to-body interactions are included in supplied volume order. Source
    cells and target centers are blocked before integration, and Hessians are
    skipped. The exact native static primitive handles interior self gradients.
    Only points, mesh references and guarded per-mesh extents persist: no global
    curl matrix or kernel cache. Kernels are recomputed for every action;
    fixed-size blocking retains quadratic pair work without per-block mesh scans;
    this is a storage-bounded baseline, not a fast multipole/compression claim.
    Fixed-vector gradients use the Euclidean transpose of the actual blocks;
    neither geometric reciprocity nor equal cell volumes are assumed.
    """
    def __init__(self,volumes,*,target_batch=32,source_batch=32):
        self.volumes=tuple(volumes)
        if not self.volumes or any(type(v) is not TetrahedralVolumeMesh for v in self.volumes):
            raise TypeError('Static curl requires native tetrahedral volumes')
        for value in (target_batch,source_batch):
            if type(value) is not int or value<1:raise ValueError('Static curl batches must be positive integers')
        reference=self.volumes[0].vertices
        if (len({v.region_id for v in self.volumes})!=len(self.volumes) or any(
                v.vertices.device!=reference.device or v.vertices.dtype!=reference.dtype for v in self.volumes)):
            raise ValueError('Static curl volumes require unique regions and matching precision/device')
        for volume in self.volumes:volume.assert_current()
        self.counts=tuple(len(v.cells) for v in self.volumes)
        self.geometry=tuple(PreparedVolumeGreenGeometry(v) for v in self.volumes)
        self.points=torch.cat([v.vertices[v.cells].mean(1) for v in self.volumes])
        self.device,self.dtype=reference.device,reference.dtype
        self.shape=(3*sum(self.counts),)*2
        self.target_batch,self.source_batch=target_batch,source_batch
        self._binding=self._state()
        self.receipt={'method':'blocked_native_static_volume_curl','target_batch':target_batch,'source_batch':source_batch,
            'global_curl_stored':False,'cross_body_interactions':True,'physical_qualification':False,
            'fast_work_qualified':False,'quadrature_convergence_certified':False}

    def _state(self):
        return (tuple(id(v) for v in self.volumes),tuple(id(g) for g in self.geometry),self.counts,id(self.points),self.points._version,
            self.shape,self.device,self.dtype,self.target_batch,self.source_batch)

    def assert_current(self):
        if self._binding!=self._state():raise ValueError('Static curl binding changed')
        for volume,geometry in zip(self.volumes,self.geometry):geometry.assert_current(volume)

    def _block(self,geometry,points,indices):
        _,gradient,_=cell_green_potentials(geometry.mesh,points,0.,order=2,include_hessian=False,cell_indices=indices,geometry=geometry)
        x,y,z=gradient.real.unbind(-1);zero=torch.zeros_like(x)
        return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(len(points),len(indices),3,3)

    def _product(self,value,adjoint):
        self.assert_current()
        if (not isinstance(value,torch.Tensor) or value.ndim<1 or value.shape[-1]!=self.shape[0]
                or value.dtype!=self.dtype or value.device!=self.device or not bool(torch.isfinite(value).all())):
            raise ValueError('Static curl requires matching finite real cell coefficients')
        if value.numel()==0:return value.clone()
        cells=value.reshape(*value.shape[:-1],sum(self.counts),3)
        result=torch.zeros_like(cells)
        offset=0
        for volume,count,geometry in zip(self.volumes,self.counts,self.geometry):
            for start in range(0,count,self.source_batch):
                end=min(start+self.source_batch,count)
                indices=torch.arange(start,end,device=self.device)
                source=slice(offset+start,offset+end)
                for target in range(0,len(self.points),self.target_batch):
                    stop=min(target+self.target_batch,len(self.points))
                    block=self._block(geometry,self.points[target:stop],indices)
                    if adjoint:
                        result[...,source,:]+=torch.einsum('qcij,...qi->...cj',block,cells[...,target:stop,:])
                    else:
                        result[...,target:stop,:]+=torch.einsum('qcij,...cj->...qi',block,cells[...,source,:])
            offset+=count
        if not bool(torch.isfinite(result).all()):raise ValueError('Nonfinite static curl action')
        self.assert_current()
        return result.reshape_as(value)

    def apply(self,value):return _Product.apply(value,self,False)
    def adjoint(self,value):return _Product.apply(value,self,True)
