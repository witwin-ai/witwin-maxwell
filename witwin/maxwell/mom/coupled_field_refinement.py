"""Streamed complete-field column checks with one shared surface solve per batch."""
import torch
from ..constants import ETA_0
from .tensor_medium import TensorMedium
from .coupled_volume_action import CoupledVolumeAction
from .fields import FieldQueryConfig


def _pair_settings(coarse,fine,config,column_batch):
    """Check every unit-current column and centroid, without a full response matrix.

    Source maps and the inverse must be shared exactly. Field quadrature
    order must increase; both complete surface and direct volume fields are summed
    before comparison. Compression errors remain additional. This successive
    order estimate is not a bound for arbitrary cancelling current combinations.
    """
    if type(coarse) is not CoupledVolumeAction or type(fine) is not CoupledVolumeAction:
        raise TypeError('Field refinement requires complete coupled actions')
    if not isinstance(config,FieldQueryConfig) or not config.adaptive:
        raise ValueError('Field refinement requires an adaptive query configuration')
    if type(column_batch) is not int or column_batch<1:
        raise ValueError('Column batch must be a positive integer')
    coarse.assert_current();fine.assert_current()
    if (coarse.inverse is not fine.inverse or len(coarse.sources)!=len(fine.sources)
            or any(a is not b for a,b in zip(coarse.sources,fine.sources))
            or coarse.field!=fine.field or coarse.sizes!=fine.sizes):
        raise ValueError('Field refinement requires identical source maps, inverse, regions and field')
    orders=[]
    for a,b,da,db in zip(coarse.fields,fine.fields,coarse.direct,fine.direct):
        if b.provider.order<=a.provider.order or b.provider.order>config.max_order:
            raise ValueError('Fine field order must increase within the configured budget')
        pa,pb=da.provider,db.provider
        if pa.tensor_kernel is not None or pb.tensor_kernel is not None:
            if (pa.tensor_kernel is not pb.tensor_kernel or pa.face_rule!=pb.face_rule
                    or pa.kernel_batch!=pb.kernel_batch):
                raise ValueError('Tensor field refinement requires identical volume kernel and integration policy')
            if pa.radial_order!=pa.order or pb.radial_order!=pb.order:
                raise ValueError('Tensor field refinement requires radial orders to follow complete field orders')
        orders.append((a.provider.order,b.provider.order))
    n=coarse.shape[0];count=n//3
    absolute=torch.cat([s.provider.binding.volume.volumes.new_full((size//3,),config.absolute_tolerance)
        / ((ETA_0 if isinstance(s.provider.binding.surface_system.bodies[s.provider.binding.volume.region_id-1].interior,TensorMedium) else abs(s.provider.binding.surface_system.bodies[s.provider.binding.volume.region_id-1].interior.eta))
           if coarse.field=='H' else 1.) for s,size in zip(coarse.sources,coarse.sizes)])
    receipt={'version':'streamed_complete_coupled_field_order_v1','field':coarse.field,
        'orders':orders,'relative_tolerance':config.relative_tolerance,
        'absolute_electric_tolerance':config.absolute_tolerance,'columns':n,'queries':count,
        'column_batch':column_batch,'checks':[],'passed':True,
        'surface_solves_shared':True,'source_unit_current_density_A_per_m2':1.,
        'physical_qualification':False,'certified_error_bound':False,
        'compression_error_estimated':False,'source_quadrature_error_estimated':False,
        'volume_discretization_error_estimated':False}
    return receipt,absolute


@torch.no_grad()
def compare_coupled_field_orders_many(pairs,*,config=FieldQueryConfig(),column_batch=8):
    """Share one surface solve across field channels and integration orders."""
    pairs=tuple(pairs)
    if not pairs or any(len(pair)!=2 for pair in pairs):
        raise ValueError('Field comparison requires nonempty coarse/fine pairs')
    settings=[_pair_settings(a,b,config,column_batch) for a,b in pairs]
    first=pairs[0][0]
    if len({a.field for a,b in pairs})!=len(pairs):
        raise ValueError('Field channels must be unique')
    for coarse,fine in pairs:
        if (coarse.inverse is not first.inverse or len(coarse.sources)!=len(first.sources)
                or any(a is not b for a,b in zip(coarse.sources,first.sources))):
            raise ValueError('Field channels must share exact source actions and inverse')
    n=first.shape[0];count=n//3
    for start in range(0,n,column_batch):
        for pair in pairs:
            for action in pair:action.assert_current()
        stop=min(start+column_batch,n)
        unit=torch.zeros(stop-start,n,device=first.device,dtype=first.dtype)
        unit[torch.arange(stop-start,device=first.device),torch.arange(start,stop,device=first.device)]=1
        local=first._split(unit);surface=first._surface(local,False)
        for (coarse,fine),(receipt,absolute) in zip(pairs,settings):
            old=coarse._decode(local,surface,False).reshape(stop-start,count,3)
            new=fine._decode(local,surface,False).reshape(stop-start,count,3)
            error=torch.linalg.vector_norm(new-old,dim=-1)
            magnitude=torch.linalg.vector_norm(new,dim=-1)
            threshold=absolute+config.relative_tolerance*magnitude
            finite=torch.isfinite(error)&torch.isfinite(magnitude)&torch.isfinite(threshold)
            accepted=finite&(error<=threshold)
            passed=bool(accepted.all())
            receipt['passed']=receipt['passed'] and passed
            ratio=torch.where(threshold>0,error/threshold,torch.where(error==0,0.,torch.inf))
            receipt['checks'].append({'column_start':start,'column_stop':stop,'passed':passed,
                'finite':bool(finite.all()),'failed_column_query_pairs':int((~accepted).sum()),
                'maximum_difference':float(error.max()),'maximum_threshold_ratio':float(ratio.max())})
    for pair in pairs:
        for action in pair:action.assert_current()
    return tuple(receipt for receipt,absolute in settings)


def compare_coupled_field_orders(coarse,fine,*,config=FieldQueryConfig(),column_batch=8):
    """Single-channel complete-field refinement with shared coarse/fine solves."""
    return compare_coupled_field_orders_many(((coarse,fine),),config=config,column_batch=column_batch)[0]
