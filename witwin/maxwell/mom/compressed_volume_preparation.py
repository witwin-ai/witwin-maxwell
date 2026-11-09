"""Adaptive preparation of induced E/H actions without dense response matrices."""
from .implicit_surface_solve import FixedSurfaceInverse
from .volume_mesh import BoundVolumeRegion
from .volume_source_refinement import prepare_source_action
from .surface_volume_hmatrix import SurfaceVolumeHConfig
from .interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction,InteriorVolumeHConfig
from .volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction,VolumeHConfig
from .volume_h2 import VolumeNestedH2Action
from .h2 import H2Config
from .coupled_volume_action import CoupledVolumeAction
from .coupled_field_refinement import compare_coupled_field_orders_many
from .coupled_original_check import check_coupled_original_columns_many
from .volume_response_matrix import VolumeMatrixConvergenceError
from .fields import FieldQueryConfig
from .tensor_medium import TensorMedium
from .dyadic_green import DyadicGreenKernel


def prepare_compressed_cell_response(inverse,bindings,*,query_config=FieldQueryConfig(),
        source_config=SurfaceVolumeHConfig(),field_config=InteriorVolumeHConfig(),
        volume_config=VolumeHConfig(),volume_h2_config=None,column_batch=8,edge_batch=32,
        query_batch=8,original_relative_tolerance=1e-8,original_absolute_tolerance=1e-12,
        source_settings=None,incident_sources=None,tensor_face_rule='triangle',tensor_kernel_batch=128,
        tensor_rhs_cache_max_bytes=0):
    """Admit source quadrature, complete E/H order differences and original rows.

    Optionally prepares a declared-source background bank. DC/HB integration,
    persistence, aggregate memory capacity and total physical error remain
    separate obligations. Every unit column is checked, but arbitrary source
    combinations and conditioning amplification are not certified.
    """
    if type(inverse) is not FixedSurfaceInverse:
        raise TypeError('Compressed preparation requires a guarded surface inverse')
    if not isinstance(query_config,FieldQueryConfig) or not query_config.adaptive:
        raise ValueError('Compressed preparation requires adaptive complete fields')
    if (any(type(v) is not int or v<1 for v in (column_batch,edge_batch,query_batch))
            or not isinstance(source_config,SurfaceVolumeHConfig)
            or not isinstance(field_config,InteriorVolumeHConfig)
            or not isinstance(volume_config,VolumeHConfig)
            or (volume_h2_config is not None and not isinstance(volume_h2_config,H2Config))):
        raise ValueError('Invalid compressed response configuration')
    inverse.assert_current();system=inverse.operator.provider.system
    bindings=tuple(bindings)
    if (not bindings or any(type(b) is not BoundVolumeRegion or b.surface_system is not system for b in bindings)
            or len({b.volume.region_id for b in bindings})!=len(bindings)):
        raise ValueError('Compressed response requires unique exact owning-system bindings')
    if (tensor_face_rule not in ('triangle','projected_duffy','distance_scaled_duffy')
            or type(tensor_kernel_batch) is not int or tensor_kernel_batch<1
            or type(tensor_rhs_cache_max_bytes) is not int or tensor_rhs_cache_max_bytes<0):
        raise ValueError('Invalid compressed tensor integration policy')
    if source_settings is not None and any(k in source_settings for k in
            ('tensor_kernel','tensor_face_rule','tensor_kernel_batch')):
        raise ValueError('Compressed tensor source policy must use the explicit preparation arguments')
    provider=inverse.operator.provider
    kernels={}
    for binding in bindings:
        region=binding.volume.region_id;medium=system.bodies[region-1].interior
        if isinstance(medium,TensorMedium):
            owned=getattr(provider,'interior_kernels',())
            kernel=owned[region-1] if len(owned)==len(system.bodies) else None
            if type(kernel) is not DyadicGreenKernel or kernel.medium is not medium:
                raise ValueError('Compressed tensor preparation requires the exact surface regional kernel')
            kernels[region]=kernel
    integration=inverse.operator.provider.config
    order=max(2,integration.order,integration.singular_order)
    receipt={'version':'adaptive_compressed_induced_response_v1','converged':False,
        'sources':[],'field_checks':[],'original_checks':[],
        'physical_qualification':False,'total_error_certified':False,
        'incident_background_included':False,'aggregate_capacity_enforced':False,
        'column_batch':column_batch,'initial_field_order':order}
    if kernels:
        receipt['tensor_volume_policy']=dict(face_rule=tensor_face_rule,kernel_batch=tensor_kernel_batch,
            radial_order_policy='follow_complete_field_order',regions=list(kernels),
            rhs_cache_max_bytes_per_handle=tensor_rhs_cache_max_bytes)
    def reject(reason):
        receipt['failure_reason']=reason
        raise VolumeMatrixConvergenceError(receipt)
    if order>=query_config.max_order:reject('initial_field_order_leaves_no_refinement')
    sources=[]
    for binding in bindings:
        try:
            kernel=kernels.get(binding.volume.region_id)
            options={} if kernel is None else dict(tensor_kernel=kernel,
                tensor_face_rule=tensor_face_rule,tensor_kernel_batch=tensor_kernel_batch)
            action,log=prepare_source_action(binding,inverse.operator.metric,config=source_config,
                **options,**(source_settings or {}))
        except VolumeMatrixConvergenceError as error:
            receipt['sources'].append(error.receipt)
            reject('source_admission_failed')
        sources.append(action);receipt['sources'].append(log)
    def fields(current_order):
        result=[]
        for field in ('E','H'):
            decoders=[];direct=[]
            for binding in bindings:
                body=system.bodies[binding.volume.region_id-1]
                kernel=kernels.get(binding.volume.region_id)
                decoders.append(InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(binding,
                    field=field,order=current_order,tensor_kernel=kernel),field_config))
                options={} if kernel is None else dict(tensor_kernel=kernel,radial_order=current_order,
                    face_rule=tensor_face_rule,kernel_batch=tensor_kernel_batch)
                volume=VolumeHierarchicalAction(VolumeCellBlocks(binding.volume,body.interior,
                    field=field,order=current_order,**options),volume_config)
                direct.append(VolumeNestedH2Action(volume,volume_h2_config) if volume_h2_config is not None else volume)
            result.append(CoupledVolumeAction(inverse,sources,decoders,direct))
        return tuple(result)
    previous=fields(order)
    while order<query_config.max_order:
        order=min(2*order,query_config.max_order);current=fields(order)
        checks=compare_coupled_field_orders_many(tuple(zip(previous,current)),config=query_config,column_batch=column_batch)
        receipt['field_checks'].append({'order':order,'channels':checks})
        if all(check['passed'] for check in checks):
            original=check_coupled_original_columns_many(current,relative_tolerance=original_relative_tolerance,
                absolute_tolerance=original_absolute_tolerance,field_config=query_config,
                column_batch=column_batch,edge_batch=edge_batch,query_batch=query_batch)
            receipt['original_checks']=original
            if not all(check['passed'] for check in original):reject('original_equation_or_field_check_failed')
            result={'electric_action':current[0],'magnetic_action':current[1],
                'bindings':bindings,'receipt':receipt}
            if incident_sources is not None:
                from .compressed_background import prepare_compressed_background
                bank=prepare_compressed_background(inverse,bindings,incident_sources,query_config=query_config,
                    edge_batch=edge_batch,query_batch=query_batch,
                    relative_tolerance=original_relative_tolerance,absolute_tolerance=original_absolute_tolerance,
                    tensor_rhs_cache_max_bytes=tensor_rhs_cache_max_bytes)
                result['background_bank']=bank
                receipt.update(incident_background_included=True,background=bank.receipt)
            receipt.update(converged=True,final_field_order=order,failure_reason=None)
            return result
        previous=current
    reject('complete_column_field_order_budget_exhausted')
