"""Weak induced-volume source matrices for the existing lifted MoM operator."""
import math
import torch
from .operator import local_geometry
from .quadrature import triangle_rule
from .volume_mesh import BoundVolumeRegion
from .vie_operator import cell_green_potentials, cell_current_field_blocks
from .linalg import apply_test_riesz_inverse



def _scaled_norm(value,dim):
    """Compute vector norms without squaring unscaled large components."""
    scale=value.abs().amax(dim=dim,keepdim=True)
    divisor=torch.where(scale==0,torch.ones_like(scale),scale)
    return torch.linalg.vector_norm(value/divisor,dim=dim)*scale.squeeze(dim)


def _column_dual_norm(operator,value):
    """Evaluate the existing Riesz dual proxy after per-column rescaling."""
    scale=value.abs().amax(dim=0)
    divisor=torch.where(scale==0,torch.ones_like(scale),scale)
    normalized=value/divisor
    inverse=apply_test_riesz_inverse(operator,normalized.mT).mT
    quadratic=(normalized.conj()*inverse).real.sum(0)
    # Reject before clamping: negative infinity must not become a zero norm.
    finite=torch.isfinite(inverse).all(0)&torch.isfinite(quadratic)
    quadratic=torch.where(finite,quadratic,torch.full_like(quadratic,float('nan')))
    return quadratic.clamp_min(0).sqrt()*scale


def _finite_metrics(*values):
    return all(bool(torch.isfinite(value).all()) for value in values)


def build_volume_source_block(operator, binding, edge_indices, cell_indices, *,
                              order=3, inner_order=6, query_batch_size=None,
                              tensor_face_rule='triangle',tensor_kernel_batch=128):
    """Owning-region selected [+<f,Hv>,-<f,Ev>] rows, cell/xyz columns.

    Rows contain requested local J edges followed by requested local M edges;
    selections retain order/repetitions. Tensor regions test regular PV volume
    fields with the owning configured dyadic kernel and transfer the static
    electric derivative to RWG divergence. Both faces of each complete RWG support
    are integrated. No unselected cell or unsupported face quadrature is done.
    """
    if not isinstance(binding,BoundVolumeRegion) or binding.surface_system is not operator.system:
        raise ValueError('Volume block requires exact owning-system binding')
    binding.assert_current()
    if any(type(v) is not int or v<2 for v in (order,inner_order)):
        raise ValueError('Volume source block orders must be integers >= 2')
    if query_batch_size is not None and (type(query_batch_size) is not int or query_batch_size<1):
        raise ValueError('Source query batch size must be a positive integer or None')
    if type(tensor_kernel_batch) is not int or tensor_kernel_batch<1:
        raise ValueError('Tensor volume kernel batch must be a positive integer')
    if tensor_face_rule not in ('triangle','projected_duffy','distance_scaled_duffy'):
        raise ValueError('Tensor volume face rule must be triangle, projected_duffy or distance_scaled_duffy')
    body=operator.system.bodies[binding.volume.region_id-1]
    basis,medium=body.basis,body.interior
    from .tensor_medium import TensorMedium
    tensor=isinstance(medium,TensorMedium)
    volume_records=[]
    if tensor:
        from .tensor_volume import cell_tensor_current_blocks,tensor_volume_static_kernels,tensor_static_cell_potentials
        operator.validate_tensor_state()
        regional_kernel=operator.body_handles[binding.volume.region_id-1].tensor_kernel
        static_electric=tensor_volume_static_kernels(medium)[0]
    for ids,count in ((edge_indices,basis.n_edges),(cell_indices,len(binding.volume.cells))):
        if (not isinstance(ids,torch.Tensor) or ids.device!=operator.device or ids.dtype!=torch.int64
                or ids.ndim!=1 or bool(((ids<0)|(ids>=count)).any())):
            raise ValueError('Selected source block indices require matching CUDA int64 in range')
    selected,inverse=torch.unique(edge_indices,sorted=True,return_inverse=True)
    cells=len(cell_indices)
    batch=min(query_batch_size or 262144,max(1,262144//max(1,cells)))
    local=torch.zeros((2*len(selected),3*cells),device=operator.device,dtype=operator.dtype)
    # Derive support from the same face/edge incidence used in weak testing,
    # without relying on the separate plus/minus lookup caches.
    faces=torch.isin(basis.face_edges,selected).any(-1).nonzero().flatten()
    if len(selected) and cells:
        lookup=torch.full((basis.n_edges,),-1,device=operator.device,dtype=torch.int64)
        lookup[selected]=torch.arange(len(selected),device=operator.device)
        edges,free,slopes=local_geometry(basis)
        for face in faces.tolist():
            positions=lookup[edges[face]];keep=positions>=0;positions=positions[keep]
            slope=slopes[face,keep];vertex=free[face,keep]
            face_points,face_weights=triangle_rule(body.mesh.triangles[face:face+1],order)
            tested_h=local.new_zeros(len(positions),cells,3);tested_e=torch.zeros_like(tested_h)
            for start in range(0,face_points.shape[1],batch):
                points=face_points[0,start:start+batch]
                weights=face_weights[0,start:start+batch].to(operator.dtype)
                values=(slope[None,:,None]*(points[:,None]-vertex[None])).to(operator.dtype)
                if tensor:
                    blocks,record=cell_tensor_current_blocks(binding.volume,points,regional_kernel,
                        order=inner_order,radial_order=inner_order,principal_value=True,cell_indices=cell_indices,
                        face_rule=tensor_face_rule,regular_only=True,kernel_batch=tensor_kernel_batch)
                    tested_h=tested_h+torch.einsum('qli,qcij,q->lcj',values,blocks[:,:,3:,:3],weights)
                    tested_e=tested_e+torch.einsum('qli,qcij,q->lcj',values,blocks[:,:,:3,:3],weights)
                    # Complete RWG supports cancel conormal edge fluxes. The
                    # physical static kernel already includes epsilon; no
                    # additional inverse constitutive factor belongs here.
                    (_,gradient,_),static_record=tensor_static_cell_potentials(binding.volume,points,static_electric,
                        cell_indices=cell_indices,principal_value=True)
                    record['weak_static_electric']=static_record
                    tested_e=tested_e-1j/medium.omega*torch.einsum('l,qcj,q->lcj',
                        (2*slope).to(operator.dtype),gradient.to(operator.dtype),weights)
                    volume_records.append(dict(face=face,query_start=start,query_count=len(points),receipt=record))
                else:
                    potential,gradient,_=cell_green_potentials(binding.volume,points,medium.k,
                        order=inner_order,principal_value=True,include_hessian=False,cell_indices=cell_indices)
                    tested_h=tested_h+(torch.linalg.cross(values[:,:,None,:],gradient[:,None,:,:],dim=-1)
                        *weights[:,None,None,None]).sum(0)
                    tested_e=tested_e+1j*medium.omega*medium.mu*torch.einsum('qlj,qc,q->lcj',values,potential,weights)
                    tested_e=tested_e-1j/(medium.omega*medium.epsilon)*torch.einsum('l,qcj,q->lcj',
                        (2*slope).to(operator.dtype),gradient,weights)
            local.index_add_(0,positions,tested_h.reshape(len(positions),3*cells))
            local.index_add_(0,positions+len(selected),-tested_e.reshape(len(positions),3*cells))
    result=local[torch.cat((inverse,inverse+len(selected)))]
    return result,{'version':'selected_tensor_weak_volume_source_block_v1' if tensor else 'selected_weak_volume_source_block_v1','region_id':body.region_id,
        'row_layout':'selected_J_then_M','column_layout':'selected_cell_xyz',
        'selected_edges':len(edge_indices),'selected_cells':cells,
        'integrated_support_faces':len(faces) if cells else 0,'query_batch_size':batch,
        'surface_testing_order':order,'volume_order':inner_order,'query_cell_work_cap':262144,
        'tensor_kernel_batch':tensor_kernel_batch,
        'hessian_evaluated':tensor,'physical_qualification':False,'quadrature_error_estimated':False,
        **({'tensor_volume_evaluations':volume_records,'dyadic_integration':regional_kernel.integration.payload(),
            'tensor_static_electric_testing':'complete_rwg_divergence'} if tensor else {}),
        'linear_contrast_added':False,'matrix_storage_bytes':result.numel()*result.element_size()}


def build_volume_source_matrix(operator, bindings, *, order=3, inner_order=6, query_batch_size=None,
                               tensor_face_rule='triangle',tensor_kernel_batch=128):
    """Map ordered cell Cartesian currents to owning global MoM boundary rows.

    Column order follows bindings, then cells, then x/y/z. Scalar E testing
    transfers the tangential derivative to full-support RWG divergence;
    tensor regions transfer the static derivative and test the regular PV fields. No incident field or additional
    linear constitutive contrast appears in this matrix.
    """
    if query_batch_size is not None and (type(query_batch_size) is not int or query_batch_size < 1):
        raise ValueError('Source query batch size must be a positive integer or None')
    if type(tensor_kernel_batch) is not int or tensor_kernel_batch<1:
        raise ValueError('Tensor volume kernel batch must be a positive integer')
    if tensor_face_rule not in ('triangle','projected_duffy','distance_scaled_duffy'):
        raise ValueError('Tensor volume face rule must be triangle, projected_duffy or distance_scaled_duffy')
    operator.system.assert_current()
    if any(type(value) is not int or value < 2 for value in (order,inner_order)):
        raise ValueError('Volume source matrix orders must be integers >= 2')
    bindings = tuple(bindings)
    used = set()
    for binding in bindings:
        if not isinstance(binding,BoundVolumeRegion) or binding.surface_system is not operator.system:
            raise ValueError('Volume matrix requires exact owning-system bindings')
        binding.assert_current()
        if binding.volume.region_id in used:
            raise ValueError('Duplicate volume matrix region')
        used.add(binding.volume.region_id)
    columns = sum(3*len(binding.volume.cells) for binding in bindings)
    matrix = torch.zeros((operator.shape[0],columns),device=operator.device,dtype=operator.dtype)
    offset,records = 0,[]
    for binding in bindings:
        index = binding.volume.region_id-1
        body = operator.system.bodies[index]
        basis,medium = body.basis,body.interior
        cells = len(binding.volume.cells)
        local,block_receipt=build_volume_source_block(operator,binding,
            torch.arange(basis.n_edges,device=operator.device),torch.arange(cells,device=operator.device),
            order=order,inner_order=inner_order,query_batch_size=query_batch_size,tensor_face_rule=tensor_face_rule,
            tensor_kernel_batch=tensor_kernel_batch)
        batch=block_receipt['query_batch_size']
        matrix[operator.system.J_slices[index],offset:offset+3*cells] = local[:basis.n_edges]
        matrix[operator.system.M_slices[index],offset:offset+3*cells] = local[basis.n_edges:]
        records.append({'region_id':body.region_id,'column_start':offset,'column_stop':offset+3*cells,'query_batch_size':batch,
                        **({'tensor_source_block':block_receipt} if block_receipt['hessian_evaluated'] else {})})
        offset += 3*cells
    return matrix,{'version':'induced_volume_weak_source_matrix_v1','column_layout':'binding_cell_xyz',
        'bodies':records,'surface_testing_order':order,'volume_order':inner_order,
        'tensor_kernel_batch':tensor_kernel_batch,
        'query_batch_size':query_batch_size,'query_cell_work_cap':262144,
        'hessian_evaluated':any('tensor_source_block' in record for record in records),
        'linear_contrast_added':False,'physical_qualification':False,'quadrature_error_estimated':False,
        'matrix_storage_bytes':matrix.numel()*matrix.element_size()}


class VolumeMatrixConvergenceError(RuntimeError):
    def __init__(self,receipt):
        super().__init__(f"Volume source matrix did not converge: {receipt['failure_reason']}")
        self.receipt = receipt


def adaptive_volume_source_matrix(operator,bindings,*,order=2,inner_order=2,max_order=32,
                                  max_inner_order=64,relative_tolerance=1e-4,absolute_tolerance=1e-12,
                                  tensor_face_rule='triangle',tensor_kernel_batch=128):
    """Refine inner and outer integration separately for every matrix column.

    Each axis receives half the requested dual-proxy tolerance. Per-column
    successive-order agreement is not a bound for arbitrary cancelling source
    combinations or a physical discretization certificate.
    """
    if (any(type(v) is not int for v in (order,inner_order,max_order,max_inner_order))
            or not 2 <= order < max_order or not 2 <= inner_order < max_inner_order
            or not all(math.isfinite(v) and v >= 0 for v in (relative_tolerance,absolute_tolerance))
            or relative_tolerance+absolute_tolerance == 0):
        raise ValueError('Invalid volume source matrix refinement settings')
    if type(tensor_kernel_batch) is not int or tensor_kernel_batch<1:
        raise ValueError('Tensor volume kernel batch must be a positive integer')
    if tensor_face_rule not in ('triangle','projected_duffy','distance_scaled_duffy'):
        raise ValueError('Tensor volume face rule must be triangle, projected_duffy or distance_scaled_duffy')
    bindings = tuple(bindings)
    receipt = {'version':'independent_inner_outer_column_dual_estimator_v1','converged':False,
        'checks':[],'evaluations':[],'physical_qualification':False,'certified_error_bound':False,
        'relative_tolerance':relative_tolerance,'absolute_tolerance_dual_proxy':absolute_tolerance,
        'initial_outer_order':order,'initial_inner_order':inner_order,
        'tensor_face_rule':tensor_face_rule,'tensor_kernel_batch':tensor_kernel_batch,
        'max_outer_order':max_order,'max_inner_order':max_inner_order}
    def evaluate(outer,inner):
        value,local = build_volume_source_matrix(operator,bindings,order=outer,inner_order=inner,
                                               tensor_face_rule=tensor_face_rule,tensor_kernel_batch=tensor_kernel_batch)
        receipt['evaluations'].append({'outer_order':outer,'inner_order':inner})
        receipt['matrix'] = local
        return value
    def check(value,old,axis,outer,inner):
        error,magnitude = (_column_dual_norm(operator,(value-old).detach()),
                           _column_dual_norm(operator,value.detach()))
        threshold = .5*(absolute_tolerance+relative_tolerance*magnitude)
        if not _finite_metrics(error,magnitude,threshold):
            receipt.update(failure_reason='nonfinite_source_error_metric',failed_axis=axis,
                           failed_outer_order=outer,failed_inner_order=inner)
            raise VolumeMatrixConvergenceError(receipt)
        passed = bool((error <= threshold).all())
        receipt['checks'].append({'axis':axis,'outer_order':outer,'inner_order':inner,'passed':passed,
            'maximum_column_difference_dual_proxy':float(error.max()) if error.numel() else 0.,
            'maximum_threshold_ratio':float((error/threshold.clamp_min(torch.finfo(error.dtype).tiny)).max()) if error.numel() else 0.})
        return passed
    def inner_converged(outer,start):
        previous = evaluate(outer,start)
        while start < max_inner_order:
            start = min(2*start,max_inner_order)
            value = evaluate(outer,start)
            if check(value,previous,'inner',outer,start):
                return value,start
            previous = value.detach()
        receipt['failure_reason'] = 'inner_order_budget_exhausted'
        raise VolumeMatrixConvergenceError(receipt)
    previous,inner = inner_converged(order,inner_order)
    while order < max_order:
        order = min(2*order,max_order)
        value,inner = inner_converged(order,max(2,inner//2))
        if check(value,previous,'outer',order,inner):
            receipt.update(converged=True,final_outer_order=order,final_inner_order=inner,failure_reason=None)
            return value,receipt
        previous = value.detach()
    receipt['failure_reason'] = 'outer_order_budget_exhausted'
    raise VolumeMatrixConvergenceError(receipt)


def prepare_cell_field_response(operator,bindings,*,query_config=None,stopping=None,**matrix_settings):
    """Prepare actual induced-current to cell-centroid E/H response matrices.

    All surface RHS share one native dense solve. Each complete surface+volume
    field column is refined as a unit current-density excitation. This is cell
    collocation, not an assertion of volume-discretization convergence.
    """
    import copy
    from dataclasses import replace
    from .fields import FieldQueryConfig,decode_interior_columns
    from .lifting_fields import decode_system
    from .linalg import StoppingRule
    from .tensor_medium import TensorMedium
    from .tensor_volume import cell_tensor_current_blocks
    from ..constants import EPSILON_0,MU_0
    stopping = StoppingRule(method='direct') if stopping is None else stopping
    if not isinstance(stopping,StoppingRule) or stopping.method != 'direct' or stopping.direct_storage != 'resident':
        raise NotImplementedError('Harmonic response preparation requires resident direct linear solves')
    bindings = tuple(bindings)
    if not bindings:
        raise ValueError('Cell response preparation requires at least one volume region')
    config = query_config or FieldQueryConfig()
    if not isinstance(config,FieldQueryConfig) or not config.adaptive:
        raise ValueError('Cell response preparation requires adaptive complete fields')
    rhs,source_receipt = adaptive_volume_source_matrix(operator,bindings,**matrix_settings)
    with operator.read_access():
        operator.refresh()
        matrix = operator.matrix.clone()
        incident_rhs = operator.build_rhs()
        all_rhs = torch.cat((incident_rhs[:,None],rhs),dim=1)
        coefficients = torch.linalg.solve(matrix,all_rhs)
        error = matrix@coefficients-all_rhs
        configured_residual = _column_dual_norm(operator,error)
        configured_threshold = stopping.atol+stopping.rtol*_column_dual_norm(operator,all_rhs)
        if (not _finite_metrics(coefficients,configured_residual,configured_threshold)
                or bool((configured_residual > configured_threshold).any())):
            raise RuntimeError('Cell response multi-RHS solve failed configured dual residual tolerance')
        residual = _scaled_norm(error,dim=0)
        threshold = 512*torch.finfo(matrix.real.dtype).eps*_scaled_norm(all_rhs,dim=0)
        if not _finite_metrics(residual,threshold) or bool((residual > threshold).any()):
            raise RuntimeError('Cell response multi-RHS solve failed its true residual check')
    points = tuple(binding.volume.vertices[binding.volume.cells].mean(1) for binding in bindings)
    count = sum(len(value) for value in points)
    order = max(2,operator.integration.order,operator.integration.singular_order)
    receipt = {'source_matrix':source_receipt,'checks':[],'converged':False,
        'configured_dual_residuals':configured_residual.detach().cpu().tolist(),
        'configured_dual_thresholds':configured_threshold.detach().cpu().tolist(),
        'surface_absolute_residuals':residual.detach().cpu().tolist(),
        'surface_residual_thresholds':threshold.detach().cpu().tolist(),
        'field_estimator':'complete_E_H_per_unit_current_column_successive_order_difference',
        'source_unit_current_density_A_per_m2':1.,'volume_sampling':'cell_centroid_collocation',
        'physical_qualification':False,'volume_discretization_error_estimated':False,
        'arbitrary_source_combination_error_certified':False,
        'tensor_field_evaluations':[],
        'magnetic_absolute_tolerance_scale':'scalar_regional_eta_or_tensor_vacuum_eta'}
    if order >= config.max_order:
        receipt['failure_reason'] = 'initial_field_order_leaves_no_refinement'
        raise VolumeMatrixConvergenceError(receipt)

    def evaluate(current_order):
        electric = matrix.new_zeros(3*count,3*count)
        magnetic = torch.zeros_like(electric)
        offset = 0
        for binding,query in zip(bindings,points):
            index = binding.volume.region_id-1
            body = operator.system.bodies[index]
            handle = copy.copy(operator.body_handles[index])
            handle.integration = replace(handle.integration,order=current_order,singular_order=current_order,
                near_max_order=max(handle.integration.near_max_order,current_order+2))
            local = torch.cat((coefficients[operator.system.J_slices[index],1:],
                               coefficients[operator.system.M_slices[index],1:]),dim=0)
            e,h = decode_interior_columns(handle,local,query)
            electric[offset:offset+3*len(query)] = e.reshape(3*len(query),3*count)
            magnetic[offset:offset+3*len(query)] = h.reshape(3*len(query),3*count)
            if isinstance(body.interior,TensorMedium):
                blocks,volume_receipt=cell_tensor_current_blocks(binding.volume,query,handle.tensor_kernel,
                    order=current_order,radial_order=current_order,
                    face_rule=matrix_settings.get('tensor_face_rule','triangle'),
                    kernel_batch=matrix_settings.get('tensor_kernel_batch',128))
                direct_e=blocks[:,:,:3,:3].permute(0,2,1,3)
                direct_h=blocks[:,:,3:,:3].permute(0,2,1,3)
                receipt['tensor_field_evaluations'].append(dict(region_id=body.region_id,order=current_order,
                    surface=handle.last_column_field_receipt,volume=volume_receipt))
            else:
                direct_e,direct_h = cell_current_field_blocks(binding.volume,query,body.interior,order=current_order)
            width = 3*len(query)
            electric[offset:offset+width,offset:offset+width] += direct_e.reshape(width,width)
            magnetic[offset:offset+width,offset:offset+width] += direct_h.reshape(width,width)
            offset += width
        return electric,magnetic

    previous = tuple(value.detach() for value in evaluate(order))
    impedances = torch.cat([matrix.real.new_full((len(query),1),
        math.sqrt(MU_0/EPSILON_0) if isinstance(operator.system.bodies[binding.volume.region_id-1].interior,TensorMedium)
        else abs(operator.system.bodies[binding.volume.region_id-1].interior.eta))
        for binding,query in zip(bindings,points)])
    while order < config.max_order:
        order = min(2*order,config.max_order)
        result = evaluate(order)
        passed,check = True,{'order':order}
        for name,value,old,absolute in zip(('E','H'),result,previous,
                (config.absolute_tolerance,config.absolute_tolerance/impedances)):
            error = _scaled_norm((value-old).reshape(count,3,-1),dim=1)
            magnitude = _scaled_norm(value.reshape(count,3,-1),dim=1)
            threshold=absolute+config.relative_tolerance*magnitude
            if not _finite_metrics(error,magnitude,threshold):
                receipt.update(failure_reason='nonfinite_complete_column_error_metric',
                               failed_field=name,failed_order=order)
                raise VolumeMatrixConvergenceError(receipt)
            accepted = bool((error <= threshold).all())
            passed = passed and accepted
            check[name] = {'passed':accepted,'maximum_column_query_difference':float(error.max().detach())}
        receipt['checks'].append(check)
        if passed:
            receipt.update(converged=True,final_order=order,failure_reason=None)
            background_e,background_h,background_logs = [],[],[]
            for binding,query in zip(bindings,points):
                e,h = decode_system(operator,coefficients[:,0],query,binding.volume.region_id,'total',query_config=config)
                background_e.append(e)
                background_h.append(h)
                background_logs.append(operator.last_field_query_log)
            receipt['background_queries'] = background_logs
            return {'electric_response':result[0],'magnetic_response':result[1],
                'background_electric':torch.cat(background_e),'background_magnetic':torch.cat(background_h),
                'surface_response':coefficients[:,1:],'background_surface':coefficients[:,0],
                'cell_volumes':torch.cat([binding.volume.volumes for binding in bindings]),
                'points':torch.cat(points),'receipt':receipt}
        previous = tuple(value.detach() for value in result)
    receipt['failure_reason'] = 'complete_column_field_order_budget_exhausted'
    raise VolumeMatrixConvergenceError(receipt)
