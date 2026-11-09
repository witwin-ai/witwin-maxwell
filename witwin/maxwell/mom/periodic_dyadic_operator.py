"""Explicit regular exterior tensor correction, not a complete periodic solver."""
import torch
from .operator import IntegrationConfig,local_geometry
from .quadrature import triangle_rule
from .periodic_operator import validate_periodic_cell
from .periodic_dyadic import REGULAR_DYADIC_KERNEL_TYPES, PeriodicDyadicConvergenceError


def assemble_periodic_dyadic_correction(system,*,reference,config=IntegrationConfig()):
    """Weakly test only nonprimary exterior image fields over every body pair.

    Rows are (H,-E), columns (J,M), stored globally as all J then all M.
    The entire free-space primary contribution and every interior contribution
    remain separate obligations. Both the strictly lossy image reference and the checked 1D/2D spectral
    regular kernel require explicit primary exclusion.
    """
    if not isinstance(reference,REGULAR_DYADIC_KERNEL_TYPES):
        raise TypeError('Periodic dyadic correction requires a supported regular dyadic kernel')
    if not isinstance(config,IntegrationConfig):
        raise TypeError('Periodic dyadic testing requires IntegrationConfig')
    validate_periodic_cell(system,reference.lattice)
    reference._validate()
    medium=reference.medium
    if any(body.interior.frequency!=medium.frequency for body in system.bodies):
        raise ValueError('Periodic tensor correction and all bodies must share one frequency')
    correction=medium.epsilon_r.new_zeros(2*system.total_edges,2*system.total_edges)
    records=[]
    for i,test_body in enumerate(system.bodies):
        test=test_body.basis
        test_geometry=local_geometry(test)
        for j,source_body in enumerate(system.bodies):
            source=source_body.basis
            geometry=(test_geometry,local_geometry(source))
            count=len(test.mesh.faces)*len(source.mesh.faces)
            receipts=[]
            for start in range(0,count,config.pair_batch):
                ids=torch.arange(start,min(start+config.pair_batch,count),device=system.device)
                rows,cols=ids//len(source.mesh.faces),ids%len(source.mesh.faces)
                blocks,receipt=_regular_panel_blocks(test,source,rows,cols,reference,config,geometry,
                    body_pair=(i,j))
                receipts.append(receipt)
                row_edges=test.face_edges[rows,:,None].expand(-1,-1,3).reshape(-1)
                col_edges=source.face_edges[cols,None,:].expand(-1,3,-1).reshape(-1)
                for row_kind,offset,sign in ((0,3,1),(1,0,-1)):
                    for col_kind,source_offset in ((0,0),(1,3)):
                        block=blocks[:,row_kind,col_kind]
                        row_slice=(system.J_slices[i],system.M_slices[i])[row_kind]
                        col_slice=(system.J_slices[j],system.M_slices[j])[col_kind]
                        correction.index_put_((row_edges+row_slice.start,col_edges+col_slice.start),
                            block.reshape(-1),accumulate=True)
            records.append({'test_region':test_body.region_id,'source_region':source_body.region_id,
                'panel_pairs':count,'image_evaluations':receipts})
    return correction,{'quantity':'nonprimary_exterior_tensor_weak_correction',
        'unknown_order':'all_body_J_then_all_body_M','testing_rows':['H','minus_E'],
        'primary_included':False,'interiors_periodic':False,'blocks':records,
        'surface_quadrature_order':config.order,'surface_quadrature_converged':False,
        'image_cutoff_converged':False,'complete_operator':False,'physical_qualification':False}


def _regular_panel_blocks(test,source,rows,cols,reference,config,geometry,*,body_pair):
    """Shared product-rule integration for nonprimary exterior panel pairs."""
    (_,test_free,test_slopes),(_,source_free,source_slopes)=geometry
    x,wx=triangle_rule(test.mesh.triangles[rows],config.order)
    y,wy=triangle_rule(source.mesh.triangles[cols],config.order)
    difference=x[:,:,None,:]-y[:,None,:,:]
    try:
        green,receipt=reference.evaluate(difference.reshape(-1,3),exclude_primary=True)
    except PeriodicDyadicConvergenceError as error:
        # Failure-only metadata identifies the exact product-rule inputs without
        # retaining a potentially large quadrature-point tensor on the host.
        receipt=dict(error.receipt,regular_assembly_context=dict(
            test_body_index=body_pair[0],source_body_index=body_pair[1],
            test_face_indices=rows.detach().cpu().tolist(),
            source_face_indices=cols.detach().cpu().tolist(),
            quadrature_order=config.order,quadrature_rule='triangle_rule',
            displacement_order='panel_pair_then_test_point_then_source_point',
            query_count=difference.numel()//3))
        raise PeriodicDyadicConvergenceError(receipt) from error
    if receipt.get('primary_image_excluded') is not True:
        raise ValueError('Periodic tensor correction cannot contain the primary singular image')
    green=green.reshape(len(rows),x.shape[1],y.shape[1],6,6)
    left=(test_slopes[rows,None,:,None]*(x[:,:,None,:]-test_free[rows,None])).to(green.dtype)
    right=(source_slopes[cols,None,:,None]*(y[:,:,None,:]-source_free[cols,None])).to(green.dtype)
    weights=(wx[:,:,None]*wy[:,None,:]).to(green.dtype)
    blocks=green.new_empty((len(rows),2,2,3,3))
    for row,offset,sign in ((0,3,1),(1,0,-1)):
        for column,source_offset in ((0,0),(1,3)):
            blocks[:,row,column]=sign*torch.einsum('bxli,bxyij,bymj,bxy->blm',left,
                green[:,:,:,offset:offset+3,source_offset:source_offset+3],right,weights)
    return blocks,receipt


def periodic_dyadic_regular_block(system,test_body_index,source_body_index,test_edges,source_edges,
                                  *,reference,config=IntegrationConfig()):
    """Integrate selected nonprimary exterior rows/columns with bounded panels.

    Returns local [selected J, selected M] ordering on both axes. No primary
    singular term, local interior or global dense matrix is constructed.
    """
    from .nearfield import _selected_edges,_local_indices
    if not isinstance(reference,REGULAR_DYADIC_KERNEL_TYPES) or not isinstance(config,IntegrationConfig):
        raise TypeError('Selected periodic tensor blocks require a regular kernel and IntegrationConfig')
    validate_periodic_cell(system,reference.lattice)
    reference._validate()
    if any(type(index) is not int or not 0<=index<len(system.bodies)
           for index in (test_body_index,source_body_index)):
        raise ValueError('Selected periodic body index is outside the system')
    if any(body.interior.frequency!=reference.medium.frequency for body in system.bodies):
        raise ValueError('Periodic tensor correction and all bodies must share one frequency')
    test=system.bodies[test_body_index].basis
    source=system.bodies[source_body_index].basis
    re,rp,rf=_selected_edges(test,test_edges)
    ce,cp,cf=_selected_edges(source,source_edges)
    nr,nc=len(re),len(ce)
    result=reference.medium.epsilon_r.new_zeros((2*nr,2*nc))
    total=len(rf)*len(cf)
    geometry=(local_geometry(test),local_geometry(source))
    maximum_pairs=0
    receipts=[]
    for start in range(0,total,config.pair_batch):
        ids=torch.arange(start,min(start+config.pair_batch,total),device=system.device)
        rows,cols=rf[ids//len(cf)],cf[ids%len(cf)]
        maximum_pairs=max(maximum_pairs,len(ids))
        blocks,receipt=_regular_panel_blocks(test,source,rows,cols,reference,config,geometry,
            body_pair=(test_body_index,source_body_index))
        receipts.append(receipt)
        ri,rv=_local_indices(test.face_edges[rows],re,rp)
        ci,cv=_local_indices(source.face_edges[cols],ce,cp)
        keep=rv[:,:,None]&cv[:,None,:]
        rr=ri[:,:,None].expand(-1,-1,3)[keep]
        cc=ci[:,None,:].expand(-1,3,-1)[keep]
        for row in range(2):
            for column in range(2):
                result.index_put_((rr+row*nr,cc+column*nc),blocks[:,row,column][keep],accumulate=True)
    return result,dict(version='periodic_dyadic_selected_regular_block_v1',
        test_region=system.bodies[test_body_index].region_id,
        source_region=system.bodies[source_body_index].region_id,panel_pairs=total,
        maximum_panel_pairs=maximum_pairs,matrix_bytes=result.numel()*result.element_size(),
        image_evaluations=receipts,surface_quadrature_order=config.order,
        primary_included=False,interiors_periodic=False,complete_operator=False,
        surface_quadrature_converged=False,physical_qualification=False)


def periodic_dyadic_system_block(system,test_body_index,source_body_index,test_edges,source_edges,
                                 *,reference,exterior_kernel,interior_kernels,config=IntegrationConfig()):
    """Compose one complete selected body pair with explicit regional kernels.

    The exterior primary and nonprimary images contribute to every body pair;
    a homogeneous interior contributes only to its own body's diagonal block.
    Returned axes are local selected J followed by selected M. No dense system
    is assembled. Fixed integration budgets do not certify physical accuracy.
    """
    from .anisotropic_operator import tensor_regional_block
    from .nearfield import scalar_regional_block
    from .dyadic_green import DyadicGreenKernel
    from .tensor_medium import TensorMedium
    if not isinstance(reference,REGULAR_DYADIC_KERNEL_TYPES) or not isinstance(config,IntegrationConfig):
        raise TypeError('Selected periodic systems require a regular kernel and IntegrationConfig')
    validate_periodic_cell(system,reference.lattice)
    reference._validate()
    if any(type(index) is not int or not 0<=index<len(system.bodies)
           for index in (test_body_index,source_body_index)):
        raise ValueError('Selected periodic body index is outside the system')
    if any(body.interior.frequency!=reference.medium.frequency for body in system.bodies):
        raise ValueError('Periodic tensor correction and all bodies must share one frequency')
    if not isinstance(exterior_kernel,DyadicGreenKernel) or exterior_kernel.medium is not reference.medium:
        raise ValueError('Selected exterior kernel must bind the exact sampled tensor medium')
    tensor_ids={body.region_id for body in system.bodies if isinstance(body.interior,TensorMedium)}
    if type(interior_kernels) is not dict or set(interior_kernels)!=tensor_ids:
        raise ValueError('Selected interior kernels must cover every tensor region exactly once')
    for body in system.bodies:
        if body.region_id in tensor_ids:
            kernel=interior_kernels[body.region_id]
            if not isinstance(kernel,DyadicGreenKernel) or kernel.medium is not body.interior:
                raise ValueError('Selected interior kernel must bind the exact sampled tensor medium')
    test,source=system.bodies[test_body_index],system.bodies[source_body_index]
    result,outside=tensor_regional_block(test.basis,source.basis,test_edges,source_edges,
        reference.medium,config,kernel=exterior_kernel)
    inside=None
    if test_body_index==source_body_index:
        if isinstance(test.interior,TensorMedium):
            block,inside=tensor_regional_block(test.basis,source.basis,test_edges,source_edges,
                test.interior,config,kernel=interior_kernels[test.region_id])
        else:
            block,inside=scalar_regional_block(test.basis,source.basis,test_edges,source_edges,
                (test.interior,),config)
        result=result+block
    regular,images=periodic_dyadic_regular_block(system,test_body_index,source_body_index,
        test_edges,source_edges,reference=reference,config=config)
    return result+regular,dict(version='periodic_dyadic_selected_system_block_v1',
        test_region=test.region_id,source_region=source.region_id,exterior_primary=outside,
        local_interior=inside,nonprimary_exterior=images,primary_included=True,
        interiors_periodic=False,complete_operator=True,physical_qualification=False)


def assemble_periodic_dyadic_system(system, *, reference, config=IntegrationConfig(),
                                    exterior_kernel=None, interior_kernels=None):
    """Compose the free-space primary, local interiors and nonprimary exterior.

    This is a fixed-cutoff experimental matrix. Complete contribution ownership
    is distinct from surface/image convergence or physical qualification.
    """
    from .anisotropic_operator import assemble_generalized_system
    if not isinstance(reference, REGULAR_DYADIC_KERNEL_TYPES):
        raise TypeError('Periodic dyadic assembly requires a supported regular dyadic kernel')
    validate_periodic_cell(system, reference.lattice)
    reference._validate()
    primary, primary_receipt = assemble_generalized_system(system, reference.medium, config,
        exterior_kernel=exterior_kernel, interior_kernels=interior_kernels)
    correction, image_receipt = assemble_periodic_dyadic_correction(system,
        reference=reference, config=config)
    return primary+correction, {
        'formulation_id': 'anisotropic_dgf_sie',
        'kernel_id': reference.kernel_id,
        'unknown_order': 'all_body_J_then_all_body_M',
        'primary': primary_receipt, 'nonprimary_exterior': image_receipt,
        'primary_included': True, 'interiors_periodic': False,
        'complete_operator': True, 'surface_quadrature_converged': False,
        'image_cutoff_converged': False, 'physical_qualification': False}


def build_periodic_dyadic_rhs(system, carrier, *, reference, config=IntegrationConfig(), spectral_config=None,
                             interior_kernels=None, rhs_cache_max_bytes=0):
    """Analytic lifted RHS with free-space constitutive contrast per body.

    The exterior analytic sheet cancellation leaves no image forcing and no
    cross-body interior term. Each mode must first pass Maxwell/Bloch checks.
    This internal assembly does not compile public source specifications.
    """
    from types import SimpleNamespace
    from .tensor_carrier import TensorIncidentCarrier
    from .dyadic_green import DyadicIntegrationConfig, DyadicGreenKernel
    from .tensor_medium import TensorMedium
    from .anisotropic_lifting import build_tensor_lifted_rhs
    from .lifting import CURRENT_KIND
    if not isinstance(reference, REGULAR_DYADIC_KERNEL_TYPES) or not isinstance(carrier, TensorIncidentCarrier):
        raise TypeError('Periodic tensor RHS requires a tensor carrier and regular dyadic kernel')
    if not isinstance(config, IntegrationConfig):
        raise TypeError('Periodic tensor RHS requires IntegrationConfig')
    spectral_config = DyadicIntegrationConfig() if spectral_config is None else spectral_config
    if not isinstance(spectral_config, DyadicIntegrationConfig):
        raise TypeError('Periodic tensor RHS requires DyadicIntegrationConfig for spectral_config')
    if type(rhs_cache_max_bytes) is not int or rhs_cache_max_bytes < 0:
        raise ValueError('Periodic tensor RHS cache budget must be a nonnegative integer')
    if interior_kernels is not None and (not isinstance(interior_kernels, dict)
            or set(interior_kernels) != {body.region_id for body in system.bodies}):
        raise ValueError('Periodic tensor RHS kernels must cover every body exactly once')
    validate_periodic_cell(system, reference.lattice)
    reference._validate()
    carrier.validate_medium(reference.medium)
    carrier.validate_lattice(reference.lattice)
    if any(body.interior.frequency != reference.medium.frequency for body in system.bodies):
        raise ValueError('Every periodic tensor region must share the incident frequency')
    rhs = reference.medium.epsilon_r.new_zeros(2*system.total_edges)
    records = []
    eye = torch.eye(3, device=rhs.device, dtype=rhs.dtype)
    for i, body in enumerate(system.bodies):
        inside = body.interior
        if not isinstance(inside, TensorMedium):
            inside = TensorMedium(inside.frequency, inside.epsilon_r*eye, inside.mu_r*eye)
        handle = SimpleNamespace(basis=body.basis, exterior=reference.medium, interior=inside,
                                 incident_carrier=carrier, integration=config,
                                 tensor_rhs_cache_max_bytes=rhs_cache_max_bytes)
        kernel = (DyadicGreenKernel(inside, spectral_config) if interior_kernels is None
                  else interior_kernels[body.region_id])
        if (not isinstance(kernel, DyadicGreenKernel) or kernel.medium is not inside
                or kernel.integration != spectral_config):
            raise ValueError('Periodic tensor RHS kernel must bind the exact body and spectral policy')
        local, receipt = build_tensor_lifted_rhs(handle, interior_kernel=kernel)
        rhs[system.J_slices[i]], rhs[system.M_slices[i]] = local[:body.basis.n_edges], local[body.basis.n_edges:]
        records.append({'region_id': body.region_id, 'rhs': receipt})
    return rhs, {'quantity': 'periodic_tensor_analytic_lifted_rhs',
        'current_kind': CURRENT_KIND, 'unknown_order': 'all_body_J_then_all_body_M',
        'incident_carrier_identity': carrier.identity, 'regions': records,
        'image_forcing_added': False, 'interior_cross_body_terms': False,
        'spectral_config': vars(spectral_config),
        'physical_qualification': False}
