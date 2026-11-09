"""Tensor AC cell-current Green integrals with analytic static contact terms."""
import torch
from .ordered_reduction import add_ordered_rows
from .dyadic_green import DyadicGreenKernel
from .static_tensor_green import StaticTensorGreen
from .static_tensor_volume import (cell_complex_static_potentials,
    ComplexStaticCellConvergenceError)
from .complex_static_tensor_green import ComplexStaticTensorGreen
from .quadrature import triangle_rule,gauss_rule,duffy_fan,distance_scaled_duffy_fan
from .fields import _pair_triangle_distance


def tensor_volume_static_kernels(medium):
    """Admit actual AC static parts before volume/HB matrix preparation.

    Analytic DC admission is separate: these are the symmetric parts of the
    actual frequency samples, not a replacement for their DC Laurent limits.
    """
    from .tensor_medium import TensorMedium
    if not isinstance(medium,TensorMedium):
        raise TypeError('Tensor volume admission requires TensorMedium')
    medium.assert_current()
    static=[]
    for kind,coefficient in zip(('electric','magnetic'),(medium.epsilon,medium.mu)):
        symmetric=(coefficient+coefficient.mT)/2
        real=not bool((symmetric.imag!=0).any())
        if real and bool(torch.linalg.cholesky_ex(symmetric.real)[1]==0):
            static.append(StaticTensorGreen(symmetric.real))
        else:
            static.append(ComplexStaticTensorGreen(medium,constitutive=kind))
    return tuple(static)


def tensor_static_cell_potentials(mesh,points,kernel,*,cell_indices=None,
                                  principal_value=False,pair_batch=4096):
    """Physical static cell dispatch with an explicit versioned inner policy.

    The complex policy checks summed cell quantities, not just panels. Outer
    weak-source and complete-field refinement remain separate obligations.
    """
    if type(pair_batch) is not int or pair_batch<1:
        raise ValueError('Static cell pair batch must be a positive integer')
    if type(kernel) is StaticTensorGreen:
        from .static_tensor_volume import cell_real_static_potentials
        return cell_real_static_potentials(mesh,points,kernel,cell_indices=cell_indices,
            principal_value=principal_value,pair_batch=pair_batch)
    if type(kernel) is not ComplexStaticTensorGreen:
        raise TypeError('Tensor static cells require a native real or complex Green kernel')
    policy=dict(relative_tolerance=1e-8,absolute_tolerance=0.,initial_order=8,max_order=128,
                pair_batch=min(pair_batch,8))
    identity=dict(version='complex_ac_static_cell_v1',**policy)
    try:
        values,receipt=cell_complex_static_potentials(mesh,points,kernel,cell_indices=cell_indices,
            principal_value=principal_value,**policy)
    except ComplexStaticCellConvergenceError as error:
        error.receipt['integration_policy']=identity
        raise
    receipt['integration_policy']=identity
    return values,receipt


def cell_tensor_current_fields(mesh,current,points,kernel,*,order=6,radial_order=6,
                               face_rule='triangle',kernel_batch=128):
    """Complete electric-current volume E/H, without a surface or incident field.

    Fixed volume orders are unestimated; a coupled caller must refine the sum
    of its surface and volume fields. Current gradients remain native Torch.
    """
    if type(kernel) is not DyadicGreenKernel:
        raise TypeError('Tensor volume fields require an owned dyadic Green kernel')
    if (not isinstance(current,torch.Tensor) or current.shape!=(len(mesh.cells),3)
            or current.device!=kernel.medium.epsilon_r.device
            or current.dtype!=kernel.medium.epsilon_r.dtype or not bool(torch.isfinite(current).all())):
        raise ValueError('Tensor volume currents must match native cell xyz, device and precision')
    blocks,receipt=cell_tensor_current_blocks(mesh,points,kernel,order=order,
        radial_order=radial_order,face_rule=face_rule,kernel_batch=kernel_batch)
    field=torch.einsum('qcij,cj->qi',blocks[:,:,:,:3],current)
    if not bool(torch.isfinite(field).all()):
        raise ValueError('Tensor volume current fields are not finite')
    receipt.update(representation='tensor_volume_electric_current_fields',
                   current_layout='cell_xyz',current_units='A_per_m2')
    return (field[:,:3],field[:,3:]),receipt


def cell_tensor_current_blocks(mesh,points,kernel,*,order=6,radial_order=6,
                               cell_indices=None,principal_value=False,pair_batch=4096,
                               kernel_batch=128,face_rule='triangle',regular_only=False):
    """Return [query,cell,6,6] physical [J,M] -> [E,H] volume blocks.

    Static diagonal Hessians include distributional contact terms analytically.
    The remaining locally integrable Maxwell kernel uses signed radial volume
    fans anchored at each observer. No origin node, epsilon cutoff or added
    constitutive contrast is used. Outer and radial orders are fixed: spectral
    convergence does not establish convergence of this volume quadrature.
    regular_only omits both static diagonal blocks for callers that integrate
    those distributions separately in a weak form; cross blocks are retained.

    Real-SPD transpose-symmetric parts use analytic cell integration; complex
    elliptic parts use explicit summed-cell error checks. Imaginary
    antisymmetric gyrotropy remains in the dynamic kernel. Nonelliptic passive
    static kernels still need a separate limiting formulation.
    """
    if type(kernel) is not DyadicGreenKernel:
        raise TypeError('Tensor volume integration requires an owned dyadic Green kernel')
    medium=kernel.medium;medium.assert_current()
    for value in (order,radial_order):
        if type(value) is not int or value<2:raise ValueError('Tensor volume orders must be integers >= 2')
    if type(kernel_batch) is not int or kernel_batch<1:
        raise ValueError('Tensor volume kernel batch must be a positive integer')
    if face_rule not in ('triangle','projected_duffy','distance_scaled_duffy'):
        raise ValueError('Tensor volume face rule must be triangle, projected_duffy or distance_scaled_duffy')
    if type(regular_only) is not bool:
        raise ValueError('Tensor volume regular_only must be boolean')
    static=tensor_volume_static_kernels(medium)
    # Shared integration owns query, selection, geometry and trace admission.
    integrated=[tensor_static_cell_potentials(mesh,points,k,cell_indices=cell_indices,
        principal_value=principal_value,pair_batch=pair_batch) for k in static]
    hessians=[values[2] for values,_ in integrated]
    cells=mesh.cells if cell_indices is None else mesh.cells[cell_indices]
    count=len(cells);queries=len(points)
    result=medium.epsilon_r.new_zeros(queries,count,6,6)
    if not regular_only:
        result[:,:,:3,:3]=1j/medium.omega*hessians[0]
        result[:,:,3:,3:]=1j/medium.omega*hessians[1]
    receipt=dict(version='tensor_cell_radial_remainder_v1' if regular_only else 'tensor_cell_static_plus_radial_remainder_v1',order=order,
        radial_order=radial_order,kernel_batch=kernel_batch,face_rule=face_rule,selected_cells=count,
        query_count=queries,contact_distribution_included=not regular_only,
        static_diagonal_included=not regular_only,
        static_cell_integrations=[dict(constitutive=kind,receipt=record)
            for kind,(_,record) in zip(('electric','magnetic'),integrated)],
        volume_quadrature_converged=False,physical_qualification=False,
        linear_contrast_added=False,spectral_evaluations=[],kernel_nodes=0)
    if not queries or not count:return result,receipt
    local=cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    panels=mesh.vertices[cells[:,local]].reshape(-1,3,3)
    normals=torch.linalg.cross(panels[:,1]-panels[:,0],panels[:,2]-panels[:,0])
    normals=normals/normals.norm(dim=-1,keepdim=True)
    origin=panels[:,0]
    relative=panels-origin[:,None]
    diameter=(panels-panels.roll(1,dims=1)).norm(dim=-1).amax(-1)
    tolerance=128*torch.finfo(points.dtype).eps*diameter
    if face_rule=='triangle':nodes,weights=triangle_rule(panels,order)
    t,w=gauss_rule(radial_order,device=points.device,dtype=points.dtype)
    dynamic=result.new_zeros(queries*count,6,6)
    # Stream node indices, so kernel_batch bounds displacements even at high
    # volume orders. Signed fans are essential for observers outside a cell.
    samples=(3 if face_rule!='triangle' else 1)*order**2*radial_order
    total=queries*len(panels)*samples
    geometry_key=None
    for start in range(0,total,kernel_batch):
        flat=torch.arange(start,min(start+kernel_batch,total),device=points.device)
        pair,node=flat//samples,flat%samples
        q,f=pair//len(panels),pair%len(panels)
        surface,radial=node//radial_order,node%radial_order
        # Use the static panel's face-PV admission, including its relative
        # geometry tolerance. A face containing the observer has zero cone
        # volume even when rotation leaves a floating-point height residual.
        # Consecutive node batches often visit the same observer/face pair.
        # Retain only that contiguous geometry interval, without detaching its
        # autograd graph or changing node order, weights or kernel batches.
        key=(start//samples,(min(start+kernel_batch,total)-1)//samples)
        if key!=geometry_key:
            unique=torch.arange(key[0],key[1]+1,device=points.device)
            uq,uf=unique//len(panels),unique%len(panels)
            on_face=_pair_triangle_distance(points[uq]-origin[uf],relative[uf])<=tolerance[uf]
            pair_height=((origin[uf]-points[uq])*normals[uf]).sum(-1)
            pair_height=torch.where(on_face,torch.zeros_like(pair_height),pair_height)
            if face_rule!='triangle':
                # At most one contiguous interval is retained. Workspace still
                # includes 3*order**2 nodes per unique observer/face pair.
                fan=distance_scaled_duffy_fan if face_rule=='distance_scaled_duffy' else duffy_fan
                fan_nodes,fan_weights=fan(points[uq],panels[uf],order)
            geometry_key=key
        inverse=pair-key[0]
        height=pair_height[inverse]
        if face_rule!='triangle':
            source=fan_nodes[inverse,surface]
            area=fan_weights[inverse,surface]
        else:
            source=nodes[f,surface]
            area=weights[f,surface]
        weight=height*area*w[radial]*t[radial].square()
        keep=weight!=0
        if not bool(keep.any()):continue
        q,f,source,radial,weight=(v[keep] for v in (q,f,source,radial,weight))
        delta=t[radial,None]*(points[q]-source)
        green,record=kernel.evaluate_regular(delta)
        dynamic=add_ordered_rows(dynamic,q*count+f//4,weight[:,None,None]*green)
        receipt['spectral_evaluations'].append(record)
        receipt['kernel_nodes']+=len(delta)
    result=result+dynamic.reshape(queries,count,6,6)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Tensor volume integrated current blocks are not finite')
    medium.assert_current();mesh.assert_current()
    return result,receipt
