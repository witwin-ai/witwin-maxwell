"""Static cell Green integrals, including distributional self terms.

These physical-volume integrals provide the static part of tensor volume
kernels. Complex AC static parts use explicit summed-cell refinement. Neither
path replaces the dynamic Maxwell remainder or establishes DC material admission.
"""
import math
import torch
from .ordered_reduction import add_ordered_rows
from .static_tensor_green import StaticTensorGreen
from .volume_mesh import TetrahedralVolumeMesh
from .fields import _pair_triangle_distance


def cell_tensor_static_potentials(mesh,points,kernel,*,cell_indices=None,
                                  principal_value=False,pair_batch=4096):
    """Return integral G, grad G and distributional Hess G as [query,cell,...].

    Near cells use analytic physical boundary integrals including contact.
    Separated cells use positive volume cubature with successive V/grad/H
    checks (rtol=1e-8, orders 8 through 128); budget exhaustion raises with
    its receipt. Use cell_real_static_potentials to retain success receipts.
    Face-interior principal values are explicit; cell edges/vertices reject.
    Selected columns preserve ordering and repetitions before integration.
    Batching bounds face-pair work, not retained output memory.
    """
    if type(kernel) is not StaticTensorGreen:
        raise TypeError('Analytic static tensor cells require a real-SPD Green kernel')
    return cell_real_static_potentials(mesh,points,kernel,cell_indices=cell_indices,
        principal_value=principal_value,pair_batch=pair_batch)[0]


def _cell_static_integrals(mesh,points,kernel,panel_integrals,*,cell_indices,
                           principal_value,pair_batch,far_order=None):
    """Shared physical boundary sum; callers own kernel and error admission."""
    if type(mesh) is not TetrahedralVolumeMesh:
        raise TypeError('Static tensor cells require a native volume mesh')
    mesh.assert_current();kernel.assert_current()
    if (mesh.vertices.device!=kernel.coefficient.device
            or mesh.vertices.dtype!=kernel.coefficient.real.dtype):
        raise ValueError('Static tensor cells and kernel must share device and precision')
    kernel._validate(points,(None,3))
    if type(principal_value) is not bool or type(pair_batch) is not int or pair_batch<1:
        raise ValueError('Static tensor cells require boolean PV and positive integer pair batch')
    cells=mesh.cells
    if cell_indices is not None:
        if (not isinstance(cell_indices,torch.Tensor) or cell_indices.device!=cells.device
                or cell_indices.dtype!=torch.int64 or cell_indices.ndim!=1
                or bool(((cell_indices<0)|(cell_indices>=len(cells))).any())):
            raise ValueError('Selected static cells require matching CUDA int64 indices in range')
        cells=cells[cell_indices]
    count=len(cells);queries=len(points)
    if not count or not queries:
        return tuple(kernel.coefficient.new_empty(queries,count,*shape) for shape in ((),(3,),(3,3)))
    local=cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    triangles=mesh.vertices[cells[:,local]].reshape(-1,3,3)
    normals=torch.linalg.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals=normals/normals.norm(dim=-1,keepdim=True)
    # Match the panel kernel's relative geometry and face-specific admission.
    origin=triangles[:,0]
    relative=triangles-origin[:,None]
    diameter=(triangles-triangles.roll(1,dims=1)).norm(dim=-1).amax(-1)
    tolerance=128*torch.finfo(points.dtype).eps*diameter
    values=kernel.coefficient.new_zeros(queries*count)
    gradients=kernel.coefficient.new_zeros(queries*count,3)
    hessians=kernel.coefficient.new_zeros(queries*count,3,3)
    far=None
    if far_order is not None:
        tetrahedra=mesh.vertices[cells]
        local_vertices=tetrahedra-tetrahedra[:,:1]
        center=local_vertices.mean(1)
        cell_diameter=(local_vertices[:,:,None]-local_vertices[:,None]).norm(dim=-1).amax(dim=(-2,-1))
        distance=(points[:,None]-tetrahedra[None,:,0]-center[None]).norm(dim=-1)
        # Every vertex is at most one diameter from the centroid, so these
        # pairs are strictly exterior and require no distributional contact.
        far=distance>=2*cell_diameter[None]
        from .dyadic_green import electrostatic_kernel
        flat=far.flatten().nonzero().flatten()
        for start in range(0,len(flat),min(pair_batch,8)):
            selected=flat[start:start+min(pair_batch,8)]
            q,c=selected//count,selected%count
            original=c if cell_indices is None else cell_indices[c]
            nodes,weights=mesh.quadrature(far_order,cell_indices=original)
            size=weights.shape[1]
            v=kernel.coefficient.new_zeros(len(selected))
            g=kernel.coefficient.new_zeros(len(selected),3)
            h=kernel.coefficient.new_zeros(len(selected),3,3)
            step=max(1,8192//len(selected))
            for first in range(0,size,step):
                displacement=points[q,None]-nodes[:,first:first+step]
                if type(kernel) is StaticTensorGreen:
                    integrands=kernel.evaluate_with_hessian(displacement.reshape(-1,3))
                    weight=weights[:,first:first+step]
                else:
                    integrands=electrostatic_kernel(displacement.reshape(-1,3),kernel.medium,constitutive=kernel.constitutive)
                    weight=weights[:,first:first+step]/kernel.units
                # Reduce each pair in a fixed node order. Atomic index_add
                # would make identical repeated columns differ by roundoff.
                v=v+(integrands[0].reshape_as(weight)*weight).sum(1)
                g=g+(integrands[1].reshape(*weight.shape,3)*weight[:,:,None]).sum(1)
                h=h+(integrands[2].reshape(*weight.shape,3,3)*weight[:,:,None,None]).sum(1)
            values=values.index_copy(0,selected,v)
            gradients=gradients.index_copy(0,selected,g)
            hessians=hessians.index_copy(0,selected,h)
    for start in range(0,queries*len(triangles),pair_batch):
        flat=torch.arange(start,min(start+pair_batch,queries*len(triangles)),device=points.device)
        q,f=flat//len(triangles),flat%len(triangles)
        if far is not None:
            keep=~far[q,f//4];q,f=q[keep],f[keep]
            if not len(q):continue
        on_face=_pair_triangle_distance(points[q]-origin[f],relative[f])<=tolerance[f]
        if not principal_value and bool(on_face.any()):
            raise ValueError('Static volume cell-face queries require explicit principal values')
        single,gradient=panel_integrals(points[q],triangles[f],principal_value=on_face)
        height=((triangles[f,0]-points[q])*normals[f]).sum(-1)
        rows=q*count+f//4
        values=add_ordered_rows(values,rows,.5*height*single)
        gradients=add_ordered_rows(gradients,rows,-single[:,None]*normals[f])
        hessians=add_ordered_rows(hessians,rows,-gradient[:,:,None]*normals[f,None,:])
    return (values.reshape(queries,count),gradients.reshape(queries,count,3),
            hessians.reshape(queries,count,3,3))


class StaticCellConvergenceError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Static cell integration exhausted its order budget')


class ComplexStaticCellConvergenceError(StaticCellConvergenceError):
    """Complex AC cell error receipt."""


class RealStaticCellConvergenceError(StaticCellConvergenceError):
    """Real SPD hybrid cell error receipt."""


def cell_real_static_potentials(mesh,points,kernel,**settings):
    """Real SPD analytic near cells and checked positive far-volume cubature."""
    if type(kernel) is not StaticTensorGreen:
        raise TypeError('Real static cells require StaticTensorGreen')
    return _adaptive_static_cell_potentials(mesh,points,kernel,**settings)


def cell_complex_static_potentials(mesh,points,kernel,**settings):
    """Complex AC physical cell integration with summed-cell error checks."""
    from .complex_static_tensor_green import ComplexStaticTensorGreen
    if type(kernel) is not ComplexStaticTensorGreen:
        raise TypeError('Complex static cells require ComplexStaticTensorGreen')
    return _adaptive_static_cell_potentials(mesh,points,kernel,**settings)


def _adaptive_static_cell_potentials(mesh,points,kernel,*,cell_indices=None,
        principal_value=False,pair_batch=8,relative_tolerance=1e-8,
        absolute_tolerance=0.,initial_order=8,max_order=128):
    """Return (potential, gradient, distributional Hessian), cell error receipt.

    The four physical boundary integrals already include the contact term.
    Each observer/cell must pass separate successive-order tests after its
    face contributions are summed. Panel convergence is not substituted for
    cell convergence. Absolute tolerance has volume-potential units; gradient
    and Hessian thresholds divide it by cell diameter and diameter squared.
    These estimates are not rigorous bounds, nor general physical acceptance.

    This integrates the AC static singular part only. The kernel's sufficient
    ellipticity admission remains required; no DC Laurent limit is inferred.
    """
    from .complex_static_tensor_green import ComplexStaticTensorGreen
    from .static_tensor_green import _static_panel_geometry
    real=type(kernel) is StaticTensorGreen
    error_type=RealStaticCellConvergenceError if real else ComplexStaticCellConvergenceError
    for value in (relative_tolerance,absolute_tolerance):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
            raise ValueError('Static cell tolerances must be finite and nonnegative')
    if relative_tolerance==0 and absolute_tolerance==0:
        raise ValueError('Static cell integration needs a nonzero tolerance')
    if type(initial_order) is not int or initial_order<2 or type(max_order) is not int or max_order<=initial_order:
        raise ValueError('Static cell integration needs increasing integer orders')
    if cell_indices is not None:
        if type(mesh) is not TetrahedralVolumeMesh:
            raise TypeError('Static tensor cells require a native volume mesh')
        mesh.assert_current()
        if (not isinstance(cell_indices,torch.Tensor) or cell_indices.device!=mesh.cells.device
                or cell_indices.dtype!=torch.int64 or cell_indices.ndim!=1
                or bool(((cell_indices<0)|(cell_indices>=len(mesh.cells))).any())):
            raise ValueError('Selected static cells require matching CUDA int64 indices in range')
        unique,inverse=torch.unique(cell_indices,sorted=True,return_inverse=True)
        if len(unique)!=len(cell_indices):
            # Integrate each physical cell once. Besides avoiding duplicate
            # work, gathering preserves identical repeated columns across
            # near/far and partial pair-batch boundaries exactly.
            identity=dict(selected_cells=len(cell_indices),unique_integrated_cells=len(unique),
                          estimator_cell_layout='query_unique_selected_cell')
            try:
                values,record=_adaptive_static_cell_potentials(mesh,points,kernel,cell_indices=unique,
                    principal_value=principal_value,pair_batch=pair_batch,relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,initial_order=initial_order,max_order=max_order)
            except StaticCellConvergenceError as error:
                error.receipt.update(identity)
                raise
            record.update(identity)
            return tuple(value[:,inverse] for value in values),record
    receipt=dict(representation='real_spd_hybrid_cell_v1' if real else 'complex_static_physical_cell_v3',
        constitutive='explicit_real_spd_coefficient' if real else kernel.constitutive,
        panel_rule='analytic_real_spd' if real else 'edge_radial_grading_and_face_boundary_lines',initial_order=initial_order,max_order=max_order,
        separated_rule='positive_tetrahedral_volume_cubature',separation_centroid_diameters=2.,
        separated_kernel_batch=8192,separated_pair_batch=min(pair_batch,8) if type(pair_batch) is int else pair_batch,
        ellipticity_admission='real_spd' if real else 'definite_real_or_strictly_positive_imaginary_symmetric_part',
        estimator='successive_order_per_summed_cell_potential_gradient_hessian',
        relative_tolerance=relative_tolerance,absolute_tolerance_potential=absolute_tolerance,
        derivative_absolute_scale='inverse_cell_diameter_power',pair_batch=pair_batch,
        principal_value=principal_value,contact_distribution_included=True,
        error_certified=False,physical_qualification=False,converged=False,checks=[])

    def integrate(order):
        def panels(x,triangles,*,principal_value):
            x,triangles,_,_,pv=_static_panel_geometry(x,triangles,principal_value)
            return kernel._integrate_panel_rule(x,triangles,pv,order)
        return _cell_static_integrals(mesh,points,kernel,kernel.panel_integrals if real else panels,cell_indices=cell_indices,
            principal_value=principal_value,pair_batch=pair_batch,far_order=order)

    order=initial_order;previous=integrate(order)
    receipt.update(query_count=previous[0].shape[0],selected_cells=previous[0].shape[1])
    if not previous[0].numel():
        receipt.update(converged=True,final_order=0,failure_reason=None)
        return previous,receipt
    cells=mesh.cells if cell_indices is None else mesh.cells[cell_indices]
    vertices=mesh.vertices[cells]
    diameter=(vertices[:,:,None]-vertices[:,None]).norm(dim=-1).amax(dim=(-2,-1))
    if not bool(torch.isfinite(diameter).all() & (diameter>0).all()):
        raise ValueError('Static cell diameter is not representable')

    def norm(value):
        flat=value.reshape(*previous[0].shape,-1)
        scale=flat.abs().amax(-1)
        safe=torch.where(scale==0,torch.ones_like(scale),scale)
        return scale*torch.linalg.vector_norm(flat/safe[:,:,None],dim=-1)

    while order<max_order:
        order=min(2*order,max_order);current=integrate(order)
        check={'order':order};passed=True;finite=True
        for power,(name,value,old) in enumerate(zip(('potential','gradient','hessian'),current,previous)):
            error=norm(value.detach()-old.detach());size=norm(value.detach())
            # Divide successively instead of forming a squared physical scale.
            absolute=torch.full_like(diameter,float(absolute_tolerance))
            for _ in range(power):absolute=absolute/diameter
            threshold=absolute[None]+relative_tolerance*size
            metrics_finite=torch.isfinite(error)&torch.isfinite(size)&torch.isfinite(threshold)
            accepted=metrics_finite&(error<=threshold)
            check[name]={'max_absolute_difference':float(error.max()),'failed_query_cells':int((~accepted).sum())}
            passed=passed and bool(accepted.all())
            finite=finite and bool(metrics_finite.all())
        receipt['checks'].append(check);receipt['final_order']=order
        if not finite:
            receipt['failure_reason']='nonfinite_cell_acceptance_metrics'
            raise error_type(receipt)
        if passed:
            kernel.assert_current();mesh.assert_current()
            receipt.update(converged=True,failure_reason=None)
            return current,receipt
        previous=tuple(value.detach() for value in current)
    receipt['failure_reason']='cell_order_budget_exhausted'
    raise error_type(receipt)
