"""Dual conductive DC preparation, recovery and decode-only HB integration."""
import copy
import torch
from .dual_static_current import DualStaticExterior
from .dual_static_snapshot import snapshot_dual_state,restore_dual_state
from .dual_preparation_snapshot import snapshot_dual_preparation,restore_dual_preparation


def dual_materials(materials):
    return all(m.electric_dc.kind=='conductive_limit' and m.magnetic_dc.kind=='conductive_limit'
               and m.electric_dc.conductivity>0 and m.magnetic_dc.conductivity>0 for m in materials)


def encode_static(static,bindings):
    payload,identity=snapshot_dual_preparation(static['current_closure'],bindings,static['floating_operator'])
    return {**{k:v for k,v in static.items() if k not in ('current_closure','floating_operator')},
            'dual_preparation':payload,'dual_binding':identity}


def restore_static(static,bindings,materials,testing_order):
    if not dual_materials(materials): raise ValueError('Dual HB requires positive electric and magnetic DC conduction in every body')
    closure,laplace=restore_dual_preparation(static.pop('dual_preparation'),bindings,
        sigma_e=[materials[b.volume.region_id-1].electric_dc.conductivity for b in bindings],
        sigma_m=[materials[b.volume.region_id-1].magnetic_dc.conductivity for b in bindings],
        expected_binding=static.pop('dual_binding'))
    if ((closure.relative_tolerance,closure.absolute_tolerance,closure.max_iterations,closure.max_backtracks)!=(1e-10,1e-12,30,12)
            or laplace.testing_order!=testing_order):
        raise ValueError('Dual preparation differs from native closure policy')
    static.update(current_closure=closure,floating_operator=laplace)


def dual_ohmic_current(closure,electric):
    sigma=closure.material_coefficients[0] if closure.streamed_rule is not None else closure.current.electric
    return sigma.reshape(-1,3)*electric


def recover_dual(closure,prepared,field,tpa,*,electric_source=None,threshold=None):
    binding=None
    if closure.streamed_rule is not None:
        import math
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold<0:
            raise ValueError('Streamed dual recovery requires the owning HB threshold')
        # Recover the actual outer HB source, not a separately resolved inner
        # nonlinear source that can differ by accepted floating-point error.
        result=closure.current.solve(electric_source,torch.zeros_like(electric_source),
                                     conductivities=closure.material_coefficients)
        defect=float((result['electric']-field[0].real).norm().detach())
        if not math.isfinite(defect) or defect>threshold:
            raise ValueError('Recovered dual DC field violates the owning HB threshold')
        result['receipt']['hb_recovery']={'method':'outer_source_linear_response',
            'threshold_V_per_m_l2':threshold,'field_residual_V_per_m_l2':defect,
            'nonlinear_root_recomputed':False,'physical_qualification':False}
    else:
        result=closure.solve(field[1:],tpa)
    exterior=DualStaticExterior(closure.current,result,prepared['bindings'][0],prepared['static']['floating_operator'])
    if closure.streamed_rule is not None:
        from .streamed_dual_state_snapshot import snapshot_streamed_dual_state
        payload,binding=snapshot_streamed_dual_state(closure.current,result,exterior,
            electric_source,torch.zeros_like(electric_source))
    else:
        payload=snapshot_dual_state(closure.current,result,exterior)
        payload['receipt']['nonlinear']=copy.deepcopy(result['nonlinear_receipt'])
    return payload,exterior.electric_layer.density.clone(),(result,exterior),binding


def decode_dual(prepared,state):
    """Restore the state against the independent identity owned by HB."""
    static=prepared['static']; closure=static['current_closure']
    if closure.streamed_rule is not None:
        from .streamed_dual_state_snapshot import restore_streamed_dual_state
        return restore_streamed_dual_state(state.dc_dual_state,closure.current,
            prepared['bindings'][0],static['floating_operator'],expected_binding=state.dc_dual_binding)
    if state.dc_dual_binding is not None:
        raise ValueError('Dense dual state cannot carry a streamed identity')
    return restore_dual_state(state.dc_dual_state,closure.current,
                              prepared['bindings'][0],static['floating_operator'])


def validate_dual(prepared,state,threshold):
    static=prepared['static']; closure=static['current_closure']
    result,exterior=decode_dual(prepared,state)
    source=state.dissipative_current[0].real.flatten()
    actual=state.cell_electric[0].real.flatten()
    if closure.streamed_rule is not None:
        from .streamed_dual_state_snapshot import validate_response_witness
        electric_source=source.reshape(-1,3)
        expected=validate_response_witness(result,electric_source,torch.zeros_like(electric_source),actual,threshold)
    else:
        expected=closure.response@source
        if not bool((actual-expected).norm()<=threshold) or not bool((result['electric'].flatten()-actual).norm()<=threshold):
            raise ValueError('Stored dual DC field violates the nonlinear coupled equation')
    # Native constitutive arrays are already checked by the outer HB reader.
    expected_j=closure.current.electric*actual+source
    expected_m=closure.current.magnetic*result['magnetic'].flatten()
    eps=torch.finfo(actual.dtype).eps
    for observed,reference,budget in ((result['electric_current'].flatten(),expected_j,
            closure.current.electric.abs().max()*threshold),
            (result['magnetic_current'].flatten(),expected_m,actual.new_zeros(()))):
        allowed=budget+4096*eps*(observed.norm()+reference.norm())
        if not bool(torch.isfinite(allowed)) or not bool((observed-reference).norm()<=allowed):
            raise ValueError('Stored dual DC currents violate the constitutive law')
    if not torch.equal(state.dc_surface_charge,exterior.electric_layer.density):
        raise ValueError('Stored dual electric layer disagrees with harmonic state')
    return expected


def query_dual(prepared,state,points,region_id,decoded=None):
    from .magnetostatic import cell_magnetostatic_current_fields
    from ..constants import MU_0
    static=prepared['static']; closure=static['current_closure']; laplace=static['floating_operator']
    if decoded is None:
        decoded=state.dc_dual_live
        if decoded is None:
            decoded=decode_dual(prepared,state)
    result,exterior=decoded
    if exterior._versions != exterior._state():
        raise RuntimeError('Dual exterior preparation or currents were mutated')
    exterior.electric_layer._assert_current()
    exterior.magnetic_layer._assert_current()
    region=laplace.system.region(region_id)
    if not isinstance(points,torch.Tensor) or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('Dual DC queries require real CUDA [Q,3] points')
    laplace._real(points,points.shape,'Dual DC query points')
    if region==0:
        fields=exterior.fields(points)
        fields.update(magnetic_flux_density_T=MU_0*fields['magnetic_A_per_m'],magnetic_flux_pole_residue_T_per_s=None)
    else:
        index=next(i for i,v in enumerate(closure.volumes) if v.region_id==region)
        block=closure.current.blocks[index]; volume=block.volume
        potentials=[]; gradients=[]
        for key,layer in (('electric_potentials',exterior.electric_layer),('magnetic_potentials',exterior.magnetic_layer)):
            potential=result[key][index]; values=[]; vectors=[]
            for batch in points.split(32):
                delta=batch[:,None]-volume.vertices[volume.cells[:,0]][None]
                local=torch.einsum('qti,tai->qta',delta,-block.gradients[:,1:]/3)
                bary=torch.cat((1-local.sum(-1,keepdim=True),local),-1)
                matches=(bary>=-128*torch.finfo(points.dtype).eps).all(-1)
                if bool((matches.sum(-1)!=1).any()):
                    raise ValueError('Dual DC queries require a unique owning cell interior')
                cell=matches.to(torch.int64).argmax(-1)
                nodal=potential[block.cell_dofs[cell]]
                weights=bary[torch.arange(len(batch),device=points.device),cell]
                values.append(((1-3*weights)*nodal).sum(-1)+layer.offsets[region-1])
                vectors.append(-torch.einsum('qa,qai->qi',nodal,block.gradients[cell]))
            potentials.append(torch.cat(values) if len(points) else points.new_empty(0))
            gradients.append(torch.cat(vectors) if len(points) else points.new_empty(0,3))
        electric,magnetic=gradients
        for v,j,m in zip(closure.volumes,result['electric_current'].split(closure.counts),result['magnetic_current'].split(closure.counts)):
            electric=electric-cell_magnetostatic_current_fields(v,m,points)[0]
            magnetic=magnetic+cell_magnetostatic_current_fields(v,j,points)[0]
        sigma=closure.current.magnetic.reshape(-1,3)[sum(closure.counts[:index]),0]
        fields={'electric_V_per_m':electric,'magnetic_A_per_m':magnetic,
                'electric_scalar_potential_V':potentials[0],'magnetic_scalar_potential_A':potentials[1],
                'magnetic_flux_density_T':None,'magnetic_flux_pole_residue_T_per_s':1j*sigma*magnetic}
    fields['physical_qualification']=False
    return fields,decoded
