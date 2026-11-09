"""Exact dual DC state records bound to already-restored numerical preparation.

No assembly or solve occurs on restore. This state component does not yet replace
whole-Result harmonic snapshots: the outer snapshot must also restore preparation
and verify the nonlinear AC/DC constitutive equation.
"""
import copy
import hashlib
import torch
from .dual_static_current import DualStaticExterior
from .harmonic_current import surface_face_potentials
from .static_floating import FloatingPotentialFields
from ..constants import EPSILON_0,MU_0

ARRAYS=('electric','magnetic','electric_current','magnetic_current')
POTENTIALS=('electric_potentials','magnetic_potentials')
LAYERS=('electric_density','electric_offsets','magnetic_density','magnetic_offsets')


def _tensor_identity(value):
    array=value.detach().cpu().contiguous().numpy()
    return {'dtype':str(array.dtype),'shape':list(array.shape),'sha256':hashlib.sha256(array.tobytes()).hexdigest()}


def preparation_record(closure,bindings,laplace):
    closure.assert_current(); laplace._assert_current()
    bindings=tuple(bindings)
    if (len(bindings)!=len(closure.volumes) or len(laplace.system.bodies)!=len(bindings)
            or sorted(v.region_id for v in closure.volumes)!=list(range(1,len(bindings)+1))):
        raise ValueError('Dual state requires complete native volume/body order')
    volumes=[]
    for binding,volume in zip(bindings,closure.volumes):
        binding.assert_current()
        if binding.volume is not volume or binding.surface_system is not laplace.system:
            raise ValueError('Dual state surface and volume preparations do not match')
        volumes.append({'region_id':volume.region_id,'vertices':_tensor_identity(volume.vertices),
                        'cells':_tensor_identity(volume.cells),'binding':_tensor_identity(binding.boundary_to_surface_vertices)})
    from .dual_current_action import StreamedDualCurrentBalance
    streamed=type(closure) is StreamedDualCurrentBalance
    if streamed:
        from .dual_action_snapshot import snapshot_dual_action
        current={'streamed':snapshot_dual_action(closure)[1]}
        recovery=()
    else:
        current={name:_tensor_identity(getattr(closure,name)) for name in
                 ('electric','magnetic','projector','points','curl','matrix')}
        recovery=tuple(_tensor_identity(value) for value in closure.recovery)
    return {'version':'streamed-dual-dc-preparation-binding-v1' if streamed else 'dual-dc-preparation-binding-v1',
            'volumes':volumes,'counts':closure.counts,'current':current,'recovery':recovery,
            'laplace_testing_order':laplace.testing_order,
            'laplace':{name:_tensor_identity(getattr(laplace,name)) for name in
                       ('triangles','normals','areas','indicator','charge_test','single_layer','mean_gradient','matrix','lu','pivots')}}


def _state_payload(closure,result,exterior):
    if exterior.closure is not closure or exterior._versions!=exterior._state():
        raise ValueError('Dual state belongs to a different or mutated exterior')
    exterior.electric_layer._assert_current(); exterior.magnetic_layer._assert_current()
    laplace=exterior.electric_layer.operator
    if exterior.magnetic_layer.operator is not laplace:
        raise ValueError('Dual state layers must share their exact surface preparation')
    payload={'schema_version':'dual-dc-state-v1','physical_qualification':False,
             'preparation':preparation_record(closure,exterior.bindings,laplace),
             'state':{name:result[name].detach().cpu().clone() for name in ARRAYS},
             'receipt':copy.deepcopy(result['receipt']),
             'exterior_receipt':copy.deepcopy(exterior.receipt)}
    payload['state'].update({name:tuple(value.detach().cpu().clone() for value in result[name]) for name in POTENTIALS})
    for prefix,layer in (('electric',exterior.electric_layer),('magnetic',exterior.magnetic_layer)):
        payload['state'][prefix+'_density']=layer.density.detach().cpu().clone()
        payload['state'][prefix+'_offsets']=layer.offsets.detach().cpu().clone()
    if not torch.equal(result['electric_current'],exterior.j) or not torch.equal(result['magnetic_current'],exterior.m):
        raise ValueError('Dual exterior and state currents differ')
    return payload


def snapshot_dual_state(closure,result,exterior):
    payload=_state_payload(closure,result,exterior)
    restore_dual_state(payload,closure,exterior.bindings,exterior.electric_layer.operator)
    return payload


def restore_dual_state(payload,closure,bindings,laplace):
    return _restore_dual_state(payload,closure,bindings,laplace)


def _restore_dual_state(payload,closure,bindings,laplace,*,curl_products=None):
    bindings=tuple(bindings)
    expected={'schema_version','physical_qualification','preparation','state','receipt','exterior_receipt'}
    if (not isinstance(payload,dict) or set(payload)!=expected or payload['schema_version']!='dual-dc-state-v1'
            or payload['physical_qualification'] is not False):
        raise ValueError('Invalid dual DC state schema or qualification claim')
    if payload['preparation']!=preparation_record(closure,bindings,laplace):
        raise ValueError('Dual DC state differs from its exact numerical preparation')
    raw=payload['state']; sample=closure.points; count=sum(closure.counts)
    if not isinstance(raw,dict) or set(raw)!=set(ARRAYS+POTENTIALS+LAYERS):
        raise ValueError('Dual DC state requires every current, potential and layer')
    def array(value,shape):
        if (not isinstance(value,torch.Tensor) or value.dtype!=sample.dtype or value.shape!=shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Dual DC state array shape, precision or finiteness mismatch')
        return value.detach().to(sample.device).clone()
    state={name:array(raw[name],(count,3)) for name in ARRAYS}
    for name in POTENTIALS:
        if not isinstance(raw[name],tuple) or len(raw[name])!=len(closure.blocks):
            raise ValueError('Dual DC state potential body count mismatch')
        state[name]=tuple(array(value,(block.n_dofs,)) for block,value in zip(closure.blocks,raw[name]))
    for name in LAYERS:
        state[name]=array(raw[name],(len(laplace.triangles) if name.endswith('density') else len(closure.blocks),))
    for name in ('receipt','exterior_receipt'):
        if not isinstance(payload[name],dict) or payload[name].get('physical_qualification') is not False:
            raise ValueError('Dual DC receipt cannot claim physical qualification')
    def close(actual,expected,scale):
        threshold=2048*torch.finfo(sample.dtype).eps*max(count,1)*scale
        if not bool(torch.isfinite(threshold).all()) or not bool((actual-expected).abs().max()<=threshold):
            raise ValueError('Dual DC state violates its prepared field or flux equations')
    j,m=state['electric_current'].flatten(),state['magnetic_current'].flatten()
    curls=((closure.curl@m,closure.curl@j) if curl_products is None else
           tuple(array(v,(count,3)).flatten() for v in curl_products))
    for name,potentials,curl,sign in (('electric','electric_potentials',curls[0],-1),
                                     ('magnetic','magnetic_potentials',curls[1],1)):
        gradient=torch.cat([block.electric(phi).flatten() for block,phi in zip(closure.blocks,state[potentials])])
        actual=state[name].flatten()
        close(actual,gradient+sign*curl,actual.abs().max()+gradient.abs().max()+curl.abs().max())
    for block,jbody,mbody in zip(closure.blocks,state['electric_current'].split(closure.counts),state['magnetic_current'].split(closure.counts)):
        for current in (jbody,mbody):
            defect=block.weak_current(current)
            scale=(block.gradients.abs()*current.abs()[:,None,:]*block.volume.volumes[:,None,None]).sum()
            close(defect,torch.zeros_like(defect),scale)
    layers=[]
    for prefix,potentials,constant in (('electric','electric_potentials',EPSILON_0),('magnetic','magnetic_potentials',MU_0)):
        boundary=torch.cat([surface_face_potentials(block,binding,phi) for block,binding,phi in sorted(zip(closure.blocks,bindings,state[potentials]),key=lambda item:item[1].volume.region_id)])
        density,offsets=state[prefix+'_density'],state[prefix+'_offsets']
        computed=laplace.single_layer@density-laplace.indicator@offsets
        scale=(laplace.single_layer.abs()@density.abs()+laplace.indicator@offsets.abs()+boundary.abs()).max()
        close(computed,boundary,scale)
        flux=laplace.charge_test@density
        close(flux,torch.zeros_like(flux),(laplace.charge_test@density.abs()).max())
        layers.append(FloatingPotentialFields(laplace,density,offsets,sample.new_tensor(constant),
                                             sample.new_zeros(len(closure.blocks)),{'physical_qualification':False}))
    exterior=object.__new__(DualStaticExterior)
    exterior.closure,exterior.bindings=closure,tuple(bindings)
    exterior.electric_layer,exterior.magnetic_layer=layers
    exterior.j,exterior.m=state['electric_current'].clone(),state['magnetic_current'].clone()
    exterior.receipt=copy.deepcopy(payload['exterior_receipt'])
    exterior._versions=exterior._state()
    result={name:state[name] for name in ARRAYS+POTENTIALS}
    result['receipt']=copy.deepcopy(payload['receipt'])
    return result,exterior
