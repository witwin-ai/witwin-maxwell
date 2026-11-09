"""Exact dual nonlinear DC preparation rebinding without numerical assembly."""
import math
import torch
from .harmonic_dual_current import DualHarmonicCurrentClosure
from .dual_static_current import DualStaticCurrentBalance
from .static_current import FaceCurrentBalance
from .harmonic_current_snapshot import FLOATING_ARRAYS,restore_floating_operator
from .dual_static_snapshot import preparation_record,_tensor_identity

PARAMETERS=('relative_tolerance','absolute_tolerance','max_iterations','max_backtracks')
CURRENT_ARRAYS=('electric','magnetic','projector','points','curl','matrix')


def _copy(value,device):
    if isinstance(value,torch.Tensor): return value.detach().to(device).clone()
    if isinstance(value,dict): return {key:_copy(item,device) for key,item in value.items()}
    if isinstance(value,tuple): return tuple(_copy(item,device) for item in value)
    if isinstance(value,list): return [_copy(item,device) for item in value]
    return value


def snapshot_dual_preparation(closure,bindings,laplace):
    closure.assert_current()
    if closure.streamed_rule is not None:
        return _snapshot_streamed_preparation(closure,bindings,laplace)
    binding={'linear':preparation_record(closure.current,bindings,laplace),
             'nonlinear_settings':{name:getattr(closure,name) for name in PARAMETERS},
             'nonlinear_response':_tensor_identity(closure.response)}
    payload={'schema_version':'dual-dc-preparation-v1','physical_qualification':False,
        'current':{name:getattr(closure.current,name) for name in CURRENT_ARRAYS},
        'recovery':closure.current.recovery,
        'blocks':tuple({name:getattr(block,name) for name in ('gradients','nodal_volume','faces','face_dofs')}
                       for block in closure.current.blocks),
        'laplace':{**{name:getattr(laplace,name) for name in FLOATING_ARRAYS},
                   'testing_order':laplace.testing_order,'body_faces':laplace.body_faces},
        'nonlinear':{**{name:getattr(closure,name) for name in PARAMETERS},'response':closure.response}}
    return _copy(payload,'cpu'),binding


def restore_dual_preparation(payload,bindings,*,sigma_e,sigma_m,expected_binding):
    """Rebind pinned arrays to native geometry and caller-sampled conductivities.

    expected_binding is the independently retained state/preparation identity,
    not a hash accepted from inside this payload. Kernel quadrature is not rerun.
    The caller still owns restored surface/volume geometry and the outer HB data.
    """
    bindings=tuple(bindings)
    if not bindings: raise ValueError('Dual preparation requires native bindings')
    for binding in bindings: binding.assert_current()
    system=bindings[0].surface_system
    if (len(bindings)!=len(system.bodies) or any(b.surface_system is not system for b in bindings)
            or sorted(b.volume.region_id for b in bindings)!=list(range(1,len(bindings)+1))
            or len(sigma_e)!=len(bindings) or len(sigma_m)!=len(bindings)):
        raise ValueError('Dual preparation requires complete ordered body bindings and material samples')
    if isinstance(payload,dict) and payload.get('schema_version')=='dual-streamed-preparation-v1':
        return _restore_streamed_preparation(payload,bindings,sigma_e,sigma_m,expected_binding)
    expected={'schema_version','physical_qualification','current','recovery','blocks','laplace','nonlinear'}
    if (not isinstance(payload,dict) or set(payload)!=expected or payload['schema_version']!='dual-dc-preparation-v1'
            or payload['physical_qualification'] is not False):
        raise ValueError('Invalid dual preparation schema or qualification')
    sample=bindings[0].volume.vertices
    raw=_copy(payload,sample.device)
    def array(value,shape,dtype=None):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape
                or value.dtype!=(sample.dtype if dtype is None else dtype) or not bool(torch.isfinite(value).all())):
            raise ValueError('Dual preparation array has invalid shape, precision or values')
    def close(actual,expected):
        scale=actual.abs().max()+expected.abs().max()
        bound=2048*torch.finfo(sample.dtype).eps*max(actual.numel()**.5,1)*scale
        if not bool(torch.isfinite(bound)) or not bool((actual-expected).abs().max()<=bound):
            raise ValueError('Dual preparation violates discrete geometry or operator equations')
    volumes=tuple(b.volume for b in bindings); counts=tuple(len(v.cells) for v in volumes)
    count=sum(counts); size=3*count
    if set(raw['current'])!=set(CURRENT_ARRAYS) or len(raw['blocks'])!=len(bindings) or len(raw['recovery'])!=len(bindings):
        raise ValueError('Incomplete dual current preparation')
    shapes={'electric':(size,),'magnetic':(size,),'projector':(size,size),'points':(count,3),
            'curl':(size,size),'matrix':(2*size,2*size)}
    for name,shape in shapes.items(): array(raw['current'][name],shape)
    for name,coefficients in (('electric',sigma_e),('magnetic',sigma_m)):
        if any(type(value) not in (float,int) or not math.isfinite(value) or value<=0 for value in coefficients):
            raise ValueError('Dual restored conductivity samples must be finite and positive')
        expected=torch.cat([sample.new_full((3*n,),value) for n,value in zip(counts,coefficients)])
        if not torch.equal(raw['current'][name],expected):
            raise ValueError('Dual preparation conductivity differs from sampled material')
    blocks=[]; projectors=[]
    for volume,record,recover in zip(volumes,raw['blocks'],raw['recovery']):
        block=_restore_face_block(volume,record,array,close)
        gradient,faces=block.gradients,block.faces
        g=sample.new_zeros(len(volume.cells),3,len(faces))
        g.scatter_(2,block.cell_dofs[:,None,:].expand(-1,3,-1),gradient.transpose(1,2))
        g=g.reshape(-1,len(faces))[:,1:]
        weighted=g.T*volume.volumes.repeat_interleave(3)[None]
        array(recover,(len(faces)-1,3*len(volume.cells)))
        close((weighted@g)@recover,weighted)
        projectors.append(torch.eye(len(g),device=sample.device,dtype=sample.dtype)-g@recover)
        blocks.append(block)
    current=object.__new__(DualStaticCurrentBalance)
    current.volumes,current.blocks,current.counts=volumes,tuple(blocks),counts
    for name,value in raw['current'].items(): setattr(current,name,value)
    current.recovery=tuple(raw['recovery'])
    close(current.projector,torch.block_diag(*projectors))
    close(current.points,torch.cat([v.vertices[v.cells].mean(1) for v in volumes]))
    eye=torch.eye(size,device=sample.device,dtype=sample.dtype); pc=current.projector@current.curl
    close(current.matrix,torch.cat((torch.cat((eye,current.electric[:,None]*pc),1),torch.cat((-current.magnetic[:,None]*pc,eye),1))))
    current._versions=current._state()
    laplace=restore_floating_operator(raw['laplace'],system,raw['laplace']['testing_order'])
    if not isinstance(expected_binding,dict) or set(expected_binding)!={'linear','nonlinear_settings','nonlinear_response'} or preparation_record(current,bindings,laplace)!=expected_binding['linear']:
        raise ValueError('Dual preparation differs from independently retained binding')
    nonlinear=raw['nonlinear']
    if set(nonlinear)!=set(PARAMETERS)|{'response'}:
        raise ValueError('Invalid dual nonlinear preparation settings')
    for name in PARAMETERS[:2]:
        if type(nonlinear[name]) not in (float,int) or not math.isfinite(nonlinear[name]) or nonlinear[name]<0:
            raise ValueError('Invalid dual nonlinear tolerance')
    if nonlinear['relative_tolerance']+nonlinear['absolute_tolerance']==0:
        raise ValueError('Dual nonlinear tolerances cannot both be zero')
    for name in PARAMETERS[2:]:
        if type(nonlinear[name]) is not int or nonlinear[name]<0: raise ValueError('Invalid dual nonlinear budget')
    response=nonlinear['response']; array(response,(size,size))
    if ({name:nonlinear[name] for name in PARAMETERS}!=expected_binding['nonlinear_settings']
            or _tensor_identity(response)!=expected_binding['nonlinear_response']):
        raise ValueError('Dual nonlinear preparation differs from retained settings or response')
    jmap=current.electric[:,None]*response+eye
    mmap=current.magnetic[:,None]*(current.projector@current.curl@jmap)
    close(current.matrix@torch.cat((jmap,mmap)),torch.cat((current.projector,torch.zeros_like(eye))))
    closure=object.__new__(DualHarmonicCurrentClosure)
    closure.current,closure.volumes,closure.counts=current,volumes,counts
    for name,value in nonlinear.items(): setattr(closure,name,value)
    closure.streamed_rule,closure.streamed_column_batch=None,8
    closure._versions=closure._state()
    return closure,laplace


def _restore_face_block(volume,record,array,close):
    sample=volume.vertices
    if not isinstance(record,dict) or set(record)!={'gradients','nodal_volume','faces','face_dofs'}:
        raise ValueError('Invalid dual face potential record')
    local=volume.cells.new_tensor([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    faces,inverse=torch.unique(volume.cells[:,local].reshape(-1,3).sort(-1).values,dim=0,return_inverse=True)
    for name,shape,dtype in (('faces',faces.shape,torch.int64),('face_dofs',volume.cells.shape,torch.int64),
                            ('gradients',(len(volume.cells),4,3),sample.dtype),('nodal_volume',(len(faces),),sample.dtype)):
        array(record[name],shape,dtype)
    if not torch.equal(record['faces'],faces) or not torch.equal(record['face_dofs'],inverse.reshape(-1,4)):
        raise ValueError('Dual face topology differs from its native volume')
    gradient=record['gradients']; tetrahedra=volume.vertices[volume.cells]
    affine=torch.einsum('tij,taj->tia',tetrahedra[:,1:]-tetrahedra[:,:1],-gradient[:,1:]/3)
    close(affine,torch.eye(3,device=sample.device,dtype=sample.dtype).expand_as(affine))
    close(gradient[:,0],-gradient[:,1:].sum(1))
    weights=sample.new_zeros(len(faces)).index_add(0,inverse,(volume.volumes[:,None].expand(-1,4)/4).flatten())
    close(record['nodal_volume'],weights)
    block=object.__new__(FaceCurrentBalance); block.volume=volume
    for name,value in record.items(): setattr(block,name,value)
    block._face_versions=tuple((id(t),t._version) for t in (block.faces,block.face_dofs))
    block._identity=(id(volume),id(gradient),gradient._version,id(block.nodal_volume),block.nodal_volume._version)
    return block


def _streamed_binding(payload):
    from .dual_action_snapshot import _identity
    from .static_projector_snapshot import _digest
    return {'schema_version':'dual-streamed-preparation-v1','sha256':_digest(_identity(payload))}


def _snapshot_streamed_preparation(closure,bindings,laplace):
    from .dual_action_snapshot import snapshot_dual_closure
    closure.assert_current();laplace._assert_current()
    payload={'schema_version':'dual-streamed-preparation-v1','physical_qualification':False,
        'linear_binding':preparation_record(closure.current,bindings,laplace),
        'blocks':tuple({name:getattr(block,name) for name in ('gradients','nodal_volume','faces','face_dofs')}
                       for block in closure.current.blocks),
        'laplace':{**{name:getattr(laplace,name) for name in FLOATING_ARRAYS},
                   'testing_order':laplace.testing_order,'body_faces':laplace.body_faces},
        'nonlinear':snapshot_dual_closure(closure)[0]}
    payload=_copy(payload,'cpu')
    return payload,_streamed_binding(payload)


def _restore_streamed_preparation(payload,bindings,sigma_e,sigma_m,expected_binding):
    from .dual_action_snapshot import restore_dual_closure,_closure_binding
    keys={'schema_version','physical_qualification','linear_binding','blocks','laplace','nonlinear'}
    if (set(payload)!=keys or payload['physical_qualification'] is not False
            or expected_binding!=_streamed_binding(payload)):
        raise ValueError('Streamed dual preparation differs from independently retained binding')
    if not isinstance(payload['blocks'],tuple) or len(payload['blocks'])!=len(bindings):
        raise ValueError('Incomplete streamed dual face preparation')
    sample=bindings[0].volume.vertices
    raw=_copy(payload,sample.device)
    def array(value,shape,dtype=None):
        if (not isinstance(value,torch.Tensor) or value.shape!=shape or value.device!=sample.device
                or value.dtype!=(sample.dtype if dtype is None else dtype) or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid streamed dual preparation array')
    def close(actual,expected):
        scale=actual.abs().max()+expected.abs().max()
        bound=2048*torch.finfo(sample.dtype).eps*max(actual.numel()**.5,1)*scale
        if not bool(torch.isfinite(bound)) or not bool((actual-expected).abs().max()<=bound):
            raise ValueError('Streamed dual preparation violates discrete geometry')
    blocks=tuple(_restore_face_block(b.volume,r,array,close) for b,r in zip(bindings,raw['blocks']))
    closure=restore_dual_closure(raw['nonlinear'],blocks,sigma_e=sigma_e,sigma_m=sigma_m,
                               expected_binding=_closure_binding(raw['nonlinear']))
    laplace=restore_floating_operator(raw['laplace'],bindings[0].surface_system,raw['laplace']['testing_order'])
    if preparation_record(closure.current,bindings,laplace)!=raw['linear_binding']:
        raise ValueError('Streamed dual surface/volume preparation binding differs')
    return closure,laplace
