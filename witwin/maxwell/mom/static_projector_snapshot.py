"""Decode-only conservative projector preparation bound to native face geometry."""
import copy
import hashlib
import json
import math
from dataclasses import asdict
import torch
from .dual_static_snapshot import _tensor_identity
from .static_current import FaceCurrentBalance,FixedStaticCurrentInverse
from .static_current_projector import StaticCurrentProjector
from .linalg import StoppingRule


def _block_record(block):
    if type(block) is not FaceCurrentBalance:raise TypeError('Projector snapshot requires a native face block')
    block._assert_current()
    v=block.volume
    return {'region_id':v.region_id,'vertices':_tensor_identity(v.vertices),'cells':_tensor_identity(v.cells),'cell_volumes':_tensor_identity(v.volumes),
        **{k:_tensor_identity(getattr(block,k)) for k in ('gradients','nodal_volume','faces','face_dofs')}}


def _digest(record):
    try:return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    except (TypeError,ValueError) as error:raise ValueError('Invalid projector snapshot metadata') from error


def _validate_bound(bound,rule,size):
    keys={'method','columns_checked','column_batch','complete','tangent_rule','sigma_min_lower','sigma_max_upper',
          'inverse_norm_upper','inverse_residual_norm_upper','numerical_singularity_cutoff','admitted','inconclusive',
          'solve_failure','underlying_action_error_certified','interval_certified','physical_qualification','fast_work_qualified'}
    if (not isinstance(bound,dict) or set(bound)!=keys or bound['method']!='complete_streamed_inverse_residual'
            or type(bound['columns_checked']) is not int or bound['columns_checked']!=size
            or type(bound['column_batch']) is not int or bound['column_batch']<1
            or bound['tangent_rule']!=asdict(rule) or bound['complete'] is not True
            or bound['admitted'] is not True or bound['inconclusive'] is not False or bound['solve_failure'] is not None
            or any(bound[k] is not False for k in ('underlying_action_error_certified','interval_certified',
                                                  'physical_qualification','fast_work_qualified'))):
        raise ValueError('Invalid stored projector inverse admission')
    values=[bound[k] for k in ('sigma_min_lower','sigma_max_upper','inverse_norm_upper',
                               'inverse_residual_norm_upper','numerical_singularity_cutoff')]
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite stored projector inverse admission')
    lower,upper,inverse,residual,cutoff=values
    if not (0<=cutoff<lower<=upper and inverse>0 and 0<=residual<1 and lower==(1-residual)/inverse):
        raise ValueError('Inconsistent stored projector inverse admission')


def snapshot_projector(projector):
    if type(projector) is not StaticCurrentProjector:raise TypeError('Native current projector required')
    projector.assert_current()
    record={'schema_version':'static-current-projector-v1','block':_block_record(projector.block),
        'rule':asdict(projector.inverse.rule),'refinement_steps':projector.inverse.refinement_steps,
        'receipt':copy.deepcopy(projector.receipt)}
    binding={'schema_version':record['schema_version'],'sha256':_digest(record)}
    # Validate the record without repeating its numerical admission.
    restore_projector(record,projector.block,expected_binding=binding)
    return record,binding


def restore_projector(record,block,*,expected_binding):
    """Restore against an independently restored/validated block, without solves.

    Binding must come from the owning preparation's identity. Historical numeric
    bounds are checked for consistency; no new conditioning claim is computed.
    """
    expected={'schema_version','block','rule','refinement_steps','receipt'}
    if (not isinstance(record,dict) or set(record)!=expected or record['schema_version']!='static-current-projector-v1'
            or expected_binding!={'schema_version':record['schema_version'],'sha256':_digest(record)}):
        raise ValueError('Projector snapshot differs from its preparation binding')
    if record['block']!=_block_record(block):raise ValueError('Projector snapshot geometry differs')
    if not isinstance(record['rule'],dict):raise ValueError('Invalid stored projector rule')
    try:rule=StoppingRule(**record['rule'])
    except (TypeError,ValueError) as error:raise ValueError('Invalid stored projector rule') from error
    if (asdict(rule)!=record['rule'] or rule.method!='gmres' or rule.preconditioner!='none'
            or rule.recycle_size or rule.fallback!='none' or type(record['refinement_steps']) is not int
            or record['refinement_steps']<0):raise ValueError('Invalid stored projector inverse policy')
    receipt=record['receipt']
    if (not isinstance(receipt,dict) or set(receipt)!={'method','tangent_bound','global_projector_stored',
            'physical_qualification','fast_work_qualified','underlying_action_error_certified','interval_certified'}
            or receipt['method']!='cellwise_weighted_current_projection'
            or any(receipt[k] is not False for k in ('global_projector_stored','physical_qualification',
                    'fast_work_qualified','underlying_action_error_certified','interval_certified'))):
        raise ValueError('Invalid stored projector receipt')
    _validate_bound(receipt['tangent_bound'],rule,block.n_dofs-1)
    bound=receipt['tangent_bound']
    cutoff=64*torch.finfo(block.volume.vertices.dtype).eps*(block.n_dofs-1)*bound['sigma_max_upper']
    if bound['numerical_singularity_cutoff']!=cutoff:raise ValueError('Invalid stored projector cutoff')
    sample=block.volume.vertices
    tangent=torch.eye(3,device=sample.device,dtype=sample.dtype).expand(len(block.volume.cells),3,3)
    inverse=FixedStaticCurrentInverse(block.tangent_action(tangent),rule,refinement_steps=record['refinement_steps'])
    restored=object.__new__(StaticCurrentProjector)
    restored.block,restored.inverse=block,inverse
    restored.shape=(3*len(block.volume.cells),)*2
    restored.device,restored.dtype=sample.device,sample.dtype
    restored._binding=(id(block),id(inverse),restored.shape,restored.device,restored.dtype)
    restored.receipt=copy.deepcopy(receipt)
    restored.assert_current()
    return restored
