"""Public policy binding on actual fixed-order tensor components, not admission."""
import copy
from dataclasses import replace
import pytest
import witwin.maxwell as mw
import torch
from test_tensor_compressed_background import inverse_for
from witwin.maxwell.mom.harmonic_runtime import (_response_settings,_validate_tensor_response_policy,
    _validate_tensor_background_policy)
from witwin.maxwell.mom.harmonic_config import HarmonicBalanceConfig
from witwin.maxwell.mom.compressed_background import prepare_compressed_background
from witwin.maxwell.mom.surface_volume_hmatrix import SurfaceVolumeBlocks,SurfaceVolumeHierarchicalAction
from witwin.maxwell.mom.interior_volume_hmatrix import InteriorVolumeBlocks,InteriorVolumeHierarchicalAction
from witwin.maxwell.mom.volume_hmatrix import VolumeCellBlocks,VolumeHierarchicalAction
from witwin.maxwell.mom.coupled_volume_action import CoupledVolumeAction
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_explicit_tensor_policy_forwarding_and_restored_component_binding():
    inverse,binding=inverse_for(True);kernel=inverse.operator.provider.interior_kernels[0]
    hb=HarmonicBalanceConfig(volumes=(binding.volume,),harmonics=1,
        tensor_face_rule='projected_duffy',tensor_kernel_batch=8192,branch_id='tensor-policy')
    config=mw.MoMConfig(operator_backend='h2',riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='block_gmres',preconditioner='row_column_norm'),harmonic_balance=hb)
    options=_response_settings(config)['compressed_settings']
    assert options['tensor_face_rule']==hb.tensor_face_rule and options['tensor_kernel_batch']==8192
    assert options['source_settings']==dict(max_order=hb.source_max_order,max_inner_order=hb.source_max_inner_order)
    source=SurfaceVolumeHierarchicalAction(SurfaceVolumeBlocks(binding,order=2,inner_order=2,
        tensor_kernel=kernel,tensor_face_rule=hb.tensor_face_rule,tensor_kernel_batch=8192))
    actions=[]
    for field in ('E','H'):
        f=InteriorVolumeHierarchicalAction(InteriorVolumeBlocks(binding,field=field,order=2,tensor_kernel=kernel))
        d=VolumeHierarchicalAction(VolumeCellBlocks(binding.volume,kernel.medium,field=field,order=2,
            radial_order=2,tensor_kernel=kernel,face_rule=hb.tensor_face_rule,kernel_batch=8192))
        actions.append(CoupledVolumeAction(inverse,(source,),(f,),(d,)))
    bank=prepare_compressed_background(inverse,(binding,),(mw.PlaneWave(),),query_config=config.query_config)
    receipt=dict(tensor_volume_policy=dict(face_rule=hb.tensor_face_rule,kernel_batch=8192,
        radial_order_policy='follow_complete_field_order',regions=[1],rhs_cache_max_bytes_per_handle=0),
        sources=[dict(tensor_source_policy=source.provider.tensor_policy())])
    part=dict(electric_action=actions[0],magnetic_action=actions[1],bindings=(binding,),background_bank=bank,receipt=receipt)
    _validate_tensor_response_policy(part,config)
    with pytest.raises(ValueError,match='tensor volume policy'):
        _validate_tensor_response_policy(part,replace(config,harmonic_balance=replace(hb,tensor_kernel_batch=128)))
    bad=copy.deepcopy(receipt);bad['sources'][0]['tensor_source_policy']['kernel_batch']=128
    with pytest.raises(ValueError,match='tensor source policy'):
        _validate_tensor_response_policy(dict(part,receipt=bad),config)
    original=copy.deepcopy(bank.receipt)
    bank.receipt['field_queries'][0]['absolute_tolerance_V_per_m']*=2
    with pytest.raises(ValueError,match='tensor background field policy'):
        _validate_tensor_response_policy(part,config)
    bank.receipt.clear();bank.receipt.update(original)
    bank.receipt['tensor_rhs_cache_max_bytes_per_handle']=1
    with pytest.raises(ValueError,match='tensor background policy'):
        _validate_tensor_response_policy(part,config)
    # Schema checks below exercise general-tensor metadata contracts only;
    # these constructed records are not field-integration evidence.
    from witwin.maxwell.mom.tensor_medium import TensorMedium
    from witwin.maxwell.constants import EPSILON_0,MU_0
    import math
    medium=TensorMedium(kernel.medium.frequency,
        torch.diag(kernel.medium.epsilon_r.new_tensor([2.,2.1,2.2])),kernel.medium.mu_r)
    log=copy.deepcopy(original['field_queries'][0])
    with pytest.raises(ValueError,match='tensor background field policy'):
        _validate_tensor_background_policy(log,medium,inverse.operator.provider.exterior,config.query_config)
    log.pop('absolute_tolerance_V_per_m');log.pop('estimator')
    log.update(representation='tensor_complete_lifted_interior_field',
        absolute_tolerance_E_V_per_m=config.query_config.absolute_tolerance,
        absolute_tolerance_H_A_per_m=config.query_config.absolute_tolerance/math.sqrt(MU_0/EPSILON_0))
    _validate_tensor_background_policy(log,medium,inverse.operator.provider.exterior,config.query_config)
    log['absolute_tolerance_H_A_per_m']*=2
    with pytest.raises(ValueError,match='tensor magnetic field policy'):
        _validate_tensor_background_policy(log,medium,inverse.operator.provider.exterior,config.query_config)
