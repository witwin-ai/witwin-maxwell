"""Complete tensor cell response through actual assembly and dense source solves."""
import pytest
import torch
import witwin.maxwell as mw
from test_tensor_volume_source import setup
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.volume_response_matrix import prepare_cell_field_response

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_tensor_scalar_limit_complete_cell_response():
    responses=[]
    for tensor in (False,True):
        decode,binding=setup(tensor)
        config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),
            dyadic_integration=decode.body_handles[0].tensor_kernel.integration if tensor else None,
            riesz_storage='compact_rwg_v1')
        operator=SystemOperator(decode.system,decode.exterior,[],config)
        prepared=prepare_cell_field_response(operator,(binding,),
            query_config=FieldQueryConfig(relative_tolerance=1e-4,absolute_tolerance=1e-10,max_order=48),
            max_order=64,tensor_face_rule='projected_duffy')
        assert prepared['receipt']['converged'] and not prepared['receipt']['physical_qualification']
        assert torch.count_nonzero(prepared['background_electric'])==0
        assert torch.count_nonzero(prepared['background_magnetic'])==0
        if tensor:
            records=prepared['receipt']['tensor_field_evaluations']
            assert records and all(x['surface']['shared_column_kernel'] for x in records)
            assert all(x['volume']['face_rule']=='projected_duffy' for x in records)
        responses.append(prepared)
    errors={key:float((responses[1][key]-responses[0][key]).norm()/responses[0][key].norm())
            for key in ('electric_response','magnetic_response','surface_response')}
    print({'tensor_scalar_limit_complete_response_errors':errors,'physical_qualification':False})
    assert max(errors.values())<.005
