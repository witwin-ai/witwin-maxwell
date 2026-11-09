"""Original full-equation checking detects corruptions beyond Krylov residuals."""
import runpy
import pytest
from witwin.maxwell.mom.coupled_original_check import check_coupled_original_columns

pytestmark=pytest.mark.skipif(not __import__('torch').cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('field',['E','H'])
def test_original_equations_fields_and_corrupted_composition(field,monkeypatch):
    action,_=runpy.run_path('tests/mom/test_coupled_field_refinement.py')['pair'](field)
    # Unequal regional cell counts [1,8], partial source/query/edge batches.
    receipt=check_coupled_original_columns(action,column_batch=4,edge_batch=16,query_batch=3)
    assert receipt['passed'] and receipt['columns']==27
    assert len(receipt['checks'])==7 and receipt['checks'][-1]['column_stop']==27
    assert all(c['original_equation_pass'] and c['original_field_pass'] for c in receipt['checks'])
    assert not receipt['physical_qualification']
    source=action.sources[0];old_source=source.apply
    monkeypatch.setattr(source,'apply',lambda value:1.01*old_source(value))
    corrupted=check_coupled_original_columns(action,column_batch=27,edge_batch=32,query_batch=8)
    assert not corrupted['passed'] and not corrupted['checks'][0]['original_equation_pass']
    monkeypatch.setattr(source,'apply',old_source)
    direct=action.direct[1];old_direct=direct.apply
    monkeypatch.setattr(direct,'apply',lambda value:1.01*old_direct(value))
    corrupted=check_coupled_original_columns(action,column_batch=27,edge_batch=32,query_batch=8)
    assert not corrupted['passed'] and not corrupted['checks'][0]['original_field_pass']
