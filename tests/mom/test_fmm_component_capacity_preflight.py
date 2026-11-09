"""Reject actual component inventories before any expansion or original quadrature."""
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis, SurfaceSystem, sphere_mesh
from witwin.maxwell.mom.physics import compile_medium
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.nearfield import ScalarSystemBlocks
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_capacity import MultipoleCapacityError

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.fixture(scope='module')
def geometry():
    exterior=compile_medium(mw.Material(),3e8)
    interior=compile_medium(mw.Material(eps_r=2.,mu_r=1.3,sigma_e=.002),3e8)
    system=SurfaceSystem.create([(i,str(i),RWGBasis(sphere_mesh(radius=.01,center=(x,0.,0.),
        subdivisions=0,device='cuda',dtype=torch.float64)),interior) for i,x in enumerate((-.5,.5))])
    return ScalarSystemBlocks(system,exterior,IntegrationConfig(order=2,singular_order=3))


def tripwires(monkeypatch):
    import witwin.maxwell.mom.fmm_system as construction
    def forbidden(*args,**kwargs):
        raise AssertionError('Component preflight reached a numerical expansion or original integration')
    monkeypatch.setattr(construction,'NestedRWGMultipoles',forbidden)
    monkeypatch.setattr(ScalarSystemBlocks,'__call__',forbidden)
    monkeypatch.setattr(ScalarSystemBlocks,'exterior_block',forbidden)
    monkeypatch.setattr(ScalarSystemBlocks,'exterior_channels',forbidden)
    planned=[];original=construction.plan_multipole_storage
    def observed(*args,**kwargs):
        record=original(*args,**kwargs);planned.append(record);return record
    monkeypatch.setattr(construction,'plan_multipole_storage',observed)
    return planned


@pytest.mark.parametrize('storage',['dense','compact','compact_shared_displacement_v1','fft_v1'])
@pytest.mark.parametrize('component',['table','near','projection'])
def test_actual_component_refuses_before_numerics_without_extra_planner(geometry,storage,component,monkeypatch,tmp_path):
    planned=tripwires(monkeypatch)
    options=dict(order=3,leaf_size=30,separation_ratio=.5,table_storage=storage,
        action_retained_budget_bytes=1024**3,table_budget_bytes=1024**3,
        near_budget_bytes=1024**3,projection_budget_bytes=1024**3)
    options[component+'_budget_bytes']=1
    with pytest.raises(MemoryError,match=component+' budget exceeded before moment setup') as error:
        MultipoleSystemActions(geometry,**options)
    assert len(planned)==1 and planned[0]['admitted']
    categories=planned[0]['categories']
    measured=(categories['m2l'] if component=='table' else categories['near'] if component=='near'
        else sum(categories[key] for key in ('centers_radii','leaf_maps','leaf_moments','shifts')))
    assert measured>1 and str(measured) in str(error.value)
    geometry.assert_current()
    (tmp_path/'preflight.json').write_text(json.dumps(dict(storage=storage,component=component,
        actual_planned_component_bytes=measured,budget_bytes=1,action_capacity_passed=True,
        one_existing_capacity_traversal_only=True,numerical_moments_constructed=False,
        original_quadrature_evaluated=False,physical_qualification=False,production_qualification=False),indent=2)+'\n')


def test_aggregate_rejection_keeps_existing_receipt_and_priority(geometry,monkeypatch):
    planned=tripwires(monkeypatch)
    with pytest.raises(MultipoleCapacityError) as error:
        MultipoleSystemActions(geometry,order=3,leaf_size=30,table_storage='compact',
            action_retained_budget_bytes=1,table_budget_bytes=1,near_budget_bytes=1,projection_budget_bytes=1)
    assert len(planned)==1 and error.value.receipt is planned[0]
    assert not error.value.receipt['admitted'] and error.value.receipt['budget_bytes']==1
    assert not error.value.receipt['peak_memory_qualified']
