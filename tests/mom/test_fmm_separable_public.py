"""Actual public FMM solves with opt-in tree shifts and exact decode replay."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('periodic',(False,True),ids=('free','ewald'))
def test_public_separable_fmm_current_queries_and_exact_result_replay(periodic,tmp_path,monkeypatch):
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01,position=(x,0.,0.)),
        mw.Material(eps_r=2.+i,mu_r=1.2+.1*i)) for i,x in enumerate((-.2,.2))],sources=[mw.PlaneWave()])
    lattice=(mw.MoMPeriodicConfig(((1.,0.,0.),),eta_per_m=2.,real_shells=2,
        reciprocal_shells=2,quadrature_order=32) if periodic else None)
    base=mw.MoMConfig(subdivisions=0,periodic=lattice,integration=IntegrationConfig(order=4,singular_order=3),
        query_config=FieldQueryConfig(adaptive=False))
    dense=mw.Simulation.mom(scene,frequency=9e7,config=base).run()
    config=replace(base,operator_backend='fmm',riesz_storage='local_rwg_v1',
        fmm=mw.FMMConfig(order=8 if periodic else 6,table_storage='compact',near_storage='full_matrix',shift_storage='separable_v1',
            leaf_size=30,separation_ratio=.25,leaf_factor_kernel='primary',leaf_factor_max_edges=16),
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    result=mw.Simulation.mom(scene,frequency=9e7,config=config).run()
    assert result.stats()['converged'] and result.stats()['error_budget']['checked']
    actions=result.surface_currents.handle.actions
    assert actions.exterior.test.shift_storage=='separable_v1'
    assert actions.exterior.test.tree.node_count>1
    assert len(actions.exterior.regions[0].far if periodic else actions.exterior.far)>0
    current=result.surface_currents.scattered_coefficients
    reference=dense.surface_currents.scattered_coefficients
    errors=[float((actual-expected).norm()/expected.norm()) for actual,expected in zip(current.chunk(2),reference.chunk(2))]
    queries=[(torch.tensor([[.1,.1,.15]],device='cuda',dtype=torch.float64),'exterior'),
        (torch.tensor([[-.2,0.,0.]],device='cuda',dtype=torch.float64),1)]
    fields=[];targets=[]
    for points,region in queries:
        value=result.query_fields(points,region_id=region,field_kind='total')
        target=dense.query_fields(points,region_id=region,field_kind='total')
        fields.append(value);targets.append(target)
    path=tmp_path/'separable-fmm.pt';result.save(path)
    # Retain producer and independent targets even if a reference gate fails.
    torch.save(dict(actual_currents=current,reference_currents=reference,
        actual_fields=fields,reference_fields=targets,queries=queries,
        current_family_errors=errors,fmm_config=config.fmm.payload()),tmp_path/'public-comparison.pt')
    print(dict(phase='public_comparison_saved',periodic=periodic,order=config.fmm.order,
        current_family_errors=errors,maximum_absolute_field_errors=[
            [float((a-b).abs().amax()) for a,b in zip(value,target)] for value,target in zip(fields,targets)]),flush=True)
    assert max(errors)<1e-8
    for value,target in zip(fields,targets):
        for a,b in zip(value,target):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-12)
    def forbidden(*args,**kwargs):raise AssertionError('Result decode rebuilt or solved a FMM operator')
    from witwin.maxwell.mom.fmm_operator import MultipoleOperator
    from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
    with monkeypatch.context() as patch:
        patch.setattr(MultipoleOperator,'__init__',forbidden)
        patch.setattr(PeriodicMultipoleOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        assert restored.surface_currents.config.fmm==config.fmm
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        for (points,region),expected in zip(queries,fields):
            for a,b in zip(restored.query_fields(points,region_id=region,field_kind='total'),expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
    from witwin.maxwell.mom.runtime import SurfaceData
    bad=copy.deepcopy(result.surface_currents.snapshot());bad['fmm_config']['parameters']['shift_storage']='dense'
    with pytest.raises(ValueError):SurfaceData.restore(bad,scene,9e7)
    print(dict(periodic=periodic,current_family_errors=errors,exact_result_replay=True,
        physical_qualification=False,complexity_qualification=False),flush=True)
