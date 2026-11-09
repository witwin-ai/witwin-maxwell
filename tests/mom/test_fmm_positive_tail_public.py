"""Public result binds an actual opt-in free-space kernel admission policy."""
import copy
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_operator import MultipoleOperator
from witwin.maxwell.mom.runtime import SurfaceData

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('rule', ('positive_tail_min_v1','first_wave_min_v1','direct_positive_tail_min_v1','direct_first_wave_min_v1'))
def test_public_automatic_positive_tail_policy_exact_state_and_identity(tmp_path,monkeypatch,rule):
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01,position=(x,0,0)),
        mw.Material(eps_r=2.+i,mu_r=1.2+.1*i)) for i,x in enumerate((-.2,.2))],sources=[mw.PlaneWave()])
    fmm=mw.FMMConfig(order=6,max_order=10,kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        kernel_bound_rule=rule,table_storage='compact',shift_storage='separable_v1',
        leaf_size=30,separation_ratio=.25,leaf_factor_kernel='primary',leaf_factor_max_edges=16)
    config=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    result=mw.Simulation.mom(scene,frequency=9e7,config=config).run()
    assert result.stats()['converged'] and result.stats()['error_budget']['checked']
    actions=result.surface_currents.handle.actions
    assert actions.order_plan['kernel_bound_rule']==rule
    assert actions.order_plan['attempts'][-1]['kernel_pass']
    assert len(actions.exterior.far)>0
    assert all(c.kernel_bound_rule==rule for _,_,c in actions.exterior.far)
    current=result.surface_currents.scattered_coefficients
    path=tmp_path/(rule+'-fmm.pt');result.save(path)
    def forbidden(*a,**k):raise AssertionError('Policy replay rebuilt or solved FMM')
    with monkeypatch.context() as patch:
        patch.setattr(MultipoleOperator,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        assert restored.surface_currents.config.fmm==fmm
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        assert restored.surface_currents.snapshot()['fmm_config']==fmm.payload()
    bad=copy.deepcopy(result.surface_currents.snapshot())
    bad['fmm_config']['parameters']['kernel_bound_rule']='bivariate_cauchy_v1'
    with pytest.raises(ValueError):SurfaceData.restore(bad,scene,9e7)
    print(dict(actual_far_tables=len(actions.exterior.far),selected_order=actions.order_plan['selected_order'],
        exact_state_policy_replay=True,physical_qualification=False),flush=True)
