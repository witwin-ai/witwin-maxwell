"""Fresh public D/K FFT solve and constructor-free exact current restoration."""
import copy
import hashlib
import json
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.fmm_operator import MultipoleOperator
from witwin.maxwell.mom.fmm_system import MultipoleSystemActions
from witwin.maxwell.mom.fmm_periodic_operator import PeriodicMultipoleOperator
from witwin.maxwell.mom.fmm_periodic_system import PeriodicMultipoleSystemActions
from witwin.maxwell.mom.runtime import SurfaceData, SystemOperator

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_public_scalar_dk_automatic_original_residual_and_exact_state_restore(tmp_path,monkeypatch):
    scene=mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.01,position=(x,0.,0.)),
        mw.Material(eps_r=2.+i,mu_r=1.2+.1*i)) for i,x in enumerate((-.2,.2))],sources=[mw.PlaneWave()])
    fmm=mw.FMMConfig(order=6,max_order=10,kernel_error_budget=mw.KernelErrorBudget(1e-6,1e-5),
        kernel_bound_rule='direct_first_wave_min_v1',table_storage='fft_v1',shift_storage='separable_v1',near_storage='scalar_dk_v1',
        leaf_size=30,separation_ratio=.25,leaf_factor_kernel='primary',leaf_factor_max_edges=16)
    config=mw.MoMConfig(subdivisions=0,integration=IntegrationConfig(order=4,singular_order=3),
        operator_backend='fmm',fmm=fmm,riesz_storage='local_rwg_v1',
        stopping=StoppingRule(method='gmres',preconditioner='rwg_diagonal',fallback='none',
            rtol=1e-12,atol=0.,maxiter=240,restart=120))
    result=mw.Simulation.mom(scene,frequency=9e7,config=config).run()
    stats=result.stats();assert stats['converged'] and stats['error_budget']['checked']
    actions=result.surface_currents.handle.actions
    assert actions.order_plan['kernel_bound_rule']=='direct_first_wave_min_v1'
    assert actions.order_plan['attempts'][-1]['kernel_pass']
    assert len(actions.exterior.far)==2
    assert all(c.receipt()['table_storage']=='fft_v1' for _,_,c in actions.exterior.far)
    assert actions.capacity_plan['table_storage']=='fft_v1'
    assert actions.capacity_plan['near_storage']=='scalar_dk_v1'
    assert actions.near_storage=='scalar_dk_v1'
    assert all(region.near.receipt()['version']=='grouped_scalar_original_dk_csr_v1' for region in (actions.exterior,*actions.interiors))
    assert fmm.payload()['version']=='cartesian_fmm_scalar_dk_config_v1'
    current=result.surface_currents.scattered_coefficients
    path=tmp_path/'scalar_dk_v1-fmm.pt';result.save(path)
    def forbidden(*a,**k):raise AssertionError('FFT replay rebuilt or solved the operator')
    with monkeypatch.context() as patch:
        for cls in (MultipoleOperator,MultipoleSystemActions,PeriodicMultipoleOperator,PeriodicMultipoleSystemActions,SystemOperator):
            patch.setattr(cls,'__init__',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        assert restored.surface_currents.config.fmm==fmm
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,current,rtol=0,atol=0)
        assert restored.surface_currents.snapshot()['fmm_config']==fmm.payload()
        for field,new in (('near_storage','full_matrix'),('table_storage','compact'),
                          ('kernel_bound_rule','bivariate_cauchy_v1'),('shift_storage','dense')):
            bad=copy.deepcopy(result.surface_currents.snapshot())
            bad['fmm_config']['parameters'][field]=new
            with pytest.raises(ValueError):SurfaceData.restore(bad,scene,9e7)
    record=dict(artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stats=stats,
        selected_order=actions.order_plan['selected_order'],far_tables=len(actions.exterior.far),
        capacity_plan=actions.capacity_plan,exact_current_restore=True,fmm_config=fmm.payload(),
        physical_qualification=False,complexity_qualification=False)
    (tmp_path/'public.json').write_text(json.dumps(record,indent=2,default=str)+'\n')
    print(record,flush=True)
