"""Actual public mixed tensor/scalar nonlinear solve and exact Result restore."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(chi2=.002,*,current=False,magnetic=False):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    first=mw.TetrahedralVolumeMesh(vertices,cells,region_id=1)
    second=mw.TetrahedralVolumeMesh(vertices+vertices.new_tensor([.2,0.,0.]),cells,region_id=2)
    materials=(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.,2.),mu_r=1.3),
        mw.Material(eps_r=2.,sigma_e=.001 if current else 0.,nonlinearity=
            (mw.NonlinearSusceptibility(chi2=chi2,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)) if current else
            mw.NonlinearSusceptibility(chi2=chi2,chi3=.001)))
    scene=mw.Scene(structures=[mw.Structure(name=name,geometry=mw.Mesh(v.boundary.vertices,v.boundary.faces,
        recenter=False,fill_mode='solid'),material=m) for name,v,m in zip(('linear','nonlinear'),(first,second),materials)],
        sources=[mw.PlaneWave()])
    hb=mw.HarmonicBalanceConfig(harmonics=2,volumes=(second,),branch_id='mixed_linear_start',pump_scales=(.5,1.))
    if magnetic:hb=replace(hb,dc_magnetic_volumes=(first,second))
    config=mw.MoMConfig(harmonic_balance=hb,dyadic_integration=DyadicIntegrationConfig(),
        integration=IntegrationConfig(order=3,singular_order=3),riesz_storage='compact_rwg_v1',
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    return scene,config,second.vertices.mean(0,keepdim=True)


@pytest.mark.parametrize('current,inner_rule,magnetic',[(False,False,False),(True,False,False),(True,True,False),(True,False,True)],ids=['spd','current','current_krylov','magnetic'])
@pytest.mark.parametrize('backend',['dense','hmatrix','h2'])
def test_public_mixed_harmonics_result_save_load_and_scene_binding(tmp_path,monkeypatch,current,inner_rule,magnetic,backend):
    scene,config,point=setup(current=current,magnetic=magnetic)
    if backend!='dense':
        from witwin.maxwell.mom.hmatrix import HMatrixConfig
        from witwin.maxwell.mom.linalg import StoppingRule
        config=replace(config,operator_backend=backend,riesz_storage='local_rwg_v1',
            hierarchical=HMatrixConfig(leaf_size=64),stopping=StoppingRule(method='block_gmres',
                preconditioner='row_column_norm',rtol=1e-11,atol=1e-20,stagnation_window=48))
    if inner_rule:
        from witwin.maxwell.mom.linalg import StoppingRule
        config=replace(config,harmonic_balance=replace(config.harmonic_balance,
            dc_tangent_rule=StoppingRule(preconditioner='none',rtol=1e-12,atol=1e-22,maxiter=30,restart=8),
            dc_tangent_column_batch=2))
    scene.add_monitor(mw.PointMonitor('nonlinear_cell',tuple(point[0].tolist()),fields=('Ex','Ey','Ez','Hx','Hy','Hz')))
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    assert result._harmonic_data.decoder.prepared['response_backend']==('dense' if backend=='dense' else 'compressed')
    assert result.frequencies==(0.,3e8,6e8)
    state=result.harmonic_state
    assert state.dc_tensor_response is not None and state.dc_surface_charge.numel()==0
    assert bool(state.cell_electric[0].norm()>0) and bool(state.cell_electric[2].norm()>0)
    dc=result.query_dc(point,region_id=2)
    ac=result.query_fields(point,region_id=2,field_kind='total',harmonic=1)
    assert state.receipt['physical_qualification'] is False
    if current:
        assert state.dc_current_regions==(2,)
        assert result.monitor('nonlinear_cell')['dc']['magnetic_available'] is magnetic
        if magnetic:
            magnetic_fields=result.query_dc_magnetic(point,region_id=2)
            assert state.dc_magnetic_volume is not None
        else:
            with pytest.raises(NotImplementedError,match='magnetic response'):
                result.query_dc_magnetic(point,region_id=2)
    path=tmp_path/'mixed.pt';result.save(path)
    native=torch.load(path,weights_only=False)['harmonic_data']
    assert native['schema_version']==(23 if magnetic else 22 if current else 21)
    def forbidden(*args,**kwargs):raise AssertionError('Public Result restore must not assemble or solve')
    from witwin.maxwell.mom.runtime import SystemOperator
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        from witwin.maxwell.mom.static_tensor_floating import StaticTensorFloatingOperator
        from witwin.maxwell.mom.harmonic_mixed import MixedHarmonicClosure
        patch.setattr(StaticTensorFloatingOperator,'__init__',forbidden)
        patch.setattr(MixedHarmonicClosure,'solve',forbidden)
        patch.setattr(DenseHarmonicBalance,'solve',forbidden)
        if backend!='dense':
            from witwin.maxwell.mom.implicit_surface_solve import FixedSurfaceInverse
            from witwin.maxwell.mom.tensor_nearfield import GeneralizedSystemBlocks
            patch.setattr(FixedSurfaceInverse,'_solve',forbidden)
            patch.setattr(GeneralizedSystemBlocks,'__call__',forbidden)
        from witwin.maxwell.mom.static_magnetic_volume import MagneticVolumeContrastResponse
        from witwin.maxwell.mom.static_tensor_green import StaticTensorGreen
        with monkeypatch.context() as inner:
            inner.setattr(MagneticVolumeContrastResponse,'__init__',forbidden)
            inner.setattr(MagneticVolumeContrastResponse,'_source',forbidden)
            inner.setattr(StaticTensorGreen,'panel_integrals',forbidden)
            restored=mw.Result.load(path,scene=scene)
        if magnetic:
            for a,b in zip(restored.query_dc_magnetic(point,region_id=2),magnetic_fields):
                torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-28)
            assert restored.monitor('nonlinear_cell')['dc']['magnetic_available'] is True
            for name in ('Hx','Hy','Hz'):
                torch.testing.assert_close(restored.monitor('nonlinear_cell')['dc']['components'][name],
                    result.monitor('nonlinear_cell')['dc']['components'][name].cpu(),rtol=0,atol=0)
        for name in ('cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(restored.harmonic_state,name),getattr(state,name),rtol=0,atol=0)
        for a,b in zip(restored.query_dc(point,region_id=2),dc):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for a,b in zip(restored.query_fields(point,region_id=2,field_kind='total',harmonic=1),ac):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for name in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            torch.testing.assert_close(restored.monitor('nonlinear_cell')['components'][name],
                result.monitor('nonlinear_cell')['components'][name].cpu(),rtol=0,atol=0)
        altered_scene,_,_=setup(chi2=.004,current=current)
        with pytest.raises(ValueError,match='Scene identity'):
            mw.Result.load(path,scene=altered_scene)
        if current:
            import witwin.maxwell.mom.tensor_harmonic_runtime as runtime
            original_materials=runtime._materials
            def changed_dc(*args,**kwargs):
                samples=list(original_materials(*args,**kwargs))
                samples[1]=replace(samples[1],electric_dc=replace(samples[1].electric_dc,conductivity=.002))
                return tuple(samples)
            with monkeypatch.context() as inner:
                inner.setattr(runtime,'_materials',changed_dc)
                with pytest.raises(ValueError,match='DC conductivity differs'):
                    mw.Result.load(path,scene=scene)
        if magnetic:
            def changed_magnetic_dc(*args,**kwargs):
                samples=list(original_materials(*args,**kwargs))
                samples[1]=replace(samples[1],magnetic_dc=replace(samples[1].magnetic_dc,finite_relative_part=1.2+0j))
                return tuple(samples)
            with monkeypatch.context() as inner:
                inner.setattr(runtime,'_materials',changed_magnetic_dc)
                with pytest.raises(ValueError,match='magnetic DC coefficient differs'):
                    mw.Result.load(path,scene=scene)
        saved=torch.load(path,weights_only=False)
        saved['harmonic_data']['harmonic_config']['source_max_order']*=2
        changed=tmp_path/'changed-policy.pt';torch.save(saved,changed)
        with pytest.raises(ValueError,match='policy differs'):
            mw.Result.load(changed,scene=scene)
        import copy
        original=torch.load(path,weights_only=False)
        for key,value in (('physical_qualification',True),('fundamental_frequency_hz',4e8),
                          ('native_data','surface'),('solver_converged',False),
                          ('harmonic_truncation_converged',True)):
            damaged=copy.deepcopy(original);damaged['metadata'][key]=value
            torch.save(damaged,changed)
            with pytest.raises(ValueError,match='metadata differs'):
                mw.Result.load(changed,scene=scene)
        damaged=copy.deepcopy(original);damaged['frequencies']=(0.,4e8,8e8)
        torch.save(damaged,changed)
        with pytest.raises(ValueError,match='frequencies must match'):
            mw.Result.load(changed,scene=scene)


def test_public_magnetic_volume_admission_before_assembly(monkeypatch):
    from witwin.maxwell.mom.runtime import SystemOperator
    def forbidden(*args,**kwargs):raise AssertionError('Invalid magnetic volumes reached AC assembly')
    monkeypatch.setattr(SystemOperator,'__init__',forbidden)
    scene,config,_=setup(current=True,magnetic=True)
    first,second=config.harmonic_balance.dc_magnetic_volumes
    missing=replace(config,harmonic_balance=replace(config.harmonic_balance,dc_magnetic_volumes=(first,)))
    with pytest.raises(ValueError,match='every body'):
        mw.Simulation.mom(scene,frequency=3e8,config=missing).prepare()
    duplicate=mw.TetrahedralVolumeMesh(second.vertices.clone(),second.cells.clone(),region_id=2)
    wrong=replace(config,harmonic_balance=replace(config.harmonic_balance,dc_magnetic_volumes=(first,duplicate)))
    with pytest.raises(ValueError,match='reuse selected'):
        mw.Simulation.mom(scene,frequency=3e8,config=wrong).prepare()
    with pytest.raises(ValueError,match='increasing region order'):
        replace(config.harmonic_balance,dc_magnetic_volumes=(second,first))
    with pytest.raises(TypeError,match='nonempty tuple'):
        replace(config.harmonic_balance,dc_magnetic_volumes=())
    scene=mw.Scene(structures=[replace(scene.structures[0],material=mw.Material(eps_r=2.,mu_r=1.3)),
        *scene.structures[1:]],sources=list(scene.sources))
    config=replace(config,dyadic_integration=None)
    with pytest.raises(NotImplementedError,match='mixed tensor current runtime'):
        mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
