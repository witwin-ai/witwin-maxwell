"""Public periodic tensor scalar limits, Bloch recovery and exact ownership."""
import copy
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.runtime import SurfaceData
from witwin.maxwell.mom.operator import IntegrationConfig
from test_generalized_system_blocks import basis_at

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def setup(dimension,tensor):
    bodies=[]
    for i,x in enumerate((-.2,.2)):
        basis=basis_at(x)
        material=(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.+i,2.+i,2.+i),
                              mu_tensor=mw.DiagonalTensor3(1.3,1.3,1.3)) if tensor else
                  mw.Material(eps_r=2.+i,mu_r=1.3))
        bodies.append(mw.Structure(mw.Mesh(basis.mesh.vertices,basis.mesh.faces,
            recenter=False,fill_mode='solid'),material,name=f'body{i}'))
    # Oblique illumination gives a nontrivial Bloch phase along the first axis.
    source=mw.PlaneWave(direction=(.3,0.,1.),polarization=(0.,1.,0.))
    from witwin.maxwell.mom.physics import compile_medium
    exterior=compile_medium(mw.Material(),3e8)
    beta=tuple(exterior.k.real*v for v in source.direction)
    periodic=mw.MoMPeriodicConfig(vectors_m=((1.,0.,0.),(0.,1.,0.))[:dimension],
        bloch_rad_per_m=(beta[0],0.,0.),reciprocal_shells=4,quadrature_order=128)
    config=mw.MoMConfig(integration=IntegrationConfig(order=2,singular_order=3),
        query_config=mw.FieldQueryConfig(adaptive=False),riesz_storage='compact_rwg_v1',periodic=periodic,
        dyadic_integration=mw.DyadicIntegrationConfig() if tensor else None)
    return mw.Scene(structures=bodies,sources=[source]),config


def test_public_three_dimensional_tensor_plane_wave_retains_spectral_pole_guard(monkeypatch):
    scene,config=setup(2,True)
    from witwin.maxwell.mom.physics import compile_medium
    exterior=compile_medium(mw.Material(),3e8)
    beta=tuple(exterior.k.real*v for v in scene.sources[0].direction)
    config=replace(config,periodic=replace(config.periodic,
        vectors_m=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),bloch_rad_per_m=beta))
    import witwin.maxwell.mom.anisotropic_operator as assembly
    def forbidden(*args,**kwargs):
        raise AssertionError('A spectral pole must reject before primary tensor assembly')
    monkeypatch.setattr(assembly,'assemble_generalized_system',forbidden)
    with pytest.raises(ValueError,match='spectral pole'):
        mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()


@pytest.mark.parametrize('dimension',(1,2))
def test_public_periodic_tensor_scalar_limit_and_archive(dimension,tmp_path,monkeypatch):
    scene,config=setup(dimension,True)
    scalar,scalar_config=setup(dimension,False)
    reference=mw.Simulation.mom(scalar,frequency=3e8,config=scalar_config).run()
    result=mw.Simulation.mom(scene,frequency=3e8,config=config).run()
    assert result.stats()['converged']
    for actual,expected in zip(result.surface_currents.scattered_coefficients.chunk(2),
                               reference.surface_currents.scattered_coefficients.chunk(2)):
        torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-13)
    data=result.surface_currents
    points={1:torch.tensor([[-.19,.01,.01],[.81,.01,.01]],device='cuda',dtype=torch.float64),
            0:torch.tensor([[.1,.2,.3],[1.1,.2,.3]],device='cuda',dtype=torch.float64)}
    expected_fields={}
    phase=torch.exp(points[0].new_tensor(config.periodic.bloch_rad_per_m[0])*1j)
    for region,point in points.items():
        actual=result.query_fields(point,region_id=region,field_kind='total')
        expected_fields[region]=actual
        expected=reference.query_fields(point,region_id=region,field_kind='total')
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-12)
            torch.testing.assert_close(a[1],phase*a[0],rtol=1e-10,atol=1e-12)
    path=tmp_path/'periodic-tensor.pt'
    result.save(path)
    payload=data.snapshot()
    assert payload['schema_version']==16
    assert payload['kernel_id']==f'quasiperiodic_ewald_{dimension}d_tensor_interiors_v1'
    assert payload['operator_metadata']['physical_qualification'] is False
    def forbidden(*args,**kwargs):
        raise AssertionError('Periodic tensor restore must not assemble or solve')
    import witwin.maxwell.mom.periodic_runtime as runtime
    import witwin.maxwell.mom.anisotropic_operator as assembly
    with monkeypatch.context() as patch:
        patch.setattr(runtime,'assemble_periodic_system',forbidden)
        patch.setattr(assembly,'assemble_generalized_system',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        restored=mw.Result.load(path,scene=scene)
        assert restored.surface_currents.config==config
        torch.testing.assert_close(restored.surface_currents.scattered_coefficients,
                                   data.scattered_coefficients,rtol=0,atol=0)
        for region,point in points.items():
            for a,b in zip(restored.query_fields(point,region_id=region,field_kind='total'),expected_fields[region]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        for mutation in ('schema','lattice','dyadic','material','operator','decoder'):
            damaged=copy.deepcopy(payload)
            if mutation=='schema':damaged['schema_version']=13
            elif mutation=='lattice':damaged['periodic_config']['real_shells']+=1
            elif mutation=='dyadic':damaged['dyadic_cache_max_bytes']+=2
            elif mutation=='material':damaged['body_table'][0]['interior']['mu_r_real_imag'][0][0][0]+=.1
            elif mutation=='operator':damaged['operator_metadata']['operator_identity']='changed'
            else:damaged['decoder_version']='periodic_analytic_lifted_fields_v1'
            with pytest.raises(ValueError):SurfaceData.restore(damaged,scene,3e8)
    kernel=data.handle.body_handles[0].tensor_kernel
    data.handle.body_handles[0].tensor_kernel=replace(kernel)
    with pytest.raises(ValueError,match='binding changed'):
        result.query_fields(points[1],region_id=1,field_kind='total')
