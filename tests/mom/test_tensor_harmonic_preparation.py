"""Actual mixed retained HB equations with regional potential/conormal DC maps."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.dyadic_green import DyadicIntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material,compile_tensor_harmonic_material
from witwin.maxwell.mom.tensor_harmonic_preparation import prepare_mixed_tensor_harmonic_balance
from witwin.maxwell.mom.harmonic_state import preparation_signature,recover_harmonic_state

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_mixed_harmonic_model_generates_dc_and_second_harmonic(monkeypatch,tmp_path):
    vertices=torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device='cuda')
    linear=TetrahedralVolumeMesh(vertices,cells,region_id=1)
    nonlinear=TetrahedralVolumeMesh(vertices+vertices.new_tensor([.2,0.,0.]),cells,region_id=2)
    tensor=compile_tensor_harmonic_material(mw.Material(epsilon_tensor=mw.DiagonalTensor3(2.,2.,2.),mu_r=1.3),3e8,2,device='cuda')
    scalar=compile_harmonic_material(mw.Material(eps_r=2.,nonlinearity=mw.NonlinearSusceptibility(chi2=.002,chi3=.001)),3e8,2)
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    config=mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),
        dyadic_integration=DyadicIntegrationConfig(),riesz_storage='compact_rwg_v1')
    operators=tuple(SystemOperator(SurfaceSystem.create([
        (0,'linear',RWGBasis(linear.boundary),tensor.linear_samples[n]),
        (1,'nonlinear',RWGBasis(nonlinear.boundary),scalar.linear_samples[n])]),
        exterior.linear_samples[n],[mw.PlaneWave()] if n==0 else [],config) for n in range(2))
    model,prepared=prepare_mixed_tensor_harmonic_balance(operators,(nonlinear,),(tensor,scalar),exterior,
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    solution=model.solve(relative_tolerance=1e-9,absolute_tolerance=1e-10)
    assert solution.receipt['converged']
    assert bool(solution.electric[0].norm()>0) and bool(solution.electric[2].norm()>0)
    assert torch.count_nonzero(solution.electric[0].imag)==0
    assert torch.count_nonzero(model.background[2])==0
    assert prepared['receipt']['physical_qualification'] is False
    source=model.sources(solution.electric)
    polarization=source.polarization[0].real
    op=prepared['static']['tensor_operator']
    direct=op.solve_uniform(vertices.new_zeros(3),bindings=prepared['bindings'][0],polarizations=(polarization,))
    for key,expected in (('potential',direct.potential),('conormal',direct.conormal),('normal_p',direct.normal_p)):
        torch.testing.assert_close(prepared['static'][key+'_response']@polarization.flatten(),expected,rtol=2e-11,atol=1e-20)
    field=direct.fields(prepared['static']['points'],2)[1]
    torch.testing.assert_close(field,solution.electric[0].real,rtol=2e-7,atol=1e-10)
    assert preparation_signature(prepared)==prepared['_signature']
    ramped=model.solve(pump_scale=.5,relative_tolerance=1e-9,absolute_tolerance=1e-10)
    ramped_p=model.sources(ramped.electric).polarization[0].real
    reference=op.solve_uniform(vertices.new_zeros(3),bindings=prepared['bindings'][0],polarizations=(ramped_p,))
    from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
    from witwin.maxwell.mom.harmonic_snapshot import snapshot
    from types import SimpleNamespace
    def forbidden(*args,**kwargs):raise AssertionError('Recovery/query must not solve the interface again')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        state=recover_harmonic_state(model,prepared,ramped)
        decoder=HarmonicFieldDecoder(prepared,state)
        assert state.dc_surface_charge.numel()==0
        assert state.receipt['dc_surface_representation']=='regional_potential_exterior_conormal_v1'
        for region,point in ((0,[.4,.2,.1]),(1,[.015,.015,.015]),(2,[.215,.015,.015])):
            query=vertices.new_tensor([point])
            for actual,expected in zip(decoder.query_dc(query,region),reference.fields(query,region)):
                torch.testing.assert_close(actual,expected,rtol=2e-10,atol=1e-12)
        e,h=decoder.query_fields(prepared['static']['points'],2,harmonic=1,
            query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
        torch.testing.assert_close(e,state.cell_electric[1],rtol=3e-4,atol=1e-8)
        assert bool(torch.isfinite(h).all())
        with pytest.raises(NotImplementedError,match='potential/conormal schema'):
            snapshot(SimpleNamespace(decoder=decoder,state=state,config=None))
    from witwin.maxwell.mom.tensor_harmonic_snapshot import (snapshot_mixed_tensor_harmonic_state,
        restore_mixed_tensor_harmonic_state)
    from witwin.maxwell.mom.static_tensor_interface import StaticTensorInterfaceOperator
    from witwin.maxwell.mom.harmonic_balance import DenseHarmonicBalance
    import copy
    payload,owner=snapshot_mixed_tensor_harmonic_state(state,prepared)
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(DenseHarmonicBalance,'solve',forbidden)
        restored=restore_mixed_tensor_harmonic_state(payload,prepared,expected_binding=owner)
        reopened=HarmonicFieldDecoder(prepared,restored)
        for name in ('frequencies','cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(restored,name),getattr(state,name),rtol=0,atol=0)
        for region,point in ((0,[.4,.2,.1]),(1,[.015,.015,.015]),(2,[.215,.015,.015])):
            query=vertices.new_tensor([point])
            for actual,expected in zip(reopened.query_dc(query,region),decoder.query_dc(query,region)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for actual,expected in zip(reopened.query_fields(prepared['static']['points'],2,harmonic=1),
                                   decoder.query_fields(prepared['static']['points'],2,harmonic=1)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    altered=copy.deepcopy(payload);altered['state']['cell_electric'][1,0,0]+=1
    with pytest.raises(ValueError,match='owning binding'):
        restore_mixed_tensor_harmonic_state(altered,prepared,expected_binding=owner)
    from witwin.maxwell.mom.volume_h_snapshot import _digest
    altered=copy.deepcopy(payload);altered['state']['total_volume_current'][1,0,0]+=1
    with pytest.raises(ValueError,match='current partition'):
        restore_mixed_tensor_harmonic_state(altered,prepared,
            expected_binding=dict(version=owner['version'],sha256=_digest(altered)))
    from dataclasses import replace
    with pytest.raises(ValueError,match='cannot discard'):
        snapshot_mixed_tensor_harmonic_state(replace(state,dc_face_potentials=(vertices.new_zeros(1),)),prepared)
    other_model=type(model)(model.background,model.polarization_response,model.current_response,
        model.cell_volumes,model.frequency,chi2=2*model.chi2,chi3=model.chi3,tpa_scale=model.tpa_scale)
    other_prepared=dict(prepared,_model_identity=id(other_model),
        nonlinear_coefficients={name:getattr(other_model,name).clone() for name in ('chi2','chi3','tpa_scale')})
    other_prepared['_signature']=preparation_signature(other_prepared)
    with pytest.raises(ValueError,match='preparation differs'):
        restore_mixed_tensor_harmonic_state(payload,other_prepared,expected_binding=owner)
    altered=copy.deepcopy(payload);altered['state']['cell_electric'][1,0,0]+=.1
    with pytest.raises(ValueError,match='constitutive source law'):
        restore_mixed_tensor_harmonic_state(altered,prepared,
            expected_binding=dict(version=owner['version'],sha256=_digest(altered)))
    from witwin.maxwell.mom.tensor_harmonic_bundle import (snapshot_mixed_tensor_harmonic_bundle,
        restore_mixed_tensor_harmonic_bundle)
    bundle,bundle_owner=snapshot_mixed_tensor_harmonic_bundle(state,prepared)
    path=tmp_path/'mixed-harmonic.pt'
    torch.save(bundle,path)
    disk=torch.load(path,weights_only=False)
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(StaticTensorInterfaceOperator,'__init__',forbidden)
        patch.setattr(SystemOperator,'__init__',forbidden)
        patch.setattr(DenseHarmonicBalance,'solve',forbidden)
        restored_model,new_prepared,new_state=restore_mixed_tensor_harmonic_bundle(
            disk,expected_binding=bundle_owner,device='cuda')
        new_decoder=HarmonicFieldDecoder(new_prepared,new_state)
        assert new_prepared is not prepared and new_prepared['operators'][0].system is not operators[0].system
        for name in ('cell_electric','polarization','total_volume_current'):
            torch.testing.assert_close(getattr(new_state,name),getattr(state,name),rtol=0,atol=0)
        for region,point in ((0,[.4,.2,.1]),(1,[.015,.015,.015]),(2,[.215,.015,.015])):
            query=vertices.new_tensor([point])
            for actual,expected in zip(new_decoder.query_dc(query,region),decoder.query_dc(query,region)):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for actual,expected in zip(new_decoder.query_fields(prepared['static']['points'],2,harmonic=1),
                                   decoder.query_fields(prepared['static']['points'],2,harmonic=1)):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(restored_model.residual(new_state.cell_electric,pump_scale=.5),
                                   model.residual(state.cell_electric,pump_scale=.5),rtol=0,atol=0)
    with pytest.raises(ValueError,match='storage budget'):
        restore_mixed_tensor_harmonic_bundle(disk,expected_binding=bundle_owner,device='cuda',max_retained_bytes=1)
    broken=copy.deepcopy(disk);broken['operators'][0]['matrix'][0,0]+=1
    with pytest.raises(ValueError,match='owning binding'):
        restore_mixed_tensor_harmonic_bundle(broken,expected_binding=bundle_owner,device='cuda')
    payload['state']['cell_electric'].add_(1)
    torch.testing.assert_close(restored.cell_electric,state.cell_electric,rtol=0,atol=0)
    state.dc_tensor_response.potential.add_(1)
    with pytest.raises(ValueError,match='changed'):decoder.query_dc(prepared['static']['points'],2)
    op.matrix[0,0].add_(1)
    with pytest.raises(ValueError,match='changed'):preparation_signature(prepared)


@pytest.mark.parametrize('fault', ('permuted', 'wrong_fundamental'))
def test_mixed_actual_frequency_axis_rejected_before_preparation(fault):
    from dataclasses import replace
    from types import SimpleNamespace
    from test_static_tensor_polarization import source
    particular,_,_=source()
    system=particular.binding.surface_system
    material=compile_harmonic_material(mw.Material(eps_r=2.),3e8,2)
    exterior=compile_harmonic_material(mw.Material(),3e8,2)
    def corrupt(sample):
        return (replace(sample,linear_samples=sample.linear_samples[::-1]) if fault=='permuted'
                else replace(sample,fundamental_frequency=4e8))
    # The placeholders expose only geometry: no assembly/refresh may be reached.
    operators=(SimpleNamespace(system=system),)*2
    with pytest.raises(ValueError,match='actual integer frequency axis'):
        prepare_mixed_tensor_harmonic_balance(operators,(particular.binding.volume,),
                                             (corrupt(material),),corrupt(exterior))


@pytest.mark.parametrize('epsilon,mu', ((2.+.3j,1.2+.1j),(-2.+0j,-1.3+0j)))
def test_exact_scalar_sample_restore_keeps_complex_passive_branch(epsilon,mu):
    from witwin.maxwell.mom.physics import Medium,_outgoing_parameters,restore_scalar_medium
    from dataclasses import asdict
    k,eta=_outgoing_parameters(3e8,epsilon,mu)
    original=Medium(3e8,epsilon,mu,k,eta,{'origin':'sampled'})
    payload=asdict(original)
    assert restore_scalar_medium(payload)==original
    payload['k']=-k
    with pytest.raises(ValueError,match='outgoing'):
        restore_scalar_medium(payload)
