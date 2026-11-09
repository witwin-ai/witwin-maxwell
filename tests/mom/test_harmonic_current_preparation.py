"""Actual public material sampling in internal current-formulation HB preparation."""
from witwin.maxwell.mom.harmonic_material import material_signature
import math
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.constants import EPSILON_0
from witwin.maxwell.mom.mesh import RWGBasis,SurfaceSystem
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.harmonic_preparation import prepare_harmonic_balance
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_conductive_tpa_material_prepares_actual_ac_and_dc_channels_once(monkeypatch,tmp_path):
    vertices = torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume = mw.TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    basis = RWGBasis(volume.boundary)
    physical = mw.Material(eps_r=2.,sigma_e=.001,nonlinearity=(
        mw.NonlinearSusceptibility(chi2=.002,chi3=.001),mw.TwoPhotonAbsorption(beta=5.)))
    material = compile_harmonic_material(physical,3e8,2)
    exterior = compile_harmonic_material(mw.Material(),3e8,2)
    config = mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),riesz_storage='compact_rwg_v1')
    operators = tuple(SystemOperator(SurfaceSystem.create([(0,'body',basis,sample)]),exterior.linear_samples[n],
        [mw.PlaneWave()] if n == 0 else [],config) for n,sample in enumerate(material.linear_samples))
    for n,operator in enumerate(operators,1):
        assert operator.interior.epsilon_r == pytest.approx(2+1j*.001/(2*math.pi*n*3e8*EPSILON_0),rel=1e-13)
    import witwin.maxwell.mom.harmonic_preparation as preparation
    def forbidden(*args,**kwargs):
        raise AssertionError('Conductive preparation must not assemble a dielectric DC map')
    with monkeypatch.context() as patch:
        patch.setattr(preparation,'prepare_static_cell_response',forbidden)
        model,prepared = prepare_harmonic_balance(operators,[volume],[material],exterior,
            query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    assert prepared['receipt']['mean_current_dc_supported']
    assert prepared['receipt']['dc_magnetic_fields_supported']
    assert torch.count_nonzero(model.polarization_response[0]) == 0
    assert torch.count_nonzero(model.current_response[0]) == 0
    torch.testing.assert_close(model.tpa_scale,vertices.new_tensor([physical.tpa_sigma_scale]),rtol=0,atol=0)
    closure = model.dc_current_closure
    torch.testing.assert_close(closure.conductivities[0],vertices.new_tensor([physical.sigma_e]),rtol=0,atol=0)
    result = model.solve_ramp((1.,),branch_id='current')[-1]
    current = physical.sigma_e*result.electric[0].real+result.sources.dissipative_current[0].real
    assert closure.blocks[0].weak_current(current).norm() < 1e-14
    assert result.receipt['converged']
    assert result.electric[2].norm() > 0
    assert not prepared['receipt']['physical_qualification']
    from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
    from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
    state = recover_harmonic_state(model,prepared,result)
    torch.testing.assert_close(state.total_volume_current[0].real,current,rtol=0,atol=0)
    assert state.receipt['formulation'] == 'coupled_sie_vie_hb_current_face_state_v1'
    assert len(state.dc_face_potentials) == 1 and state.dc_potential_offsets.shape == (1,)
    decoder = HarmonicFieldDecoder(prepared,state)
    inside = vertices.mean(0,keepdim=True)
    outside = vertices.new_tensor([[.16,.1,.09]])
    with monkeypatch.context() as patch:
        patch.setattr(closure,'solve',forbidden)
        patch.setattr(prepared['static']['floating_operator'],'solve_face_means',forbidden)
        patch.setattr(torch.linalg,'solve',forbidden)
        from witwin.maxwell.mom.static_magnetic import MagnetostaticInterfaceResponse
        patch.setattr(MagnetostaticInterfaceResponse,'__init__',forbidden)
        magnetic,flux = decoder.query_dc_magnetic(outside,0)
        from witwin.maxwell.constants import MU_0
        torch.testing.assert_close(flux,MU_0*magnetic,rtol=0,atol=0)
        potential,electric = decoder.query_dc(inside,1)
        torch.testing.assert_close(electric,state.cell_electric[0].real,rtol=1e-7,atol=1e-10)
        expected = state.dc_face_potentials[0].mean()+state.dc_potential_offsets[0]
        torch.testing.assert_close(potential,expected.reshape(1),rtol=1e-12,atol=1e-15)
        external = decoder.query_dc(outside,0)
        assert all(torch.isfinite(value).all() for value in external)
        ac_e,ac_h = decoder.query_fields(inside,1,harmonic=1,query_config=FieldQueryConfig(
            relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
        assert (ac_e-state.cell_electric[1]).norm()/state.cell_electric[1].norm() < 3e-4
        assert torch.isfinite(ac_h).all()
    from dataclasses import replace
    from witwin.maxwell.mom.harmonic_runtime import HarmonicData
    from witwin.maxwell.mom.harmonic_snapshot import snapshot,restore
    from witwin.maxwell.mom.runtime import _signature
    from witwin.maxwell.mom.static_current import FaceCurrentBalance
    from witwin.maxwell.mom.static_floating import FloatingPotentialOperator
    scene = mw.Scene(structures=[mw.Structure(name='body',geometry=mw.Mesh(volume.boundary.vertices,
        volume.boundary.faces,recenter=False,fill_mode='solid'),material=physical)],sources=[mw.PlaneWave()])
    signature = (_signature(scene,exterior.linear_samples[0],(material.linear_samples[0],)),
                 tuple(material_signature(s.material) for s in scene.structures if s.enabled),3e8)
    saved_config = replace(config,harmonic_balance=mw.HarmonicBalanceConfig(
        harmonics=2,volumes=(volume,),branch_id='current'))
    data = HarmonicData(prepared,state,saved_config,signature)
    monitors = (mw.PointMonitor('mixed',tuple(outside[0].tolist()),fields=('Ex','Hx','Hy','Hz')),
                mw.PointMonitor('magnetic_copy',tuple(outside[0].tolist()),fields=('Hx','Hy','Hz')))
    magnetic_calls = []
    original_query = data.query_dc_magnetic
    def recorded_query(points,region):
        magnetic_calls.append((len(points),region))
        return original_query(points,region)
    with monkeypatch.context() as patch:
        patch.setattr(data,'query_dc_magnetic',recorded_query)
        monitor_values,monitor_receipt = data.sample_point_monitors(monitors)
    assert magnetic_calls == [(1,0)]
    assert monitor_receipt['unique_monitor_positions'] == 1
    assert monitor_values['magnetic_copy']['dc']['potential_V'] is None
    assert monitor_values['magnetic_copy']['dc']['magnetic_available']
    reference_h,_ = original_query(outside,0)
    for index,name in enumerate(('Hx','Hy','Hz')):
        for monitor in monitor_values.values():
            torch.testing.assert_close(monitor['dc']['components'][name],reference_h[0,index],rtol=0,atol=0)
    payload = snapshot(data)
    torch.save(payload,tmp_path/'current.pt')
    assert payload['schema_version'] == 5 and payload['payload_kind'] == 'coupled_sie_vie_hb_current'
    reference = [decoder.query_dc(inside,1),decoder.query_dc(outside,0)]
    magnetic_reference = decoder.query_dc_magnetic(outside,0)
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',forbidden)
        patch.setattr(torch.linalg,'lu_factor',forbidden)
        patch.setattr(torch.linalg,'lu_solve',forbidden)
        patch.setattr(FaceCurrentBalance,'__init__',forbidden)
        patch.setattr(FloatingPotentialOperator,'__init__',forbidden)
        patch.setattr(MagnetostaticInterfaceResponse,'__init__',forbidden)
        restored = restore(torch.load(tmp_path/'current.pt',weights_only=False),scene)
        for query,region,expected in zip((inside,outside),(1,0),reference):
            for actual,value in zip(restored.decoder.query_dc(query,region),expected):
                torch.testing.assert_close(actual,value,rtol=0,atol=0)
        for actual,value in zip(restored.decoder.query_dc_magnetic(outside,0),magnetic_reference):
            torch.testing.assert_close(actual,value,rtol=0,atol=0)
        again = snapshot(restored)
    for key in ('dc_surface_charge','dc_potential_offsets','total_volume_current','dc_magnetic_density','dc_magnetic_source_mean'):
        torch.testing.assert_close(again['state'][key],payload['state'][key],rtol=0,atol=0)
    import copy
    damaged = copy.deepcopy(payload)
    damaged['state']['dc_potential_offsets'].add_(.1)
    with pytest.raises(ValueError,match='floating state'):
        restore(damaged,scene)
    damaged = copy.deepcopy(payload)
    damaged['state']['dc_magnetic_density'].add_(.1)
    with pytest.raises(ValueError,match='magnetic density'):
        restore(damaged,scene)
    state.dc_face_potentials[0].add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        decoder.query_dc(inside,1)
    print({'residual':result.receipt['final_residual'],'dc_norm':float(result.electric[0].norm().detach()),
           'tpa_scale':physical.tpa_sigma_scale,'dc_conductivity':physical.sigma_e,'physical_qualification':False})
