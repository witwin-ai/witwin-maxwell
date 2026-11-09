"""Actual dielectric SIE-VIE response wiring; qualification gates stay open."""
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.mesh import SurfaceMesh,RWGBasis,SurfaceSystem
from witwin.maxwell.mom.volume_mesh import TetrahedralVolumeMesh
from witwin.maxwell.mom.runtime import SystemOperator
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.harmonic_material import compile_harmonic_material
from witwin.maxwell.mom.harmonic_preparation import prepare_harmonic_balance

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_actual_dielectric_hb_generates_dc_and_second_harmonic(monkeypatch):
    vertices = torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume = TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    basis = RWGBasis(SurfaceMesh(volume.boundary.vertices.clone(),volume.boundary.faces.clone()))
    material = compile_harmonic_material(mw.Material(eps_r=2.,nonlinearity=mw.NonlinearSusceptibility(
        chi2=.002,chi3=.001)),3e8,2)
    exterior = compile_harmonic_material(mw.Material(),3e8,2)
    config = mw.MoMConfig(integration=IntegrationConfig(order=3,singular_order=3),riesz_storage='compact_rwg_v1')
    operators = tuple(SystemOperator(SurfaceSystem.create([(0,'dielectric',basis,material.linear_samples[n])]),
        exterior.linear_samples[n],[mw.PlaneWave()] if n == 0 else [],config) for n in range(2))
    model,prepared = prepare_harmonic_balance(operators,[volume],[material],exterior,
        query_config=FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48))
    assert torch.count_nonzero(model.background[2]) == 0
    result = model.solve(relative_tolerance=1e-9,absolute_tolerance=1e-10)
    assert result.receipt['converged']
    assert torch.linalg.vector_norm(result.electric[0]) > 0
    assert torch.linalg.vector_norm(result.electric[2]) > 0
    assert torch.count_nonzero(result.electric[0].imag) == 0
    assert not prepared['receipt']['physical_qualification']
    assert not result.receipt['harmonic_truncation_converged']
    from witwin.maxwell.mom.harmonic_state import recover_harmonic_state
    ramped = model.solve(pump_scale=.5)
    state = recover_harmonic_state(model,prepared,ramped)
    torch.testing.assert_close(state.frequencies,vertices.new_tensor([0.,3e8,6e8]),rtol=0,atol=0)
    assert torch.count_nonzero(state.total_volume_current[0]) == 0
    assert state.dc_surface_charge.norm() > 0
    for n,operator in enumerate(operators):
        part = prepared['harmonics'][n]
        expected_rhs = .5*operator.build_rhs()
        # The physical weak source matrix is recovered from its solved columns.
        induced_rhs = operator.matrix@part['surface_response']@state.total_volume_current[n+1].flatten()
        torch.testing.assert_close(operator.matrix@state.surface_coefficients[n],expected_rhs+induced_rhs,
                                   rtol=1e-10,atol=1e-12)
    from witwin.maxwell.mom.harmonic_fields import HarmonicFieldDecoder
    decoder = HarmonicFieldDecoder(prepared,state)
    settings = FieldQueryConfig(relative_tolerance=1e-5,absolute_tolerance=1e-10,max_order=48)
    for n in (1,2):
        electric,magnetic = decoder.query_fields(prepared['harmonics'][n-1]['points'],1,harmonic=n,query_config=settings)
        error = torch.linalg.vector_norm(electric-state.cell_electric[n])/torch.linalg.vector_norm(state.cell_electric[n])
        assert error < 3e-4
        assert torch.isfinite(magnetic).all()
    def unexpected_solve(*args,**kwargs):
        raise AssertionError('Field queries must not repeat the interface solve')
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'solve',unexpected_solve)
        potential,electric = decoder.query_dc(prepared['static']['points'],1)
    torch.testing.assert_close(electric,state.cell_electric[0].real,rtol=1e-8,atol=1e-10)
    assert potential.dtype == vertices.dtype
    with pytest.raises(ValueError,match='positive harmonic'):
        decoder.query_fields(prepared['static']['points'],1,harmonic=0)
    original_order = prepared['dc_testing_order']
    prepared['dc_testing_order'] = original_order+2
    with pytest.raises(RuntimeError,match='mutated'):
        decoder.query_dc(prepared['static']['points'],1)
    prepared['dc_testing_order'] = original_order
    original_epsilon = prepared['dc_permittivities']
    prepared['dc_permittivities'] = (2*original_epsilon[0],original_epsilon[1])
    with pytest.raises(RuntimeError,match='mutated'):
        decoder.query_dc(prepared['static']['points'],1)
    prepared['dc_permittivities'] = original_epsilon
    # Old response maps cannot be paired with changed decoding physics, even
    # after the decoder has cached its own local copies.
    from dataclasses import replace
    mutations = [
        (operators[0],'incident_carrier',replace(operators[0].incident_carrier,waves=())),
        (operators[0],'integration',replace(operators[0].integration,order=4)),
        (operators[0].body_handles[0],'interior',operators[0].exterior),
        (operators[0].body_handles[0],'incident_carrier',operators[1].incident_carrier),
        (operators[0],'body_handles',()),
    ]
    for owner,name,replacement in mutations:
        original = getattr(owner,name)
        try:
            setattr(owner,name,replacement)
            with pytest.raises(RuntimeError,match='mutated'):
                decoder.query_dc(prepared['static']['points'],1)
            with pytest.raises(RuntimeError,match='changed'):
                recover_harmonic_state(model,prepared,ramped)
        finally:
            setattr(owner,name,original)
    # Extend only the new frequency, preserving every existing physical map.
    import witwin.maxwell.mom.harmonic_preparation as preparation
    extended_material = compile_harmonic_material(mw.Material(eps_r=2.,nonlinearity=mw.NonlinearSusceptibility(
        chi2=.002,chi3=.001)),3e8,3)
    extended_exterior = compile_harmonic_material(mw.Material(),3e8,3)
    third = SystemOperator(SurfaceSystem.create([(0,'dielectric',basis,extended_material.linear_samples[2])]),
                           extended_exterior.linear_samples[2],[],config)
    original_prepare = preparation.prepare_cell_field_response
    calls = []
    def only_new(operator,*args,**kwargs):
        assert operator is third
        calls.append(operator)
        return original_prepare(operator,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(preparation,'prepare_static_cell_response',unexpected_solve)
        patch.setattr(preparation,'prepare_cell_field_response',only_new)
        extended_model,extended = prepare_harmonic_balance((*operators,third),[volume],
            [extended_material],extended_exterior,query_config=settings,reuse_prepared=prepared)
    assert calls == [third]
    assert extended['static'] is prepared['static']
    assert all(a is b for a,b in zip(extended['harmonics'][:2],prepared['harmonics']))
    assert extended['receipt']['reused_positive_harmonics'] == 2
    warm = torch.cat((result.electric,result.electric.new_zeros(1,1,3)))
    expanded = extended_model.solve(warm)
    assert expanded.receipt['converged'] and expanded.electric[3].norm() > 0
    assert not expanded.receipt['harmonic_truncation_converged']
    with pytest.raises(ValueError,match='DC material'):
        prepare_harmonic_balance((*operators,third),[volume],[extended_material],extended_exterior,
            dc_testing_order=6,reuse_prepared=prepared)
    print({'extension_residual':expanded.receipt['final_residual'],
           'third_harmonic_norm':float(expanded.electric[3].norm()),'reused_harmonics':2})
    original_field = ramped.electric.detach().clone()
    state.cell_electric.add_(1)
    with pytest.raises(RuntimeError,match='mutated'):
        decoder.query_dc(prepared['static']['points'],1)
    torch.testing.assert_close(ramped.electric,original_field,rtol=0,atol=0)
    with torch.no_grad():
        prepared['harmonics'][0]['surface_response'].add_(1)
    with pytest.raises(RuntimeError,match='changed'):
        recover_harmonic_state(model,prepared,ramped)
    print({'residual':result.receipt['final_residual'],'dc_norm':float(result.electric[0].norm()),
           'second_harmonic_norm':float(result.electric[2].norm()),'physical_qualification':False})
