"""Public budgeted cutoff flow and strict failure without redundant preparation."""
from dataclasses import replace
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.fields import FieldQueryConfig
from witwin.maxwell.mom.harmonic_runtime import HarmonicCutoffError

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def test_public_cutoff_reuses_prefix_records_success_and_budget_exhaustion(tmp_path,monkeypatch):
    vertices = torch.tensor([[0.,0.,0.],[.07,0.,0.],[0.,.07,0.],[0.,0.,.07]],device='cuda',dtype=torch.float64)
    volume = mw.TetrahedralVolumeMesh(vertices,torch.tensor([[0,1,2,3]],device='cuda'),region_id=1)
    scene = mw.Scene(structures=[mw.Structure(geometry=mw.Mesh(volume.boundary.vertices,volume.boundary.faces,
        recenter=False,fill_mode='solid'),material=mw.Material(eps_r=2.))],sources=[mw.PlaneWave()])
    policy = mw.HarmonicCutoffConfig(max_harmonics=3)
    config = mw.MoMConfig(harmonic_balance=mw.HarmonicBalanceConfig(harmonics=1,volumes=(volume,),
        branch_id='zero_nonlinearity',cutoff=policy),integration=IntegrationConfig(order=3,singular_order=3),
        riesz_storage='compact_rwg_v1',query_config=FieldQueryConfig(relative_tolerance=1e-5,
            absolute_tolerance=1e-10,max_order=48))
    prepared = mw.Simulation.mom(scene,frequency=3e8,config=config).prepare()
    background = prepared.solver.model.background.clone()
    import witwin.maxwell.mom.harmonic_preparation as preparation
    original = preparation.prepare_cell_field_response
    calls = []
    def added_only(operator,*args,**kwargs):
        calls.append(operator.exterior.frequency)
        return original(operator,*args,**kwargs)
    def forbidden(*args,**kwargs):
        raise AssertionError('Cutoff extension must not repeat static preparation')
    initial = .3*background
    initial_copy = initial.clone()
    import witwin.maxwell.mom.harmonic_convergence as convergence
    compare=convergence.compare_harmonic_cells
    compared=[]
    def capture(previous,current,**settings):
        compared.append((previous,current))
        return compare(previous,current,**settings)
    with pytest.raises(ValueError,match='real DC'):
        prepared.run(initial_harmonic_electric=initial+1j)
    with monkeypatch.context() as patch:
        patch.setattr(preparation,'prepare_cell_field_response',added_only)
        patch.setattr(preparation,'prepare_static_cell_response',forbidden)
        patch.setattr(convergence,'compare_harmonic_cells',capture)
        result = prepared.run(initial_harmonic_electric=initial)
    initial.add_(1)
    torch.testing.assert_close(result.solver_stats['initial_harmonic_electric'],initial_copy,rtol=0,atol=0)
    assert result.solver_stats['branch_initialization'] == 'explicit_harmonic_cell_field'
    assert result.solver_stats['continuation'][0]['initialization'] == 'explicit_state'
    assert calls == [6e8,9e8]
    assert result.frequencies == (0.,3e8,6e8,9e8)
    torch.testing.assert_close(result.harmonic_state.cell_electric[:2],background,rtol=0,atol=0)
    assert torch.count_nonzero(result.harmonic_state.cell_electric[2:]) == 0
    cutoff = result.solver_stats['cutoff']
    assert cutoff['cell_truncation_estimate_passed'] and len(cutoff['checks']) == 2
    assert cutoff['checks'][-1]['successive_passes'] == 2
    assert not cutoff['physical_qualification'] and not cutoff['harmonic_truncation_converged']
    result.save(tmp_path/'cutoff.pt')
    restored = mw.Result.load(tmp_path/'cutoff.pt',scene=scene)
    assert restored.solver_stats['cutoff'] == cutoff
    torch.testing.assert_close(restored.solver_stats['initial_harmonic_electric'],initial_copy.cpu(),rtol=0,atol=0)
    assert restored._harmonic_data.config.harmonic_balance.cutoff == policy
    assert restored.frequencies == result.frequencies
    with pytest.raises(ValueError,match='must be finite'):
        compare(*compared[-1],relative_tolerance=1.79e308,absolute_tolerance=1.79e308)
    # Exhaust the budget at the current cutoff without re-solving any field.
    with pytest.raises(HarmonicCutoffError) as error:
        prepared.solver._extend_cutoff(result._harmonic_data,policy)
    assert error.value.receipt['failure_reason'] == 'harmonic_budget_exhausted'
    same,failed = prepared.solver._extend_cutoff(result._harmonic_data,replace(policy,failure='record'))
    assert same is result._harmonic_data and not failed['cell_truncation_estimate_passed']
    assert failed['checks'] == []
    for values in ({'max_harmonics':True},{'successive_passes':0},{'tail_absolute_tolerance':0.},{'failure':'ignore'}):
        with pytest.raises(ValueError):
            replace(policy,**values)
