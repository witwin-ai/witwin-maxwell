"""Genuine TPA nonzero Graph capture and detached disk-resume controls."""
from pathlib import Path
import json
import os
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state
from witwin.maxwell.fdtd.runtime.stepping import _make_field_update_runner
from tests.gradients.test_fdtd_adjoint_materials import _tpa_scene,_TPA_BETA


def _require_cuda():
    if not torch.cuda.is_available():raise RuntimeError('TPA actual qualification cannot silently skip CUDA.')


def _output(tmp_path, name):
    return Path(os.environ.get('TPA_UNITS_CASES', str(tmp_path))) / name


def test_nonzero_tpa_graph_capture_keeps_physical_and_CPML_state(tmp_path):
    _require_cuda()
    execution=mw.Simulation.fdtd(_tpa_scene(_TPA_BETA),frequencies=[1e9],
        run_time=mw.TimeConfig(time_steps=200),full_field_dft=False).prepare()
    solver=execution.solver
    solver.Ez.fill_(2.);solver.Hx.fill_(.1)
    for name in ('psi_ex_y','psi_ex_z','psi_ey_x','psi_ey_z','psi_ez_x','psi_ez_y',
                 'psi_hx_y','psi_hx_z','psi_hy_x','psi_hy_z','psi_hz_x','psi_hz_y'):
        if torch.is_tensor(getattr(solver,name,None)):getattr(solver,name).fill_(.2)
    assert solver.tpa_enabled and solver.nonlinear_general_enabled
    before=capture_checkpoint_state(solver,step=0)
    assert bool((before.tensors['Ez']!=0).any()) and before.schema.cpml_state_names
    _make_field_update_runner(solver,True)
    assert solver._cuda_graph_active is True
    after=capture_checkpoint_state(solver,step=0)
    assert after.schema==before.schema and after.tensors.keys()==before.tensors.keys()
    for name in before.tensors:
        torch.testing.assert_close(after.tensors[name],before.tensors[name],rtol=0,atol=0,msg=lambda m:name+': '+m)
    output=_output(tmp_path,'nonzero-capture');output.mkdir(parents=True,exist_ok=False)
    torch.save(dict(before=before,after=after),output/'physical-state.pt')
    (output/'witness.json').write_text(json.dumps(dict(actual_cuda_graph_active=True,
        actual_tpa_enabled=True,physical_state_bitwise_unchanged=True,
        instantaneous_tpa_has_no_accumulated_polarization_memory=True),indent=2)+'\n')


@pytest.mark.parametrize('graph',[False,True])
def test_tpa_real_disk_checkpoint_matches_direct_physics_and_spectrum(graph,tmp_path):
    _require_cuda()
    scene=_tpa_scene(_TPA_BETA)
    simulation=mw.Simulation.fdtd(scene,frequencies=[1e9],cuda_graph=graph,
        run_time=mw.TimeConfig(time_steps=200),full_field_dft=False,
        spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False))
    output=_output(tmp_path,f'resume-graph{int(graph)}');output.mkdir(parents=True,exist_ok=False)
    direct=simulation.prepare().run()
    assert direct.stats()['steps_run']==200 and direct.stats()['cuda_graph_active']==graph
    # One Point field at one frequency intentionally uses native point sampling,
    # without a dedicated grouped-observer graph. Full-field DFT is disabled,
    # so there is no separately captured running-DFT tail either.
    assert direct.stats()['observer_cuda_graph_active'] is False
    assert direct.stats()['tail_graph_active'] is False
    assert direct.solver._single_observer_engine is None
    assert direct.solver.dft_enabled is False
    state=capture_checkpoint_state(direct.solver,step=200)
    spectrum=direct.monitor('probe')['data'].detach().cpu().clone()
    assert torch.isfinite(spectrum).all() and bool((spectrum.abs()>0).any())
    direct.save(output/'direct.pt')
    execution=simulation.prepare();checkpoint=execution.run_until(100)
    assert checkpoint.step==100 and checkpoint.total_steps==200
    checkpoint.save(output/'checkpoint.pt')
    del execution,checkpoint
    loaded=mw.FDTDResumeCheckpoint.load(output/'checkpoint.pt',map_location='cpu')
    resumed=simulation.prepare().run(resume_from=loaded)
    after=capture_checkpoint_state(resumed.solver,step=200)
    assert after.schema==state.schema
    for name in state.tensors:
        torch.testing.assert_close(after.tensors[name],state.tensors[name],rtol=0,atol=0,msg=lambda m:name+': '+m)
    torch.testing.assert_close(resumed.monitor('probe')['data'].cpu(),spectrum,rtol=0,atol=0)
    assert resumed.stats()['steps_run']==resumed.stats()['time_steps']==200
    assert resumed.stats()['cuda_graph_active']==graph
    assert resumed.stats()['observer_cuda_graph_active'] is False
    assert resumed.stats()['tail_graph_active'] is False
    assert resumed.solver._single_observer_engine is None
    assert resumed.solver.dft_enabled is False
    assert all(count==200 for count in resumed.stats()['observer_sample_counts'])
    resumed.save(output/'resumed.pt')
    torch.save(dict(direct=state,resumed=after,spectrum=spectrum),output/'physical-state.pt')
    public=mw.Result.load(output/'resumed.pt',scene=scene,map_location='cpu')
    torch.testing.assert_close(public.monitor('probe')['data'],spectrum,rtol=0,atol=0)
    (output/'witness.json').write_text(json.dumps(dict(graph=graph,steps=200,checkpoint_step=100,
        physical_state_and_spectrum_bitwise=True,actual_nonlinear_enabled=True,
        field_graph_active=resumed.stats()['cuda_graph_active'],observer_graph_active=resumed.stats()['observer_cuda_graph_active'],
        tail_graph_active=resumed.stats()['tail_graph_active'],
        observation_counts=resumed.stats()['observer_sample_counts']),indent=2)+'\n')
