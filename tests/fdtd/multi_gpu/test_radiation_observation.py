"""Distributed radiation faces retain owned Yee samples and exact quadrature."""
import math
import numpy as np
import pytest
import torch
import witwin.maxwell as mw


def _simulation(*,parallel=False,copper=False,pml=True,sampled=None,steps=96,
                frequency=1e9,extra_monitors=(),nonuniform=False,substrate=False,
                full_field_dft=False):
    structures=[]
    if copper:
        structures.append(mw.Structure(mw.Box(position=torch.tensor((0.,0.,0.),dtype=torch.float64),
            size=torch.tensor((.02,.02,0.),dtype=torch.float64)),
            mw.CopperFoil(thickness=2e-6,frequency_range=(.5*frequency,2*frequency)),name='foil'))
    if substrate:
        structures.append(mw.Structure(mw.Box(position=(0.,0.,-.005),size=(.02,.02,.01)),
            mw.Material(eps_r=2.2,sigma_e=.02,debye_poles=(mw.DebyePole(delta_eps=.8,tau=1e-10),)),name='substrate'))
    grid=mw.GridSpec.uniform(.005)
    if nonuniform:
        x=(-.03,-.025,-.02,-.015,-.01,-.005,0.,.004,.009,.014,.02,.025,.03)
        grid=mw.GridSpec.custom(x,np.linspace(-.03,.03,13),np.linspace(-.03,.03,13))
    frequencies=(frequency,1.2*frequency)
    surface=mw.ClosedSurfaceMonitor.box('surface',position=(0.,0.,0.),size=(.04,)*3,
        frequencies=frequencies,sample_spacing=sampled)
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.03,.03),)*3),grid=grid,
        boundary=mw.BoundarySpec.pml(num_layers=3) if pml else mw.BoundarySpec.none(),
        structures=structures,sources=[mw.PointDipole(position=(-.005,0.,-.005),polarization='Ez',profile='ideal',
            source_time=mw.CW(frequency=frequency),name='drive')],monitors=(surface,*extra_monitors),device='cuda:0')
    config=mw.FDTDParallelConfig(devices=('cuda:0','cuda:1'),transport='cuda_p2p',overlap=True,gather_fields=full_field_dft) if parallel else None
    return mw.Simulation.fdtd(scene,frequencies=frequencies,parallel=config,cuda_graph=False,
        full_field_dft=full_field_dft,run_time=mw.TimeConfig(time_steps=steps,dt=5e-12),spectral_sampler=mw.SpectralSampler(window='none'))


@pytest.mark.parametrize('copper,pml',[(False,False),(False,True),(True,True)])
def test_owned_closed_surface_nf2ff_and_conductor_loss_match_single_domain(cuda_p2p_devices,copper,pml):
    expected=_simulation(copper=copper,pml=pml).run()
    actual=_simulation(copper=copper,pml=pml,parallel=True).run()
    _compare_surface(actual,expected)
    if copper:
        for key in ('surface_power','dissipated_energy','stored_energy'):
            torch.testing.assert_close(torch.as_tensor(actual.conductor_losses['foil'][key]),torch.as_tensor(expected.conductor_losses['foil'][key]),rtol=4e-4,atol=1e-12)


def _compare_surface(actual,expected,name='surface'):
    measured,reference=actual.monitor(name),expected.monitor(name)
    for face,expected_face in zip(measured['faces'].values(),reference['faces'].values()):
        assert face['face_label']==expected_face['face_label']
        for axis in 'xyz':
            if axis in expected_face:
                np.testing.assert_array_equal(face[axis],expected_face[axis])
        for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            if component in expected_face:
                torch.testing.assert_close(face[component],expected_face[component],rtol=5e-5,atol=5e-6)
        from witwin.maxwell.fdtd.quadrature import _compute_plane_flux
        torch.testing.assert_close(_compute_plane_flux(face),_compute_plane_flux(expected_face),rtol=3e-4,atol=1e-9)
    angles=dict(theta=torch.linspace(0.,math.pi,7,device='cuda:0',dtype=torch.float64),phi=torch.linspace(0.,2*math.pi,9,device='cuda:0',dtype=torch.float64))
    from witwin.maxwell.postprocess.antenna import _far_fields_from_result
    frequencies=torch.as_tensor(measured['frequencies'],device='cuda:0',dtype=torch.float64)
    first=_far_fields_from_result(actual,surface=name,frequencies=frequencies,**angles)
    second=_far_fields_from_result(expected,surface=name,frequencies=frequencies,**angles)
    scale=max(float(second[component].abs().max()) for component in ('e_theta','e_phi'))
    assert scale>0
    for component in ('e_theta','e_phi'):
        torch.testing.assert_close(first[component],second[component],rtol=2e-4,atol=2e-5*scale)


@pytest.mark.parametrize('sampled',[.006,.009,.03])
@pytest.mark.parametrize('copper',[False,True])
def test_sampled_closed_surface_has_unique_current_cross_shard_gathers(cuda_p2p_devices,sampled,copper):
    expected=_simulation(sampled=sampled,copper=copper).run()
    actual=_simulation(sampled=sampled,copper=copper,parallel=True).run()
    _compare_surface(actual,expected)
    assert actual.solver.parallel_stats['observation_halo_bytes_per_step']>0


@pytest.mark.parametrize('sampled',[None,.006])
def test_distributed_radiation_and_copper_resume_restore_all_spectra(cuda_p2p_devices,tmp_path,sampled):
    simulation=_simulation(parallel=True,copper=True,sampled=sampled)
    reference=simulation.run()
    checkpoint=_simulation(parallel=True,copper=True,sampled=sampled).prepare().run_until(47)
    assert checkpoint.signature['additive_j_injection']=='n+1/2'
    assert any(name.startswith('rank1/') for name in checkpoint.physics.tensors)
    checkpoint.save(tmp_path/'distributed.pt')
    restored=mw.FDTDResumeCheckpoint.load(tmp_path/'distributed.pt',map_location='cpu')
    actual=_simulation(parallel=True,copper=True,sampled=sampled).prepare().run(resume_from=restored)
    _compare_surface(actual,reference)
    assert actual.solver.parallel_stats['resumed_from_step']==47
    assert actual.solver.parallel_stats['executed_steps']==96-47
    actual.save(tmp_path/'radiation-result.pt')
    saved=mw.Result.load(tmp_path/'radiation-result.pt',scene=simulation.scene,map_location='cuda:0')
    assert saved.solver is None
    _compare_surface(saved,reference)
    for key in ('surface_power','dissipated_energy','stored_energy'):
        torch.testing.assert_close(torch.as_tensor(actual.conductor_losses['foil'][key]),torch.as_tensor(reference.conductor_losses['foil'][key]),rtol=3e-5,atol=1e-12)


@pytest.mark.parametrize('convention',[None,'n'])
def test_distributed_resume_refuses_old_additive_j_timing(cuda_p2p_devices,tmp_path,convention):
    checkpoint=_simulation(parallel=True,steps=8).prepare().run_until(3)
    payload=checkpoint._snapshot()
    payload['signature'].pop('additive_j_injection')
    if convention is not None:
        payload['signature']['additive_j_injection']=convention
    payload['signature']['logical_fingerprint']='old fingerprint'
    torch.save(payload,tmp_path/'old-timing.pt')
    target=_simulation(parallel=True,steps=8).prepare()
    before=[shard.solver.Ez.clone() for shard in target.solver.shards]
    with pytest.raises(NotImplementedError,match=r'checkpoint predates the additive-J n\+1/2 source timing; re-run from the start'):
        target.run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path/'old-timing.pt'))
    for shard,saved in zip(target.solver.shards,before):
        assert torch.equal(shard.solver.Ez,saved)


@pytest.mark.parametrize('sampled',(None,.006))
def test_nonuniform_lossy_substrate_and_copper_radiation_match_single_domain(cuda_p2p_devices,sampled):
    expected=_simulation(copper=True,substrate=True,nonuniform=True,sampled=sampled).run()
    actual=_simulation(copper=True,substrate=True,nonuniform=True,sampled=sampled,parallel=True).run()
    _compare_surface(actual,expected)


def test_sampled_observation_halo_is_physically_required(cuda_p2p_devices,monkeypatch):
    from witwin.maxwell.fdtd.distributed.observations import DistributedObservationHalos
    expected=_simulation(sampled=.006,pml=False).run()
    with monkeypatch.context() as patch:
        patch.setattr(DistributedObservationHalos,'exchange',lambda self:None)
        incomplete=_simulation(sampled=.006,pml=False,parallel=True).run()
    differences=[]
    for face in expected.monitor('surface')['faces']:
        reference=expected.monitor('surface')['faces'][face]
        actual=incomplete.monitor('surface')['faces'][face]
        for component in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            if component in reference:
                scale=reference[component].abs().max().clamp_min(1e-8)
                differences.append(float((actual[component]-reference[component]).abs().max()/scale))
    assert max(differences)>.01


@pytest.mark.parametrize('axis',('y','z'))
def test_cross_shard_mode_assembly_and_disk_resume(cuda_p2p_devices,tmp_path,axis):
    position=(0.,.02,0.) if axis=='y' else (0.,0.,.02)
    size=(.04,0.,.04) if axis=='y' else (.04,.04,0.)
    monitor=mw.ModeMonitor('mode',position=position,size=size,polarization='Ex',frequencies=(1e10,1.2e10))
    options=dict(frequency=1e10,extra_monitors=(monitor,))
    expected=_simulation(**options).run()
    reference=_simulation(parallel=True,**options).run()
    checkpoint=_simulation(parallel=True,**options).prepare().run_until(43)
    checkpoint.save(tmp_path/'modal.pt')
    restored=mw.FDTDResumeCheckpoint.load(tmp_path/'modal.pt',map_location='cpu')
    actual=_simulation(parallel=True,**options).prepare().run(resume_from=restored)
    for result in (reference,actual):
        _compare_surface(result,expected)
        for index in range(2):
            for key in ('mode_power','total_power','amplitude_forward','amplitude_backward','power_forward','power_backward'):
                torch.testing.assert_close(result.monitor('mode',freq_index=index)[key],expected.monitor('mode',freq_index=index)[key],rtol=5e-4,atol=1e-7)


@pytest.mark.parametrize('tamper',('shape','dtype','non_tensor','quadrature','dt'))
def test_malformed_checkpoint_rejects_before_any_field_mutation(cuda_p2p_devices,tamper):
    from dataclasses import replace
    checkpoint=_simulation(parallel=True,copper=True,sampled=.006).prepare().run_until(47)
    tensors=dict(checkpoint.physics.tensors)
    if tamper=='shape':
        tensors['rank1/Ex']=tensors['rank1/Ex'][:-1]
    elif tamper=='dtype':
        tensors['rank1/Ex']=tensors['rank1/Ex'].double()
    elif tamper=='non_tensor':
        tensors['rank1/Ex']=1.
    malformed=replace(checkpoint,physics=replace(checkpoint.physics,tensors=tensors))
    if tamper=='dt':
        malformed=replace(malformed,dt=checkpoint.dt*1.01)
    prepared=_simulation(parallel=True,copper=True,sampled=.009 if tamper=='quadrature' else .006).prepare()
    before={(shard.rank,name):getattr(shard.solver,name).clone()
            for shard in prepared.solver.shards for name in ('Ex','Ey','Ez','Hx','Hy','Hz')}
    with pytest.raises((ValueError,TypeError),match={'shape':'shape','dtype':'dtype','non_tensor':'torch.Tensor','quadrature':'configuration','dt':'dt'}[tamper]):
        prepared.run(resume_from=malformed)
    for shard in prepared.solver.shards:
        for name in ('Ex','Ey','Ez','Hx','Hy','Hz'):
            assert torch.equal(getattr(shard.solver,name),before[shard.rank,name])


def test_full_field_dft_disk_resume_retains_global_owned_spectra(cuda_p2p_devices,tmp_path):
    options=dict(parallel=True,copper=True,full_field_dft=True)
    expected=_simulation(**options).run()
    checkpoint=_simulation(**options).prepare().run_until(41)
    checkpoint.save(tmp_path/'full.pt')
    actual=_simulation(**options).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path/'full.pt',map_location='cpu'))
    _compare_surface(actual,expected)
    for component in ('Ex','Ey','Ez'):
        torch.testing.assert_close(actual.tensor(component),expected.tensor(component),rtol=4e-5,atol=1e-7)


@pytest.mark.parametrize('kind',('circuit','delayed_network','crossing_wire'))
def test_distributed_owner_histories_and_sampled_spectra_resume_together(cuda_p2p_devices,tmp_path,kind):
    if kind=='circuit':
        from tests.fdtd.multi_gpu.test_circuit_owner import _two_shard_circuit_scene
        make_scene=_two_shard_circuit_scene
        frequency=3e9
        extent=.016
    elif kind=='delayed_network':
        from tests.fdtd.multi_gpu.test_network_owner import _delayed_two_shard_network_scene
        make_scene=_delayed_two_shard_network_scene
        frequency=3e9
        extent=.016
    else:
        from tests.fdtd.multi_gpu.test_wire_owner import _straight_crossing_scene,_FREQUENCY
        make_scene=_straight_crossing_scene
        frequency=_FREQUENCY
        extent=.08
    def simulation():
        scene=make_scene()
        surface=mw.ClosedSurfaceMonitor.box('surface',size=(2*extent,)*3,frequencies=(frequency,),sample_spacing=extent/3)
        scene=scene.clone(monitors=(*scene.monitors,surface))
        return mw.Simulation.fdtd(scene,frequencies=(frequency,),
            parallel=mw.FDTDParallelConfig(devices=cuda_p2p_devices,transport='cuda_p2p',gather_fields=False),
            cuda_graph=False,full_field_dft=False,run_time=mw.TimeConfig(time_steps=200),
            spectral_sampler=mw.SpectralSampler(window='none' if kind=='crossing_wire' else 'hanning'))
    expected=simulation().run()
    checkpoint=simulation().prepare().run_until(97)
    checkpoint.save(tmp_path/(kind+'.pt'))
    actual=simulation().prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path/(kind+'.pt'),map_location='cpu'))
    _compare_surface(actual,expected)
    for name,port in expected.ports.items():
        torch.testing.assert_close(actual.port(name).voltage,port.voltage,rtol=2e-6,atol=1e-10)
        torch.testing.assert_close(actual.port(name).current,port.current,rtol=2e-6,atol=1e-10)
    for name,circuit in expected.circuits.items():
        torch.testing.assert_close(actual.circuit(name).node_voltages,circuit.node_voltages,rtol=2e-6,atol=1e-10)
        torch.testing.assert_close(actual.circuit(name).branch_currents,circuit.branch_currents,rtol=2e-6,atol=1e-10)
    for name,network in expected.embedded_networks.items():
        for key in ('port_energy','absorbed_energy','generated_energy'):
            np.testing.assert_allclose(actual.embedded_networks[name].metadata[key],network.metadata[key],rtol=2e-6,atol=1e-20)
    if kind=='crossing_wire':
        torch.testing.assert_close(actual.monitor('wire_state').current,expected.monitor('wire_state').current,rtol=2e-6,atol=1e-10)
        torch.testing.assert_close(actual.monitor('wire_state').charge,expected.monitor('wire_state').charge,rtol=2e-6,atol=1e-10)


@pytest.mark.parametrize('sampled',(None,.005))
def test_actual_finite_copper_patch_circuit_radiation_and_resume_match_single_domain(cuda_p2p_devices,tmp_path,sampled,record_property):
    import json
    import time
    from benchmark.scenes.antenna.printed_patch import printed_patch_scene
    def simulation(parallel):
        frequency=3.2e9
        scene=printed_patch_scene(lateral_dl=.002,substrate_cells=3,air_dl=.003,
            frequencies=(frequency,),metal='copper',device='cuda:0')
        circuit=mw.Circuit('probe_feed')
        node=circuit.node('feed')
        circuit.add(mw.Resistor('R',node,circuit.ground,50.))
        circuit.add(mw.CurrentSource('I',node,circuit.ground,0.,waveform=mw.SineWaveform(0.,1e-3,frequency)))
        circuit.bind_port('feed',positive=node,negative=circuit.ground)
        radiation=scene.monitors[0]
        monitor=mw.ClosedSurfaceMonitor.box('radiation',position=tuple((lo+hi)/2 for lo,hi in radiation.bounds),
            size=tuple(hi-lo for lo,hi in radiation.bounds),
            frequencies=(frequency,),sample_spacing=sampled)
        scene=scene.clone(circuits=(circuit,),monitors=(monitor,))
        config=mw.FDTDParallelConfig(devices=cuda_p2p_devices,transport='cuda_p2p',gather_fields=False) if parallel else None
        return mw.Simulation.fdtd(scene,frequencies=(frequency,),parallel=config,cuda_graph=False,
            full_field_dft=False,run_time=mw.TimeConfig(time_steps=192),spectral_sampler=mw.SpectralSampler(window='none'))
    start=time.perf_counter()
    expected=simulation(False).run()
    single_wall=time.perf_counter()-start
    start=time.perf_counter()
    actual=simulation(True).run()
    distributed_wall=time.perf_counter()-start
    _compare_surface(actual,expected,name='radiation')
    checkpoint=simulation(True).prepare().run_until(91)
    checkpoint.save(tmp_path/'patch.pt')
    start=time.perf_counter()
    resumed=simulation(True).prepare().run(resume_from=mw.FDTDResumeCheckpoint.load(tmp_path/'patch.pt',map_location='cpu'))
    resume_wall=time.perf_counter()-start
    _compare_surface(resumed,expected,name='radiation')
    for result in (actual,resumed):
        torch.testing.assert_close(result.port('feed').voltage,expected.port('feed').voltage,rtol=5e-5,atol=1e-10)
        torch.testing.assert_close(result.port('feed').current,expected.port('feed').current,rtol=5e-5,atol=1e-10)
        for name,loss in expected.conductor_losses.items():
            torch.testing.assert_close(result.conductor_losses[name]['surface_power'],loss['surface_power'],rtol=5e-4,atol=1e-16)
    def stats(result,wall):
        parallel=getattr(result.solver,'parallel_stats',{})
        return dict(wall_including_prepare_s=wall,solve_loop_s=float(result.solver.last_solve_elapsed_s),
            resumed_from_step=parallel.get('resumed_from_step',0),executed_steps=parallel.get('executed_steps',192),
            halo_bytes_per_step=parallel.get('halo_bytes_per_step',0),halo_bytes_total=parallel.get('halo_bytes_total',0),
            observation_halo_bytes_per_step=parallel.get('observation_halo_bytes_per_step',0))
    evidence=dict(qualification='physics/state parity only; scale performance remains open',
        sample_spacing_m=sampled,planned_steps=192,single=stats(expected,single_wall),
        two_gpu=stats(actual,distributed_wall),resumed=stats(resumed,resume_wall))
    (tmp_path/'patch-execution.json').write_text(json.dumps(evidence,indent=2)+'\n')
    record_property('patch_execution',json.dumps(evidence,sort_keys=True))
