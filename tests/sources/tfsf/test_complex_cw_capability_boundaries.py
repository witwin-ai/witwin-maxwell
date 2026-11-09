"""Actual preparation rejects execution combinations lacking complex coupling."""
import pytest
import torch
import witwin.maxwell as mw

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA scene preparation.')


def scene():
    return mw.Scene(domain=mw.Domain(bounds=((- .03,.03),)*3),grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.bloch((0.,5.,-7.)),device='cuda:0',
        sources=(mw.PointDipole(name='drive',position=(0.,0.,0.),polarization='Ez',
            source_time=mw.CW(3e9,phase=.47)),))


def simulation(problem):
    return mw.Simulation.fdtd(problem,frequency=3e9,run_time=mw.TimeConfig(time_steps=20),
        spectral_sampler=mw.SpectralSampler(window='none'))


def test_complex_cw_native_batch_remains_explicitly_unsupported():
    batch=mw.Simulation.batch((simulation(scene()),),
        execution=mw.BatchExecution(devices=('cuda:0',),batch_size=1,graph='off'))
    with pytest.raises(NotImplementedError,match='complex_fields_enabled'):
        batch.prepare()


def test_complex_cw_lumped_branch_allows_passive_termination_but_rejects_drive():
    problem=scene()
    problem.add_port(mw.LumpedPort(name='feed',positive=(-.01,0.,.005),negative=(-.01,0.,-.005),
        voltage_path=mw.AxisPath('z'),current_surface=mw.Box(position=(-.01,0.,-.0025),size=(.015,.015,0.)),
        termination=mw.SeriesRLC(r=50.,l=1e-9,c=1e-12)))
    prepared = simulation(problem).prepare()
    assert prepared.solver._port_runtimes[0].lumped.inductor_current_imag.item() == 0.
    driven = mw.Simulation.fdtd(problem,frequency=3e9,run_time=mw.TimeConfig(time_steps=20),
        excitations=mw.PortExcitation('feed'))
    with pytest.raises(NotImplementedError,match='Driven lumped port.*v2'):
        driven.prepare()


def test_complex_cw_spatial_execution_requires_complex_halo_exchange():
    distributed=mw.Simulation.fdtd(scene(),frequency=3e9,run_time=mw.TimeConfig(time_steps=20),
        parallel=mw.FDTDParallelConfig(devices=('cuda:0','cuda:1'),result_device='cuda:0',transport='cuda_p2p'))
    # The semantic guard runs before hardware preflight or shard allocation;
    # this test requires only its scene's one CUDA device.
    with pytest.raises(ValueError,match='complex halo exchange'):
        distributed.prepare()
