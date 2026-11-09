"""Long RF scalar-clock accuracy, independent of the field-update solver."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from witwin.maxwell.fdtd.runtime import spectral


def solver(window,steps, *, complex_fields=False):
    value=SimpleNamespace(dt=.5681529297e-15,device='cpu',verbose=False,
        Ex=torch.zeros((1,1,1)),_dft_batched_fields={},complex_fields_enabled=complex_fields,
        _normalize_source=False,_normalization_source_time=None,
        _time_plan={'response':'cw_steady_state','spectral_start_step':0})
    spectral.enable_dft(value,(2.45e9,3.2e9,10e9),window_type=window,end_step=steps,components=('Ex',))
    return value


@pytest.mark.parametrize('window',('none','hanning','ramp'))
def test_long_small_step_rf_tables_track_closed_form_unit_phasor_and_window(window):
    steps=120000;value=solver(window,steps)
    assert spectral.build_dft_step_tables(value,steps)
    n=np.arange(steps,dtype=np.float64)
    phase=2*np.pi*(n[:,None]+1)*np.array(value.dft_frequencies)[None]*value.dt
    position=n/steps
    if window=='none':weight=np.ones(steps)
    elif window=='hanning':weight=.5*(1-np.cos(2*np.pi*position))
    else:weight=np.where(position<.1,.5*(1-np.cos(np.pi*position/.1)),1.)
    real=value._dft_weighted_cos_table.numpy();imag=value._dft_weighted_sin_table.numpy()
    assert np.max(abs(real-weight[:,None]*np.cos(phase))) < 6e-8
    assert np.max(abs(imag-weight[:,None]*np.sin(phase))) < 6e-8
    assert np.max(abs(value._dft_window_normalization_values-weight.sum())) < 2e-8
    assert real.dtype==imag.dtype==np.float32


def test_host_scalar_clock_uses_same_rounded_weights_as_captured_table(monkeypatch):
    steps=512;table=solver('hanning',steps);host=solver('hanning',steps)
    assert spectral.build_dft_step_tables(table,steps)
    rows=[]
    monkeypatch.setattr(spectral,'_launch_dft_accumulation',lambda _,real,imag,**kwargs:rows.append((real.numpy().copy(),imag.numpy().copy())))
    for n in range(steps):spectral.accumulate_dft(host,n)
    real=np.stack([row[0] for row in rows]);imag=np.stack([row[1] for row in rows])
    assert np.array_equal(real,table._dft_weighted_cos_table.numpy())
    assert np.array_equal(imag,table._dft_weighted_sin_table.numpy())
    np.testing.assert_array_equal(host._dft_window_normalization_values,table._dft_window_normalization_values)


def test_complex_host_clock_remains_unit_length_at_late_rf_phase(monkeypatch):
    value=solver('none',1000000,complex_fields=True)
    late=800000
    phase=2*np.pi*np.array(value.dft_frequencies)*late*value.dt
    value._dft_phase_cos=np.cos(phase);value._dft_phase_sin=np.sin(phase)
    monkeypatch.setattr(spectral,'_launch_dft_accumulation',lambda *args,**kwargs:None)
    for n in range(late,late+2048):spectral.accumulate_dft(value,n)
    actual=value._dft_phase_cos+1j*value._dft_phase_sin
    expected=np.exp(2j*np.pi*np.array(value.dft_frequencies)*(late+2048)*value.dt)
    assert np.max(abs(actual-expected)) < 5e-12


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA FDTD required')
@pytest.mark.parametrize('graph',(False,True))
def test_actual_full_field_and_native_point_spectra_share_long_cw_clock(graph):
    import witwin.maxwell as mw
    scene=mw.Scene(domain=mw.Domain(bounds=((-0.2,0.2),)*3),grid=mw.GridSpec.uniform(.05),
        boundary=mw.BoundarySpec.pml(num_layers=6),device='cuda',
        sources=(mw.PointDipole(position=(-.1,0.,0.),polarization='Ex',width=.05,source_time=mw.CW(1e9)),),
        monitors=(mw.PointMonitor('native',position=(.075,0.,0.),fields=('Ex',)),))
    result=mw.Simulation.fdtd(scene,frequency=1e9,run_time=mw.TimeConfig(time_steps=8192),
        spectral_sampler=mw.SpectralSampler(window='hanning'),full_field_dft=True,cuda_graph=graph).run()
    payload=result.monitor('native');index=tuple(payload['field_indices']['Ex'])
    full=result.field('Ex').reshape((-1,*result.solver.Ex.shape))[0][index]
    point=payload['components']['Ex'].reshape(())
    assert float(point.abs()) > 1e-8
    torch.testing.assert_close(full,point.to(full),rtol=8e-6,atol=1e-9)
