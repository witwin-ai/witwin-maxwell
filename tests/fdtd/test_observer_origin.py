"""Physical Yee labels: the approved independent observer-origin design gate."""
import cmath, math
import numpy as np
import pytest
from tests.support.observer_origin_probe import run, port_table_probe, port_unit_probe

@pytest.fixture(scope='module')
def records():
    return [run(kind,normalized=normalized) for kind in ('J','M') for normalized in (False,True)]

def test_revision_recurrence_control(records):
    for row in records:
        for measurement in row['measurements'].values():
            assert min(measurement['current_relative_error'],measurement['physical_relative_error']) < 3e-6

@pytest.mark.parametrize('kind',['J','M'])
@pytest.mark.parametrize('normalized',[False,True])
def test_physical_origin_recurrence(records,kind,normalized):
    row=next(r for r in records if r['kind']==kind and r['normalized']==normalized)
    for measurement in row['measurements'].values():
        assert measurement['physical_relative_error'] < 3e-6

def test_N1_origin_independent_of_source():
    row=run('M',1)
    theta=2*math.pi*row['frequency']*row['dt']
    for comp,shift in [('Ex',1.),('Hy',.5)]:
        ratio=complex(*row['origin'][comp]['ratio_to_twice_final'])
        assert abs(ratio-cmath.exp(1j*theta*shift)) < 2e-14

def test_port_midpoint_origin_table():
    row=port_table_probe()['payload']
    for key in ('voltage_weights','current_weights'):
        assert np.max(np.abs(np.asarray(row[key])-np.asarray(row['expected_physical_weights']))) < 2e-14

def test_exact_physical_step_tables():
    from tests.support.observer_origin_probe import mw, torch
    scene=mw.Scene(domain=mw.Domain(bounds=((0.,.1),(0.,.1),(0.,.4))),grid=mw.GridSpec.uniform(.05),device='cpu',boundary=mw.BoundarySpec.pec())
    scene.add_source(mw.PointDipole(position=(.025,.05,.1),polarization='Ex',source_time=mw.CW(frequency=2e9)))
    s=mw.Simulation.fdtd(scene,frequencies=(2e9,),backend='cpu_reference',cuda_graph=False,full_field_dft=False,run_time=mw.TimeConfig(time_steps=7,dt=9.5238e-12),spectral_sampler=mw.SpectralSampler(window='none',normalize_source=False)).prepare().solver
    fs=(1.5e9,2e9,2.2e9)
    s.enable_dft(fs,window_type='none',end_step=7,components=('Ex',))
    assert s.build_dft_step_tables(7)
    angle=2*np.pi*np.asarray(fs,dtype=np.float64)*s.dt
    dc=np.cos(angle);ds=np.sin(angle)
    c=dc.copy();si=ds.copy()
    ec=np.empty((7,3),dtype=np.float32);es=np.empty((7,3),dtype=np.float32)
    for n in range(7):
        ec[n]=c;es[n]=si
        c,si=c*dc-si*ds,si*dc+c*ds
    assert torch.equal(s._dft_weighted_cos_table,torch.from_numpy(ec))
    assert torch.equal(s._dft_weighted_sin_table,torch.from_numpy(es))

@pytest.fixture(scope='module')
def port_units():
    return port_unit_probe()

def test_port_source_circuit_and_curl_control(port_units):
    p=port_units
    for actual,expected in [(p['records'][0]['I'],p['independent_N1_I']),(p['records'][0]['V'],p['independent_N1_V']),(p['records'][0]['E'],p['independent_N1_E']),(p['records'][1]['H'],p['independent_N2_H'])]:
        assert abs(actual-expected)/abs(expected) < 3e-6

def test_port_driven_VI_physical_origin(port_units):
    p=port_units
    for key in ('first_V_ratio','first_I_ratio'):
        assert abs(complex(*p[key])-cmath.exp(.5j*p['theta'])) < 2e-14

def test_port_driven_EH_physical_origin(port_units):
    p=port_units;one,two=p['records']
    e=complex(*one['reported_fields']['Ex'])/(2*one['E'])
    h=complex(*two['reported_fields']['Hy'])/two['H']
    assert abs(e-cmath.exp(1j*p['theta'])) < 2e-14
    assert abs(h-cmath.exp(1.5j*p['theta'])) < 2e-14
