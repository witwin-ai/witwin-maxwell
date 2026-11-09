"""Absolute pure-CW complex-field amplitudes against real periodic controls."""
import functools
import cmath
import math
import pytest
import torch
import witwin.maxwell as mw

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires native CUDA FDTD.')
FREQUENCY=1e9
AMPLITUDE=3.
PHASE=.47


def _rotate(values, axis):
    result = [None] * 3
    for index, value in enumerate(values):
        result[(index + axis) % 3] = value
    return tuple(result)


@functools.lru_cache(maxsize=None)
def measure(source_kind='plane',*,bloch=False,normalized=False,phase=PHASE,axis=0,direction=1,pml=12):
    transverse='bloch' if bloch else 'periodic'
    normal='xyz'[axis]
    e_name='E'+'xyz'[(axis+2)%3]
    h_name='H'+'xyz'[(axis+1)%3]
    boundary={name: 'pml' if name == normal else transverse for name in 'xyz'}
    point=_rotate((direction*.025,0.,0.),axis)
    plane_fields=tuple(kind+name for kind in 'EH' for name in 'xyz' if name != normal)
    scene=mw.Scene(domain=mw.Domain(bounds=_rotate(((- .12,.12),(-.01,.01),(-.01,.01)),axis)),
        grid=mw.GridSpec.uniform(.005),device='cuda',
        boundary=mw.BoundarySpec.faces(default='none',**boundary,
            bloch_wavevector=(0.,0.,0.),num_layers=pml),
        monitors=(mw.PointMonitor('probe',point,fields=(e_name,h_name)),
                  mw.FinitePlaneMonitor('power',position=point,size=_rotate((0.,.01,.01),axis),
                      fields=plane_fields,compute_flux=True)))
    if source_kind=='plane':
        source=mw.PlaneWave(direction=_rotate((float(direction),0.,0.),axis),polarization=_rotate((0.,0.,1.),axis),
            injection=mw.TFSF.slab(axis=normal,bounds=(-.08,.08)),
            source_time=mw.CW(FREQUENCY,amplitude=AMPLITUDE,phase=phase))
    else:
        source=mw.UniformCurrentSource(center=_rotate((-direction*.04,0.,-.0025),axis),size=(.005,)*3,polarization=e_name,
            source_time=mw.CW(FREQUENCY,amplitude=1e-4,phase=phase))
    scene.add_source(source)
    result=mw.Simulation.fdtd(scene, absorber="cpml",frequency=FREQUENCY,full_field_dft=True,
        run_time=mw.TimeConfig.auto(transient_cycles=16,steady_cycles=12),
        spectral_sampler=mw.SpectralSampler(window='hanning',normalize_source=normalized)).run()
    return result


@pytest.mark.parametrize('normalized',(False,True))
@pytest.mark.parametrize('source_kind',('plane','current'))
def test_zero_bloch_cw_matches_real_periodic_amplitude_and_power(source_kind,normalized,record_property):
    periodic=measure(source_kind,bloch=False,normalized=normalized)
    complex_result=measure(source_kind,bloch=True,normalized=normalized)
    for component in ('Ez','Hy'):
        reference=torch.as_tensor(periodic.monitor('probe')[component])
        actual=torch.as_tensor(complex_result.monitor('probe')[component])
        record_property(component+'_ratio',repr(complex((actual/reference).reshape(()))))
        torch.testing.assert_close(actual,reference,rtol=2e-4,atol=1e-10)
    reference=torch.as_tensor(periodic.monitor('power')['flux'])
    actual=torch.as_tensor(complex_result.monitor('power')['flux'])
    record_property('power_ratio',repr(float((actual/reference).reshape(()))))
    torch.testing.assert_close(actual,reference,rtol=4e-4,atol=1e-14)
    if source_kind=='plane':
        amplitude=torch.as_tensor(complex_result.monitor('probe')['Ez']).abs().reshape(())
        expected = AMPLITUDE
        torch.testing.assert_close(amplitude,amplitude.new_tensor(expected),rtol=3e-3,atol=1e-5)
        physical_power = AMPLITUDE**2 / (2*376.730313668) * .01**2
        torch.testing.assert_close(actual,actual.new_tensor(physical_power),rtol=3e-3,atol=1e-14)
        dt=complex_result.stats()['dt']
        k=2*math.asin(math.sin(math.pi*FREQUENCY*dt)*.005/(299792458.*dt))/.005
        expected_phasor=AMPLITUDE*cmath.exp(1j*(k*.025-PHASE))
        measured=torch.as_tensor(complex_result.monitor('probe')['Ez'])
        record_property('absolute_center_referenced_E_ratio',repr(complex(measured/expected_phasor)))
        torch.testing.assert_close(measured,measured.new_tensor(expected_phasor),rtol=2e-4,atol=1e-8)


@pytest.mark.parametrize('normalized',(False,True))
def test_analytic_plane_declared_phase_rotates_fields_without_changing_power(normalized):
    baseline=measure('plane',bloch=True,normalized=normalized,phase=0.)
    shifted=measure('plane',bloch=True,normalized=normalized,phase=PHASE)
    for component in ('Ez','Hy'):
        reference=torch.as_tensor(baseline.monitor('probe')[component])
        actual=torch.as_tensor(shifted.monitor('probe')[component])
        torch.testing.assert_close(actual,reference*cmath.exp(-1j*PHASE),rtol=2e-5,atol=1e-9)
    torch.testing.assert_close(torch.as_tensor(shifted.monitor('power')['flux']),
        torch.as_tensor(baseline.monitor('power')['flux']),rtol=2e-5,atol=1e-14)


@pytest.mark.parametrize('axis',range(3))
@pytest.mark.parametrize('direction',(-1,1))
@pytest.mark.parametrize('pml',(8,16))
def test_cw_injection_center_phase_is_independent_of_padded_seed_location(axis,direction,pml,record_property):
    reference=measure('plane',bloch=True,axis=axis,direction=direction,pml=12)
    actual=measure('plane',bloch=False,axis=axis,direction=direction,pml=pml)
    for component in ('E'+'xyz'[(axis+2)%3], 'H'+'xyz'[(axis+1)%3]):
        expected=torch.as_tensor(reference.monitor('probe')[component])
        measured=torch.as_tensor(actual.monitor('probe')[component])
        record_property(component+'_ratio',repr(complex((measured/expected).reshape(()))))
        torch.testing.assert_close(measured,expected,rtol=2e-4,atol=1e-9)
    torch.testing.assert_close(torch.as_tensor(actual.monitor('power')['flux']),
        torch.as_tensor(reference.monitor('power')['flux']),rtol=4e-4,atol=1e-14)
    dt=actual.stats()['dt']
    k=2*math.asin(math.sin(math.pi*FREQUENCY*dt)*.005/(299792458.*dt))/.005
    electric_name='E'+'xyz'[(axis+2)%3]
    magnetic_name='H'+'xyz'[(axis+1)%3]
    measured=torch.as_tensor(actual.monitor('probe')[electric_name])
    expected=AMPLITUDE*cmath.exp(1j*(k*.025-PHASE))
    torch.testing.assert_close(measured,measured.new_tensor(expected),rtol=2e-4,atol=1e-8)
    from witwin.maxwell.fdtd.coords import component_coords
    coordinates=component_coords(actual.prepared_scene,magnetic_name)
    native_index=actual.monitor('probe')['field_indices'][magnetic_name][axis]
    position=float(coordinates[axis][native_index])
    measured=torch.as_tensor(actual.monitor('probe')[magnetic_name])
    expected=-direction*AMPLITUDE/376.730313668*cmath.exp(1j*(k*direction*position-PHASE))
    torch.testing.assert_close(measured,measured.new_tensor(expected),rtol=2e-4,atol=1e-10)
