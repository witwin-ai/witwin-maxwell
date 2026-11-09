"""Actual two-polarization film spectra under analytic and real CW drives."""
import math
import pytest
import torch
import witwin.maxwell as mw

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA FDTD.')


def run_film(*,bloch):
    transverse='bloch' if bloch else 'periodic'
    material=mw.CopperFoil(thickness=2e-6,conductivity=1e4,frequency_range=(.5e9,1.5e9))
    scene=mw.Scene(domain=mw.Domain(bounds=((- .12,.12),(-.01,.01),(-.015,.015))),
        grid=mw.GridSpec.uniform(.005),device='cuda',
        boundary=mw.BoundarySpec.faces(default='none',x='pml',y=transverse,z=transverse,
            bloch_wavevector=(0.,0.,0.),num_layers=12),
        structures=(mw.Structure(name='foil',geometry=mw.Box(position=(0.,0.,0.),size=(0.,.02,.03)),material=material),),
        monitors=(mw.PointMonitor('probe',(.025,0.,0.),fields=('Ey','Ez','Hy','Hz')),))
    scene.add_source(mw.PlaneWave(direction=(1.,0.,0.),polarization=(0.,.6,.8),
        injection=mw.TFSF.slab(axis='x',bounds=(-.08,.08)),
        source_time=mw.CW(1e9,amplitude=3.,phase=.47)))
    result=mw.Simulation.fdtd(scene, absorber="cpml",frequency=1e9,full_field_dft=True,
        run_time=mw.TimeConfig.auto(transient_cycles=16,steady_cycles=12),
        spectral_sampler=mw.SpectralSampler(window='hanning',normalize_source=True)).run()
    return result,material


def test_two_polarization_copper_phasors_and_absolute_work_match_real_periodic(tmp_path,record_property):
    periodic,_=run_film(bloch=False)
    actual,material=run_film(bloch=True)
    for component in ('Ey','Ez','Hy','Hz'):
        torch.testing.assert_close(torch.as_tensor(actual.monitor('probe')[component]),
            torch.as_tensor(periodic.monitor('probe')[component]),rtol=3e-4,atol=1e-9)
    record=actual.conductor_losses['foil']
    expected=periodic.conductor_losses['foil']
    torch.testing.assert_close(record['surface_power'],expected['surface_power'],rtol=6e-4,atol=1e-15)
    assert record['energy_representation']=='combined_real_imaginary_algorithmic_state'
    assert expected['energy_representation']=='physical_real_field'
    independent=record['surface_power'].new_zeros(1)
    dt=actual.stats()['dt']
    angle=math.pi*1e9*dt
    warped=math.tan(angle)/(math.pi*dt)
    admittance=material.surface_admittance([warped]).cpu()
    traces=record['bulk_electric_traces']
    assert {trace['component'] for trace in traces}=={'Ey','Ez'}
    # The rectangular aperture gives the two native polarizations distinct
    # support lengths. Pairing the wrong imaginary bank cannot pass this test.
    assert traces[0]['indices'].shape != traces[1]['indices'].shape
    for trace in traces:
        tile=next(tile for tile in record['surface_power_traces'] if tile['component']==trace['component'])
        field=trace['electric_fields']
        independent += (.5*torch.einsum('fim,fij,fjm->fm',field.conj(),admittance,field).real*
                        tile['area']).sum(-1)*math.cos(angle)**2
    record_property('independent_modal_admittance_heat_relative_error',repr(float(
        (record['surface_power']/independent-1).abs().max())))
    torch.testing.assert_close(record['surface_power'],independent,rtol=1e-3,atol=1e-15)
    path=tmp_path/'analytic-film.pt'
    actual.save(path)
    loaded=mw.Result.load(path,scene=actual.scene,map_location='cuda:0')
    assert loaded.conductor_losses['foil']['energy_representation']==record['energy_representation']
    torch.testing.assert_close(loaded.conductor_losses['foil']['surface_power'].cpu(),record['surface_power'])
