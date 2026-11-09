"""CPU preparation for fixed-normal dipole observations, without propagation."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.compiler.monitors import compile_fdtd_observers
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.fdtd.observers import resolve_plane_observer
from witwin.maxwell.scene import prepare_scene
from tests._fresh_interpreter import run_without_cuda

PATH=Path(__file__).resolve().parents[3]/'docs/assessments/printed-dipole-2026-10-22/validate.py'
if not PATH.exists():
    pytest.skip('repository evidence not in this checkout: docs/assessments/printed-dipole-2026-10-22/validate.py',allow_module_level=True)
spec=importlib.util.spec_from_file_location('fixed_dipole_validation',PATH)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
FREQUENCIES=(2.2e9,3e9,3.8e9)


@pytest.mark.parametrize('metal',['pec','copper'])
def test_native_planes_retain_real_dipole_geometry_and_cover_all_compiled_sigma(metal):
    run_without_cuda(__file__,'_check_native_planes',metal)


def _check_native_planes(metal):
    config=dict(lateral_dl=.001,substrate_cells=4,metal=metal,dielectric_sigma=.001)
    scene,schema,bounds=driver.loss_observation_geometry(config,FREQUENCIES,'native_planes')
    original=driver.printed_dipole_scene(device='cpu',frequencies=FREQUENCIES,**config)
    assert bounds==scene.domain.bounds==original.domain.bounds
    for name in ('name','positive_terminal','negative_terminal','reference_plane','reference_impedance'):
        assert getattr(scene.ports[0],name)==getattr(original.ports[0],name)
    assert scene.ports[0].integration_path.axis==original.ports[0].integration_path.axis
    for name in ('voltage_aperture','current_surface'):
        a,b=[getattr(s.ports[0],name) for s in (scene,original)]
        torch.testing.assert_close(a.position,b.position,rtol=0,atol=0)
        torch.testing.assert_close(a.size,b.size,rtol=0,atol=0)
    for axis in 'xyz':np.testing.assert_array_equal(getattr(scene.grid,axis+'_coords'),getattr(original.grid,axis+'_coords'))
    assert scene.monitors[:len(original.monitors)]==original.monitors
    assert not any(isinstance(monitor,mw.PointMonitor) for monitor in scene.monitors)
    monitor=next(m for m in scene.monitors if m.name=='line_roi_bulk')
    assert monitor.frequencies is None and monitor.quadrature=='physical_intersection'
    prepared=prepare_scene(scene);compiled=compile_power_loss_monitor(prepared,monitor)
    records={plane['name']:plane for selection in schema.values() for plane in selection['planes']}
    observers={r['monitor_name']:r for r in compile_fdtd_observers(scene) if r['monitor_name'] in records}
    assert set(observers)==set(records)
    for name,record in records.items():
        samples,_=resolve_plane_observer(prepared,observers[name])
        assert samples==({'plane_index':record['normal_index'],'plane_position':record['normal_position_m'],'weight':1.},)
    for component,selection in schema.items():
        expected=torch.zeros_like(compiled.component_masks[component])
        expected[compiled.component_masks[component]]=compiled.conductivity[component]!=0
        needed=set(torch.nonzero(expected.any(0).any(0)).flatten().tolist())
        assert needed.issubset({plane['normal_index'] for plane in selection['planes']})
        assert selection['include_flux'] is False


@pytest.mark.parametrize('observation,sigma',[('full',.001),('full',0.),('native_planes',0.)])
def test_default_and_lossless_paths_require_no_extra_observers(observation,sigma):
    assert driver.loss_observation_geometry({'dielectric_sigma':sigma},FREQUENCIES,observation)==(None,None,None)


def test_unknown_loss_observation_is_rejected():
    with pytest.raises(ValueError,match='Loss observation'):
        driver.loss_observation_geometry({},FREQUENCIES,'interpolated')


@pytest.mark.parametrize('case',['medium_air05_pml48_final','medium_air05_pml48_time16',
    'medium_air05_pml48_z16','medium_air05_pml48_y025','medium_air05_pml48_mesh_fine',
    'medium_air05_pml48_mesh_finer','medium_air05_pml48_sigma','medium_air05_pml48_combined'])
def test_new_queue_configurations_retain_exterior_normal_samples_and_finite_feed(case):
    config,duration=driver.case_configuration(case,FREQUENCIES)
    reference,_=driver.case_configuration('medium_air05_pml48_final',FREQUENCIES)
    actual=driver.printed_dipole_scene(device='cpu',frequencies=FREQUENCIES,**config)
    original=driver.printed_dipole_scene(device='cpu',frequencies=FREQUENCIES,**reference)
    a,b=[np.asarray(scene.grid.z_coords) for scene in (actual,original)]
    np.testing.assert_array_equal(a[abs(a)>.002],b[abs(b)>.002])
    for field in ('voltage_aperture','current_surface'):
        x,y=[getattr(scene.ports[0],field) for scene in (actual,original)]
        torch.testing.assert_close(x.position,y.position,rtol=0,atol=0)
        torch.testing.assert_close(x.size,y.size,rtol=0,atol=0)
    assert actual.domain.bounds==original.domain.bounds
    assert duration==(16e-9 if case.endswith('time16') else 12e-9)


@pytest.mark.parametrize('level,cells,step',[('fine',16,.00025),('finer',32,.000125)])
@pytest.mark.parametrize('material,metal,sigma',[('copper','copper',0.),('sigma','pec',.001),('combined','copper',.001)])
def test_fine_material_controls_use_the_declared_joint_mesh(level,cells,step,material,metal,sigma):
    config,duration=driver.case_configuration(f'medium_air05_pml48_mesh_{level}_{material}',FREQUENCIES)
    assert config['lateral_dl']==step and config['coupon_y_dl']==step and config['substrate_cells']==cells
    assert config.get('metal','pec')==metal and config.get('dielectric_sigma',0.)==sigma
    assert duration==12e-9
