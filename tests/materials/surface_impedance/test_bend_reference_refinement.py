"""Independent comparison contracts reject fitted or mismatched controls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).parents[3]/'scripts/dev/fdtd/bend_reference_refinement.py'
spec=importlib.util.spec_from_file_location('reference_trend',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(spacing):
    theta=np.linspace(0,np.pi,17);phi=np.linspace(0,2*np.pi,33)
    wave=np.sin(theta)[:,None]*np.ones((1,len(phi)))
    report=dict(schema='independent-cylindrical-bent-pcb-reference/1',
        source_binding_before={'pin':'fixed'},source_binding_after={'pin':'fixed'},
        execution_driver_sha256='a',analysis_driver_sha256='a',current=dict(real=2.,imag=0.),
        E_theta=dict(real=wave.tolist(),imag=(wave*.3).tolist()),E_phi=dict(real=(wave*0).tolist(),imag=(wave*0).tolist()),
        source_normalized_radial_flux=[-1.,-1.],theta=theta.tolist(),phi=phi.tolist(),
        curvature_m_inverse=3.,frequency_hz=1e9,duration_s=96e-9,metal_model='pec',known_current_peak_a=.001,
        phase_center_world_m=[0.,0.,0.],reference_phase_center_cylindrical=[1/3,-np.pi/2,0.],
        source_identity={'head':'fixed'},spacing_m=spacing,radiated_spectral_w_s2=4.,directivity=1.5,
        impedance=dict(real=50.,imag=5.),raw_input_manifest=dict(source='matched-current',metal='pec',vacuum=False,
            current_amplitude=.001,duration=96e-9,field_dft_oversampling=8,grid_construction='domain_then_pml',
            canonical_grid_faces=True,source_helper_sha256='source'))
    return report


def test_known_current_normalization_does_not_fit_amplitude_or_phase():
    coarse=fixture(.01);fine=fixture(.005)
    fine['current']['real']=4.
    for key in ('E_theta','E_phi'):
        for part in ('real','imag'):fine[key][part]=(2*np.asarray(fine[key][part])).tolist()
    fine['radiated_spectral_w_s2']=16.
    match=module.compare_pair(coarse,fine)
    assert match['complex_radiation_transfer_relative_l2_change']<1e-15
    assert match['radiated_power_per_amp_squared_relative_change']==0
    matched=deepcopy(fine)
    # A physical absolute amplitude change must remain visible.
    fine['E_theta']['real']=(1.01*np.asarray(fine['E_theta']['real'])).tolist()
    changed=module.compare_pair(coarse,fine)
    assert changed['complex_radiation_transfer_relative_l2_change']>.009
    for key in ('E_theta','E_phi'):
        field=module.complex_value(matched[key])*np.exp(.03j)
        matched[key]=dict(real=field.real.tolist(),imag=field.imag.tolist())
    phase=module.compare_pair(coarse,matched)['complex_radiation_transfer_relative_l2_change']
    np.testing.assert_allclose(phase,2*np.sin(.03/2),rtol=1e-13)


@pytest.mark.parametrize('reason',('source_mutation','physical_source','current_balance'))
def test_reject_unpaired_or_unqualified_reference(reason):
    coarse=fixture(.01);fine=deepcopy(fixture(.005))
    if reason=='source_mutation':fine['source_binding_after']['pin']='changed'
    elif reason=='physical_source':fine['raw_input_manifest']['current_amplitude']=.002
    else:fine['source_normalized_radial_flux'][0]=-.9
    with pytest.raises(ValueError):module.compare_pair(coarse,fine)
