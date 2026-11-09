"""Independent cylindrical reference geometry and coherent physical power."""
import math
import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from scripts.dev.fdtd.bend_pcb_external_reference import chart_to_cylinder,cylindrical_world,fixture
from scripts.dev.fdtd.bend_cylinder_native_control import coherent_reference_heat
from scripts.dev.fdtd.copper_cylinder_reference import shell_scattering,MU_0


@pytest.mark.parametrize('curvature',(1.,3.,5.))
def test_independent_cylindrical_fixture_has_same_physical_positions_frames_and_aperture(curvature):
    model=fixture(.01,curvature)
    points=np.array(((0.,0.,0.),(.07,.02,-.03),(-.08,-.035,.04),(.01,.01,0.)))
    chart=mw.BendCoordinates(curvature=curvature)
    np.testing.assert_allclose(cylindrical_world(chart_to_cylinder(points,curvature),curvature),
        chart.position(torch.from_numpy(points)).numpy(),rtol=2e-14,atol=3e-16)
    angles=chart_to_cylinder(points,curvature)[:,1]
    radial=np.stack((np.cos(angles),np.sin(angles),np.zeros_like(angles)),-1)
    angular=np.stack((-np.sin(angles),np.cos(angles),np.zeros_like(angles)),-1)
    J=chart.jacobian(torch.from_numpy(points)).numpy()
    np.testing.assert_allclose(J[:,:,1],-radial,rtol=2e-14,atol=3e-16)
    np.testing.assert_allclose(J[:,:,0]/(1-curvature*points[:,1,None]),angular,rtol=2e-14,atol=3e-16)
    lo,hi=model['aperture'];r=(lo[0]+hi[0])/2
    area=r*(hi[1]-lo[1])*(hi[2]-lo[2])
    assert area==pytest.approx((1-curvature*.01)*.02*.02,rel=2e-14)
    assert all(np.all(np.diff(axis)>0) for axis in model['grid'])
    assert model['grid'][0][0]>0


@pytest.mark.parametrize('phase',(0.,.5*math.pi,math.pi,1.5*math.pi))
def test_annulus_heat_reference_retains_coherent_counterpropagating_interference(phase):
    frequency,sigma=1e9,5.8e7
    thickness=math.sqrt(2/(2*math.pi*frequency*MU_0*sigma))
    scattering=shell_scattering(.3,thickness,frequency,sigma,order=1,polarization='te')
    incoming=np.array((1.,np.exp(1j*phase)))
    actual=coherent_reference_heat(scattering,np.stack((incoming.real,incoming.imag),-1))
    incident_power=np.vdot(incoming,incoming).real
    outward=scattering@incoming
    expected=(incident_power-np.vdot(outward,outward).real)/incident_power
    assert actual==pytest.approx(expected,rel=5e-11,abs=3e-16)
    assert actual>0
    if phase==0:
        diagonal_only=float(np.mean(1-(abs(scattering)**2).sum(0)))
        assert abs(actual-diagonal_only)/actual>.1


def test_declared_domain_grid_extends_boundary_cell_widths_without_padded_resampling():
    from witwin.maxwell.scene import PreparedScene
    spacing=.01;curvature=3.
    model=fixture(spacing,curvature,grid_construction='domain_then_pml')
    scene=PreparedScene(mw.Scene(domain=mw.Domain(bounds=((-0.12,.12),(-.1,.1),(-.12,.12))),
        grid=mw.GridSpec.uniform(spacing),boundary=mw.BoundarySpec.pml(num_layers=8),device='cpu'))
    np.testing.assert_array_equal(model['grid'][0],1/curvature-np.asarray(scene.y_nodes64)[::-1])
    np.testing.assert_array_equal(model['grid'][1],curvature*np.asarray(scene.x_nodes64)-math.pi/2)
    np.testing.assert_array_equal(model['grid'][2],np.asarray(scene.z_nodes64))
    nominal=fixture(spacing,curvature)
    for actual,old in zip(model['grid'],nominal['grid']):
        np.testing.assert_allclose(actual,old,rtol=0,atol=2e-15)
    for key in ('ground','patch','substrate','aperture','surface'):
        np.testing.assert_array_equal(model[key],nominal[key])


def test_ulp_canonicalization_retains_declared_grid_faces_without_finite_displacement():
    original=fixture(.01,3.,grid_construction='domain_then_pml')
    canonical=fixture(.01,3.,grid_construction='domain_then_pml',canonical_grid_faces=True)
    assert canonical['face_roundoff']
    for entry in canonical['face_roundoff']:
        assert abs(entry['delta'])<=entry['tolerance']
        assert entry['canonical'] in canonical['grid'][entry['axis']]
    for name in ('ground','patch','substrate','aperture','surface'):
        np.testing.assert_allclose(canonical[name],original[name],rtol=0,atol=3e-15)
    assert canonical['patch'][0][0]==canonical['patch'][1][0]


def test_operator_component_permutation_reverses_radial_not_angular_axis():
    from scripts.dev.fdtd.bend_material_operator_audit import reference_to_chart_edges
    values=np.indices((3,5,6,7))
    encoded=1000*values[0]+100*values[1]+10*values[2]+values[3]
    for component in range(3):
        actual=reference_to_chart_edges(encoded,component)
        external=(1,0,2)[component]
        shape=[5,6,7];shape[external]-=1
        expected_shape=(shape[1],shape[0],shape[2])
        assert actual.shape==expected_shape
        for phi in range(shape[1]):
            for u in range(shape[0]):
                for z in range(shape[2]):
                    assert actual[phi,u,z]==1000*external+100*(shape[0]-1-u)+10*phi+z


@pytest.mark.parametrize('module_name',(
    'scripts.dev.fdtd.bend_pcb_external_reference',
    'scripts.dev.fdtd.bend_pcb_matched_current_control',
    'scripts.dev.fdtd.bend_cylinder_native_control'))
def test_physical_controllers_reject_mutable_unbound_source(tmp_path,monkeypatch,module_name):
    import importlib
    module=importlib.import_module(module_name)
    path=tmp_path/'unbound'/'scripts'/'dev'/'fdtd'/'control.py'
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(module,'__file__',str(path))
    with pytest.raises(RuntimeError,match='archive'):
        if module_name.endswith('bend_pcb_external_reference'):
            module.reference_binding(tmp_path)
        else:
            module.source_binding()
