"""Printed-line geometry and independent TEM reference contracts."""
import numpy as np
import pytest
import torch
from benchmark.scenes.rf.printed_lines import printed_line_scene,analytic_homogeneous_line,KINDS,C0


@pytest.mark.parametrize("kind",KINDS)
@pytest.mark.parametrize("dx",[.0025,.00125,.000625])
def test_printed_line_refinement_keeps_physical_conductors_and_layers_fixed(kind,dx):
    baseline=printed_line_scene(kind,device="cpu")
    actual=printed_line_scene(kind,dx=dx,device="cpu")
    assert actual.domain.bounds==baseline.domain.bounds
    assert actual.metadata["pml_physical_thickness_m"]==baseline.metadata["pml_physical_thickness_m"]
    for a,b in zip(actual.structures,baseline.structures):
        assert a.name==b.name
        for name in ("position","size","vertices"):
            if hasattr(a.geometry,name):
                assert torch.equal(getattr(a.geometry,name),getattr(b.geometry,name))
        if hasattr(a.geometry,"coordinate"):
            assert a.geometry.coordinate==b.geometry.coordinate
            assert a.geometry.physical_thickness==35e-6
    for a,b in zip(actual.ports,baseline.ports):
        assert a.position==b.position and a.reference_plane==b.reference_plane and a.size==b.size
        assert [m.voltage_path for m in a.modes]==[m.voltage_path for m in b.modes]


@pytest.mark.parametrize("kind",["stripline","cpw"])
def test_independent_homogeneous_reference_has_exact_tem_speed_and_passive_loss(kind):
    frequency=np.array([1e9,1.5e9,2e9])
    reference=analytic_homogeneous_line(kind,frequency,eps_r=2.25)
    np.testing.assert_allclose(reference["beta_rad_per_m"],2*np.pi*frequency*1.5/C0,rtol=1e-9)
    assert np.all(reference["alpha_np_per_m"]==0)
    lossy=analytic_homogeneous_line(kind,frequency,eps_r=2.25,sigma_e=.001)
    assert np.all(lossy["alpha_np_per_m"]>0)
    assert np.all(lossy["characteristic_impedance_ohm"].real>0)
    assert np.all(lossy["characteristic_impedance_ohm"].imag<0)
    assert np.all(lossy["beta_rad_per_m"]>reference["beta_rad_per_m"])


@pytest.mark.parametrize("dx",[.00125,.000625,.0003125])
def test_layered_transverse_refinement_retains_axial_source_and_physical_pml(dx):
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.ports import compile_port_geometry
    scene=printed_line_scene("cpw",dx=dx,longitudinal_dl=.0025,device="cpu")
    reference=printed_line_scene("cpw",device="cpu")
    for a,b in zip(scene.structures,reference.structures):
        for name in ("position","size","vertices"):
            if hasattr(a.geometry,name):
                assert torch.equal(getattr(a.geometry,name),getattr(b.geometry,name))
    prepared=prepare_scene(scene)
    assert prepared.Nx < 190
    for axis,half in (("x",.060),("y",.060),("z",.050)):
        values=getattr(prepared,axis+"_nodes64")
        assert np.isclose(values[0],-half-.020,atol=1e-12)
        assert np.isclose(values[-1],half+.020,atol=1e-12)
    port=scene.ports[0]
    contour=port.modes[0].current_contour
    assert np.isclose(float(contour.position[0]),-.020+.00125)
    assert port.reference_plane==-.020


def test_asymmetric_pair_paddle_leaves_both_modal_reference_cross_sections_fixed():
    symmetric=printed_line_scene("differential_pair",device="cpu")
    asymmetric=printed_line_scene("differential_pair",pair_perturbation=True,device="cpu")
    assert len(asymmetric.structures)==len(symmetric.structures)+1
    paddle=asymmetric.structures[-1]
    assert paddle.name=="positive_signal_paddle"
    assert float(paddle.geometry.vertices[:,0].abs().max()) < abs(symmetric.ports[0].position[0])
    for a,b in zip(symmetric.ports,asymmetric.ports):
        assert a.position==b.position and a.reference_plane==b.reference_plane
        for ma,mb in zip(a.modes,b.modes):
            assert ma.voltage_path==mb.voltage_path
            assert torch.equal(ma.current_contour.position,mb.current_contour.position)
            assert torch.equal(ma.current_contour.size,mb.current_contour.size)


@pytest.mark.parametrize("kind",KINDS)
def test_graded_outer_mesh_keeps_physical_geometry_native_paths_and_pml_fixed(kind):
    from witwin.maxwell.scene import prepare_scene
    from witwin.maxwell.compiler.waveports import compile_waveport_cross_section
    uniform=printed_line_scene(kind,dx=.000625,longitudinal_dl=.0025,device="cpu")
    graded=printed_line_scene(kind,dx=.000625,longitudinal_dl=.0025,
                              transverse_refinement_radius=.020,device="cpu")
    a,b=prepare_scene(uniform),prepare_scene(graded)
    np.testing.assert_array_equal(a.x_nodes64,b.x_nodes64)
    for axis in "yz":
        fine,coarse=[np.asarray(getattr(v,axis+"_nodes64")) for v in (a,b)]
        np.testing.assert_array_equal(fine[abs(fine)<=.020+1e-12],coarse[abs(coarse)<=.020+1e-12])
        np.testing.assert_allclose(np.diff(coarse)[[0,-1]],.0025,atol=1e-12)
        assert len(coarse)<len(fine)
    for x,y in zip(uniform.structures,graded.structures):
        for key in ("position","size","vertices"):
            if hasattr(x.geometry,key):assert torch.equal(getattr(x.geometry,key),getattr(y.geometry,key))
    for x,y in zip(uniform.ports,graded.ports):
        assert x.position==y.position and x.size==y.size and x.reference_plane==y.reference_plane
        ca,cb=[compile_waveport_cross_section(s,p,device="cpu") for s,p in ((a,x),(b,y))]
        for ma,mb in zip(ca.modes,cb.modes):
            np.testing.assert_array_equal(ma.voltage_weights,mb.voltage_weights)
            for va,vb in zip(ma.current_weights,mb.current_weights):np.testing.assert_array_equal(va,vb)
            assert abs(float(ma.voltage_weights.sum()))>0


@pytest.mark.parametrize("radius",[.010,.0201,.050,float("nan")])
def test_invalid_graded_transverse_radius_is_explicit(radius):
    with pytest.raises(ValueError,match="transverse_refinement_radius"):
        printed_line_scene("cpw",dx=.000625,longitudinal_dl=.0025,
                           transverse_refinement_radius=radius,device="cpu")


@pytest.mark.parametrize("separation",[.40,.48])
def test_long_line_two_length_control_preserves_cross_section_and_pml_continuation(separation):
    from witwin.maxwell.scene import prepare_scene
    scene=printed_line_scene("cpw",port_separation=separation,axial_half_length=.30,device="cpu")
    prepared=prepare_scene(scene)
    assert np.isclose(prepared.x_nodes64[0],-.320,atol=1e-12)
    assert np.isclose(prepared.x_nodes64[-1],.320,atol=1e-12)
    assert scene.ports[0].reference_plane==-separation/2
    for structure in scene.structures:
        if hasattr(structure.geometry,"vertices"):
            assert structure.geometry.vertices[:,0].min() < prepared.x_nodes64[0]
            assert structure.geometry.vertices[:,0].max() > prepared.x_nodes64[-1]
    signal=next(s for s in scene.structures if s.name=="signal")
    assert torch.equal(signal.geometry.vertices[:,1].unique(),torch.tensor([-.005,.005],dtype=torch.float64))


@pytest.mark.parametrize("kind",KINDS)
def test_finite_copper_line_keeps_the_actual_foil_and_explicit_approximate_launch(kind):
    scene=printed_line_scene(kind,metal="copper",device="cpu")
    reference=printed_line_scene(kind,device="cpu")
    for foil,ideal in zip(scene.structures[1:],reference.structures[1:]):
        assert foil.material.is_copper_foil
        assert foil.material.thickness==35e-6
        assert not getattr(foil.material,"is_pec",False)
        assert float(foil.geometry.size[2])==0.
        assert float(foil.geometry.position[2])==ideal.geometry.coordinate
        assert foil.geometry.position.dtype==torch.float64
        assert float(foil.geometry.size[1])==pytest.approx(float(ideal.geometry.vertices[:,1].max()-ideal.geometry.vertices[:,1].min()))
    assert all(mode.conductor_model=="good_conductor" for port in scene.ports for mode in port.modes)
