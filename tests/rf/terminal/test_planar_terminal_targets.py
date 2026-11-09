"""Named planar terminals retain actual polygon topology and native contacts."""
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import compile_port_geometry
from witwin.maxwell.scene import prepare_scene


def _box(position, size):
    return mw.Box(position=torch.tensor(position,dtype=torch.float64),
                  size=torch.tensor(size,dtype=torch.float64))


def _sheet(name,axis,coordinate,metal="pec",holes=()):
    if metal=="copper":
        position=[0.,0.,0.];position[axis]=coordinate
        size=[.006,.006,.006];size[axis]=0.
        geometry=_box(position,size)
        material=mw.CopperFoil(thickness=35e-6,frequency_range=(.1e9,3.e9))
    else:
        geometry=mw.PlanarPEC.rectangle(((-.003,.003),(-.003,.003)),
            normal_axis="xyz"[axis],coordinate=coordinate,holes=holes)
        material=mw.Material.pec()
    return mw.Structure(geometry,material,name=name)


def _scene(axis=2,metal="pec",*,holes=(),aperture=True,extras=(),device="cpu",center=None):
    center=np.zeros(3) if center is None else np.array(center,dtype=float)
    center[axis]=.0005
    size=np.full(3,.005);size[axis]=0.
    ap_size=np.full(3,.002);ap_size[axis]=.004
    ap_position=center.copy();ap_position[axis]=0.
    port=mw.TerminalPort("feed",mw.TerminalRef("signal"),mw.TerminalRef("ground"),
        mw.AxisPath("xyz"[axis]),.0005,current_surface=_box(center,size),
        voltage_aperture=_box(ap_position,ap_size) if aperture else None)
    return mw.Scene(domain=mw.Domain(bounds=((- .008,.008),)*3),
        grid=mw.GridSpec.uniform(.001),boundary=mw.BoundarySpec.none(),device=device,
        structures=(_sheet("signal",axis,.002,metal,holes),_sheet("ground",axis,-.002,metal),*extras),
        ports=(port,))


@pytest.mark.parametrize("axis",range(3))
@pytest.mark.parametrize("metal",("pec","copper"))
def test_three_axis_planar_named_targets_compile_real_parallel_contacts(axis,metal):
    scene=prepare_scene(_scene(axis,metal))
    compiled=compile_port_geometry(scene,scene.ports[0],device="cpu")
    assert compiled.metadata["positive_contact_structure"]=="signal"
    assert compiled.metadata["native_parallel_path_count"]==9
    assert compiled.metadata["native_voltage_edge_count"]==36
    assert "disjoint" in compiled.metadata["native_connectivity"]


@pytest.mark.parametrize("axis",range(3))
def test_center_polygon_hole_is_not_a_named_contact(axis):
    hole=((-.0005,-.0005),(.0005,-.0005),(.0005,.0005),(-.0005,.0005))
    with pytest.raises(ValueError,match="polygon hole|slot"):
        _scene(axis,holes=(hole,))


def test_finite_aperture_cannot_cover_an_interior_hole_with_valid_corners():
    hole=((.0003,.0003),(.0007,.0003),(.0007,.0007),(.0003,.0007))
    with pytest.raises(ValueError,match="voltage_aperture.*polygon hole"):
        _scene(holes=(hole,))


@pytest.mark.parametrize("axis",range(3))
@pytest.mark.parametrize("metal",("pec","copper"))
def test_remote_via_short_is_detected_without_touching_the_feed_ray(axis,metal):
    center=[0.,0.,0.];center[(axis+1)%3]=.002
    size=[.001,.001,.001];size[axis]=.004
    via=mw.Structure(_box(center,size),mw.Material.pec(),name="remote_short")
    with pytest.raises(ValueError,match="shorted.*connected conductor"):
        _scene(axis,metal,extras=(via,))


@pytest.mark.parametrize("axis",range(3))
def test_a_floating_zero_thickness_sheet_cannot_be_crossed_by_the_gap(axis):
    scene=prepare_scene(_scene(axis,extras=(_sheet("floating",axis,0.),)))
    with pytest.raises(ValueError,match="gap intersects conductor 'floating'"):
        compile_port_geometry(scene,scene.ports[0],device="cpu")


def _coplanar(metal="pec",explicit=True):
    structures=[]
    for name,bounds in (("signal",(-.004,-.001)),("ground",(.001,.004))):
        if metal=="pec":
            geometry=mw.PlanarPEC.rectangle((bounds,(-.003,.003)),coordinate=0.)
            material=mw.Material.pec()
        else:
            geometry=_box(((bounds[0]+bounds[1])/2,0.,0.),(bounds[1]-bounds[0],.006,0.))
            material=mw.CopperFoil(thickness=35e-6,frequency_range=(.1e9,3e9))
        structures.append(mw.Structure(geometry,material,name=name))
    options={"current_surface":_box((.0005,0.,0.),(0.,.005,.003)),
             "voltage_aperture":_box((0.,0.,0.),(.002,.002,0.))} if explicit else {}
    port=mw.TerminalPort("feed",mw.TerminalRef("signal"),mw.TerminalRef("ground"),mw.AxisPath("x"),.0005,**options)
    return mw.Scene(domain=mw.Domain(bounds=((- .008,.008),)*3),grid=mw.GridSpec.uniform(.001),
                    boundary=mw.BoundarySpec.none(),device="cpu",structures=structures,ports=(port,))


@pytest.mark.parametrize("metal",("pec","copper"))
def test_coplanar_feed_has_explicit_loop_and_only_open_gap_rows(metal):
    with pytest.raises(ValueError,match="coplanar sheets require explicit"):
        _coplanar(metal,False)
    scene=prepare_scene(_coplanar(metal))
    compiled=compile_port_geometry(scene,scene.ports[0],device="cpu")
    assert compiled.metadata["native_parallel_path_count"]==3
    assert compiled.metadata["native_voltage_edge_count"]==6


def test_a_probe_reaches_the_named_patch_through_a_real_finite_contact():
    probe=mw.Structure(_box((0.,0.,.0015),(.002,.002,.001)),mw.Material.pec(),name="probe")
    scene=prepare_scene(_scene(extras=(probe,),aperture=False))
    assert scene.ports[0].positive[2]==pytest.approx(.001)
    compiled=compile_port_geometry(scene,scene.ports[0],device="cpu")
    assert compiled.metadata["positive_contact_structure"]=="probe"
    assert compiled.metadata["positive_component"]==["probe","signal"]


def _hole_via_scene(dx):
    hole=((- .001,-.001),(.001,-.001),(.001,.001),(-.001,.001))
    signal=_sheet("signal",2,.002,holes=(hole,))
    via=mw.Structure(_box((0.,0.,.00025),(.0015,.0015,.0045)),mw.Material.pec(),name="via_in_hole")
    # Actual clearance is 0.25 mm on both coarse and fine meshes.
    width=.005 if dx==.001 else .00525
    port=mw.TerminalPort("feed",mw.TerminalRef("signal"),mw.TerminalRef("ground"),mw.AxisPath("z"),.0005,
        current_surface=_box((-.002,0.,.0005),(width,width,0.)))
    return mw.Scene(domain=mw.Domain(bounds=((- .008,.008),)*3),grid=mw.GridSpec.anisotropic(dx,dx,.001),
        boundary=mw.BoundarySpec.none(),device="cpu",structures=(signal,_sheet("ground",2,-.002),via),ports=(port,))


def test_actual_polygon_hole_prevents_aabb_contact_and_coarse_grid_does_not_short_disjoint_via():
    coarse=prepare_scene(_hole_via_scene(.001))
    assert coarse.ports[0]._terminal_audit["positive_component"]==["signal"]
    # The old short expectation depended on the pre-node-rule midpoint comb.
    compiled=compile_port_geometry(coarse,coarse.ports[0],device="cpu")
    assert compiled.metadata["negative_component"]==["ground","via_in_hole"]
    assert "disjoint" in compiled.metadata["native_connectivity"]
    fine=prepare_scene(_hole_via_scene(.00025))
    compiled=compile_port_geometry(fine,fine.ports[0],device="cpu")
    assert compiled.metadata["negative_component"]==["ground","via_in_hole"]


def test_concave_terminal_requires_a_path_on_the_actual_footprint():
    polygon=mw.PlanarPEC(((-.003,-.003),(.003,-.003),(.003,.003),(.001,.003),(.001,-.001),
                         (-.001,-.001),(-.001,.003),(-.003,.003)),coordinate=.002)
    signal=mw.Structure(polygon,mw.Material.pec(),name="signal")
    base=_scene(aperture=False)
    with pytest.raises(ValueError,match="polygon hole|slot"):
        base.clone(structures=(signal,base.structures[1]))
    centered=_scene(aperture=False,center=(-.002,0.,0.))
    scene=prepare_scene(centered.clone(structures=(signal,centered.structures[1])))
    compiled=compile_port_geometry(scene,scene.ports[0],device="cpu")
    assert compiled.metadata["positive_contact_structure"]=="signal"


def test_point_only_return_contact_is_explicitly_unqualified():
    # Two boxes meet only on an edge, connecting the upper/lower sheets through
    # a zero-area path. The physical contact cannot be assumed from its AABB.
    first=mw.Structure(_box((.0015,.0015,.001),(.001,.001,.002)),mw.Material.pec(),name="first")
    second=mw.Structure(_box((.0025,.0025,-.001),(.001,.001,.002)),mw.Material.pec(),name="second")
    with pytest.raises(ValueError,match="unqualified zero-area"):
        _scene(extras=(first,second))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires CUDA")
@pytest.mark.parametrize("axis",range(3))
@pytest.mark.parametrize("metal",("pec","copper"))
def test_actual_planar_named_feed_matches_coordinate_feed_on_gpu(axis,metal,tmp_path):
    scene=_scene(axis,metal,device="cuda")
    named=scene.ports[0]
    explicit=mw.LumpedPort("feed",positive=named.positive,negative=named.negative,
        voltage_path=named.voltage_path,current_surface=named.current_surface,
        reference_plane=named.reference_plane,voltage_aperture=named.voltage_aperture)
    results=[]
    for port in (named,explicit):
        simulation=mw.Simulation.fdtd(scene.clone(ports=(port,)),frequency=1e9,
            run_time=mw.TimeConfig(time_steps=256),full_field_dft=False,
            spectral_sampler=mw.SpectralSampler(window="none"),excitations=mw.PortExcitation("feed"))
        results.append(simulation.run())
    for component in ("Ex","Ey","Ez","Hx","Hy","Hz"):
        assert torch.equal(getattr(results[0].solver,component),getattr(results[1].solver,component))
    for channel in ("voltage","current"):
        a=getattr(results[0].port("feed"),channel);b=getattr(results[1].port("feed"),channel)
        assert torch.equal(a,b)
        assert a.abs().max()>0
    if metal=="copper":
        assert sum(float(v["dissipated_energy"]) for v in results[0].conductor_losses.values())>0
    audit=results[0].port("feed").metadata["terminal_audit"]
    assert audit["positive_target"]=="signal"
    assert audit["native_parallel_path_count"]==9
    if axis==2 and metal=="copper":
        path=tmp_path/"named-copper-terminal.pt"
        results[0].save(path)
        loaded=mw.Result.load(path,scene=results[0].scene,map_location="cuda:0")
        assert loaded.port("feed").metadata["terminal_audit"]==audit


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires CUDA")
def test_gpu_named_polygon_with_hole_uses_a_valid_offcenter_contact():
    hole=((- .0005,-.0005),(.0005,-.0005),(.0005,.0005),(-.0005,.0005))
    scene=_scene(holes=(hole,),aperture=False,center=(-.002,0.,0.),device="cuda")
    result=mw.Simulation.fdtd(scene,frequency=1e9,run_time=mw.TimeConfig(time_steps=256),
        spectral_sampler=mw.SpectralSampler(window="none"),excitations=mw.PortExcitation("feed")).run()
    assert result.port("feed").voltage.abs().max()>0
    assert torch.isfinite(result.solver.Ez).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires CUDA")
def test_gpu_preparation_rejects_underresolved_short_without_a_gap_repair():
    # The 0.5 mm physical gap still shorts the volume nodes at z=0 and z=1 mm.
    upper=mw.Structure(_box((.002,0.,.002),(.001,.002,.0025)),mw.Material.pec(),name="upper_via")
    lower=mw.Structure(_box((.002,0.,-.001),(.001,.002,.0025)),mw.Material.pec(),name="lower_via")
    scene=_scene(extras=(upper,lower),device="cuda")
    with pytest.raises(ValueError,match="native Yee conductor ownership shorts"):
        mw.Simulation.fdtd(scene,frequency=1e9,run_time=mw.TimeConfig(time_steps=1),
                           excitations=mw.PortExcitation("feed")).prepare()
