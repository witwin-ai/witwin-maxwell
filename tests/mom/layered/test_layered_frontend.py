"""Process/GDS round trip, strict boundaries and sliver-avoidance gates."""

from dataclasses import replace
import gc
import time

import gdstk
import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from tests.support.rfic_fixtures import SWEEP, fixture_fa
from tests.support.rfic_frontend_fixtures import process_fa, write_fa_gds

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


@pytest.fixture(autouse=True)
def resource_gate(request,record_property):
    if request.node.get_closest_marker("cuda") is None:
        yield
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    record_property("seconds",seconds)
    record_property("peak_allocated_bytes",peak)
    assert peak < 6*2**30


def test_gds_reference_arrays_and_database_units(tmp_path):
    f = fixture_fa()
    process = process_fa(f)
    library = gdstk.Library(unit=1e-3, precision=1e-9)
    top = library.new_cell("TOP")
    child = library.new_cell("TUBE")
    child.add(gdstk.rectangle((-.002,-.002),(.002,.002),layer=3))
    top.add(gdstk.Reference(child,origin=(.01,.02),columns=3,rows=2,spacing=(.01,.02)))
    # A metal polygon is sufficient to inspect the scene; no solver is entered.
    top.add(gdstk.rectangle((0,0),(.1,.1),layer=1))
    path = tmp_path/"array.gds"
    library.write_gds(str(path))
    scene = process.scene_from_gds(path)
    vias = [s.geometry for s in scene.structures if isinstance(s.geometry,mw.Box)]
    assert len(vias) == 6
    actual = sorted(tuple(np.round(v.position.detach().numpy()[:2]/1e-6,7)) for v in vias)
    assert actual == [(x,y) for x in (10.,20.,30.) for y in (20.,40.)]
    assert all(torch.allclose(v.size[:2],torch.full((2,),4e-6,dtype=torch.float64),atol=1e-18,rtol=0) for v in vias)


@pytest.mark.parametrize("case,exception,message", [
    ("unpaired",ValueError,"partner terminal"),
    ("unknown",ValueError,"Unknown GDS"),
    ("badlabel",ValueError,"form name"),
    ("badmetal",ValueError,"unknown metal"),
    ("duplicate",ValueError,"Duplicate pin"),
    ("diagonalport",ValueError,"axis-aligned"),
])
def test_process_rejections(tmp_path,case,exception,message):
    process = process_fa(fixture_fa())
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0,0),(10,10),layer=1))
    if case == "unpaired":
        cell.add(gdstk.Label("p:+:lower",(1,1),layer=10))
    elif case == "unknown":
        cell.add(gdstk.rectangle((0,0),(1,1),layer=99,datatype=2))
    elif case == "badlabel":
        cell.add(gdstk.Label("p",(1,1),layer=10))
    elif case == "badmetal":
        cell.add(gdstk.Label("p:+:unknown",(1,1),layer=10))
    elif case == "duplicate":
        cell.add(gdstk.Label("p:+:lower",(1,1),layer=10),
                 gdstk.Label("p:+:lower",(2,1),layer=10))
    elif case == "diagonalport":
        cell.add(gdstk.Label("p:+:lower",(1,1),layer=10),
                 gdstk.Label("p:-:lower",(2,2),layer=10))
    path = tmp_path/"reject.gds"
    library.write_gds(str(path))
    with pytest.raises(exception,match=message):
        process.scene_from_gds(path)


@pytest.mark.parametrize("outline", [
    ((0,0),(2,0),(2,1),(1,1),(1,2),(0,2)),
    # One database-square notch in a large bounding rectangle is not a rectangle or a regular
    # polygon (an area-relative tolerance would wrongly admit it); it keeps its exact vertices.
    ((.001,0),(10000,0),(10000,10000),(0,10000),(0,.001),(.001,.001)),
], ids=["l_shape", "notch"])
def test_irregular_gds_via_imports_as_exact_extruded_polygon(tmp_path,outline):
    """B2: an irregular via outline imports as a z-axis ExtrudedPolygon on the database grid, plane to plane."""
    from witwin.core import ExtrudedPolygon
    from witwin.maxwell.mom.layered.planar import via_record
    process = process_fa(fixture_fa())
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0,0),(10000,10000),layer=1))
    cell.add(gdstk.Polygon(outline,layer=3))
    path = tmp_path/"via.gds"
    library.write_gds(str(path))
    scene = process.scene_from_gds(path)
    vias = [s.geometry for s in scene.structures if isinstance(s.geometry,ExtrudedPolygon)]
    assert len(vias) == 1 and vias[0].axis == "z" and vias[0].vertices.dtype == torch.float64
    planes = sorted(m.z for m in process.metals)
    record = via_record(vias[0],planes)
    assert (record.z0,record.z1) == (planes[0],planes[-1])
    expected = np.rint(np.array(outline)/1e-3)*1e-3*1e-6
    actual = np.array(record.outline)
    assert len(actual) == len(expected)
    assert np.abs(actual[:,None]-expected[None]).max(-1).min(1).max() <= 1e-15


@pytest.mark.parametrize("turns",(1,2,3))
def test_quarter_turn_references_at_origin(tmp_path,turns):
    fixture = fixture_fa()
    library = gdstk.Library(unit=1e-6,precision=1e-9)
    child = library.new_cell("CHILD")
    child.add(gdstk.rectangle((0,0),(10,2),layer=1))
    top = library.new_cell("TOP")
    top.add(gdstk.Reference(child,rotation=turns*np.pi/2))
    path = tmp_path/"rotated.gds"
    library.write_gds(str(path))
    scene = process_fa(fixture).scene_from_gds(path)
    rotation = np.array((((0,-1),(1,0)),((-1,0),(0,-1)),((0,1),(-1,0))))[turns-1]
    expected = np.array(((0,0),(10,0),(10,2),(0,2))) @ rotation.T * 1e-6
    actual = scene.structures[0].geometry.vertices.numpy()
    assert set(map(tuple,actual)) == set(map(tuple,expected))


def test_explicit_ports_and_nonstrict_layers(tmp_path):
    fixture = fixture_fa()
    path = write_fa_gds(fixture,tmp_path/"explicit.gds")
    process = replace(process_fa(fixture),pin_layer=None)
    scene = process.scene_from_gds(path,ports=fixture.scene.ports,strict=False)
    assert [p.name for p in scene.ports] == ["p1","p2"]
    assert scene.ports[0] is fixture.scene.ports[0]


def test_multiple_gds_maps_union_on_one_plane(tmp_path):
    fixture = fixture_fa()
    process = process_fa(fixture)
    process = replace(process,metals=(*process.metals,
        mw.mom.LayerMap("top_ground",2,305e-6,2e-6,mw.Material.pec(),datatype=1)))
    # Use identical physical materials across independently declared mappings.
    process = replace(process,metals=tuple(replace(m,material=mw.Material.pec()) for m in process.metals))
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0,0),(10,10),layer=2),
             gdstk.rectangle((5,0),(15,10),layer=2,datatype=1))
    path = tmp_path/"maps.gds"
    library.write_gds(str(path))
    scene = process.scene_from_gds(path)
    assert len(scene.structures) == 1
    geometry = scene.structures[0].geometry
    assert float(geometry.vertices[:,0].min()) == 0
    assert abs(float(geometry.vertices[:,0].max())-15e-6) < 1e-18


def test_process_multi_plane_vias_and_same_plane_rejection():
    f = fixture_fa()
    process = process_fa(f)
    # B2: a via may pass an intermediate plane (it must clear or land in that metal at meshing).
    replace(process,metals=(*process.metals,mw.mom.LayerMap("middle",4,303e-6,1e-6,mw.Material.pec())))
    with pytest.raises(ValueError,match="distinct"):
        replace(process,vias=(mw.mom.ViaMap(3,("lower","lower"),mw.Material.pec()),))


def test_gds_non_manhattan_metal_and_regular_polygon_vias(tmp_path):
    """45-degree metal imports as PlanarPEC; regular-polygon vias import as z-axis Prism."""
    process = process_fa(fixture_fa())
    library = gdstk.Library(unit=1e-6,precision=1e-9)
    cell = library.new_cell("TOP")
    octagon = [(100*np.cos(np.pi/8+k*np.pi/4),100*np.sin(np.pi/8+k*np.pi/4)) for k in range(8)]
    for layer in (1,2):
        cell.add(gdstk.Polygon(octagon,layer=layer))
    via = [(3*np.cos(k*np.pi/4)+10,3*np.sin(k*np.pi/4)-5) for k in range(8)]
    cell.add(gdstk.Polygon(via,layer=3))
    path = tmp_path/"octagon.gds"
    library.write_gds(str(path))
    scene = process.scene_from_gds(path)
    sheets = [s.geometry for s in scene.structures if getattr(s.geometry,"kind",None) == "planar_pec"]
    assert len(sheets) == 2 and all(len(g.vertices) == 8 for g in sheets)
    prisms = [s.geometry for s in scene.structures if isinstance(s.geometry,mw.geometry.Prism)]
    assert len(prisms) == 1 and prisms[0].num_sides == 8 and prisms[0].axis == "z"
    from witwin.maxwell.mom.layered.planar import via_record
    planes = sorted(m.z for m in process.metals)
    record = via_record(prisms[0],planes)
    assert (record.z0,record.z1) == (planes[0],planes[-1])
    expected = np.array(via)*1e-6
    actual = np.array(record.outline)
    distance = np.abs(actual[:,None]-expected[None]).max(-1).min(1)
    # GDS stores the outline on its 1 nm database grid; Prism is regular within that grid.
    assert distance.max() <= 1e-9


@requires_cuda
@pytest.mark.cuda
def test_fa_gds_round_trip(tmp_path,record_property):
    fixture = fixture_fa()
    path = write_fa_gds(fixture,tmp_path/"fa.gds")
    scene = process_fa(fixture).scene_from_gds(path)
    assert sum(len(s.geometry.holes) for s in scene.structures if isinstance(s.geometry,mw.geometry.PlanarPEC)) == 1
    first = fixture.simulation((1e9,10e9)).run()
    second = mw.Simulation.mom(scene,frequencies=(1e9,10e9),config=fixture.config).run()
    error = float((first.network.s-second.network.s).abs().max())
    record_property("max_S_error",error)
    record_property("bitwise",torch.equal(first.network.s,second.network.s))
    assert torch.equal(mw.mom.surface_currents(first).layout.mesh.vertices,mw.mom.surface_currents(second).layout.mesh.vertices)
    assert torch.equal(mw.mom.surface_currents(first).layout.mesh.faces,mw.mom.surface_currents(second).layout.mesh.faces)
    assert torch.equal(first.network.s,second.network.s)
    # Every sampling cell has the same exact union membership despite a
    # different rectangle cover of the Boolean-unioned polygons.
    assert first.network.metadata["unknowns"] == second.network.metadata["unknowns"]
    for result in (first,second):
        assert mw.mom.surface_currents(result).mesh_quality["max_aspect_ratio"] <= 20
    record_property("mesh_quality",str(mw.mom.surface_currents(second).mesh_quality))


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("start",range(0,25,5))
def test_fa_jitter_clustering_quality(tmp_path,start,record_property):
    fixture = fixture_fa()
    tolerance = 5e-9
    options = replace(fixture.config.layered,breakpoint_tolerance_m=tolerance)
    config = mw.mom.MoMConfig(layered=options)
    path = write_fa_gds(fixture,tmp_path/"jitter.gds",jitter=True)
    scene = process_fa(fixture,tolerance=tolerance).scene_from_gds(path)
    frequencies = SWEEP[start:start+5]
    first = mw.Simulation.mom(fixture.scene,frequencies=frequencies,config=config).run()
    second = mw.Simulation.mom(scene,frequencies=frequencies,config=config).run()
    record_property("frequencies_Hz",frequencies)
    error = float((first.network.s-second.network.s).abs().max())
    quality = mw.mom.surface_currents(second).mesh_quality
    record_property("max_S_error",error)
    for name,value in quality.items():
        record_property(name,value)
    assert error <= 1e-3
    assert quality == second.network.metadata["mesh_quality"]
    assert quality["max_aspect_ratio"] <= 20
    assert quality["min_angle_degrees"] > 2
    assert quality["smallest_edge_m"] >= 0.5e-6
    assert set(mw.mom.surface_currents(first).layout.mesh.vertices[:,2].tolist()) == set(
        mw.mom.surface_currents(second).layout.mesh.vertices[:,2].tolist())


def test_scene_from_gds_names_the_rfic_extra_when_gdstk_is_missing(tmp_path,monkeypatch):
    """gdstk is in the optional extra 'rfic': without it the layered GDS import says how to install it."""
    import sys
    fixture = fixture_fa()
    process = process_fa(fixture)
    path = write_fa_gds(fixture,tmp_path/"fa.gds")
    monkeypatch.setitem(sys.modules,"gdstk",None)
    with pytest.raises(ImportError,match=r"pip install witwin-maxwell\[rfic\]"):
        process.scene_from_gds(path)
