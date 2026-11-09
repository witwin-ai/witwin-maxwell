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
    assert seconds < 120
    assert peak < 6*2**30


def test_gds_reference_arrays_and_database_units(tmp_path):
    f = fixture_fa(device="cpu")
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
    scene = process.scene_from_gds(path, device="cpu")
    vias = [s.geometry for s in scene.structures if isinstance(s.geometry,mw.Box)]
    assert len(vias) == 6
    actual = sorted(tuple(np.round(v.position.detach().numpy()[:2]/1e-6,7)) for v in vias)
    assert actual == [(x,y) for x in (10.,20.,30.) for y in (20.,40.)]
    assert all(torch.allclose(v.size[:2],torch.full((2,),4e-6,dtype=torch.float64),atol=1e-18,rtol=0) for v in vias)


@pytest.mark.parametrize("case,exception,message", [
    ("diagonal",NotImplementedError,"Manhattan"),
    ("via",NotImplementedError,"non-rectangular"),
    ("via_notch",NotImplementedError,"non-rectangular"),
    ("unpaired",ValueError,"partner terminal"),
    ("unknown",ValueError,"Unknown GDS"),
    ("badlabel",ValueError,"form name"),
    ("badmetal",ValueError,"unknown metal"),
    ("duplicate",ValueError,"Duplicate pin"),
    ("diagonalport",ValueError,"axis-aligned"),
])
def test_process_rejections(tmp_path,case,exception,message):
    process = process_fa(fixture_fa(device="cpu"))
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0,0),(10,10),layer=1))
    if case == "diagonal":
        cell.add(gdstk.Polygon(((0,0),(2,0),(1,1)),layer=2))
    elif case == "via":
        cell.add(gdstk.Polygon(((0,0),(2,0),(2,1),(1,1),(1,2),(0,2)),layer=3))
    elif case == "via_notch":
        # One database-square notch in a large bounding rectangle must reject;
        # an area-relative tolerance would incorrectly admit this polygon.
        cell.add(gdstk.Polygon(((.001,0),(10000,0),(10000,10000),(0,10000),
                                (0,.001),(.001,.001)),layer=3))
    elif case == "unpaired":
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
        process.scene_from_gds(path,device="cpu")


@pytest.mark.parametrize("turns",(1,2,3))
def test_quarter_turn_references_at_origin(tmp_path,turns):
    fixture = fixture_fa(device="cpu")
    library = gdstk.Library(unit=1e-6,precision=1e-9)
    child = library.new_cell("CHILD")
    child.add(gdstk.rectangle((0,0),(10,2),layer=1))
    top = library.new_cell("TOP")
    top.add(gdstk.Reference(child,rotation=turns*np.pi/2))
    path = tmp_path/"rotated.gds"
    library.write_gds(str(path))
    scene = process_fa(fixture).scene_from_gds(path,device="cpu")
    rotation = np.array((((0,-1),(1,0)),((-1,0),(0,-1)),((0,1),(-1,0))))[turns-1]
    expected = np.array(((0,0),(10,0),(10,2),(0,2))) @ rotation.T * 1e-6
    actual = scene.structures[0].geometry.vertices.numpy()
    assert set(map(tuple,actual)) == set(map(tuple,expected))


def test_explicit_ports_and_nonstrict_layers(tmp_path):
    fixture = fixture_fa(device="cpu")
    path = write_fa_gds(fixture,tmp_path/"explicit.gds")
    process = replace(process_fa(fixture),pin_layer=None)
    scene = process.scene_from_gds(path,ports=fixture.scene.ports,strict=False,device="cpu")
    assert [p.name for p in scene.ports] == ["p1","p2"]
    assert scene.ports[0] is fixture.scene.ports[0]


def test_multiple_gds_maps_union_on_one_plane(tmp_path):
    fixture = fixture_fa(device="cpu")
    process = process_fa(fixture)
    process = replace(process,metals=(*process.metals,
        mw.LayerMap("top_ground",2,305e-6,2e-6,mw.Material.pec(),datatype=1)))
    # Use identical physical materials across independently declared mappings.
    process = replace(process,metals=tuple(replace(m,material=mw.Material.pec()) for m in process.metals))
    library = gdstk.Library()
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((0,0),(10,10),layer=2),
             gdstk.rectangle((5,0),(15,10),layer=2,datatype=1))
    path = tmp_path/"maps.gds"
    library.write_gds(str(path))
    scene = process.scene_from_gds(path,device="cpu")
    assert len(scene.structures) == 1
    geometry = scene.structures[0].geometry
    assert float(geometry.vertices[:,0].min()) == 0
    assert abs(float(geometry.vertices[:,0].max())-15e-6) < 1e-18


def test_process_nonconsecutive_plane_rejection():
    f = fixture_fa(device="cpu")
    process = process_fa(f)
    with pytest.raises(NotImplementedError,match="consecutive"):
        replace(process,metals=(*process.metals,mw.LayerMap("middle",4,303e-6,1e-6,mw.Material.pec())))


@pytest.mark.cuda
def test_fa_gds_round_trip(tmp_path,record_property):
    fixture = fixture_fa()
    path = write_fa_gds(fixture,tmp_path/"fa.gds")
    scene = process_fa(fixture).scene_from_gds(path)
    assert sum(len(s.geometry.holes) for s in scene.structures if isinstance(s.geometry,mw.PlanarPEC)) == 1
    first = fixture.simulation((1e9,10e9)).run()
    second = mw.Simulation.mom(scene,frequencies=(1e9,10e9),config=fixture.config).run()
    error = float((first.network.s-second.network.s).abs().max())
    record_property("max_S_error",error)
    record_property("bitwise",torch.equal(first.network.s,second.network.s))
    assert torch.equal(first.surface_currents.layout.mesh.vertices,second.surface_currents.layout.mesh.vertices)
    assert torch.equal(first.surface_currents.layout.mesh.faces,second.surface_currents.layout.mesh.faces)
    assert torch.equal(first.network.s,second.network.s)
    # Every sampling cell has the same exact union membership despite a
    # different rectangle cover of the Boolean-unioned polygons.
    assert first.network.metadata["unknowns"] == second.network.metadata["unknowns"]
    for result in (first,second):
        assert result.surface_currents.mesh_quality["max_aspect_ratio"] <= 20
    record_property("mesh_quality",str(second.surface_currents.mesh_quality))


@pytest.mark.cuda
@pytest.mark.parametrize("start",range(0,25,5))
def test_fa_jitter_clustering_quality(tmp_path,start,record_property):
    fixture = fixture_fa()
    tolerance = 5e-9
    options = replace(fixture.config.layered,breakpoint_tolerance_m=tolerance)
    config = mw.MoMConfig(layered=options)
    path = write_fa_gds(fixture,tmp_path/"jitter.gds",jitter=True)
    scene = process_fa(fixture,tolerance=tolerance).scene_from_gds(path)
    frequencies = SWEEP[start:start+5]
    first = mw.Simulation.mom(fixture.scene,frequencies=frequencies,config=config).run()
    second = mw.Simulation.mom(scene,frequencies=frequencies,config=config).run()
    record_property("frequencies_Hz",frequencies)
    error = float((first.network.s-second.network.s).abs().max())
    quality = second.surface_currents.mesh_quality
    record_property("max_S_error",error)
    for name,value in quality.items():
        record_property(name,value)
    assert error <= 1e-3
    assert quality == second.network.metadata["mesh_quality"]
    assert quality["max_aspect_ratio"] <= 20
    assert quality["min_angle_degrees"] > 2
    assert quality["smallest_edge_m"] >= 0.5e-6
    assert set(first.surface_currents.layout.mesh.vertices[:,2].tolist()) == set(
        second.surface_currents.layout.mesh.vertices[:,2].tolist())
