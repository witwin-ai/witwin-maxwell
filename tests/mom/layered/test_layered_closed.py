"""Closed conductor geometry, SIBC, junction and public network contracts."""
import gc
import math
import time

import numpy as np
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.layered.mesh import mesh_closed_conductors
from witwin.maxwell.mom.layered.network import _closed_geometry
from tests.support.rfic_closed_references import strip_fixture,network_metrics,SIGMA,UM,STRIP_DC_OHM

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


@pytest.fixture(autouse=True)
def budget(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    yield
    torch.cuda.synchronize()
    record_property("seconds",time.perf_counter()-start)
    record_property("peak_allocated_bytes",torch.cuda.max_memory_allocated())
    assert torch.cuda.max_memory_allocated() < 6*2**30


def small_mesh(bridge=True):
    planes=torch.tensor([1.,3.],device="cuda",dtype=torch.float64)*UM
    boxes=[(0.,4*UM,0.,4*UM,UM,3*UM)]
    bridges=[(4*UM,8*UM,UM,3*UM,3*UM)] if bridge else []
    return mesh_closed_conductors(boxes,bridges,planes,4*UM,
        cuts=[([6*UM],[])],h_edge=None,growth_ratio=1.5,interfaces=())


def test_extrusion_holes_and_via_face_gaps():
    planes=torch.tensor([0.,2.,4.,6.],device="cuda",dtype=torch.float64)*UM
    # Square annuli at two levels and a via inside their left-hand strips.
    rectangles=[tuple(v*UM for v in r) for z in (1.,5.) for r in
                ((0.,8.,0.,2.,z),(0.,8.,6.,8.,z),(0.,2.,2.,6.,z),(6.,8.,2.,6.,z))]
    materials=[(SIGMA,2*UM,None)]*len(rectangles)
    via=(.5*UM,1.5*UM,3*UM,5*UM,UM,5*UM)
    stack=mw.mom.LayerStack((mw.mom.Layer("air",100*UM),),mw.mom.HalfSpace(),mw.mom.HalfSpace())
    boxes,_,shortened=_closed_geometry(rectangles,materials,[via],[],stack)
    assert shortened==[(.5*UM,1.5*UM,3*UM,5*UM,2*UM,4*UM)]
    layout=mesh_closed_conductors(boxes,[],planes,4*UM,cuts=[],h_edge=None,
                                  growth_ratio=1.5,interfaces=())
    topology=layout.mesh.topology
    assert topology.closed and topology.manifold_vertices and topology.orientable
    assert bool((topology.face_count==2).all())
    # Closed union volume is two rings plus the exposed via gap.
    triangles=layout.mesh.triangles.cpu().numpy()
    volume=np.einsum("ij,ij->i",triangles[:,0],np.cross(triangles[:,1],triangles[:,2])).sum()/6
    assert abs(volume/(2*(64-16)*2*UM**3+1*2*2*UM**3)-1)<1e-12
    centroids=layout.mesh.centroids
    assert not bool(((centroids[:,0]>2*UM)&(centroids[:,0]<6*UM)&
                     (centroids[:,1]>2*UM)&(centroids[:,1]<6*UM)&
                     (layout.triangle_planes>=0)).any())


def test_junction_gram_and_charge_against_cpu_quadrature(record_property):
    layout=small_mesh()
    basis,mesh=layout.basis,layout.mesh
    assert mesh.topology.junction_edges>0
    assert mesh.topology.face_count.max()==3
    tri=mesh.triangles.cpu().numpy()
    edges=basis.face_edges.cpu().numpy()
    signs=basis.face_signs.cpu().numpy()
    local=basis.local_slots.cpu().numpy()
    lengths=basis.edge_lengths.cpu().numpy()
    area=mesh.areas.cpu().numpy()
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    points=np.einsum("qv,tvd->tqd",bary,tri)
    free=np.take_along_axis(tri,local[...,None],axis=1)
    values=(points[:,:,None]-free[:,None])*((signs*lengths[edges]/(2*area[:,None]))[:,None,:,None])
    expected=np.einsum("tqid,tqjd,t->tij",values,values,area/3)
    actual=basis.mass_blocks().cpu().numpy()
    relative=np.max(abs(actual-expected))/np.max(abs(expected))
    assert relative<1e-12
    record_property("gram_max_relative",relative)
    from witwin.maxwell.mom.layered.build import build_extension
    build_extension()
    identity=torch.eye(basis.n_edges,device="cuda",dtype=torch.float64)
    div=torch.ops.witwin_maxwell_mom_layered.divergence(
        identity,basis.face_edges,basis.face_signs,basis.edge_lengths)
    assert div.sum(0).abs().max()<1e-18
    for _ in range(2):
        assert torch.equal(basis.mass_blocks(),basis.mass_blocks())


@pytest.mark.parametrize("roughness",[None,mw.media.HammerstadRoughness(rms_height=.4*UM),
    mw.media.HurayRoughness.cannonball(.2*UM),mw.media.GroisseRoughness(rms_height=.4*UM)])
def test_single_face_sibc_all_faces_and_ideal_bridge(roughness):
    from witwin.maxwell.mom.layered.loss import prepare_closed_loss,sheet_impedance
    layout=small_mesh()
    frequencies=torch.tensor([1e6,1e9,1e11],device="cuda",dtype=torch.float64)
    material=mw.media.LossyMetalMedium(conductivity=SIGMA,roughness=roughness)
    loss=prepare_closed_loss(layout,[(SIGMA,2*UM,roughness)],frequencies)
    physical=layout.face_owners>=0
    expected=torch.tensor([material.surface_impedance(2*math.pi*f) for f in frequencies.tolist()],
                           device="cuda",dtype=torch.complex128)
    assert torch.allclose(loss.impedance[:,physical],expected[:,None],rtol=1e-12,atol=0)
    assert (loss.impedance[:,~physical]==0).all()
    thick=sheet_impedance(frequencies[-1:],SIGMA,20*UM,roughness)
    assert torch.allclose(expected[-1:],2*thick,rtol=1e-7,atol=0)
    # The native in-place loss fold must include both RWGs on a junction triangle.
    n=layout.basis.n_edges
    native=torch.zeros((len(frequencies),n,n),device="cuda",dtype=torch.complex128)
    torch.ops.witwin_maxwell_mom_layered.loss_fold(
        native,loss.blocks.contiguous(),loss.columns.contiguous(),loss.supports.contiguous(),loss.impedance,
        torch.arange(n,device="cuda"),frequencies)
    # The fold stores R/(-i*mu0*w) beside the magnetic potential.
    native=native*(-1j/(8.8541878128e-12*299792458.0**2)*2*math.pi*frequencies[:,None,None])
    for f in range(len(frequencies)):
        assembled=torch.zeros_like(native[f])
        loss.add_to(assembled,f)
        assert torch.allclose(native[f],assembled,rtol=1e-14,atol=1e-24)
        assert torch.allclose(native[f],native[f].T,rtol=1e-13,atol=1e-24)


def test_public_complete_repeat_and_pec_passivity(record_property):
    first=None
    for _ in range(3):
        scene,config=strip_fixture(maximum_um=80.,edge_um=10.)
        result=mw.Simulation.mom(scene,frequencies=(1e8,1e9),config=config).run()
        if first is None:
            first=(result.network.s.clone(),mw.mom.surface_currents(result).currents.clone())
            record_property("metrics",str(network_metrics(result)))
        else:
            assert torch.equal(first[0],result.network.s)
            assert torch.equal(first[1],mw.mom.surface_currents(result).currents)
        assert result.metadata["conductor_model"]=="closed"
        assert len(result.metadata["conductor_faces"])==5
        assert result.network.metadata["port_attachment"]=="top_face"
        layout=mw.mom.surface_currents(result).layout
        horizontal=layout.triangle_planes>=0
        assert torch.equal(layout.mesh.centroids[horizontal,2],
            result.solver.prepared.table.planes[layout.triangle_planes[horizontal]])
        assert torch.linalg.matrix_norm(result.network.s,ord=2).max()<=1+1e-9
        del result
        gc.collect();torch.cuda.empty_cache()
    scene,config=strip_fixture(maximum_um=80.,edge_um=10.,lossy=False)
    result=mw.Simulation.mom(scene,frequencies=1e6,config=config).run()
    assert result.solver.prepared.loss is None
    assert abs(result.network.s.abs().item()-1)<1e-8


def test_closed_sibc_dc_degradation_is_explicit(record_property):
    # SIBC R -> 0 as sqrt(f); a finite conductivity bar has finite DC R.
    # The model must not silently replace SIBC by the two-face sheet formula.
    values={}
    for model in ("sheet","closed"):
        scene,config=strip_fixture(model=model,maximum_um=80.,edge_um=10.)
        result=mw.Simulation.mom(scene,frequencies=(1e4,1e6),config=config).run()
        values[model]=network_metrics(result)["R_ohm"]
        del result
        gc.collect();torch.cuda.empty_cache()
    record_property("R_ohm",str(values))
    record_property("analytic_DC_ohm",STRIP_DC_OHM)
    assert abs(values["sheet"][0]/STRIP_DC_OHM-1)<.03
    assert values["closed"][0]<.1*STRIP_DC_OHM
    assert values["closed"][1]>5*values["closed"][0]


def test_closed_geometry_negative_boundaries():
    stack=mw.mom.LayerStack((mw.mom.Layer("air",100.),),mw.mom.HalfSpace(),mw.mom.HalfSpace())
    r=[(0.,4.,0.,4.,10.)]
    with pytest.raises(ValueError,match="conductor_model"):
        mw.mom.LayeredSurfaceConfig(bridge_width_m=1.,max_edge_length_m=1.,conductor_model="volume")
    with pytest.raises(NotImplementedError,match="positive physical_thickness"):
        _closed_geometry(r,[(None,0.,None)],[],[],stack)
    # Vertical port: the upper terminal plane (z=20) has no body (R6-4 narrowed guard).
    with pytest.raises(NotImplementedError,match="Closed vertical ports need terminal bodies"):
        _closed_geometry(r,[(None,2.,None)],[],[(0,0.,0.,1.,10.,20.,15.,.5,1.)],stack)
    with pytest.raises(NotImplementedError,match="positive face gap"):
        _closed_geometry(r+[(0.,4.,0.,4.,11.)],[(None,2.,None)]*2,[],[],stack)
    with pytest.raises(NotImplementedError,match="uniform terminal thickness"):
        _closed_geometry(r+[(0.,4.,0.,4.,20.),(4.,8.,0.,4.,20.)],
            [(None,2.,None),(None,2.,None),(None,4.,None)],[(1.,7.,1.,3.,10.,20.)],[],stack)
    with pytest.raises(NotImplementedError,match="identical top-face"):
        _closed_geometry(r+[(6.,10.,0.,4.,10.)],[(None,2.,None),(None,4.,None)],[],
            [((4.,6.,1.,3.,10.),10.,0,5.,2.,1.)],stack)
    with pytest.raises(ValueError,match="identical materials"):
        _closed_geometry(r+r,[(SIGMA,2.,None),(SIGMA/2,2.,None)],[],[],stack)


def test_two_port_closed_reciprocity_and_complete_repeat(record_property):
    from tests.mom.layered.test_layered_network import coplanar_strips
    from tests.mom.layered.test_layered_vias import vacuum
    from witwin.maxwell.mom.layered.network import prepare_layered_network,solve_prepared_network
    rectangles,ports=coplanar_strips(100*UM,10*UM,10*UM,2*UM,100*UM)
    first=None
    for _ in range(3):
        prepared=prepare_layered_network(vacuum(),rectangles,ports,(1e9,10e9),
            bridge_width=2*UM,max_edge=25*UM,sheet_materials=[(SIGMA,2*UM,None)]*len(rectangles),
            conductor_model="closed")
        network,current=solve_prepared_network(prepared)
        y=network.to_y()
        reciprocity=((y-y.transpose(-1,-2)).abs().max()/y.abs().max()).item()
        assert reciprocity<1e-10
        assert torch.linalg.matrix_norm(network.s,ord=2).max()<=1+1e-9
        assert torch.linalg.eigvalsh(y.real).min()>0
        record_property("reciprocity",reciprocity)
        if first is None:
            first=(network.s.clone(),current.clone())
        else:
            assert torch.equal(first[0],network.s)
            assert torch.equal(first[1],current)
        del prepared,network,current,y
        gc.collect();torch.cuda.empty_cache()


def test_interface_plane_matches_upper_displaced_limit(record_property):
    from witwin.maxwell.mom.layered.green import LayeredGreenTable
    stack=mw.mom.LayerStack((mw.mom.Layer("substrate",40*UM,11.9,10.),
                        mw.mom.Layer("oxide",10*UM,4.1)),mw.mom.HalfSpace(11.9,10.),mw.mom.HalfSpace())
    interface=stack.interfaces()[-1]
    heights=(48*UM,interface,interface+1e-11)
    table=LayeredGreenTable.build(stack,heights,(1e9,),100*UM,device="cuda",extract_distance=30*UM)
    rho=torch.logspace(-6,-4,21,device="cuda",dtype=torch.float64)
    exact=table.evaluate(interface,48*UM,rho)
    limit=table.evaluate(interface+1e-11,48*UM,rho)
    relative=((exact-limit).abs().amax(-1)/exact.abs().amax(-1)).max().item()
    assert relative<1e-5
    record_property("upper_interface_limit_relative",relative)
    rectangle=[(0.,10*UM,0.,10*UM,49*UM)]
    boxes,_,_=_closed_geometry(rectangle,[(None,2*UM,None)],[],[],stack)
    assert boxes[0][-1]==interface


def test_native_junction_gather_against_cpu_support_sum():
    from witwin.maxwell.mom.layered.assembly import assemble_layered
    from witwin.maxwell.mom.layered.green import LayeredGreenTable
    from tests.mom.layered.test_layered_vias import vacuum
    layout=small_mesh()
    table=LayeredGreenTable.build(vacuum(),(UM,3*UM),(1e9,),20*UM,device="cuda",
                                  via_spans=((UM,3*UM),),extract_distance=1e-3)
    tiles=assemble_layered(layout,table,_tiles=True).cpu().numpy()
    vector,scalar=assemble_layered(layout,table,separate=True)
    basis=layout.basis
    supports=torch.stack((basis.plus_faces,basis.minus_faces),-1).cpu().numpy()
    slots=layout.support_slots.cpu().numpy()
    areas=layout.mesh.areas.cpu().numpy()
    lengths=basis.edge_lengths.cpu().numpy()
    slopes=lengths[:,None]*np.array([1.,-1.])[None]/(2*areas[supports])
    expected=np.zeros((basis.n_edges,basis.n_edges),dtype=np.complex128)
    for a in (0,1):
        for b in (0,1):
            t=supports[:,a,None];u=supports[None,:,b]
            i=slots[:,a,None];j=slots[None,:,b]
            expected+=slopes[:,a,None]*slopes[None,:,b]*tiles[t,u,3*i+j]
    actual=vector.cpu().numpy()
    assert np.max(abs(actual-expected))<1e-12*np.max(abs(expected))
    expected_scalar=tiles[:,:,9]/(areas[:,None]*areas[None,:])
    assert np.max(abs(scalar.cpu().numpy()-expected_scalar))<1e-12*np.max(abs(expected_scalar))


def test_pec_thin_limit_inductance_trend(record_property):
    values=[]
    for model,thickness in (("sheet",2.),("closed",2.),("closed",.5),("closed",.1)):
        scene,config=strip_fixture(model=model,thickness_um=thickness,lossy=False,
                                  maximum_um=40.,edge_um=5.)
        result=mw.Simulation.mom(scene,frequencies=1e8,config=config).run()
        values.append(network_metrics(result)["L_H"][0])
        del result
        gc.collect();torch.cuda.empty_cache()
    errors=[abs(v-values[0]) for v in values[1:]]
    assert errors[0]>errors[1]>errors[2]
    record_property("sheet_and_closed_2_0p5_0p1um_L_H",values)


def test_default_sheet_live_frozen_numeric_controls(record_property):
    # Original source-bound receipts remain untouched and reject source changes.
    # This separate control checks their numbers without renewing provenance.
    import json
    from pathlib import Path
    from tests.support.rfic_fixtures import fixture_fa
    from tests.support.rfic_r6b_references import plate_fixture,FREQUENCIES
    data=Path(__file__).with_name("data")
    evidence=json.loads((data/"fixture_r6_native_evidence.json").read_text())
    expected=evidence["coarse"]["none"]
    fixture=fixture_fa(mesh=expected["mesh"])
    f=(1e9,expected["metrics"]["peak_frequency_Hz"])
    result=fixture.simulation(f).run()
    actual=network_metrics(result)
    assert abs(actual["L_H"][0]/expected["metrics"]["L_1GHz_H"]-1)<1e-6
    assert abs(actual["Q"][1]/expected["metrics"]["Q_peak"]-1)<1e-6
    assert result.network.metadata["conductor_model"]=="sheet"
    del result
    gc.collect();torch.cuda.empty_cache()
    expected=json.loads((data/"fixture_r6b_shunt_evidence.json").read_text())["native_coarse"]
    scene,config=plate_fixture(40.,10.)
    result=mw.Simulation.mom(scene,frequencies=FREQUENCIES,config=config).run()
    reference=np.array(expected["y_real"])+1j*np.array(expected["y_imag"])
    actual=result.network.to_y().cpu().numpy()
    error=np.max(abs(actual-reference))/np.max(abs(reference))
    assert error<1e-10
    record_property("default_shunt_Y_relative",error)


def test_four_face_junction_has_complete_conservative_space():
    from witwin.maxwell.mom.layered.mesh import ClosedConductorMesh,JunctionRWGBasis
    from witwin.maxwell.mom.topology import classify_surface
    from witwin.maxwell.mom.layered.build import build_extension
    mesh=ClosedConductorMesh.__new__(ClosedConductorMesh)
    mesh.vertices=torch.tensor(((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),(0.,1.,0.),
                                (-1.,0.,0.),(0.,-1.,0.)),device="cuda",dtype=torch.float64)
    mesh.faces=torch.tensor(((0,1,2),(0,1,3),(0,1,4),(0,1,5)),device="cuda",dtype=torch.int64)
    mesh.topology=classify_surface(mesh.faces,len(mesh.vertices))
    mesh._topology_version=(id(mesh.faces),mesh.faces._version)
    basis=JunctionRWGBasis(mesh)
    assert basis.n_edges==3
    assert mesh.topology.face_count.max()==4
    identity=torch.eye(3,device="cuda",dtype=torch.float64)
    div=torch.ops.witwin_maxwell_mom_layered.divergence(
        identity,basis.face_edges,basis.face_signs,basis.edge_lengths)
    assert torch.equal(div.sum(0),torch.zeros(3,device="cuda",dtype=torch.float64))
    assert torch.linalg.matrix_rank(div).item()==3
    gram=basis.mass_blocks()
    # Arbitrary independent coefficients must dissipate positive energy, even
    # though all three functions share the first triangle of the junction.
    local=identity[:,basis.face_edges]
    energy=torch.einsum("fti,tij,ftj->f",local,gram,local)
    assert (energy>0).all()


def test_closed_r6_reference_comparison_is_evidence_only(record_property):
    import json
    from pathlib import Path
    data=Path(__file__).with_name("data")
    evidence=json.loads((data/"fixture_r4b_closed_comparison.json").read_text())
    original=json.loads((data/"fixture_r6_reference.json").read_text())
    assert evidence["qualification"]=="evidence_only"
    assert evidence["target_relative"]==.015
    assert evidence["original_r6_targets"]==original["targets"]
    assert not evidence["all_targets_met"]
    # Values are locked numerically (D-P1): test_layered_evidence_lock checks the lock records.
    comparisons=[]
    for case in evidence["cases"]:
        comparisons.append(dict(name=case["name"],unknowns=case["unknowns"],samples=case["samples"]))
    record_property("closed_vs_original_reference",json.dumps(comparisons))
