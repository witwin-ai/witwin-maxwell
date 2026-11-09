"""R3b bent currents, conforming vias, vertical ports and network gates."""

import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import LayeredGreenTable, assemble_layered
from witwin.maxwell.mom.layered.mesh import mesh_conductors, validate_vias
from witwin.maxwell.mom.layered.network import prepare_layered_network, solve_prepared_network
from witwin.maxwell.mom.operator import IntegrationConfig
from witwin.maxwell.mom.single_layer import assemble_single_layer
from tests.mom.layered.test_layered_network import gap_port, abcd
from tests.support.layered_via_references import rectangular_loop_inductance
from tests.support.coupled_line_references import hammerstad_jensen_microstrip

pytestmark = pytest.mark.cuda
C0, EPS0 = 299792458., 8.8541878128e-12


def vacuum():
    return mw.LayerStack((mw.Layer("air", 1e-3),), mw.HalfSpace(), mw.HalfSpace())


def bent_sheet():
    side, z0, z1 = 200e-6, 100e-6, 300e-6
    table = LayeredGreenTable.build(vacuum(), (z0,z1), (10e9,), 1.05*math.sqrt(2)*side,
                                   via_spans=((z0,z1),), extract_distance=1e-3, device="cuda")
    layout = mesh_conductors([(0,side,0,side,z0)], table.planes, side/5,
                             bridges=[(0,0.,0.,side,z0,z1,(z0+z1)/2,side/2,1.)])
    return layout, table


def test_bent_homogeneous_single_layer(record_property):
    layout, table = bent_sheet()
    z = assemble_layered(layout, table)
    k = 2*math.pi*10e9/C0
    config = IntegrationConfig(order=8, singular_order=16, near_tolerance=1e-7,
                               near_max_order=32, pair_batch=4, far_pair_batch=64)
    vector, scalar, _ = assemble_single_layer(((layout.basis,0,0),), k, config)
    div = layout.basis.divergence.to(torch.complex128)
    reference = -1j*(2*math.pi*10e9)/(EPS0*C0**2)*(vector-div.T@scalar@div/k**2)
    whole = (torch.linalg.vector_norm(z-reference)/torch.linalg.vector_norm(reference)).item()
    basis, mesh = layout.basis, layout.mesh
    supports = torch.stack((basis.plus_faces,basis.minus_faces),-1)
    centres = mesh.centroids[supports]
    edge2 = (mesh.triangles-mesh.triangles.roll(1,1)).square().sum(-1).amax(-1)
    d2 = (centres[:,None,:,None]-centres[None,:,None]).square().sum(-1)
    threshold = 4*torch.maximum(edge2[supports][:,None,:,None],edge2[supports][None,:,None])
    far = (d2>=threshold).all(dim=(-2,-1))
    far_error = ((z[far]-reference[far]).abs()/reference[far].abs()).max().item()
    reciprocal = ((z-z.T).abs().max()/z.abs().max()).item()
    record_property("whole_frobenius",whole)
    record_property("far_entry_error",far_error)
    record_property("transpose_error",reciprocal)
    assert whole < 1e-5
    assert far_error < 1e-9
    assert reciprocal < 1e-12
    for _ in range(2):
        assert torch.equal(z,assemble_layered(layout,table))


def vertical_loop(frequencies=(1e6,1e8,1e9), *, lossy=False):
    length, height, width, via = 200e-6, 200e-6, 10e-6, 4e-6
    za, zb, gap = 100e-6, 300e-6, width
    rectangles = [(-width,length+width,-width/2,width/2,zb),
                  (-width,length/2-gap/2,-width/2,width/2,za),
                  (length/2+gap/2,length+width,-width/2,width/2,za)]
    vias = [(-via/2,via/2,-via/2,via/2,za,zb),
            (length-via/2,length+via/2,-via/2,via/2,za,zb)]
    port = gap_port("loop",(length/2+gap/2,0,za),(length/2-gap/2,0,za),"x",width)
    prepared = prepare_layered_network(vacuum(),rectangles,[port],frequencies,
                                       bridge_width=width,max_edge=20e-6,edge_cell=2e-6,vias=vias,
                                       via_materials=[(5.8e7,0.,None)]*2 if lossy else None)
    return prepared, rectangles, vias, port, (length,height,width,via)


def test_vertical_loop_inductance_kcl_and_flatness(record_property):
    prepared, _, vias, _, dimensions = vertical_loop()
    network, currents = solve_prepared_network(prepared)
    inductance = -network.to_z()[:,0,0].imag/(2*math.pi*prepared.table.frequencies)
    expected = rectangular_loop_inductance(*dimensions)
    relative = abs(inductance[0].item()/expected-1)
    flat = ((inductance.max()-inductance.min())/inductance[0]).item()
    basis, mesh = prepared.layout.basis, prepared.layout.mesh
    a,b = mesh.vertices[basis.edges[:,0]],mesh.vertices[basis.edges[:,1]]
    flow = mesh.centroids[basis.minus_faces]-mesh.centroids[basis.plus_faces]
    terminal = (prepared.excitation[:,0]*currents[0,:,0]).sum()
    kcl = []
    # At 1 MHz the loop is magnetostatic to (k*a)^2 < 2e-11. At higher
    # frequencies displacement current permits physical variation along it.
    for x0,x1,y0,y1,za,zb in vias:
        selected = ((a[:,2]==b[:,2]) & (a[:,2]>za) & (a[:,2]<zb)
                    & (a[:,0]>=x0)&(a[:,0]<=x1)&(b[:,0]>=x0)&(b[:,0]<=x1)
                    & (a[:,1]>=y0)&(a[:,1]<=y1)&(b[:,1]>=y0)&(b[:,1]<=y1))
        for z in a[selected,2].unique():
            cross = selected & (a[:,2]==z)
            current = (torch.sign(flow[cross,2])*basis.edge_lengths[cross]*currents[0,cross,0]).sum()
            kcl.append(abs(abs((current/terminal).item())-1))
    record_property("inductance_H",inductance.tolist())
    record_property("reference_H",expected)
    record_property("inductance_relative",relative)
    record_property("flatness",flat)
    record_property("kcl_max",max(kcl))
    assert relative < .03
    assert flat < .002
    assert max(kcl) < 1e-8
    chunked,chunked_currents=solve_prepared_network(prepared,frequency_chunk_size=1)
    assert torch.equal(network.s,chunked.s)
    assert torch.equal(currents,chunked_currents)
    assert prepared.layout.mesh.topology.junction_edges == 0
    assert bool((mesh.normals[prepared.layout.triangle_planes<0,2]==0).all())
    topology=mesh.topology
    a,b=mesh.vertices[topology.edges[:,0]],mesh.vertices[topology.edges[:,1]]
    for x0,x1,y0,y1,za,zb in vias:
        for z in (za,zb):
            rim=((a[:,2]==z)&(b[:,2]==z)&(a[:,0]>=x0)&(a[:,0]<=x1)
                 &(b[:,0]>=x0)&(b[:,0]<=x1)&(a[:,1]>=y0)&(a[:,1]<=y1)
                 &(b[:,1]>=y0)&(b[:,1]<=y1))
            assert bool(rim.any())
            assert bool((topology.face_count[rim]==2).all())
    record_property("peak_allocated_bytes",torch.cuda.max_memory_allocated())


def test_vertical_port_to_ground(record_property):
    # f*h/c < 8.5e-5: transverse fields are quasi-static. Two identical
    # bridge error boxes cancel in the two-line trace extraction.
    h, frequency, width = 254e-6, 100e6, 254e-6
    stack = mw.LayerStack((mw.Layer("substrate",h,9.8),),"pec",mw.HalfSpace())
    matrices = []
    for length in (2e-3,4e-3):
        rectangles = [(0.,length,-width/2,width/2,h)]
        ports = [gap_port(name,(x,0,h),(x,0,0.),"z",width)
                 for name,x in (("p1",0.),("p2",length))]
        prepared = prepare_layered_network(stack,rectangles,ports,(frequency,),
                                           bridge_width=width,max_edge=h/2,edge_cell=width/32)
        network,_ = solve_prepared_network(prepared)
        record_property(f"unknowns_length_{length}",prepared.layout.basis.n_edges)
        record_property(f"z_admission_length_{length}",prepared.table.vertical.z_interpolation_error.item())
        assert prepared.layout.grounded
        assert bool((prepared.layout.basis.minus_faces<0).any())
        matrices.append(abcd(network.to_z()[0]))
    difference = matrices[1]@torch.linalg.inv(matrices[0])
    gamma = torch.arccosh(torch.trace(difference)/2).item()
    eps_eff = (abs(gamma.imag)/2e-3*C0/(2*math.pi*frequency))**2
    z0 = abs(torch.sqrt(difference[0,1]/difference[1,0]).item())
    expected_eps,expected_z0 = hammerstad_jensen_microstrip(1.,9.8)
    record_property("eps_eff",eps_eff)
    record_property("Z0",z0)
    record_property("eps_eff_reference",expected_eps)
    record_property("Z0_reference",expected_z0)
    assert abs(eps_eff/expected_eps-1) < .01
    assert abs(z0/expected_z0-1) < .02


def test_via_placement_rejections():
    rectangles = [(0.,2.,0.,2.,z) for z in (0.,1.,2.)]
    with pytest.raises(ValueError,match="endpoints"):
        validate_vias(rectangles,[(.5,1.5,.5,1.5,0.,.9)])
    with pytest.raises(NotImplementedError,match="consecutive"):
        validate_vias(rectangles,[(.5,1.5,.5,1.5,0.,2.)])
    with pytest.raises(ValueError,match="footprint"):
        validate_vias(rectangles,[(-.1,1.5,.5,1.5,0.,1.)])
    with pytest.raises(ValueError,match="increasing"):
        validate_vias(rectangles,[(1.,.5,.5,1.5,0.,1.)])
    with pytest.raises(NotImplementedError,match="junction"):
        validate_vias(rectangles,[(.5,1.5,.5,1.5,0.,1.)]*2)
    with pytest.raises(ValueError,match="footprint rim"):
        validate_vias(rectangles,[(0.,1.,.5,1.5,0.,1.)])


def underpass(lossy=False):
    stack = mw.LayerStack((mw.Layer("silicon",300e-6,11.9,10.),
                           mw.Layer("oxide",4e-6,4.1)),mw.HalfSpace(11.9,10.),mw.HalfSpace())
    za,zb = 300.5e-6,302.5e-6
    sheets = [(-20e-6,10e-6,-5e-6,5e-6,zb),(90e-6,120e-6,-5e-6,5e-6,zb),
              (-5e-6,105e-6,-5e-6,5e-6,za),(-20e-6,120e-6,-20e-6,-10e-6,zb)]
    vias = [(-2e-6,2e-6,-2e-6,2e-6,za,zb),(98e-6,102e-6,-2e-6,2e-6,za,zb)]
    ports = [gap_port(name,(x,-5e-6,zb),(x,-10e-6,zb),"y",4e-6)
             for name,x in (("p1",-15e-6),("p2",115e-6))]
    return prepare_layered_network(stack,sheets,ports,(10e9,),bridge_width=4e-6,max_edge=10e-6,
                                    vias=vias,via_materials=[(5.8e7,0.,None)]*2 if lossy else None)


def test_rfic_underpass_reciprocity_passivity_loss_and_repeat(record_property):
    prepared = underpass()
    network,_ = solve_prepared_network(prepared)
    y = network.to_y()
    reciprocal = (torch.linalg.matrix_norm(y-y.transpose(-1,-2))/torch.linalg.matrix_norm(y)).max().item()
    passive = torch.linalg.matrix_norm(network.s,ord=2).max().item()
    for _ in range(2):
        again,_ = solve_prepared_network(underpass())
        assert torch.equal(network.s,again.s)
        assert torch.equal(y,again.to_y())
    lossy,_ = solve_prepared_network(underpass(True))
    # Re Z11 is dominated by the lossy silicon; the series path Re(-1/Y21) isolates
    # the via walls. Skin-effect estimate for two 4x4x2 um copper vias at 10 GHz:
    # 2*Rs*h/perimeter (uniform wall current; corner crowding raises it slightly).
    r0,r1 = ((-1/n.to_y()[0,0,1]).real.item() for n in (network,lossy))
    sigma,frequency = 5.8e7,10e9
    rs = 1/(sigma*math.sqrt(2/(2*math.pi*frequency*4e-7*math.pi*sigma)))
    expected = 2*rs*2e-6/16e-6
    record_property("reciprocity",reciprocal)
    record_property("largest_singular_S",passive)
    record_property("PEC_series_R_ohm",r0)
    record_property("lossy_via_series_R_ohm",r1)
    record_property("via_series_R_estimate_ohm",expected)
    assert reciprocal < 1e-10
    assert passive <= 1+1e-9
    assert abs((r1-r0)/expected-1) < .25


def public_via_scene(material=None):
    # Binary-exact coordinates preserve the declared plane/Box endpoint identity.
    u = 2.**-20
    za,zb = 64*u,128*u
    sheets = [(0.,32*u,-8*u,8*u,za),(0.,32*u,-8*u,8*u,zb),
              (40*u,64*u,-8*u,8*u,za)]
    via = (12*u,20*u,-4*u,4*u,za,zb)
    geometry = mw.Box(position=torch.tensor((16*u,0.,96*u),dtype=torch.float64),
                      size=torch.tensor((8*u,8*u,64*u),dtype=torch.float64))
    structures = [mw.Structure(mw.PlanarPEC([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
                  normal_axis="z",coordinate=z),mw.Material.pec()) for x0,x1,y0,y1,z in sheets]
    structures.append(mw.Structure(geometry,mw.Material.pec() if material is None else material))
    scene = mw.Scene(background=vacuum(),structures=structures)
    port = gap_port("p",(40*u,0.,za),(32*u,0.,za),"x",8*u)
    scene.add_port(port)
    return scene,sheets,via,port


def test_public_box_via_equals_internal_and_rejects_material(record_property):
    scene,sheets,via,port = public_via_scene()
    u=2.**-20
    config = mw.MoMConfig(layered=mw.LayeredSurfaceConfig(bridge_width_m=8*u,max_edge_length_m=16*u))
    result = mw.Simulation.mom(scene,frequencies=(1e9,),config=config).run()
    internal = prepare_layered_network(vacuum(),sheets,[port],(1e9,),bridge_width=8*u,
                                       max_edge=16*u,vias=[via])
    reference,_ = solve_prepared_network(internal)
    assert torch.equal(result.network.s,reference.s)
    record_property("public_unknowns",internal.layout.basis.n_edges)
    invalid,_,_,_ = public_via_scene(mw.Material(eps_r=2.))
    with pytest.raises(ValueError,match="via requires"):
        mw.Simulation.mom(invalid,frequency=1e9,config=config).prepare()
    misplaced,_,_,_ = public_via_scene()
    misplaced.structures[-1].geometry.size[2] /= 2
    with pytest.raises(ValueError,match="endpoints"):
        mw.Simulation.mom(misplaced,frequency=1e9,config=config).prepare()


def test_public_float32_box_via_snaps_to_planes():
    # A float32 micrometre Box misses the planes by ~1e-11 m; compilation snaps
    # bounds within its rounding scale to the declared sheet coordinates.
    za,zb = 300.5e-6,302.5e-6
    stack = mw.LayerStack((mw.Layer("silicon",300e-6,11.9,10.),mw.Layer("oxide",4e-6,4.1)),
                          mw.HalfSpace(11.9,10.),mw.HalfSpace())
    scene = mw.Scene(background=stack)
    scene.add_port(gap_port("p",(0.,0.,za),(0.,0.,zb),"z",4e-6))
    for z in (za,zb):
        scene.add_structure(mw.Structure(mw.PlanarPEC([(0.,-5e-6),(30e-6,-5e-6),(30e-6,5e-6),(0.,5e-6)],
                                                      normal_axis="z",coordinate=z),mw.Material.pec()))
    scene.add_structure(mw.Structure(mw.Box(position=(15e-6,0.,(za+zb)/2),size=(4e-6,4e-6,zb-za)),
                                     mw.Material.pec()))
    assert scene.structures[-1].geometry.position.dtype == torch.float32
    from witwin.maxwell.mom.layered.runtime import compile_layered_scene
    *_,vias,_ = compile_layered_scene(scene)
    assert (vias[0][4],vias[0][5]) == (za,zb)


def test_wall_quadrature_precision_calibration(record_property):
    u=2.**-20
    table=LayeredGreenTable.build(vacuum(),(4*u,8*u),(10e9,),10*u,device="cuda",
                                  via_spans=((4*u,8*u),),extract_distance=1e-3)
    tube=mesh_conductors([(-3*u,3*u,-3*u,3*u,z) for z in (4*u,8*u)],
                         table.planes,2*u,vias=[(-u,u,-u,u,4*u,8*u)])
    stack=mw.LayerStack((mw.Layer("silicon",300*u,11.9,10.),mw.Layer("oxide",4*u,4.1)),
                         mw.HalfSpace(11.9,10.),mw.HalfSpace())
    za,zb=300.5*u,302.5*u
    layered=LayeredGreenTable.build(stack,(za,zb),(10e9,),10*u,device="cuda",
                                    via_spans=((za,zb),),extract_distance=4*u)
    layered_tube=mesh_conductors([(-3*u,3*u,-3*u,3*u,z) for z in (za,zb)],
                                 layered.planes,2*u,vias=[(-u,u,-u,u,za,zb)])
    for fixture,(layout,green) in (("bend",bent_sheet()),("tube",(tube,table)),
                                    ("layered_tube",(layered_tube,layered))):
        reference=assemble_layered(layout,green,far_order=12,near_order=32,_tiles=True)
        actual=assemble_layered(layout,green,far_dtype=torch.float32,_network_accuracy=True,_tiles=True)
        wall=layout.triangle_planes<0
        centres=layout.mesh.centroids
        sizes=(layout.mesh.triangles-layout.mesh.triangles.roll(1,1)).square().sum(-1).amax(-1)
        ratio=((centres[:,None]-centres[None,:]).square().sum(-1)/
               torch.maximum(sizes[:,None],sizes[None,:])).sqrt()
        normal=layout.mesh.normals
        parallel=(normal[:,None]*normal[None,:]).sum(-1).abs()>.5
        # Entry-error calibration: maximum block-entry error divided by maximum
        # reference block entry in each pair/distance class, V and P separately.
        for name,pairs in (("sheet_wall",wall[:,None]^wall[None,:]),
                           ("wall_wall_parallel",wall[:,None]&wall[None,:]&parallel),
                           ("wall_wall_perpendicular",wall[:,None]&wall[None,:]&~parallel)):
            for lo,hi in ((0.,2.),(2.,4.),(4.,8.),(8.,float("inf"))):
                selected=pairs&(ratio>=lo)&(ratio<hi)
                if not bool(selected.any()):
                    continue
                for term,sl in (("V",slice(0,9)),("P",slice(9,10))):
                    ref=reference[selected][:,sl]
                    error=((actual[selected][:,sl]-ref).abs().max()/ref.abs().max()).item()
                    record_property(f"{fixture}_{name}_{lo:g}_{hi:g}_{term}",error)
                    assert error < 1e-6


def test_ground_half_rwg_three_complete_runs(record_property):
    u=2.**-20
    stack=mw.LayerStack((mw.Layer("air",128*u),),"pec",mw.HalfSpace())
    sheets=[(0.,64*u,-4*u,4*u,64*u)]
    ports=[gap_port(name,(x,0.,64*u),(x,0.,0.),"z",8*u)
           for name,x in (("p1",0.),("p2",64*u))]
    first=None
    for _ in range(3):
        prepared=prepare_layered_network(stack,sheets,ports,(1e6,1e9),
                                         bridge_width=8*u,max_edge=16*u)
        network,current=solve_prepared_network(prepared)
        assert prepared.layout.grounded
        half=prepared.layout.basis.minus_faces<0
        assert half.sum().item()==2
        if first is None:
            first=(network.s,current)
        else:
            assert torch.equal(first[0],network.s)
            assert torch.equal(first[1],current)
    record_property("half_functions",int(half.sum()))
    record_property("unknowns",prepared.layout.basis.n_edges)


def test_closed_form_static_cross_integrals(record_property):
    from tests.support.layered_via_references import static_cross_triangle
    from witwin.maxwell.mom.layered.assembly import triangle_cross_integrals
    horizontal=((0.,0.,0.),(1.,0.,0.),(.2,.8,0.))
    vertical=((0.,0.,-1.),(0.,1.,-1.),(0.,.2,0.))
    cases=[]
    for x,y in ((.25,.2),(.5,0.),(0.,0.),(1.1,-.1)):
        for z in (0.,.001,.1,1.,-.1):
            cases.append((horizontal,(x,y,z),1. if z>=0 else -1.))
    for x in (0.,1e-8,.001,.1,1.,-.1):
        for z in (0.,.1):
            cases.append((vertical,(x,.2,z),1.))
            cases.append((tuple((y,x,z) for x,y,z in vertical),(.2,x,z),1.))
    right=((0.,0.,-1.),(0.,1.,-1.),(0.,0.,0.))
    for x,y,z in ((.001,.2,0.),(.1,.2,.1),(-.1,.4,0.),(0.,.2,0.)):
        cases.append((right,(x,y,z),1.))
        cases.append((tuple((a,b,-c) for a,b,c in right),(x,y,-z),-1.))
    triangles=torch.tensor([c[0] for c in cases],dtype=torch.float64,device="cuda")
    observers=torch.tensor([c[1] for c in cases],dtype=torch.float64,device="cuda")
    beta=torch.tensor([c[2] for c in cases],dtype=torch.float64,device="cuda")
    import numpy as np
    reference=torch.as_tensor(np.stack([static_cross_triangle(*c) for c in cases]),
                              dtype=torch.float64,device="cuda")
    first=triangle_cross_integrals(triangles,observers,beta)
    relative=((first-reference).abs().amax(-1)/reference.abs().amax(-1)).max().item()
    record_property("cross_primitive_relative",relative)
    assert relative < 1e-10
    shift=triangles.new_tensor((.002,-.003,.0003))
    scaled=triangle_cross_integrals(triangles*1e-6+shift,observers*1e-6+shift,beta)
    small=((scaled-reference*1e-12).abs().amax(-1)/(reference.abs().amax(-1)*1e-12)).max().item()
    record_property("shifted_micro_cross_relative",small)
    assert small < 1e-10
    for _ in range(2):
        assert torch.equal(first,triangle_cross_integrals(triangles,observers,beta))
