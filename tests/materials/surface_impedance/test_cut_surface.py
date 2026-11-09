"""Physical cut geometry and independent positive-metric PEC constraints."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.cut_surface import clipped_surface_quadrature, compile_integrated_traces
from witwin.maxwell.fdtd.cut_surface import ElectricTraceProjection, apply_cut_surface_constraints

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native cut surfaces require CUDA.")


def _mesh(*, aligned=False):
    vertices = torch.tensor(((-.008, -.008, 0.), (.008, -.008, 0.),
                             (.008, .008, 0.), (-.008, .008, 0.)), dtype=torch.float64, device="cuda")
    if not aligned:
        vertices[:, 2] = .3*vertices[:, 0]+.001
    return mw.SurfaceMesh(vertices, ((0, 1, 2), (0, 2, 3)))


def _simulation(mesh=None, *, graph=False, source=()):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),)*3), grid=mw.GridSpec.uniform(.005),
        boundary=mw.BoundarySpec.none(), device="cuda:0",
        structures=(mw.Structure(geometry=_mesh() if mesh is None else mesh, material=mw.Material(pec=True)),),
        sources=source)
    return mw.Simulation.fdtd(scene, frequency=1.e9, cuda_graph=graph, full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="none"), run_time=mw.TimeConfig(time_steps=64, dt=5.e-12))


def test_true_clipping_preserves_oblique_area_and_linear_moment():
    mesh = _mesh()
    vertices, triangles = mesh.to_mesh()
    nodes = (torch.linspace(-.04, .04, 17, dtype=torch.float64, device="cuda"),)*3
    points, weight, _cells, normals = clipped_surface_quadrature(vertices, triangles, nodes)
    expected_area = .016**2*math.sqrt(1+.3**2)
    torch.testing.assert_close(weight.sum(), weight.new_tensor(expected_area), rtol=2.e-14, atol=0.)
    torch.testing.assert_close((weight[:, None]*points).sum(0)/weight.sum(),
                               points.new_tensor((0., 0., .001)), rtol=0., atol=1.e-17)
    assert torch.all(normals[:, 2] > 0)
    compiled = compile_integrated_traces(vertices, triangles, nodes)
    assert compiled["cell_count"] > 1
    assert compiled["constraint_count"] >= 2*compiled["cell_count"]


def test_projection_matches_independent_kkt_and_preserves_constrained_work():
    generator = torch.Generator(device="cuda").manual_seed(712)
    matrix = torch.randn((7, 19), device="cuda", dtype=torch.float64, generator=generator)
    matrix[-1] = matrix[0]+matrix[1]
    weight = torch.linspace(.1, 3., 19, device="cuda", dtype=torch.float64)
    electric = torch.randn(19, device="cuda", dtype=torch.float64, generator=generator)
    before = electric.clone()
    inverse = 1/weight
    gram = (matrix*inverse)@matrix.T
    expected = before-inverse*(matrix.T@torch.linalg.pinv(gram, hermitian=True)@(matrix@before))
    projection = ElectricTraceProjection(matrix.to_sparse(), weight)
    projection.apply(electric)
    assert projection.rank == 6
    torch.testing.assert_close(electric, expected, rtol=1.e-12, atol=1.e-13)
    torch.testing.assert_close(matrix@electric, torch.zeros(7, device="cuda", dtype=torch.float64), rtol=0., atol=2.e-13)
    assert torch.sum(weight*electric.square()) <= torch.sum(weight*before.square())
    projected = electric.clone()
    projection.apply(electric)
    torch.testing.assert_close(electric, projected, rtol=1.e-12, atol=1.e-13)
    # A constraint reaction does exactly zero work on every admissible midpoint.
    reaction = weight*(before-projected)
    assert float(torch.abs(torch.dot(reaction, projected))) < 1.e-13


def test_oblique_prepare_and_native_projection_residual():
    solver = _simulation().prepare().solver
    runtime = solver._cut_surface
    assert runtime is not None
    assert runtime["surface_count"] == 1
    for name in ("Ex", "Ey", "Ez"):
        getattr(solver, name).normal_()
    apply_cut_surface_constraints(solver)
    # Native fields are float32, while rank revelation and constraints use float64.
    gathered = runtime["electric"].clone()
    for name, group in runtime["groups"].items():
        gathered[group["positions"]] = getattr(solver, name).flatten()[group["linear_indices"]].double()
    matrix = runtime["projection"].matrix
    residual = torch.sparse.mm(matrix, gathered[:, None]).norm()
    scale = matrix.values().norm()*gathered.norm()
    assert float(residual/scale) < 1.e-7


def test_exact_planar_surface_lowers_to_existing_native_pec_edges():
    mesh = _mesh(aligned=True)
    scene = _simulation(mesh).scene
    lowered = mesh.planar_pec(scene)
    assert lowered is not None
    first = _simulation(mesh).prepare().solver
    second = _simulation(lowered).prepare().solver
    assert first._cut_surface is None
    for name in ("cex_curl", "cey_curl", "cez_curl"):
        torch.testing.assert_close(getattr(first, name), getattr(second, name), rtol=0, atol=0)


def test_bent_surface_has_positive_true_area_without_occupancy():
    mesh = mw.SurfaceMesh.bent_rectangle(width=.02, length=.016, radius=.03, segments=12, device="cuda")
    solver = _simulation(mesh).prepare().solver
    expected = 2*.03*math.sin(.02/(2*.03*12))*12*.016
    assert float(solver._cut_surface["physical_area"]) == pytest.approx(expected, rel=1.e-13)
    with pytest.raises(NotImplementedError, match="zero volume"):
        mesh.signed_distance(None, None, None)


def test_overlapping_source_is_rejected_without_deleting_its_work():
    source = mw.PointDipole(position=(0., 0., 0.), profile="ideal", polarization="Ex", source_time=mw.CW(1.e9))
    with pytest.raises(NotImplementedError, match="joint source/constraint"):
        _simulation(source=(source,)).prepare()


@pytest.mark.parametrize("bent", (False, True))
def test_source_free_full_yee_modified_energy_is_conserved(bent):
    from witwin.maxwell.fdtd.runtime.stepping import update_magnetic_fields, update_electric_fields
    mesh = mw.SurfaceMesh.bent_rectangle(width=.02, length=.016, radius=.03, segments=12,
                                         device="cuda") if bent else _mesh()
    solver = _simulation(mesh).prepare().solver
    generator = torch.Generator(device="cuda").manual_seed(33)
    electric_weight, magnetic_weight = {}, {}
    for axis, name in enumerate(("Ex", "Ey", "Ez")):
        value = getattr(solver, name)
        value[3:-3, 3:-3, 3:-3] = torch.randn(value[3:-3, 3:-3, 3:-3].shape,
            generator=generator, device="cuda")
        widths = [torch.as_tensor(getattr(solver.scene, coordinate+("_primal64" if i == axis else "_dual64")),
                                  device="cuda") for i, coordinate in enumerate(("dx", "dy", "dz"))]
        electric_weight[name] = getattr(solver, "eps_"+name).double()*widths[0][:, None, None]*widths[1][None, :, None]*widths[2][None, None, :]
    for axis, name in enumerate(("Hx", "Hy", "Hz")):
        widths = [torch.as_tensor(getattr(solver.scene, coordinate+("_dual64" if i == axis else "_primal64")),
                                  device="cuda") for i, coordinate in enumerate(("dx", "dy", "dz"))]
        magnetic_weight[name] = getattr(solver, "mu_"+name).double()*widths[0][:, None, None]*widths[1][None, :, None]*widths[2][None, None, :]
    apply_cut_surface_constraints(solver)
    energies = []
    for _ in range(512):
        previous_h = {name: getattr(solver, name).clone() for name in magnetic_weight}
        update_magnetic_fields(solver, solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
        # Exact leapfrog invariant: electric integer-time energy plus the product
        # of the two magnetic half-time states, rather than a simultaneous norm.
        energy = sum((weight*getattr(solver, name).double().square()).sum()
                     for name, weight in electric_weight.items())
        energy += sum((weight*previous_h[name].double()*getattr(solver, name).double()).sum()
                      for name, weight in magnetic_weight.items())
        energies.append(energy)
        update_electric_fields(solver, solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
        apply_cut_surface_constraints(solver)
    energies = torch.stack(energies)
    relative_drift = (energies-energies[0]).abs().max()/energies[0]
    print(dict(bent=bent, steps=512, maximum_relative_modified_energy_drift=float(relative_drift)))
    assert float(relative_drift) < 5.e-5


def test_oblique_and_bent_scene_graph_match_eager_and_resume():
    source = mw.PointDipole(position=(-.025, 0., -.02), profile="ideal", polarization="Ez", source_time=mw.CW(1.e9))
    eager = _simulation(source=(source,)).run()
    graph = _simulation(graph=True, source=(source,)).run()
    assert graph.solver._cuda_graph_active
    for name in ("Ex", "Ey", "Ez"):
        torch.testing.assert_close(graph.tensor(name), eager.tensor(name), rtol=1.e-6, atol=1.e-8)
    checkpoint = _simulation(source=(source,)).prepare().run_until(27)
    resumed = _simulation(graph=True, source=(source,)).prepare().run(resume_from=checkpoint)
    for name in ("Ex", "Ey", "Ez"):
        torch.testing.assert_close(resumed.tensor(name), eager.tensor(name), rtol=1.e-6, atol=1.e-8)


def test_cyclic_rotation_commutes_with_physical_projection():
    first = _simulation().prepare().solver
    mesh = _mesh()
    vertices, triangles = mesh.to_mesh()
    rotated = mw.SurfaceMesh(vertices[:, (1, 2, 0)], triangles)
    second = _simulation(rotated).prepare().solver
    generator = torch.Generator(device="cuda").manual_seed(128)
    for old, new in (("Ex", "Ez"), ("Ey", "Ex"), ("Ez", "Ey")):
        getattr(first, old).normal_(generator=generator)
        getattr(second, new).copy_(getattr(first, old).permute(1, 2, 0))
    apply_cut_surface_constraints(first)
    apply_cut_surface_constraints(second)
    for old, new in (("Ex", "Ez"), ("Ey", "Ex"), ("Ez", "Ey")):
        torch.testing.assert_close(getattr(second, new), getattr(first, old).permute(1, 2, 0), rtol=1.e-5, atol=3.e-7)


def test_resume_rejects_changed_cut_geometry_before_field_mutation():
    checkpoint = _simulation().prepare().run_until(27)
    mesh = _mesh()
    mesh.vertices[:, 2] += .0002
    prepared = _simulation(mesh).prepare()
    before = prepared.solver.Ez.clone()
    with pytest.raises(ValueError, match="layout|does not match"):
        prepared.run(resume_from=checkpoint)
    torch.testing.assert_close(prepared.solver.Ez, before, rtol=0, atol=0)


def test_cut_surface_native_graph_batch_has_private_projection_buffers_and_repeat_parity():
    simulations = [_simulation(source=(mw.PointDipole(name="drive", position=(-.025, 0., -.02),
        profile="ideal", polarization="Ez", source_time=mw.CW(1.e9, amplitude=amplitude)),))
        for amplitude in (1., 2.)]
    references = [simulation.run() for simulation in simulations]
    batch = mw.Simulation.batch(simulations, variation=mw.BatchVariation(sources=("drive",)),
        outputs=mw.BatchOutput(fields=("Ez",)), execution=mw.BatchExecution(devices=("cuda:0",),
        batch_size=2, graph="required", graph_steps=7))
    with batch.prepare() as prepared:
        first = prepared.run().results()
        graphs = tuple(prepared._graphs.values())
        pointers = [solver._cut_surface["electric"].data_ptr() for solver in prepared._slot.solvers]
        assert len(set(pointers)) == 2
        second = prepared.run().results()
        assert graphs == tuple(prepared._graphs.values())
        assert pointers == [solver._cut_surface["electric"].data_ptr() for solver in prepared._slot.solvers]
    for index, reference in enumerate(references):
        torch.testing.assert_close(first[index].tensor("Ez"), reference.tensor("Ez"), rtol=1.e-5, atol=1.e-8)
        torch.testing.assert_close(second[index].tensor("Ez"), first[index].tensor("Ez"), rtol=0, atol=0)


def test_overlapping_port_and_finite_copper_surface_fail_preparation():
    simulation = _simulation()
    simulation.scene.ports = (mw.LumpedPort(name="feed", positive=(0., 0., .005),
        negative=(0., 0., -.005), voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.))),)
    with pytest.raises(NotImplementedError, match="joint port/constraint"):
        simulation.prepare()
    # Finite copper SurfaceMesh is owned by the conformal two-sided thin-sheet
    # runtime, which prepares on CUDA but still rejects ports in the scene.
    simulation = _simulation()
    simulation.scene.structures = (mw.Structure(geometry=_mesh(),
        material=mw.CopperFoil(thickness=2.e-6, frequency_range=(.5e9, 2.e9))),)
    prepared = simulation.prepare()
    assert prepared.solver._thin_sheets and prepared.solver._cut_surface is None
    simulation = _simulation()
    simulation.scene.structures = (mw.Structure(geometry=_mesh(),
        material=mw.CopperFoil(thickness=2.e-6, frequency_range=(.5e9, 2.e9))),)
    simulation.scene.ports = (mw.LumpedPort(name="feed", positive=(0., 0., .005),
        negative=(0., 0., -.005), voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.))),)
    # Sheet gap feeds are admitted (D6); any other port beside a conformal sheet is rejected.
    with pytest.raises(NotImplementedError, match="support only sheet gap feeds as ports"):
        simulation.prepare()


def test_independent_pec_sphere_reference_satisfies_boundary_and_maxwell_curls():
    import numpy as np
    from scripts.dev.fdtd.pec_cut_surface_reference import pec_sphere_fields
    radius, k, eta = .02, 62.87535066, 376.730313668
    directions = np.asarray(((.3, .4, .5), (-.2, .4, -.3), (0., 0., 1.)))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    points = radius*directions
    scattered_e, _ = pec_sphere_fields(radius, k, points)
    total_e = scattered_e.copy()
    total_e[:, 0] += np.exp(1j*k*points[:, 2])
    np.testing.assert_allclose(np.cross(directions, total_e), 0., rtol=0., atol=1.e-13)
    exterior = 2*points.copy()
    exterior[2] = (.02, -.03, .04)
    delta = 1.e-7
    jac_e, jac_h = [], []
    for direction in np.eye(3):
        ep, hp = pec_sphere_fields(radius, k, exterior+delta*direction)
        em, hm = pec_sphere_fields(radius, k, exterior-delta*direction)
        jac_e.append((ep-em)/(2*delta))
        jac_h.append((hp-hm)/(2*delta))
    curl = lambda j: np.stack((j[1][:, 2]-j[2][:, 1], j[2][:, 0]-j[0][:, 2], j[0][:, 1]-j[1][:, 0]), -1)
    e, h = pec_sphere_fields(radius, k, exterior)
    np.testing.assert_allclose(curl(jac_e), 1j*k*eta*h, rtol=2.e-8, atol=1.e-8)
    np.testing.assert_allclose(curl(jac_h), -1j*k/eta*e, rtol=2.e-8, atol=1.e-10)


def test_clipped_geometry_and_zero_tangent_trace_survive_subdivision_and_normal_reversal():
    original = _mesh()
    vertices, faces = original.to_mesh()
    center = vertices.mean(0, keepdim=True)
    refined = mw.SurfaceMesh(torch.cat((vertices, center)), ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    reversed_mesh = mw.SurfaceMesh(vertices, faces.flip(1))
    solvers = [_simulation(mesh).prepare().solver for mesh in (original, refined, reversed_mesh)]
    generator = torch.Generator(device="cuda").manual_seed(591)
    for name in ("Ex", "Ey", "Ez"):
        getattr(solvers[0], name).normal_(generator=generator)
        for solver in solvers[1:]:
            getattr(solver, name).copy_(getattr(solvers[0], name))
    for solver in solvers:
        apply_cut_surface_constraints(solver)
    # Reversing normals preserves the same RWG trial space. A different
    # triangulation changes that finite current space, so compare physical area
    # and an independently known admissible trace rather than random projections.
    for name in ("Ex", "Ey", "Ez"):
        torch.testing.assert_close(getattr(solvers[2], name), getattr(solvers[0], name), rtol=2.e-5, atol=5.e-7)
    for solver in solvers:
        torch.testing.assert_close(solver._cut_surface["physical_area"], solvers[0]._cut_surface["physical_area"], rtol=2.e-14, atol=0.)
        for name, value in (("Ex", -.3), ("Ey", 0.), ("Ez", 1.)):
            getattr(solver, name).fill_(value)
        apply_cut_surface_constraints(solver)
        for name, value in (("Ex", -.3), ("Ey", 0.), ("Ez", 1.)):
            torch.testing.assert_close(getattr(solver, name), torch.full_like(getattr(solver, name), value), rtol=1.e-6, atol=1.e-7)
    nodes = (torch.linspace(-.04, .04, 17, dtype=torch.float64, device="cuda"),)*3
    with pytest.raises(NotImplementedError, match="explicit 2 row cap"):
        compile_integrated_traces(vertices, faces, nodes, maximum_constraints=2)


def test_closed_surface_queries_use_true_contact_and_winding_without_voxelization():
    from witwin.maxwell.mom.mesh import sphere_mesh
    sphere = sphere_mesh(.02, subdivisions=1)
    mesh = mw.SurfaceMesh(sphere.vertices, sphere.faces)
    points = torch.cat((sphere.vertices[:1], sphere.vertices.new_tensor(((0., 0., 0.), (.04, 0., 0.)))))
    interior, contact = mesh.classify_query_points(points)
    assert contact.tolist() == [True, False, False]
    assert interior[1:].tolist() == [True, False]


def test_cut_surface_tfsf_injection_support_is_checked_before_stepping():
    def simulation(bound):
        source = mw.PlaneWave(direction=(0.,0.,1.),polarization=(1.,0.,0.),source_time=mw.CW(1.e9),
                              injection=mw.TFSF(bounds=((-bound,bound),)*3))
        return _simulation(source=(source,))
    assert simulation(.025).prepare().solver._cut_surface is not None
    with pytest.raises(NotImplementedError, match="joint source/constraint"):
        simulation(.01).prepare()


def test_cut_surface_declines_geometry_derivatives_instead_of_detaching_them():
    width = torch.tensor(.02, device="cuda", requires_grad=True)
    with pytest.raises(NotImplementedError, match="fixed geometry"):
        mw.SurfaceMesh.bent_rectangle(width=width, length=.016, radius=.03, device="cuda")
    vertices, faces = _mesh().to_mesh()
    with pytest.raises(NotImplementedError, match="fixed geometry"):
        mw.SurfaceMesh(vertices.requires_grad_(), faces)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible CUDA devices")
def test_distributed_cut_surfaces_fail_before_unowned_projection():
    scene = _simulation().scene
    config = mw.FDTDParallelConfig(devices=("cuda:0","cuda:1"),transport="cuda_p2p",overlap=False,gather_fields=True)
    with pytest.raises(NotImplementedError,match="surface-current ownership"):
        mw.Simulation.fdtd(scene,frequency=1.e9,parallel=config).prepare()
