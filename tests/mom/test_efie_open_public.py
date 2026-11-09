"""Open-surface PEC EFIE through Scene -> Simulation -> Result (CUDA)."""
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.constants import C_0 as C0, MU_0
from witwin.maxwell.mom.linalg import StoppingRule

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

FREQUENCY = 3e8
WAVELENGTH = C0/FREQUENCY


def plate_scene(sources, side=0.5*WAVELENGTH, monitors=(), name='plate'):
    return mw.Scene(structures=[mw.Structure(mw.Box(size=(side, 0.8*side, 0)), mw.Material.pec(), name=name)],
                    sources=list(sources), monitors=list(monitors), device='cuda')


def config(edge=0.1*WAVELENGTH/2, **kwargs):
    return mw.MoMConfig(open_surface=mw.OpenSurfaceConfig(max_edge_length_m=edge), **kwargs)


def plane_wave(direction=(0., 0., -1.), polarization=(1., 0., 0.)):
    return mw.PlaneWave(direction=direction, polarization=polarization, source_time=mw.CW(frequency=FREQUENCY))


def unit(vector):
    v = torch.tensor(vector, dtype=torch.float64)
    return tuple((v/v.norm()).tolist())


def sphere_rule(n_theta=24, n_phi=48):
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = 2*math.pi*np.arange(n_phi)/n_phi
    ct, ph = np.meshgrid(x, phi, indexing='ij')
    st = np.sqrt(1-ct**2)
    directions = np.stack((st*np.cos(ph), st*np.sin(ph), ct), -1).reshape(-1, 3)
    weights = np.repeat(w, n_phi)*2*math.pi/n_phi
    return (torch.tensor(directions, device='cuda'), torch.tensor(weights, device='cuda'))


def test_single_layer_blocks_reproduce_the_pmchwt_electric_block_on_a_closed_surface():
    from witwin.maxwell.mom.mesh import RWGBasis, sphere_mesh
    from witwin.maxwell.mom.open_surface import OpenRWGBasis, OpenSurfaceMesh
    from witwin.maxwell.mom.operator import assemble_region_weak
    from witwin.maxwell.mom.physics import compile_medium
    from witwin.maxwell.mom.single_layer import assemble_single_layer, global_divergence
    mesh = sphere_mesh(0.1, 1)
    closed, sheet = RWGBasis(mesh), OpenRWGBasis(OpenSurfaceMesh(mesh.vertices, mesh.faces))
    medium = compile_medium(mw.Material(), 1e9, background=True)
    matrix, _ = assemble_region_weak(closed, medium)
    n = closed.n_edges
    reference = -matrix[n:, :n]/(1j*2*math.pi*1e9*MU_0)
    vector, scalar, _ = assemble_single_layer([(sheet, 0, 0)], medium.k)
    divergence = global_divergence([(sheet, 0, 0)], vector.dtype)
    actual = vector-divergence.T@scalar@divergence/medium.k**2
    assert float((actual-reference).abs().max()/reference.abs().max()) < 1e-12


def test_public_plate_solve_is_symmetric_converged_and_identified():
    result = mw.Simulation.mom(plate_scene([plane_wave()]), frequency=FREQUENCY, config=config()).run()
    data = result.surface_currents
    handle = result.solver.handle
    assert handle.metadata['formulation_id'] == 'efie_open_pec_exp_minus_iwt_v1'
    assert handle.metadata['basis_id'] == 'open_rwg_interior_edges_v1'
    assert handle.metadata['kernel_id'] == 'free_space_helmholtz' and handle.metadata['operator_backend'] == 'dense'
    assert result.solver_stats['converged']
    assert data.J.shape == (handle.system.total_edges,) and data.formulation == 'efie_open_pec'
    # Galerkin EFIE with identical trial/test spaces is complex symmetric.
    z = handle.matrix
    assert float((z-z.T).abs().max()/z.abs().max()) < 1e-12
    assert data.bodies[0].name == 'plate' and data.bodies[0].J.shape == data.J.shape


def test_lossless_plate_satisfies_the_optical_theorem():
    result = mw.Simulation.mom(plate_scene([plane_wave(unit((0.3, -0.2, -1.0)), unit((1.0, 0.0, 0.3)))]),
                               frequency=FREQUENCY, config=config()).run()
    sections = result.surface_currents.plane_wave_cross_sections(*sphere_rule())
    scattering, extinction = float(sections['scattering_m2']), float(sections['extinction_m2'])
    assert scattering > 0
    assert abs(extinction-scattering)/scattering < 5e-3


def test_bistatic_reciprocity_under_source_receiver_swap():
    a_dir, a_pol = unit((0.2, 0.1, -1.0)), unit((1.0, 0.0, 0.2))
    b_dir, b_pol = unit((-0.5, 0.3, -0.8)), unit((0.0, 0.8, 0.3))
    first = mw.Simulation.mom(plate_scene([plane_wave(a_dir, a_pol)]), frequency=FREQUENCY, config=config()).run()
    second = mw.Simulation.mom(plate_scene([plane_wave(b_dir, b_pol)]), frequency=FREQUENCY, config=config()).run()
    # F_a(-b_dir) . b_pol = F_b(-a_dir) . a_pol for unit-amplitude plane waves.
    fa, _ = first.surface_currents.far_field(-torch.tensor([b_dir], device='cuda', dtype=torch.float64))
    fb, _ = second.surface_currents.far_field(-torch.tensor([a_dir], device='cuda', dtype=torch.float64))
    left = (fa[0]*torch.tensor(b_pol, device='cuda')).sum()
    right = (fb[0]*torch.tensor(a_pol, device='cuda')).sum()
    assert float((left-right).abs()/left.abs()) < 1e-6


def test_point_dipole_reciprocity_through_the_sheet():
    p1, r1 = unit((1.0, 0.2, 0.3)), (0.05, -0.1, 0.35)
    p2, r2 = unit((0.1, 1.0, -0.4)), (-0.12, 0.08, -0.3)

    def dipole(position, moment):
        return mw.PointDipole(position=position, polarization=moment, profile='ideal',
                              source_time=mw.CW(frequency=FREQUENCY))
    first = mw.Simulation.mom(plate_scene([dipole(r1, p1)]), frequency=FREQUENCY, config=config()).run()
    second = mw.Simulation.mom(plate_scene([dipole(r2, p2)]), frequency=FREQUENCY, config=config()).run()
    e1, _ = first.query_fields(torch.tensor([r2], device='cuda', dtype=torch.float64), region_id=0, field_kind='scattered')
    e2, _ = second.query_fields(torch.tensor([r1], device='cuda', dtype=torch.float64), region_id=0, field_kind='scattered')
    left = (e1[0]*torch.tensor(p2, device='cuda')).sum()
    right = (e2[0]*torch.tensor(p1, device='cuda')).sum()
    assert float((left-right).abs()/left.abs()) < 1e-4


def test_near_sheet_fields_obey_the_current_sheet_jump_relations_and_monitors_match():
    from witwin.maxwell.constants import EPSILON_0
    result = mw.Simulation.mom(plate_scene([plane_wave(unit((0.3, 0.1, -1.0)), unit((1.0, 0.0, 0.3)))]),
                               frequency=FREQUENCY, config=config(edge=0.05*WAVELENGTH)).run()
    data = result.surface_currents
    mesh = data.mesh
    face = int(torch.argmin(mesh.centroids[:, :2].norm(dim=-1)))
    foot = mesh.centroids[face]
    height = 1e-4*WAVELENGTH
    normal = mesh.normals[face]
    points = torch.stack((foot+height*normal, foot-height*normal))
    electric, magnetic = result.query_fields(points, region_id=0, field_kind='total')
    current = data.evaluate_currents(foot[None], face_indices=torch.tensor([face], device='cuda'))[0]
    normal_c = normal.to(current.dtype)
    jump_h = magnetic[0]-magnetic[1]
    # n x (H+ - H-) = J and (E+ - E-) . n = rho_s / eps0 with rho_s = div J / (i omega).
    error_h = torch.linalg.cross(normal_c, jump_h)-current
    assert float(error_h.norm()/current.norm()) < 2e-2
    basis = data.basis
    charge = complex((data.J[basis.face_edges[face]]*(basis.face_signs[face]*basis.edge_lengths[basis.face_edges[face]]
                                                      / mesh.areas[face])).sum())/(1j*2*math.pi*FREQUENCY)
    jump_e = complex(((electric[0]-electric[1])*normal_c).sum())
    assert abs(jump_e-charge/EPSILON_0)/abs(charge/EPSILON_0) < 2e-2
    tangential = (electric[0]-electric[1])-normal_c*((electric[0]-electric[1])*normal_c).sum()
    assert float(tangential.norm()) < 2e-2*abs(charge/EPSILON_0)
    with pytest.raises(ValueError, match='on the surface'):
        result.query_fields(foot[None], region_id=0, field_kind='total')
    probe = mw.PointMonitor(name='above', position=tuple(points[0].tolist()), fields=('Ex', 'Hy'))
    monitored = mw.Simulation.mom(plate_scene(result.scene.sources, monitors=[probe]), frequency=FREQUENCY,
                                  config=config(edge=0.05*WAVELENGTH)).run()
    torch.testing.assert_close(monitored.monitor('above')['Ex'], electric[0, 0], rtol=1e-10, atol=0)
    torch.testing.assert_close(monitored.monitor('above')['Hy'], magnetic[0, 1], rtol=1e-10, atol=0)


def test_right_preconditioner_hook_accepts_the_single_current_layout():
    scene = plate_scene([plane_wave()])
    direct = mw.Simulation.mom(scene, frequency=FREQUENCY, config=config()).run()
    gmres = StoppingRule(method='gmres', rtol=1e-10, maxiter=2000, restart=200, preconditioner='none')
    for factory in (mw.IdentityRightPreconditioner(), mw.LeafBlockRightPreconditioner(16)):
        result = mw.Simulation.mom(scene, frequency=FREQUENCY,
                                   config=config(stopping=gmres, right_preconditioner=factory)).run()
        assert result.solver_stats['converged']
        assert result.solver_stats['krylov_method'] == 'flexible_right_gmres'
        error = (result.surface_currents.J-direct.surface_currents.J).norm()/direct.surface_currents.J.norm()
        assert float(error) < 1e-7


@pytest.mark.parametrize('mutate, error, match', [
    (lambda s, c: (s, mw.MoMConfig(operator_backend='hmatrix', riesz_storage='local_rwg_v1',
                                    stopping=StoppingRule(method='gmres'),
                                    open_surface=c.open_surface)), NotImplementedError, 'hierarchical'),
    (lambda s, c: (s, mw.MoMConfig(riesz_storage='compact_rwg_v1', open_surface=c.open_surface)),
     ValueError, 'riesz_storage'),
    (lambda s, c: (s, mw.MoMConfig()), ValueError, 'max_edge_length_m'),
    (lambda s, c: (mw.Scene(structures=[*s.structures, mw.Structure(mw.Sphere(position=(0, 0, 2), radius=.1),
                                                                     mw.Material(eps_r=2))],
                            sources=s.sources, device='cuda'), c), NotImplementedError, 'multiregion'),
    (lambda s, c: (mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material.pec())],
                            sources=s.sources, device='cuda'), c), NotImplementedError, 'combined-field'),
    (lambda s, c: (mw.Scene(structures=[mw.Structure(mw.Box(size=(.2, .2, 0)), mw.Medium2D(sigma_s=1e-3))],
                            sources=s.sources, device='cuda'), c), NotImplementedError, 'impedance'),
    (lambda s, c: (plate_scene([mw.PointDipole(position=(0, 0, 0.01), polarization=(1, 0, 0), profile='ideal',
                                               source_time=mw.CW(frequency=FREQUENCY))]), c),
     ValueError, 'panel radii'),
    (lambda s, c: (mw.Scene(structures=s.structures, sources=s.sources, boundary=mw.BoundarySpec.periodic(),
                            device='cuda'), c), ValueError, 'unbounded'),
])
def test_unsupported_combinations_fail_closed(mutate, error, match):
    scene, cfg = mutate(plate_scene([plane_wave()]), config())
    with pytest.raises(error, match=match):
        mw.Simulation.mom(scene, frequency=FREQUENCY, config=cfg).prepare()


def test_open_surface_config_is_rejected_by_pmchwt_and_t_junction_mesh_by_efie():
    sphere = mw.Scene(structures=[mw.Structure(mw.Sphere(radius=.1), mw.Material(eps_r=2))],
                      sources=[plane_wave()], device='cuda')
    with pytest.raises(ValueError, match='open PEC sheets'):
        mw.Simulation.mom(sphere, frequency=FREQUENCY, config=config()).prepare()
    vertices = torch.tensor([(0., 0, 0), (.1, 0, 0), (0, .1, 0), (0, -.1, 0), (0, 0, .1)], dtype=torch.float64)
    faces = torch.tensor([(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    tee = mw.Scene(structures=[mw.Structure(mw.Mesh(vertices, faces, fill_mode='surface', recenter=False),
                                            mw.Material.pec())], sources=[plane_wave()], device='cuda')
    with pytest.raises(NotImplementedError, match='junction'):
        mw.Simulation.mom(tee, frequency=FREQUENCY, config=config()).prepare()


def test_explicit_surface_mesh_matches_the_box_sheet(tmp_path):
    from witwin.maxwell.mom.open_surface import rectangular_plate_mesh
    side, width = 0.5, 0.375  # exactly representable in the float32 Box size
    plate = rectangular_plate_mesh((side, width), (10, 8))
    mesh = mw.Mesh(plate.vertices.cpu(), plate.faces.cpu(), fill_mode='surface', recenter=False)
    by_mesh = mw.Simulation.mom(mw.Scene(structures=[mw.Structure(mesh, mw.Material.pec())],
                                         sources=[plane_wave()], device='cuda'),
                                frequency=FREQUENCY, config=mw.MoMConfig()).run()
    box = mw.Scene(structures=[mw.Structure(mw.Box(size=(side, width, 0)), mw.Material.pec())],
                   sources=[plane_wave()], device='cuda')
    by_box = mw.Simulation.mom(box, frequency=FREQUENCY, config=config(edge=side/10)).run()
    torch.testing.assert_close(by_box.surface_currents.mesh.vertices, plate.vertices, rtol=0, atol=0)
    scale = by_mesh.surface_currents.J.abs().max()
    torch.testing.assert_close(by_mesh.surface_currents.J/scale, by_box.surface_currents.J/scale, rtol=0, atol=1e-11)
    with pytest.raises(NotImplementedError, match='do not support snapshot persistence'):
        by_box.surface_currents.snapshot()
    with pytest.raises(NotImplementedError, match='do not support snapshot persistence'):
        by_box.save(tmp_path/'sheet.pt')


def two_plates(sources, gap=0.15*WAVELENGTH, side=0.4*WAVELENGTH):
    plates = [mw.Structure(mw.Box(position=(0.0, 0.0, z), size=(side, 0.75*side, 0)), mw.Material.pec(), name=name)
              for z, name in ((gap/2, 'upper'), (-gap/2, 'lower'))]
    return mw.Scene(structures=plates, sources=list(sources), device='cuda')


def test_two_parallel_sheets_solve_against_lu_with_reciprocity_and_power_balance():
    from witwin.maxwell.mom.linalg import solve_reference
    a_dir, a_pol = unit((0.2, 0.1, -1.0)), unit((1.0, 0.0, 0.2))
    b_dir, b_pol = unit((-0.5, 0.3, -0.8)), unit((0.0, 0.8, 0.3))
    gmres = StoppingRule(method='gmres', rtol=1e-10, maxiter=2000, restart=2000, preconditioner='none')
    first = mw.Simulation.mom(two_plates([plane_wave(a_dir, a_pol)]), frequency=FREQUENCY,
                              config=config(stopping=gmres)).run()
    handle = first.solver.handle
    assert first.solver_stats['converged'] and len(handle.system.bodies) == 2
    assert [b.name for b in first.surface_currents.bodies] == ['upper', 'lower']
    z = handle.matrix
    assert float((z-z.T).abs().max()/z.abs().max()) < 1e-12
    reference, _ = solve_reference(handle, first.solver.rhs, StoppingRule(method='direct'))
    assert float((first.surface_currents.J-reference).norm()/reference.norm()) < 1e-8
    second = mw.Simulation.mom(two_plates([plane_wave(b_dir, b_pol)]), frequency=FREQUENCY, config=config()).run()
    fa, _ = first.surface_currents.far_field(-torch.tensor([b_dir], device='cuda', dtype=torch.float64))
    fb, _ = second.surface_currents.far_field(-torch.tensor([a_dir], device='cuda', dtype=torch.float64))
    left = (fa[0]*torch.tensor(b_pol, device='cuda')).sum()
    right = (fb[0]*torch.tensor(a_pol, device='cuda')).sum()
    assert float((left-right).abs()/left.abs()) < 1e-6
    sections = second.surface_currents.plane_wave_cross_sections(*sphere_rule())
    scattering, extinction = float(sections['scattering_m2']), float(sections['extinction_m2'])
    assert scattering > 0 and abs(extinction-scattering)/scattering < 5e-3
