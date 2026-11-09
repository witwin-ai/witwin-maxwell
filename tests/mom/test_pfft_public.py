"""Public pFFT backend: precorrection, PMCHWT action, admission and fail-closed combinations."""
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.mesh import sphere_mesh
from witwin.maxwell.mom.nearfield import scalar_regional_channels
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
FREQUENCY = 299792458.0
INTEGRATION = IntegrationConfig(order=4, singular_order=4)


def _scene():
    structures = []
    for centre, eps in (((-.4, 0., 0.), 4.), ((.45, .1, 0.), 2.5)):
        mesh = sphere_mesh(radius=.25, frequency=5, center=centre, device='cuda')
        structures.append(mw.Structure(mw.Mesh(mesh.vertices, mesh.faces, recenter=False, fill_mode='solid'),
                                       mw.Material(eps_r=eps)))
    return mw.Scene(structures=structures, sources=[mw.PlaneWave()])


def _pfft_config(**kwargs):
    return mw.MoMConfig(integration=INTEGRATION, operator_backend='pfft', riesz_storage='local_rwg_v1',
                        # Original sampled rows see the accelerator error (about 1e-3 here), so their
                        # budget sits above it; the compressed dual residual is checked tightly.
                        pfft=mw.PFFTConfig(**{'points_per_wavelength': 15., 'order': 3,
                                              'solve_dual_tolerance': 1e-4, 'original_sample_tolerance': 3e-2,
                                              **kwargs.pop('pfft', {})}),
                        # complex64 grids put a floor near 1e-6 under the achievable residual.
                        stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal', rtol=1e-5, atol=0.,
                                              maxiter=2000, restart=300, stagnation_window=300), **kwargs)


def _body_lu(context):
    blocks = []
    for body in context.system.bodies:
        edges = torch.arange(body.edge_offset, body.edge_offset+body.basis.n_edges, device=context.device)
        blocks.append((context.coefficient_indices(edges), torch.linalg.lu_factor(context.principal_block(edges))))

    def apply(residual):
        result = torch.empty_like(residual)
        for index, (factor, pivot) in blocks:
            result[index] = torch.linalg.lu_solve(factor, pivot, residual[index, None])[:, 0]
        return result
    return apply


def _relative(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))


@pytest.fixture(scope='module')
def prepared():
    dense = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=mw.MoMConfig(
        integration=INTEGRATION, riesz_storage='compact_rwg_v1',
        stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))).prepare()
    fast = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=_pfft_config()).prepare()
    return dense, fast


def test_precorrection_exact_entries_match_per_block_integration(prepared):
    _, fast = prepared
    handle = fast.solver.handle
    region = handle.actions.interiors[0]
    correction = region.precorrection
    body = handle.system.bodies[0]
    pick = torch.linspace(0, len(correction.rows)-1, 64, device='cuda').long()
    rows, cols = correction.rows[pick], correction.cols[pick]
    grid_d, grid_k = region.region.grid_entries(rows, cols)
    geometry = (handle.provider.geometry[0],)*2
    for i in range(len(rows)):
        d, k, _ = scalar_regional_channels(body.basis, body.basis, rows[i:i+1], cols[i:i+1], body.interior,
                                           INTEGRATION, geometry=geometry)
        assert abs(complex(correction.delta_d[pick[i]]+grid_d[i]-d[0, 0])) <= 1e-14*abs(complex(d[0, 0]))+1e-30
        assert abs(complex(correction.delta_k[pick[i]]+grid_k[i]-k[0, 0])) <= 1e-14*max(abs(complex(k[0, 0])),
                                                                                   abs(complex(d[0, 0])))+1e-30


def test_system_action_and_adjoint_match_dense(prepared):
    dense, fast = prepared
    generator = torch.Generator(device='cuda').manual_seed(4)
    n = fast.solver.handle.shape[0]
    x, y = (torch.randn(n, generator=generator, device='cuda', dtype=torch.complex128) for _ in range(2))
    exact = dense.solver.handle.apply(x)
    assert _relative(fast.solver.handle.apply(x), exact) < 3e-3
    handle = fast.solver.handle
    left = torch.vdot(y, handle.apply(x))
    right = torch.vdot(handle.adjoint(y), x)
    # The adjoint is the exact conjugate transpose of the same algebra; agreement is
    # limited by the complex64 grid transforms of the default configuration.
    assert abs(complex(left-right)) <= 1e-6*abs(complex(left))


def test_public_solve_matches_dense_lu_with_admission(prepared):
    dense, _ = prepared
    reference = dense.run()
    config = _pfft_config(right_preconditioner=mw.FunctionRightPreconditioner(_body_lu, name='body_lu'))
    result = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=config).run()
    stats = result.stats()
    assert stats['iterations'] < 60, (stats['reason'], stats['relative_residual'])
    assert stats['converged'] and stats['error_budget']['checked']
    assert stats['error_budget']['original_sampled_pass']
    x, x_ref = result.surface_currents.scattered_coefficients, reference.surface_currents.scattered_coefficients
    directions = torch.nn.functional.normalize(torch.randn((64, 3), device='cuda', dtype=torch.float64), dim=-1)
    far, _ = result.surface_currents.far_field(directions)
    far_ref, _ = reference.surface_currents.far_field(directions)
    assert _relative(far, far_ref) < 5e-3
    assert _relative(x, x_ref) < 2e-2
    with pytest.raises(NotImplementedError):
        result.surface_currents.snapshot()


def test_right_preconditioner_hook_on_pfft():
    config = _pfft_config(right_preconditioner=mw.LeafBlockRightPreconditioner(32))
    result = mw.Simulation.mom(_scene(), frequency=FREQUENCY, config=config).run()
    stats = result.stats()
    assert stats['converged'] and stats['krylov_method'] == 'flexible_right_gmres', (stats['reason'], stats.get('relative_residual'))
    assert stats['right_preconditioner']['block_source'] == 'hierarchical'


def test_unsupported_combinations_fail_closed():
    good = dict(operator_backend='pfft', riesz_storage='local_rwg_v1',
                stopping=StoppingRule(method='gmres', preconditioner='rwg_diagonal'))
    with pytest.raises(ValueError):
        mw.MoMConfig(**{**good, 'stopping': StoppingRule(method='direct')})
    with pytest.raises(ValueError):
        mw.MoMConfig(**{**good, 'stopping': StoppingRule(method='gmres', preconditioner='row_norm')})
    with pytest.raises(ValueError):
        mw.MoMConfig(**{**good, 'dtype': torch.complex64})
    with pytest.raises(ValueError):
        mw.MoMConfig(**good, fmm=mw.FMMConfig())
    with pytest.raises(NotImplementedError):
        mw.MoMConfig(**good, periodic=mw.MoMPeriodicConfig(vectors_m=((1., 0., 0.),)))
    with pytest.raises(ValueError):
        mw.MoMConfig(pfft=mw.PFFTConfig())
    with pytest.raises(NotImplementedError, match='hmatrix/h2/direct_h2/fmm/pfft'):
        mw.MoMConfig(**{**good, 'operator_backend': 'precorrected'})
    with pytest.raises(ValueError):
        mw.PFFTConfig(points_per_wavelength=2.)
    payload = mw.PFFTConfig(order=4).payload()
    assert mw.PFFTConfig.restore(payload) == mw.PFFTConfig(order=4)
    with pytest.raises(ValueError):
        mw.PFFTConfig.restore({**payload, 'version': 'pfft_precorrected_config_v0'})


def test_grid_spacing_is_tied_to_the_mesh_and_recorded():
    from witwin.maxwell.mom.pfft_config import MINIMUM_SPACING_TO_EDGE
    from witwin.maxwell.mom.physics import compile_medium
    mesh = sphere_mesh(radius=.25, frequency=5, device='cuda')
    from witwin.maxwell.mom.mesh import RWGBasis, SurfaceMesh
    basis = RWGBasis(SurfaceMesh(mesh.vertices, mesh.faces))
    edge = float(basis.edge_lengths.mean())
    vacuum = compile_medium(mw.Material(), FREQUENCY, background=True)
    wavelength = 2*3.141592653589793/complex(vacuum.k).real
    spec = mw.PFFTConfig().grid_spec(vacuum, [basis])
    assert spec.order == 3 and spec.mean_edge == pytest.approx(edge)
    assert spec.spacing == pytest.approx(min(wavelength/15, edge)) and spec.spacing_rule in ('wavelength', 'mesh')
    # A tighter target raises the stencil order; the spacing never drops below a quarter edge.
    assert mw.PFFTConfig(target_accuracy=1e-4).grid_spec(vacuum, [basis]).order == 4
    dense = compile_medium(mw.Material(eps_r=400.), FREQUENCY)
    floor = mw.PFFTConfig().grid_spec(dense, [basis])
    assert floor.spacing == pytest.approx(MINIMUM_SPACING_TO_EDGE*edge) and floor.spacing_rule == 'quarter_edge_floor'
    with pytest.raises(ValueError, match='quarter|0.25'):
        mw.PFFTConfig(points_per_wavelength=wavelength/(0.2*edge)).grid_spec(vacuum, [basis])
    with pytest.raises(ValueError):
        mw.PFFTConfig(target_accuracy=1e-6)


def test_prepared_receipt_records_the_mesh_tied_grid(prepared):
    _, fast = prepared
    receipt = fast.solver.handle.persistence_receipt()['construction']
    projection = receipt['regions']['exterior']['projection']
    assert projection['spacing_rule'] == 'explicit_points_per_wavelength'
    assert projection['spacing_to_mean_edge'] > 0.25
    assert receipt['configuration']['spacing_rule'] == 'pfft_mesh_tied_spacing_v1'


@pytest.mark.parametrize('kind', ['rounded_sliver_cube', 'thin_slab'])
def test_precorrection_follows_the_dense_rules_on_sliver_meshes(kind):
    import math
    from witwin.maxwell.mom.kernels import FREE_SPACE
    from witwin.maxwell.mom.operator import _assemble_scalar_regions
    from witwin.maxwell.mom.primitive_mesh import rounded_box_mesh, slab_mesh
    from witwin.maxwell.mom.quadrature import anisotropic_touching_pairs
    h = 1/math.sqrt(6.)/8  # metres at 3e8 Hz, the polyhedral study's coarse level
    if kind == 'rounded_sliver_cube':
        # One arc segment per 45 degrees on r = 2 mm edges: 25:1 strips.
        surface = rounded_box_mesh((.2,)*3, .002, edge_length=h, max_arc_angle=math.pi/4)
    else:
        surface = slab_mesh(.3, .3, .01, edge_length=h)
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(surface.vertices, surface.faces, recenter=False,
                                                      fill_mode='solid'), mw.Material(eps_r=2.))],
                     sources=[mw.PlaneWave(source_time=mw.CW(frequency=3e8))])
    fast = mw.Simulation.mom(scene, frequency=3e8, config=_pfft_config(pfft={'points_per_wavelength': None,
                                                                             'order': None})).prepare()
    handle = fast.solver.handle
    basis, n = handle.system.bodies[0].basis, handle.system.total_edges
    triangles = basis.mesh.triangles
    assert bool(anisotropic_touching_pairs(triangles, triangles).any())
    for region in (handle.actions.exterior, handle.actions.interiors[0]):
        correction, medium = region.precorrection, region.medium
        grid_d, grid_k = region.region.grid_entries(correction.rows, correction.cols)
        matrix, metadata = _assemble_scalar_regions(basis, (medium,), INTEGRATION, kernel=FREE_SPACE)
        assert metadata['quadrature_error_estimate'].get('touching_anisotropic')
        dense_d = matrix[:n, n:][correction.rows, correction.cols]/(1j*medium.omega*medium.epsilon)
        dense_k = matrix[:n, :n][correction.rows, correction.cols]
        scale = dense_d.abs().max()
        assert float((correction.delta_d+grid_d-dense_d).abs().max()/scale) < 1e-12
        assert float((correction.delta_k+grid_k-dense_k).abs().max()/dense_k.abs().max()) < 1e-12
        del matrix
    dense = mw.Simulation.mom(scene, frequency=3e8, config=mw.MoMConfig(
        integration=INTEGRATION, riesz_storage='compact_rwg_v1',
        stopping=StoppingRule(method='direct', rtol=1e-10, atol=0.))).prepare()
    x = torch.randn(2*n, generator=torch.Generator(device='cuda').manual_seed(9), device='cuda',
                    dtype=torch.complex128)
    assert _relative(handle.apply(x), dense.solver.handle.apply(x)) < 3e-3


def test_pfft_prepared_rhs_matches_dense_lifting_and_batch_receipt():
    from witwin.maxwell.mom.lifting import build_lifted_rhs
    surface = sphere_mesh(radius=.5, frequency=4, device='cuda')
    vertices = surface.vertices
    scene = mw.Scene(structures=[mw.Structure(mw.Mesh(vertices, surface.faces, recenter=False, fill_mode='solid'),
                                            mw.Material(eps_r=4.))], sources=[mw.PlaneWave()])
    fast = mw.Simulation.mom(scene, frequency=FREQUENCY,
        config=_pfft_config(pfft={'points_per_wavelength': None, 'order': None})).prepare()
    handle = fast.solver.handle
    dense, _ = build_lifted_rhs(handle.body_handles[0])
    assert _relative(fast.solver.rhs, dense) < 3e-3
    batched, (decoder,) = handle.build_rhs_batch((scene.sources,))
    assert _relative(fast.solver.rhs, batched[0]) < 1e-14
    assert handle.metadata['rhs_identity'] == decoder.metadata['rhs_identity']
    without_time = lambda record: {k: v for k, v in record.items() if k != 'rhs_seconds'}
    assert without_time(handle.metadata['rhs_bodies'][0]) == without_time(decoder.metadata['rhs_bodies'][0])


def test_pfft_batched_leaf_blocks_and_factors_match_selected_blocks(prepared):
    from witwin.maxwell.mom.preconditioning import PreconditionerContext, bind_right_preconditioner
    _, fast = prepared
    handle = fast.solver.handle
    context = PreconditionerContext(handle, FREQUENCY)
    offset = handle.system.bodies[1].edge_offset
    # Include cross-body entries and reverse one block's edge order.
    edges = torch.tensor([[0, 1, 2, offset, offset+1], [offset+2, offset+1, 3, 2, 1]], device='cuda')
    provider = handle.provider
    before = list(provider.metadata['triangle_pair_counts'])
    original = torch.stack([context.principal_block(group) for group in edges])
    selected_counts = [a-b for a, b in zip(provider.metadata['triangle_pair_counts'], before)]
    before = list(provider.metadata['triangle_pair_counts'])
    batched = provider._principal_blocks_many(edges, entry_budget=50)
    assert _relative(batched, original) < 1e-14
    assert [a-b for a, b in zip(provider.metadata['triangle_pair_counts'], before)] == selected_counts

    factory = mw.LeafBlockRightPreconditioner(32)
    groups = context.leaf_partition(32)
    residual = torch.randn(handle.shape[0], device='cuda', dtype=torch.complex128,
                           generator=torch.Generator(device='cuda').manual_seed(18))
    expected = torch.empty_like(residual)
    for group in groups:
        index = context.coefficient_indices(group)
        factor, pivot = torch.linalg.lu_factor(context.principal_block(group))
        expected[index] = torch.linalg.lu_solve(factor, pivot, residual[index, None])[:, 0]
    construction = handle.actions.receipt()
    bound, receipt = bind_right_preconditioner(factory, handle, FREQUENCY)
    assert _relative(bound.apply(residual), expected) < 1e-14
    assert handle.actions.receipt() == construction
    assert {k: v for k, v in receipt.items() if k != 'setup_seconds'} == {
        'version': 'leaf_block_jacobi_right_preconditioner_v1', 'blocks': len(groups), 'max_block_edges': 32,
        'size_classes': len({len(group) for group in groups}),
        'retained_bytes': sum((2*len(group))**2*16+2*len(group)*12 for group in groups),
        'block_source': 'hierarchical', 'principal_block_requests': len(groups),
        'krylov_method': 'flexible_right_gmres', 'admission': 'unchanged_true_residual'}
