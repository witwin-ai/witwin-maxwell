import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule
from witwin.maxwell.mom.operator import IntegrationConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def scene(geometry=None, eps=2.25, **kwargs):
    return mw.Scene(structures=[mw.Structure(geometry=geometry or mw.Sphere(radius=.1),
                    material=mw.Material(eps_r=eps))], sources=[mw.PlaneWave(direction=(0, 0, 1), polarization=(1, 0, 0))], **kwargs)


def config(**kwargs):
    return mw.mom.MoMConfig(subdivisions=0, integration=IntegrationConfig(order=2, singular_order=3), **kwargs)


def test_shared_simulation_result_cuda_queries_and_persistence(tmp_path):
    physical = scene()
    simulation = mw.Simulation.mom(physical, frequencies=3e8, config=config())
    assert isinstance(simulation, mw.Simulation)
    prepared = simulation.prepare()
    result = prepared.run()
    assert isinstance(result, mw.Result) and result.method == 'mom'
    assert mw.mom.surface_currents(result).J_scattered.is_cuda and mw.mom.surface_currents(result).M_scattered.shape == (30,)
    assert result.stats()['converged']
    points = torch.tensor([[.3, .2, .4]], dtype=torch.float64, device='cuda')
    e, h = result.query_fields(points, region_id='exterior', field_kind='total')
    assert torch.isfinite(e).all() and torch.isfinite(h).all()
    f = mw.mom.far_field(result, points / torch.linalg.vector_norm(points, dim=-1, keepdim=True))[0].values[0]
    assert f.is_cuda and f.dtype == torch.complex128
    path = tmp_path / 'surface.pt'
    result.save(path)
    payload = torch.load(path, weights_only=False)
    assert 'matrix' not in payload['surface_data']
    restored = mw.Result.load(path, scene=physical)
    assert torch.equal(mw.mom.surface_currents(restored).scattered_coefficients, mw.mom.surface_currents(result).scattered_coefficients)
    er, hr = restored.query_fields(points, region_id='exterior', field_kind='total')
    torch.testing.assert_close(er, e, rtol=5e-13, atol=1e-15)
    torch.testing.assert_close(hr, h, rtol=5e-13, atol=1e-15)
    with pytest.raises(ValueError, match='does not match'):
        mw.Result.load(path, scene=scene(eps=3))
    for action in [lambda: result.fields, lambda: result.field(), lambda: result.field('EX'), lambda: result.prepared_scene]:
        with pytest.raises(NotImplementedError, match='RWG'):
            action()
    with pytest.raises(RuntimeError, match='only once'):
        prepared.run()


def test_scene_module_and_ellipsoid_transform():
    class Definition(mw.SceneModule):
        def to_scene(self):
            return scene(mw.geometry.Ellipsoid(position=(.03, -.02, .01), radii=(.1, .08, .06)))
    result = mw.Simulation.mom(Definition(), frequencies=3e8, config=config()).run()
    assert mw.mom.surface_currents(result).mesh.vertices.shape == (12, 3)
    torch.testing.assert_close(mw.mom.surface_currents(result).mesh.vertices.mean(0),
        torch.tensor([.03, -.02, .01], dtype=torch.float64, device='cuda'), rtol=1e-6, atol=1e-8)


@pytest.mark.host_validation
def test_fail_closed_public_scope():
    with pytest.raises(ValueError, match='is not supported'):
        mw.Simulation.mom(scene(mw.geometry.Pyramid(base_size=.1, height=.1)), frequencies=3e8, config=config()).prepare()
    with pytest.raises(ValueError, match='monitors'):
        mw.Simulation.mom(scene(monitors=[mw.PlaneMonitor(name="probe", axis="x", position=.4)]), frequencies=3e8, config=config()).prepare()
    with pytest.raises(ValueError, match='CUDA'):
        mw.Simulation.mom(scene(), frequencies=3e8, config=config(device='cpu')).prepare()
    with pytest.raises(ValueError, match='positive'):
        mw.Simulation.mom(scene(), frequencies=0, config=config()).prepare()
    with pytest.raises(TypeError, match='MoMConfig'):
        mw.Simulation.mom(scene(), frequencies=3e8, config=mw.FDFDConfig())


def test_shared_mesh_surface_compilation_and_exact_persistence(tmp_path):
    from witwin.maxwell.mom.mesh import deformed_sphere_mesh
    mesh = deformed_sphere_mesh(radius=.1, subdivisions=0, harmonics={(2, 0): .05})
    geometry = mw.Mesh(mesh.vertices, mesh.faces, position=(.03, -.02, .01),
                       recenter=False, fill_mode='solid', device='cuda')
    physical = scene(geometry)
    result = mw.Simulation.mom(physical, frequencies=3e8, config=config()).run()
    expected = geometry._transform_mesh_verts(geometry.vertices.to(dtype=torch.float64)*geometry.scale.to(dtype=torch.float64))
    assert torch.equal(mw.mom.surface_currents(result).mesh.vertices, expected)
    assert mw.mom.surface_currents(result).mesh.vertices.dtype == torch.float64
    path = tmp_path/'mesh.pt'
    result.save(path)
    restored = mw.Result.load(path, scene=physical)
    assert torch.equal(mw.mom.surface_currents(restored).mesh.vertices, expected)
    assert torch.equal(mw.mom.surface_currents(restored).scattered_coefficients, mw.mom.surface_currents(result).scattered_coefficients)
    with pytest.raises(ValueError, match='closed'):
        mw.Simulation.mom(scene(mw.Mesh(mesh.vertices, mesh.faces[:-1], recenter=False, fill_mode='solid')),
                          frequencies=3e8, config=config()).prepare()
    with pytest.raises(ValueError, match='recenter=False'):
        mw.Simulation.mom(scene(mw.Mesh(mesh.vertices, mesh.faces, fill_mode='solid')),
                          frequencies=3e8, config=config()).prepare()


# Scenes are immutable, so only the simulation's frequency or configuration can change under a prepared surface.
@pytest.mark.parametrize('mutation', ['frequency', 'configuration'])
def test_prepared_surface_rejects_changed_frequency_or_configuration(mutation):
    physical = scene()
    simulation = mw.Simulation.mom(physical, frequencies=3e8, config=config())
    prepared = simulation.prepare()
    if mutation == 'frequency':
        simulation.frequency = 4e8
    else:
        simulation.config = mw.mom.MoMConfig(subdivisions=1)
    with pytest.raises(ValueError):
        prepared.run()
