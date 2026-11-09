"""Diagonal magnetic CUDA operators against stencil and manufactured references."""
import math
import cupy as cp
import pytest
import torch
import witwin.maxwell as mw
from witwin.maxwell.fdfd.magnetic import assemble_magnetic_operator
from witwin.maxwell.fdfd.observations import compile_projection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def scene():
    result = mw.Scene(domain=mw.Domain(bounds=((-0.12,.12),)*3),
                      grid=mw.GridSpec.custom(*([-.12,-.04,.03,.12],)*3),
                      boundary=mw.BoundarySpec.pml(num_layers=2,strength=2),device='cuda')
    result.add_source(mw.PointDipole(position=(-.04,0.,0.),polarization='Ez',width=.02,
                                    source_time=mw.CW(frequency=1e9)))
    return result


def solver_for(candidate):
    return mw.Simulation.fdfd(candidate,frequency=1e9,
           solver=mw.GMRES(solver_type='direct',precision='double')).prepare().solver


def test_factorized_curl_matches_fast_stencil_and_constant_mu_scaling():
    solver = solver_for(scene())
    solver._ensure_system_matrix()
    reference = solver.A_matrix.toarray()
    factored = assemble_magnetic_operator(solver).toarray()
    cp.testing.assert_allclose(factored,reference,rtol=2e-12,atol=1e-10)
    solver.material_mu_edges = {k:torch.full_like(v,2.3,dtype=torch.float64)
                                for k,v in solver.material_mu_edges.items()}
    magnetic = assemble_magnetic_operator(solver).toarray()
    expected = (reference-cp.eye(reference.shape[0])*solver.k0**2)/2.3+cp.eye(reference.shape[0])*solver.k0**2
    cp.testing.assert_allclose(magnetic,expected,rtol=2e-12,atol=1e-10)
    cp.testing.assert_allclose(magnetic,magnetic.T,rtol=2e-12,atol=1e-10)


def test_variable_mu_manufactured_constant_magnetic_curl():
    solver = solver_for(scene());solver._ensure_material_components()
    prepared = solver.scene
    x = torch.tensor(prepared.x_nodes64,device='cuda')
    mu = (1+x[:,None,None]).expand(prepared.Nx,prepared.Ny,prepared.Nz).contiguous()
    from witwin.maxwell.fdfd.magnetic import magnetic_samples
    solver.material_mu_edges = {k[1:]: v for k, v in magnetic_samples(
        {'x':torch.ones_like(mu),'y':torch.ones_like(mu),'z':mu}).items()}
    matrix = assemble_magnetic_operator(solver)
    field = torch.zeros(prepared.N_vector_total,device='cuda',dtype=torch.complex128)
    ey = field[prepared.N_ex:prepared.N_ex+prepared.N_ey].reshape(prepared.Nx_ey,prepared.Ny_ey,prepared.Nz_ey)
    ey[:] = (1+x[:,None,None]).square()
    scale = solver._ensure_symmetrization_scale()
    actual = torch.from_dlpack((matrix@(cp.from_dlpack(field)*scale))/scale)
    # Away from PML/outer closure, (d Ey/dx)/mu_z = 2, so curl(mu^-1 curl E)=0.
    # Rows on the PML interface node still couple to the stretched H sample half
    # a cell inside the layer, so the unstretched interior starts one node further in.
    residual = (actual-solver.k0**2*field)[prepared.N_ex:prepared.N_ex+prepared.N_ey].reshape(ey.shape)
    inner = slice(prepared.pml_thickness+1,-(prepared.pml_thickness+1))
    assert float(residual[inner,:,inner].abs().max())<1e-9


def test_diagonal_mu_magnetic_observation_and_fixed_background_local_response():
    background = scene()
    background.add_structure(mw.Structure(name='magnet',geometry=mw.Box(size=(10.,)*3),
        material=mw.Material(mu_tensor=mw.DiagonalTensor3(2.,4.,8.))))
    solver = solver_for(background)
    solver._ensure_system_matrix()
    prepared = solver.scene
    projection = mw.FieldProjection('hy',component='Hy',position=(0.,0.,0.))
    q = compile_projection((projection,),prepared,torch.complex128,1e9)
    field = torch.zeros(prepared.N_vector_total,device='cuda',dtype=torch.complex128)
    ez = field[prepared.N_ex+prepared.N_ey:].reshape(prepared.Nx_ez,prepared.Ny_ez,prepared.Nz_ez)
    ez[:] = prepared.x[:,None,None]
    actual = torch.sparse.mm(q,field[:,None])[0,0]
    expected = actual.new_tensor(1j/(2*math.pi*1e9*4*math.pi*1e-7*4))
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-12)
    config = mw.LocalResponseConfig(mw.LocalDesignRegion(bounds=((-0.05,.04),)*3),(projection,),tolerance=1e-8)
    with mw.Simulation.fdfd(background,frequency=1e9,solver=mw.GMRES(solver_type='direct',precision='double'),local_response=config).prepare() as response:
        candidate = background.clone()
        candidate.add_structure(mw.Structure(name='edit',geometry=mw.Box(size=(.04,)*3),
            material=mw.Material(eps_r=3.,mu_tensor=mw.DiagonalTensor3(2.,4.,8.))))
        result = response.run(scene=candidate,full_fields=True)
        full = solver_for(candidate);full._ensure_system_matrix()
        d = torch.from_dlpack(full._ensure_symmetrization_scale())
        expected = full.solve_physical_rhs(torch.from_dlpack(full._build_source_vector_yee())/d,tol=1e-8)
        actual = torch.cat([result.tensor(c).flatten() for c in ('Ex','Ey','Ez')])
        torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-8)
        torch.testing.assert_close(result.monitor('hy')['value'],torch.sparse.mm(q,expected[:,None])[0,0],rtol=1e-7,atol=1e-8)
        full._release_direct_solver()
        candidate.add_structure(mw.Structure(name='bad_mu',geometry=mw.Box(size=(.04,)*3),material=mw.Material(mu_r=5.)))
        with pytest.raises(ValueError,match='unchanged background permeability'):
            response.run(scene=candidate)


def test_trainable_permeability_fails_in_full_and_local_material_gradients():
    background = scene()
    background.subpixel = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic")
    candidate = background.clone()
    density = torch.zeros((1,1,1),device='cuda',requires_grad=True)
    candidate.add_material_region(mw.MaterialRegion(name='magnetic_edit',geometry=mw.Box(size=(.04,)*3),
        density=density,eps_bounds=(1.,2.),mu_bounds=(1.,2.)))
    with pytest.raises(NotImplementedError,match='fixed permeability'):
        mw.Simulation.fdfd(candidate,frequency=1e9,solver=mw.GMRES(solver_type='direct')).run()
    config = mw.LocalResponseConfig(mw.LocalDesignRegion(bounds=((-0.05,.04),)*3),
        (mw.FieldProjection('probe',position=(.03,.03,.03)),),tolerance=1e-8)
    with mw.Simulation.fdfd(background,frequency=1e9,solver=mw.GMRES(solver_type='direct',precision='double'),local_response=config).prepare() as response:
        with pytest.raises(NotImplementedError,match='fixed permeability'):
            response.run(scene=candidate)


def test_magnetic_conductivity_recompiles_on_frequency_change():
    background = scene()
    background.add_structure(mw.Structure(name='magnetic_loss',geometry=mw.Box(size=(10.,)*3),
                                         material=mw.Material(mu_r=1.5,sigma_m=.05)))
    solver = solver_for(background)
    for frequency in (1e9,1.5e9):
        solver.set_frequency(frequency)
        solver._ensure_system_matrix()
        actual = solver.material_mu_components['x']
        expected = 1.5+1j*.05/(2*math.pi*frequency*4*math.pi*1e-7)
        torch.testing.assert_close(actual,torch.full_like(actual,expected),rtol=1e-6,atol=1e-8)
        solver.solve(max_iter=100,tol=1e-8)
        assert solver.converged and bool(torch.isfinite(solver.E_field[0]).all())
    solver._release_direct_solver()


def test_passive_conductivity_and_cw_phase_use_radiating_convention():
    from witwin.maxwell.compiler.sources import compile_fdfd_sources
    from witwin.maxwell.constants import EPSILON_0
    background = scene()
    background.sources = [mw.PointDipole(position=(0.,0.,0.),polarization='Ez',
                          source_time=mw.CW(frequency=1e9,amplitude=2.,phase=math.pi/2))]
    assert abs(compile_fdfd_sources(background,default_frequency=1e9)[0]['polarization'][2]+2j)<1e-12
    background.add_structure(mw.Structure(name='loss',geometry=mw.Box(size=(10.,)*3),
        material=mw.Material(eps_r=2.,sigma_e=.1)))
    solver = solver_for(background);solver._ensure_material_components()
    epsilon = solver.material_eps_components['x']
    expected = 2+1j*.1/(2*math.pi*1e9*EPSILON_0)
    torch.testing.assert_close(epsilon,torch.full_like(epsilon,expected),rtol=1e-6,atol=1e-8)
    assert bool((epsilon.imag>0).all())
