"""Independent manufactured-field and physical-RHS checks for graded Yee grids."""
import cupy as cp
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_solver(pec=False, pml=False):
    axes = ([-.12, -.08, -.035, 0., .025, .06, .12],
            [-.1, -.065, -.02, .01, .05, .1],
            [-.09, -.055, -.015, .02, .045, .09])
    scene = mw.Scene(domain=mw.Domain(bounds=tuple((a[0], a[-1]) for a in axes)),
                     grid=mw.GridSpec.custom(*axes), device="cuda",
                     boundary=mw.BoundarySpec.pml(num_layers=2, strength=3.) if pml else mw.BoundarySpec.none())
    if pec:
        scene.add_structure(mw.Structure(name="metal", geometry=mw.Box(size=(.06, .06, .06)),
                                         material=mw.Material(pec=True)))
    return mw.Simulation.fdfd(scene, frequency=1e9,
                             solver=mw.GMRES(solver_type="direct", precision="double")).prepare().solver


def test_nonuniform_manufactured_curl_curl_and_reciprocity():
    solver = make_solver()
    solver._ensure_system_matrix()
    scene = solver.scene
    d = solver._ensure_symmetrization_scale()
    A = solver.A_matrix
    assert float(cp.max(cp.abs((A - A.T).data))) < 1e-9
    # E_y=x^2 has -curl(curl(E))_y=2 in the interior.
    e = cp.zeros(A.shape[0], dtype=cp.complex128)
    ey = e[scene.N_ex:scene.N_ex+scene.N_ey].reshape(scene.Nx_ey,scene.Ny_ey,scene.Nz_ey)
    ey[:] = cp.asarray(scene.x_nodes64)[:,None,None]**2
    action = (A @ (d*e))/d - solver.k0**2*e
    y = action[scene.N_ex:scene.N_ex+scene.N_ey].reshape(ey.shape)
    cp.testing.assert_allclose(y[1:-1,:,1:-1], 2., atol=1e-10)
    # E_x=x*y has -curl(curl(E))_y=-1.
    e[:] = 0
    ex = e[:scene.N_ex].reshape(scene.Nx_ex,scene.Ny_ex,scene.Nz_ex)
    ex[:] = cp.asarray(scene.x_half64)[:,None,None]*cp.asarray(scene.y_nodes64)[None,:,None]
    action = (A @ (d*e))/d - solver.k0**2*e
    y = action[scene.N_ex:scene.N_ex+scene.N_ey].reshape(ey.shape)
    cp.testing.assert_allclose(y[1:-1,:,1:-1], -1., atol=1e-10)


@pytest.mark.parametrize("pml", [False, True])
def test_physical_block_rhs_pec_and_owned_storage(pml):
    solver = make_solver(pec=True, pml=pml)
    solver._ensure_system_matrix()
    mask = torch.from_dlpack(solver._ensure_pec_mask())
    assert bool(mask.any())
    torch.manual_seed(7)
    rhs = torch.randn(mask.numel(), 3, dtype=torch.complex128, device="cuda")
    rhs[mask] = 0
    rhs[:,2] = 0
    fields = solver.solve_physical_rhs(rhs, tol=1e-9)
    copy = fields.clone()
    factor = solver._direct_solver
    solver.solve_physical_rhs(rhs * 2, tol=1e-9)
    assert solver._direct_solver is factor
    assert torch.equal(copy,fields)
    assert torch.count_nonzero(fields[mask]) == 0
    assert torch.count_nonzero(fields[:,2]) == 0
    d = solver._ensure_symmetrization_scale()[:,None]
    actual = solver.A_matrix @ (d * cp.from_dlpack(fields)) / d
    cp.testing.assert_allclose(actual, cp.from_dlpack(rhs), rtol=1e-8, atol=1e-8)
    rhs[mask] = 1
    with pytest.raises(ValueError, match="PEC"):
        solver.solve_physical_rhs(rhs)
    solver._release_direct_solver()


@pytest.mark.parametrize('engine',('cg','gmres','bicgstab','tfqmr','idr','sqmr'))
def test_multi_rhs_preserves_selected_iterative_engine(engine,monkeypatch):
    from witwin.maxwell.fdfd import krylov
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.1,.1),)*3),grid=mw.GridSpec.uniform(.1),
                     boundary=mw.BoundarySpec.none(),device='cuda')
    solver = mw.Simulation.fdfd(scene,frequency=1e10,
        solver=mw.GMRES(solver_type=engine,precision='double')).prepare().solver
    solver._ensure_system_matrix()
    called = []
    original = krylov.solve
    def tracked(name,*args,**kwargs):
        called.append(name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(krylov,'solve',tracked)
    rhs = torch.randn((solver.scene.N_vector_total,2),device='cuda',dtype=torch.complex128)
    actual = solver.solve_physical_rhs(rhs,tol=1e-9,max_iter=200)
    scale = torch.from_dlpack(solver._ensure_symmetrization_scale())
    matrix = torch.from_dlpack(solver.A_matrix.toarray())
    expected = torch.linalg.solve(matrix,rhs*scale[:,None])/scale[:,None]
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-10)
    if engine not in ('cg','gmres'):
        assert called == [engine,engine]
