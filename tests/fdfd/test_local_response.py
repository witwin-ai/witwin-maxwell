"""Public CUDA local responses compared to a separately assembled full solve."""
import cupy as cp
import pytest
import torch
import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def setup_response(batch=7):
    scene = mw.Scene(subpixel_samples=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"), domain=mw.Domain(bounds=((-0.12, .12),)*3),
                     grid=mw.GridSpec.custom(*([-.12,-.07,-.02,.025,.07,.12],)*3),
                     boundary=mw.BoundarySpec.pml(num_layers=2, strength=3), device="cuda")
    scene.add_source(mw.PointDipole(position=(-.04,0.,0.), polarization="Ez", width=.02,
                                    source_time=mw.CW(frequency=1e9)))
    config = mw.LocalResponseConfig(
        mw.LocalDesignRegion(bounds=((-0.025,.03),)*3),
        (mw.FieldProjection('probe', position=(.07,.07,0.)),),
        rhs_batch_size=batch, tolerance=1e-9)
    simulation = mw.Simulation.fdfd(scene, frequency=1e9,
                 solver=mw.GMRES(solver_type="direct", precision="double"), local_response=config)
    return simulation.prepare()


def test_precompute_and_projection_agree_with_full_reassembly():
    with setup_response() as response:
        scene = response.simulation.scene.clone()
        scene.add_structure(mw.Structure(name="design", geometry=mw.Box(size=(.02,.02,.02)),
                                         material=mw.Material(eps_r=2.5)))
        result = response.run(scene=scene, full_fields=True)
        # Build a fresh sparse operator from the full candidate Scene.
        solver = mw.Simulation.fdfd(scene, frequency=1e9,
                     solver=mw.GMRES(solver_type="direct", precision="double")).prepare().solver
        solver._ensure_system_matrix()
        d = torch.from_dlpack(solver._ensure_symmetrization_scale())
        rhs = torch.from_dlpack(solver._build_source_vector_yee()) / d
        actual = solver.solve_physical_rhs(rhs, tol=1e-9)
        predicted = torch.cat([result.tensor(c).flatten() for c in ('Ex','Ey','Ez')])
        torch.testing.assert_close(predicted, actual, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(result.raw_output['projections'],
            torch.sparse.mm(response._projection, actual[:,None])[:,0], rtol=1e-7, atol=1e-8)
        assert result.raw_output['local_field'].is_cuda
        solver._release_direct_solver()


def test_explicit_contrast_dense_reference_and_context_guards():
    with setup_response() as response:
        torch.manual_seed(8)
        delta = torch.rand_like(response._u0.real).to(response.dtype) * 20
        result = response.run(delta=delta)
        solver = response.solver
        A = solver.A_matrix.copy()
        design = cp.from_dlpack(response.design_indices)
        # Similarity leaves diagonal contrasts unchanged.
        from cupyx.scipy.sparse import diags
        diagonal = cp.zeros(A.shape[0], dtype=A.dtype)
        diagonal[design] = cp.from_dlpack(delta)
        matrix = (A + diags(diagonal)).toarray()
        d = solver._ensure_symmetrization_scale()
        actual = cp.linalg.solve(matrix, cp.from_dlpack(response._rhs) * d) / d
        torch.testing.assert_close(result.raw_output['local_field'], torch.from_dlpack(actual)[response.design_indices], rtol=1e-8, atol=1e-8)
        response.simulation.scene.boundary = mw.BoundarySpec.none()
        with pytest.raises(ValueError, match="Background"):
            response.run(delta=delta)
    with pytest.raises(RuntimeError, match="closed"):
        response.run(delta=delta)


def test_low_rank_trials_accept_reject_and_generation():
    with setup_response() as response:
        response.config = mw.LocalResponseConfig(response.config.design, response.config.observations,
                                                tolerance=1e-9, rebuild_interval=3)
        torch.manual_seed(42)
        delta = torch.zeros_like(response._u0)
        pec = torch.zeros_like(delta, dtype=torch.bool)
        first = response.trial(delta=delta)
        for iteration in range(15):
            candidate = delta.clone(); candidate_pec = pec.clone()
            index = iteration % delta.numel()
            candidate[index] = (iteration + 1) * (3 - .2j)
            if iteration % 4 == 0:
                candidate_pec[index] = ~candidate_pec[index]
            inverse_before = response._inverse.clone()
            trial = response.trial(delta=candidate, pec=candidate_pec)
            expected = response.run(delta=candidate, pec=candidate_pec)
            torch.testing.assert_close(trial.projections, expected.raw_output['projections'], rtol=1e-8, atol=1e-8)
            assert trial.rank <= 1
            assert torch.equal(inverse_before, response._inverse)
            if iteration % 3:
                actual = response.accept(trial)
                torch.testing.assert_close(actual.raw_output['local_field'], expected.raw_output['local_field'], rtol=1e-8, atol=1e-8)
                delta, pec = candidate, candidate_pec
                with pytest.raises(ValueError, match="stale"):
                    response.accept(trial)
        with pytest.raises(ValueError, match="stale"):
            response.accept(first)


def test_pec_candidate_matches_independent_full_elimination():
    with setup_response() as response:
        delta = torch.zeros_like(response._u0)
        pec = torch.zeros_like(delta, dtype=torch.bool)
        pec[::7] = True
        delta[1::5] = 80 - 2j
        result = response.run(delta=delta, pec=pec, full_fields=True)
        matrix = torch.from_dlpack(response.solver.A_matrix.toarray()).clone()
        index = response.design_indices
        matrix[index,index] += delta
        mask = index[pec]
        matrix[mask,:] = 0; matrix[:,mask] = 0; matrix[mask,mask] = 1
        d = torch.from_dlpack(response.solver._ensure_symmetrization_scale())
        rhs = response._rhs * d
        rhs[mask] = 0
        reference = torch.linalg.solve(matrix, rhs) / d
        actual = torch.cat([result.tensor(c).flatten() for c in ('Ex','Ey','Ez')])
        torch.testing.assert_close(actual, reference, rtol=1e-8, atol=1e-8)
        assert float(actual[mask].abs().max()) < 1e-8


def test_complex_material_gradient_and_full_field_adjoint():
    with setup_response() as response:
        delta = torch.full_like(response._u0, 30-2j, requires_grad=True)
        result = response.run(delta=delta, full_fields=True)
        probe = result.raw_output['projections']
        electric = torch.cat([result.tensor(c).flatten() for c in ('Ex','Ey','Ez')])
        loss = probe.abs().square().sum() + .01 * electric.abs().square().sum()
        gradient, = torch.autograd.grad(loss, delta)
        # Independent dense full-system differentiable solve on the GPU.
        matrix = torch.from_dlpack(response.solver.A_matrix.toarray()).clone()
        full_delta = torch.zeros(matrix.shape[0], dtype=delta.dtype, device=delta.device)
        full_delta = full_delta.index_copy(0, response.design_indices, delta)
        d = torch.from_dlpack(response.solver._ensure_symmetrization_scale())
        actual = torch.linalg.solve(matrix + torch.diag(full_delta), response._rhs*d) / d
        projected = torch.sparse.mm(response._projection, actual[:,None])[:,0]
        reference_loss = projected.abs().square().sum() + .01 * actual.abs().square().sum()
        reference_gradient, = torch.autograd.grad(reference_loss, delta)
        torch.testing.assert_close(gradient, reference_gradient, rtol=2e-7, atol=1e-9)
        direction = torch.randn_like(delta)
        step = 1e-3
        positive = response.run(delta=delta.detach()+step*direction).raw_output['projections'].abs().square().sum()
        negative = response.run(delta=delta.detach()-step*direction).raw_output['projections'].abs().square().sum()
        projected_gradient, = torch.autograd.grad(response.run(delta=delta).raw_output['projections'].abs().square().sum(), delta)
        torch.testing.assert_close((positive-negative)/(2*step), (projected_gradient.conj()*direction).real.sum(), rtol=2e-5, atol=1e-8)


def test_scene_module_material_gradient_survives_repeated_updates():
    with setup_response() as response:
        class Design(mw.SceneModule):
            def __init__(self):
                super().__init__()
                self.density = torch.nn.Parameter(torch.full((1,1,1),.4,device='cuda'))
            def to_scene(self):
                scene = response.simulation.scene.clone()
                scene.add_material_region(mw.MaterialRegion('density',mw.Box(size=(.08,.08,.08)),
                                                           self.density,eps_bounds=(1.,3.)))
                return scene
        model = Design()
        for _ in range(2):
            value = response.run(scene=model).raw_output['projections'].abs().square().sum()
            gradient, = torch.autograd.grad(value, model.density)
            assert torch.isfinite(gradient).all() and float(gradient.abs().max()) > 0
            step = .002
            with torch.no_grad():
                model.density += step
            plus = response.run(scene=model).raw_output['projections'].abs().square().sum().detach()
            with torch.no_grad():
                model.density -= 2*step
            minus = response.run(scene=model).raw_output['projections'].abs().square().sum().detach()
            torch.testing.assert_close(gradient.sum().double(), (plus-minus)/(2*step), rtol=.005, atol=1e-6)
            with torch.no_grad():
                model.density += step + .01


def test_multifrequency_binary_optimization_and_atomic_acceptance(tmp_path):
    with setup_response() as base:
        simulation = mw.Simulation.fdfd(base.simulation.scene,frequencies=(.9e9,1.1e9),
            solver=mw.GMRES(solver_type='direct',precision='double'),local_response=base.config)
        with simulation.prepare() as sweep:
            zero = torch.zeros((2,sweep.design_indices.numel()),dtype=torch.complex128,device='cuda')
            trial = sweep.trial(delta=zero)
            result = sweep.accept(trial)
            assert result.raw_output['projections'].shape == (2,1)
            torch.testing.assert_close(result.monitor('probe',freq_index=1)['value'],
                                       result.raw_output['projections'][1,0])
            path = tmp_path/'response.pt'
            result.save(path)
            restored = mw.Result.load(path,scene=simulation.scene)
            torch.testing.assert_close(restored.monitor('probe',freq_index=1)['value'].to('cuda'),
                                       result.raw_output['projections'][1,0])
            initial_generation = tuple(r._generation for r in sweep.responses)
            with pytest.raises(ValueError,match='stale'):
                sweep.accept(trial)
            assert initial_generation == tuple(r._generation for r in sweep.responses)
            groups = [torch.tensor([i,i+1],device='cuda',dtype=torch.int64) for i in range(3)]
            optimized = sweep.optimize_binary(groups,lambda values: values.abs().square().sum(),
                                               max_passes=3,seed=3,mode='dielectric',epsilon_contrast=.5)
            assert optimized.attempts >= 3
            assert bool((optimized.objective_history[1:] < optimized.objective_history[:-1]).all())
            assert optimized.tiles.is_cuda
            assert optimized.result.solver_stats['converged']
            sweep.save_cache(tmp_path/'sweep')
        with simulation.prepare(response_cache=tmp_path/'sweep') as restored:
            assert all(child.cache_hit for child in restored.responses)
            torch.testing.assert_close(restored.run(delta=zero).raw_output['projections'],
                                       result.raw_output['projections'],rtol=0,atol=0)


def test_singular_candidate_and_snapshot_mutation_fail_without_state_change():
    with setup_response() as response:
        delta = torch.zeros_like(response._u0)
        delta[0] = -1/response._green[0,0]
        inverse = response._inverse.clone()
        with pytest.raises(RuntimeError,match='singular'):
            response.trial(delta=delta)
        assert torch.equal(response._inverse,inverse)
        from dataclasses import replace
        response.simulation.config = replace(response.simulation.config,
            solver=replace(response.simulation.config.solver, tol=2*response.simulation.config.solver.tol))
        with pytest.raises(ValueError,match='configuration'):
            response.run(delta=torch.zeros_like(delta))


def test_full_field_backward_after_close_releases_reopened_factors():
    response = setup_response()
    delta = torch.full_like(response._u0,10.,requires_grad=True)
    result = response.run(delta=delta,full_fields=True)
    loss = result.tensor('Ez').abs().square().sum()
    response.close()
    loss.backward()
    assert delta.grad is not None and bool(torch.isfinite(delta.grad).all())
    assert response.solver._direct_solver is None


def test_response_cache_roundtrip_skips_factorization_and_preserves_gradients(tmp_path, monkeypatch):
    with setup_response() as original:
        simulation = original.simulation
        delta = torch.linspace(1.,20.,original.design_indices.numel(),device='cuda',dtype=torch.float64).to(torch.complex128).requires_grad_()
        expected = original.run(delta=delta,full_fields=True)
        expected_gradient = torch.autograd.grad(expected.raw_output['projections'].abs().square().sum(),delta)[0]
        original.accept(original.trial(delta=delta.detach()))
        path = original.save_cache(tmp_path/'background.pt')
        solver_type = type(original.solver)
    def forbidden(*args,**kwargs):
        raise AssertionError('A restored projection cache must not factorize the background.')
    monkeypatch.setattr(solver_type,'_build_direct_solver',forbidden)
    with simulation.prepare(response_cache=path) as restored:
        assert restored.cache_hit and restored.solver._direct_solver is None
        assert not bool(restored._accepted_delta.any())
        actual = restored.run(delta=delta)
        torch.testing.assert_close(actual.raw_output['projections'],expected.raw_output['projections'],rtol=0,atol=0)
        gradient = torch.autograd.grad(actual.raw_output['projections'].abs().square().sum(),delta)[0]
        torch.testing.assert_close(gradient,expected_gradient,rtol=0,atol=0)
    monkeypatch.undo()
    with simulation.prepare(response_cache=path) as restored:
        actual = restored.run(delta=delta,full_fields=True)
        assert restored.solver._direct_solver is not None
        torch.testing.assert_close(actual.tensor('Ez'),expected.tensor('Ez'),rtol=1e-9,atol=1e-10)


def test_response_cache_rejects_corruption_and_different_operator(tmp_path):
    with setup_response() as original:
        path = original.save_cache(tmp_path/'valid.pt')
        simulation = original.simulation
    payload = torch.load(path,weights_only=True)
    payload['arrays']['_green'][0,0] += 1
    corrupt = tmp_path/'corrupt.pt';torch.save(payload,corrupt)
    with pytest.raises(ValueError,match='checksum'):
        simulation.prepare(response_cache=corrupt)
    different = mw.Simulation.fdfd(simulation.scene,frequency=1.1e9,
        solver=simulation.config.solver,local_response=simulation.config.local_response)
    with pytest.raises(ValueError,match='current operator'):
        different.prepare(response_cache=path)


def test_scene_module_can_define_reference_and_resume_cached_material_gradients(tmp_path):
    with setup_response() as base:
        background,config = base.simulation.scene,base.config
    class Design(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.density = torch.nn.Parameter(torch.full((1,1,1),.4,device='cuda'))
        def to_scene(self):
            candidate = background.clone()
            candidate.add_material_region(mw.MaterialRegion('density',mw.Box(size=(.08,)*3),
                                         self.density,eps_bounds=(1.,3.)))
            return candidate
    model = Design()
    simulation = mw.Simulation.fdfd(model,frequency=1e9,
        solver=mw.GMRES(solver_type='direct',precision='double'),local_response=config)
    with simulation.prepare() as original:
        loss = original.run().monitor('probe')['value'].abs().square()
        expected_loss = loss.detach().clone()
        expected, = torch.autograd.grad(loss,model.density)
        path = original.save_cache(tmp_path/'module-reference.pt')
    with simulation.prepare(response_cache=path) as restored:
        loss = restored.run().monitor('probe')['value'].abs().square()
        torch.testing.assert_close(loss.detach(),expected_loss,rtol=0,atol=0)
        gradient, = torch.autograd.grad(loss,model.density)
        # Parameters are float32 even though fields are complex128. CUDA
        # backward accumulation can differ by a ULP; cached forward data is exact.
        torch.testing.assert_close(gradient,expected,rtol=2*torch.finfo(model.density.dtype).eps,atol=0)
        assert bool(gradient.abs().max()>0)
        with torch.no_grad():
            model.density.add_(.02)
        updated = restored.run().monitor('probe')['value'].abs().square()
        updated.backward()
        assert bool(torch.isfinite(model.density.grad).all())
