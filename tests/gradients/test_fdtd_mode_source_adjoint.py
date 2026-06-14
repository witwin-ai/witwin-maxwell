import pytest
import torch

import witwin.maxwell as mw
import witwin.maxwell.fdtd.excitation.modes as mode_solver
from witwin.maxwell.fdtd.adjoint import _FDTDGradientBridge
from witwin.maxwell.fdtd.adjoint.dispatch import reverse_step
from tests.gradients import fdtd_adjoint_baselines as adjoint_baselines


def _build_mode_simulation(model, *, time_steps=32):
    return mw.Simulation.fdtd(
        model,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=False,
    )


class _DensityModeSourceScene(mw.SceneModule):
    def __init__(self, init=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((1, 1, 1), float(init), device="cuda"))

    def to_scene(self):
        density = torch.sigmoid(self.logits)
        scene = mw.Scene(
            domain=mw.Domain(bounds=((-0.72, 0.72), (-0.60, 0.60), (-0.60, 0.60))),
            grid=mw.GridSpec.uniform(0.12),
            boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
            device="cuda",
        )
        scene.add_structure(
            mw.Structure(
                name="waveguide",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.20, 0.36, 0.36)),
                material=mw.Material(eps_r=8.0),
            )
        )
        scene.add_material_region(
            mw.MaterialRegion(
                name="design",
                geometry=mw.Box(position=(-0.12, 0.0, 0.0), size=(0.24, 0.12, 0.12)),
                density=density,
                eps_bounds=(8.0, 12.0),
                mu_bounds=(1.0, 1.0),
            )
        )
        scene.add_source(
            mw.ModeSource(
                position=(-0.36, 0.0, 0.0),
                size=(0.0, 0.36, 0.36),
                polarization="Ez",
                source_time=mw.CW(frequency=1.0e9, amplitude=40.0),
                name="mode0",
            )
        )
        scene.add_monitor(mw.PointMonitor("probe", (0.24, 0.0, 0.0), fields=("Ez",)))
        return scene


def _mode_probe_loss(model):
    result = _build_mode_simulation(model).run()
    data = result.monitor("probe")["data"]
    loss = torch.abs(data) ** 2
    return result, data, loss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
def test_fdtd_gradient_bridge_mode_source_backpropagates_to_material_density():
    model = _DensityModeSourceScene(init=0.0).cuda()

    result, data, loss = _mode_probe_loss(model)
    assert torch.is_tensor(data)
    assert data.is_cuda
    assert data.requires_grad

    loss.backward()

    assert model.logits.grad is not None
    assert model.logits.grad.shape == model.logits.shape
    assert torch.isfinite(model.logits.grad).all()
    assert float(torch.max(torch.abs(model.logits.grad)).item()) > 0.0
    assert result.monitor("probe")["data"].grad_fn is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
def test_fdtd_gradient_bridge_mode_source_matches_central_difference_and_uses_python_reference_cpml():
    model = _DensityModeSourceScene(init=0.0).cuda()

    _result, _data, loss = _mode_probe_loss(model)
    loss.backward()
    backward_grad = model.logits.grad.detach().clone()
    assert torch.isfinite(backward_grad).all()

    delta = 1.0e-2
    with torch.no_grad():
        base = model.logits.detach().clone()
        model.logits.copy_(base + delta)
    _, _, loss_plus = _mode_probe_loss(model)
    with torch.no_grad():
        model.logits.copy_(base - delta)
    _, _, loss_minus = _mode_probe_loss(model)
    with torch.no_grad():
        model.logits.copy_(base)

    finite_difference = (loss_plus - loss_minus) / (2.0 * delta)
    assert torch.allclose(
        backward_grad,
        torch.full_like(backward_grad, finite_difference.item()),
        rtol=1.5e-1,
        atol=1e-10,
    )

    bridge = _FDTDGradientBridge(_build_mode_simulation(model))
    bridge.forward(tuple(bridge.material_inputs))
    profile = bridge.backward_profile()

    assert profile["reverse_backend_counts"].get("python_reference_cpml", 0) == bridge._time_steps
    assert profile["reverse_backend_counts"].get("torch_vjp", 0) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for CUDA FDTD")
def test_mode_source_reverse_step_matches_python_reference_for_checkpoint_replay():
    model = _DensityModeSourceScene(init=0.0).cuda()
    bridge = _FDTDGradientBridge(_build_mode_simulation(model, time_steps=10))
    bridge.forward(tuple(bridge.material_inputs))

    solver = bridge._last_solver
    assert solver is not None
    checkpoint = bridge._last_checkpoints[0]
    states = mw.fdtd.adjoint._replay_segment_states(solver, checkpoint, 0, 2)
    forward_state = states[1]
    time_value = solver.dt
    adjoint_state = {
        name: torch.randn_like(tensor)
        for name, tensor in forward_state.items()
    }

    eps_ex = solver.eps_Ex.detach().clone().requires_grad_(True)
    eps_ey = solver.eps_Ey.detach().clone().requires_grad_(True)
    eps_ez = solver.eps_Ez.detach().clone().requires_grad_(True)
    solver._mode_source_explicit_vjp_remaining = 2
    actual = reverse_step(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )

    expected_resolved_source_terms = adjoint_baselines.resolved_source_term_lists(
        solver,
        eps_ex,
        eps_ey,
        eps_ez,
    )
    expected = adjoint_baselines.reverse_step_cpml_python_reference(
        solver,
        forward_state,
        adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=expected_resolved_source_terms,
    )
    expected = adjoint_baselines.accumulate_source_term_gradients(
        expected,
        solver=solver,
        adjoint_state=adjoint_state,
        time_value=time_value,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=expected_resolved_source_terms,
    )

    assert actual.backend == "python_reference_cpml"
    assert expected.backend == "python_reference_cpml"
    for name in forward_state:
        assert torch.allclose(
            actual.pre_step_adjoint[name],
            expected.pre_step_adjoint[name],
            rtol=2.0e-4,
            atol=2.0e-5,
            equal_nan=True,
        )
    assert torch.allclose(
        actual.grad_eps_ex,
        expected.grad_eps_ex,
        rtol=2.0e-4,
        atol=2.0e-5,
        equal_nan=True,
    )
    assert torch.allclose(
        actual.grad_eps_ey,
        expected.grad_eps_ey,
        rtol=2.0e-4,
        atol=2.0e-5,
        equal_nan=True,
    )
    assert torch.allclose(
        actual.grad_eps_ez,
        expected.grad_eps_ez,
        rtol=2.0e-4,
        atol=2.0e-5,
        equal_nan=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for slang FDTD")
def test_fdtd_gradient_bridge_mode_source_supports_sparse_implicit_mode_solver(monkeypatch):
    model = _DensityModeSourceScene(init=0.0).cuda()

    dense_result, dense_data, dense_loss = _mode_probe_loss(model)
    dense_loss.backward()
    dense_grad = model.logits.grad.detach().clone()
    model.logits.grad = None

    monkeypatch.setattr(mode_solver, "_DENSE_EIGEN_LIMIT", 1)

    sparse_result, sparse_data, sparse_loss = _mode_probe_loss(model)
    sparse_loss.backward()

    assert torch.is_tensor(sparse_data)
    assert sparse_data.is_cuda
    assert sparse_data.requires_grad
    assert sparse_result.monitor("probe")["data"].grad_fn is not None
    assert torch.allclose(model.logits.grad, dense_grad, rtol=2.5e-2, atol=1e-5)
    assert abs(complex(sparse_data.detach().cpu().item())) == pytest.approx(
        abs(complex(dense_data.detach().cpu().item())),
        rel=1e-2,
        abs=1e-6,
    )
