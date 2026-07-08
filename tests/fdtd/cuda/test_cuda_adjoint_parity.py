from __future__ import annotations

import pytest
import torch

from tests.gradients import fdtd_adjoint_baselines as adjoint_baselines
from tests.gradients.test_fdtd_adjoint_bridge import (
    _bloch_reverse_state_shapes,
    _cpml_reverse_state_shapes,
    _fake_bloch_reverse_solver,
    _fake_cpml_reverse_solver,
    _fake_standard_reverse_solver,
    _move_solver_tensors_to_cuda,
)
from witwin.maxwell.fdtd.adjoint import core as adjoint_core
from witwin.maxwell.fdtd.adjoint.dispatch import reverse_step, resolve_fdtd_adjoint_backend_name
from witwin.maxwell.fdtd.boundary import BOUNDARY_NONE, BOUNDARY_PEC
from witwin.maxwell.fdtd.checkpoint import checkpoint_schema
from tests.fdtd.cuda._parity_backend import backend
from witwin.maxwell.fdtd.cuda import get_compiled_extension


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native adjoint backend tests.")
requires_extension_build = pytest.mark.skipif(
    not bool(__import__("os").environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD")),
    reason="native CUDA extension build is opt-in for adjoint kernel tests",
)


def _forbid_slang_reverse(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("native CUDA adjoint dispatch must not call Slang reverse kernels")

    monkeypatch.setattr(adjoint_core, "_reverse_step_standard_slang", fail)
    monkeypatch.setattr(adjoint_core, "_reverse_step_cpml_slang", fail)
    monkeypatch.setattr(adjoint_core, "_reverse_step_bloch_slang", fail)
    monkeypatch.setattr(adjoint_core, "_reverse_step_dispersive_slang", fail)


def test_cuda_backend_selects_python_adjoint_reference_by_default(monkeypatch):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND", raising=False)

    assert resolve_fdtd_adjoint_backend_name() == "python"


def test_cuda_standard_reverse_step_matches_python_reference_without_slang(monkeypatch):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    _forbid_slang_reverse(monkeypatch)
    torch.manual_seed(101)
    solver = _move_solver_tensors_to_cuda(_fake_standard_reverse_solver())
    forward_state = {
        "Ex": torch.randn(2, 4, 5, device="cuda", dtype=torch.float32),
        "Ey": torch.randn(3, 3, 5, device="cuda", dtype=torch.float32),
        "Ez": torch.randn(3, 4, 4, device="cuda", dtype=torch.float32),
        "Hx": torch.randn(3, 3, 4, device="cuda", dtype=torch.float32),
        "Hy": torch.randn(2, 4, 4, device="cuda", dtype=torch.float32),
        "Hz": torch.randn(2, 3, 5, device="cuda", dtype=torch.float32),
    }
    adjoint_state = {name: torch.randn_like(tensor) for name, tensor in forward_state.items()}
    eps_ex = torch.full_like(forward_state["Ex"], 2.3, requires_grad=True)
    eps_ey = torch.full_like(forward_state["Ey"], 2.7, requires_grad=True)
    eps_ez = torch.full_like(forward_state["Ez"], 3.1, requires_grad=True)

    actual = reverse_step(
        solver,
        forward_state,
        adjoint_state,
        time_value=0.0,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    expected = adjoint_baselines.reverse_step_standard_python_reference(
        solver,
        forward_state,
        adjoint_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    torch.cuda.synchronize()

    assert actual.backend == "python_reference_standard"
    for name in forward_state:
        torch.testing.assert_close(actual.pre_step_adjoint[name], expected.pre_step_adjoint[name], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ex, expected.grad_eps_ex, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ey, expected.grad_eps_ey, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ez, expected.grad_eps_ez, rtol=1.0e-5, atol=1.0e-6)


def test_cuda_cpml_reverse_step_matches_python_reference_without_slang(monkeypatch):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    _forbid_slang_reverse(monkeypatch)
    torch.manual_seed(103)
    solver = _move_solver_tensors_to_cuda(_fake_cpml_reverse_solver())
    state_shapes = _cpml_reverse_state_shapes()
    forward_state = {
        name: torch.randn(state_shapes[name], device="cuda", dtype=torch.float32)
        for name in checkpoint_schema(solver).state_names
    }
    adjoint_state = {name: torch.randn_like(tensor) for name, tensor in forward_state.items()}
    eps_ex = torch.full_like(forward_state["Ex"], 2.3, requires_grad=True)
    eps_ey = torch.full_like(forward_state["Ey"], 2.7, requires_grad=True)
    eps_ez = torch.full_like(forward_state["Ez"], 3.1, requires_grad=True)

    actual = reverse_step(
        solver,
        forward_state,
        adjoint_state,
        time_value=0.0,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    expected = adjoint_baselines.reverse_step_cpml_python_reference(
        solver,
        forward_state,
        adjoint_state,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    torch.cuda.synchronize()

    assert actual.backend == "python_reference_cpml"
    for name in forward_state:
        torch.testing.assert_close(actual.pre_step_adjoint[name], expected.pre_step_adjoint[name], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ex, expected.grad_eps_ex, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ey, expected.grad_eps_ey, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ez, expected.grad_eps_ez, rtol=1.0e-5, atol=1.0e-6)


@requires_extension_build
def test_cuda_bloch_reverse_step_uses_native_module_without_slang_jit(monkeypatch):
    import slangtorch

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_ADJOINT_BACKEND", "slang")
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")

    def fail_load_module(*args, **kwargs):
        raise AssertionError("native CUDA Bloch adjoint must not call slangtorch.loadModule")

    monkeypatch.setattr(slangtorch, "loadModule", fail_load_module)
    torch.manual_seed(105)
    solver = _move_solver_tensors_to_cuda(_fake_bloch_reverse_solver())
    state_shapes = _bloch_reverse_state_shapes()
    forward_state = {
        name: torch.randn(state_shapes[name], device="cuda", dtype=torch.float32)
        for name in checkpoint_schema(solver).state_names
    }
    adjoint_state = {
        name: torch.randn_like(tensor)
        for name, tensor in forward_state.items()
    }
    eps_ex = torch.rand(state_shapes["Ex"], device="cuda", dtype=torch.float32) + 2.0
    eps_ey = torch.rand(state_shapes["Ey"], device="cuda", dtype=torch.float32) + 2.0
    eps_ez = torch.rand(state_shapes["Ez"], device="cuda", dtype=torch.float32) + 2.0

    expected = adjoint_baselines.reverse_step_bloch_python_reference(
        solver,
        {name: tensor.clone() for name, tensor in forward_state.items()},
        {name: tensor.clone() for name, tensor in adjoint_state.items()},
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
        resolved_source_terms=([], [], []),
    )
    actual = reverse_step(
        solver,
        {name: tensor.clone() for name, tensor in forward_state.items()},
        {name: tensor.clone() for name, tensor in adjoint_state.items()},
        time_value=0.0,
        eps_ex=eps_ex,
        eps_ey=eps_ey,
        eps_ez=eps_ez,
    )
    torch.cuda.synchronize()

    assert actual.backend == "slang_bloch"
    for name in expected.pre_step_adjoint:
        torch.testing.assert_close(actual.pre_step_adjoint[name], expected.pre_step_adjoint[name], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ex, expected.grad_eps_ex, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ey, expected.grad_eps_ey, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual.grad_eps_ez, expected.grad_eps_ez, rtol=1.0e-5, atol=1.0e-6)


@requires_extension_build
def test_native_adjoint_standard_h_decay_kernel_matches_reference():
    ext = get_compiled_extension()
    torch.manual_seed(107)
    adj_mid = torch.randn((4, 3, 2), device="cuda", dtype=torch.float32)
    decay = torch.rand_like(adj_mid) * 0.2 + 0.8
    actual = torch.empty_like(adj_mid)

    ext.reverse_magnetic_adjoint_decay(actual, adj_mid, decay)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, adj_mid * decay, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_adjoint_standard_electric_reverse_kernels_match_torch_dispatcher(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(129)
    ex = torch.randn((2, 4, 5), device="cuda", dtype=torch.float32)
    ey = torch.randn((3, 3, 5), device="cuda", dtype=torch.float32)
    ez = torch.randn((3, 4, 4), device="cuda", dtype=torch.float32)
    hx = torch.randn((3, 3, 4), device="cuda", dtype=torch.float32)
    hy = torch.randn((2, 4, 4), device="cuda", dtype=torch.float32)
    hz = torch.randn((2, 3, 5), device="cuda", dtype=torch.float32)
    ex_curl = torch.rand_like(ex) * 0.3 + 0.1
    ey_curl = torch.rand_like(ey) * 0.3 + 0.1
    ez_curl = torch.rand_like(ez) * 0.3 + 0.1
    inv_dx, inv_dy, inv_dz = 0.7, 0.8, 0.9

    expected_hx = torch.empty_like(hx)
    expected_hy = torch.empty_like(hy)
    expected_hz = torch.empty_like(hz)
    backend._reverse_electric_hx_standard(
        AdjHxMid=expected_hx,
        AdjHxPost=hx,
        AdjEyPost=ey,
        AdjEzPost=ez,
        EyCurl=ey_curl,
        EzCurl=ez_curl,
        invDy=inv_dy,
        invDz=inv_dz,
    )
    backend._reverse_electric_hy_standard(
        AdjHyMid=expected_hy,
        AdjHyPost=hy,
        AdjExPost=ex,
        AdjEzPost=ez,
        ExCurl=ex_curl,
        EzCurl=ez_curl,
        invDx=inv_dx,
        invDz=inv_dz,
    )
    backend._reverse_electric_hz_standard(
        AdjHzMid=expected_hz,
        AdjHzPost=hz,
        AdjExPost=ex,
        AdjEyPost=ey,
        ExCurl=ex_curl,
        EyCurl=ey_curl,
        invDx=inv_dx,
        invDy=inv_dy,
    )

    actual_hx = torch.empty_like(hx)
    actual_hy = torch.empty_like(hy)
    actual_hz = torch.empty_like(hz)
    ext.reverse_electric_adjoint_to_hx_standard(actual_hx, hx, ey, ez, ey_curl, ez_curl, inv_dy, inv_dz)
    ext.reverse_electric_adjoint_to_hy_standard(actual_hy, hy, ex, ez, ex_curl, ez_curl, inv_dx, inv_dz)
    ext.reverse_electric_adjoint_to_hz_standard(actual_hz, hz, ex, ey, ex_curl, ey_curl, inv_dx, inv_dy)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_hx, expected_hx, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hy, expected_hy, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hz, expected_hz, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
@pytest.mark.parametrize(
    "boundary_modes",
    [
        (BOUNDARY_NONE, BOUNDARY_NONE, BOUNDARY_NONE, BOUNDARY_NONE),
        (BOUNDARY_PEC, BOUNDARY_PEC, BOUNDARY_PEC, BOUNDARY_PEC),
    ],
)
def test_native_adjoint_standard_magnetic_reverse_kernels_match_torch_dispatcher(monkeypatch, boundary_modes):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(131)
    ex = torch.randn((2, 4, 5), device="cuda", dtype=torch.float32)
    ey = torch.randn((3, 3, 5), device="cuda", dtype=torch.float32)
    ez = torch.randn((3, 4, 4), device="cuda", dtype=torch.float32)
    hx = torch.randn((3, 3, 4), device="cuda", dtype=torch.float32)
    hy = torch.randn((2, 4, 4), device="cuda", dtype=torch.float32)
    hz = torch.randn((2, 3, 5), device="cuda", dtype=torch.float32)
    ex_decay = torch.rand_like(ex) * 0.2 + 0.7
    ey_decay = torch.rand_like(ey) * 0.2 + 0.7
    ez_decay = torch.rand_like(ez) * 0.2 + 0.7
    ex_curl = torch.rand_like(ex) * 0.3 + 0.1
    ey_curl = torch.rand_like(ey) * 0.3 + 0.1
    ez_curl = torch.rand_like(ez) * 0.3 + 0.1
    hx_curl = torch.rand_like(hx) * 0.3 + 0.1
    hy_curl = torch.rand_like(hy) * 0.3 + 0.1
    hz_curl = torch.rand_like(hz) * 0.3 + 0.1
    eps_ex = torch.rand_like(ex) + 2.0
    eps_ey = torch.rand_like(ey) + 2.0
    eps_ez = torch.rand_like(ez) + 2.0
    inv_dx, inv_dy, inv_dz = 0.7, 0.8, 0.9
    low_a, high_a, low_b, high_b = boundary_modes

    expected_ex = torch.empty_like(ex)
    expected_ey = torch.empty_like(ey)
    expected_ez = torch.empty_like(ez)
    expected_grad_ex = torch.empty_like(ex)
    expected_grad_ey = torch.empty_like(ey)
    expected_grad_ez = torch.empty_like(ez)
    backend._reverse_magnetic_ex_standard(
        AdjExPrev=expected_ex,
        GradEpsEx=expected_grad_ex,
        AdjExPost=ex,
        AdjHyMid=hy,
        AdjHzMid=hz,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        EpsEx=eps_ex,
        HyMid=hy,
        HzMid=hz,
        HyCurl=hy_curl,
        HzCurl=hz_curl,
        invDy=inv_dy,
        invDz=inv_dz,
        yLowBoundaryMode=low_a,
        yHighBoundaryMode=high_a,
        zLowBoundaryMode=low_b,
        zHighBoundaryMode=high_b,
    )
    backend._reverse_magnetic_ey_standard(
        AdjEyPrev=expected_ey,
        GradEpsEy=expected_grad_ey,
        AdjEyPost=ey,
        AdjHxMid=hx,
        AdjHzMid=hz,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        EpsEy=eps_ey,
        HxMid=hx,
        HzMid=hz,
        HxCurl=hx_curl,
        HzCurl=hz_curl,
        invDx=inv_dx,
        invDz=inv_dz,
        xLowBoundaryMode=low_a,
        xHighBoundaryMode=high_a,
        zLowBoundaryMode=low_b,
        zHighBoundaryMode=high_b,
    )
    backend._reverse_magnetic_ez_standard(
        AdjEzPrev=expected_ez,
        GradEpsEz=expected_grad_ez,
        AdjEzPost=ez,
        AdjHxMid=hx,
        AdjHyMid=hy,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        EpsEz=eps_ez,
        HxMid=hx,
        HyMid=hy,
        HxCurl=hx_curl,
        HyCurl=hy_curl,
        invDx=inv_dx,
        invDy=inv_dy,
        xLowBoundaryMode=low_a,
        xHighBoundaryMode=high_a,
        yLowBoundaryMode=low_b,
        yHighBoundaryMode=high_b,
    )

    actual_ex = torch.empty_like(ex)
    actual_ey = torch.empty_like(ey)
    actual_ez = torch.empty_like(ez)
    actual_grad_ex = torch.empty_like(ex)
    actual_grad_ey = torch.empty_like(ey)
    actual_grad_ez = torch.empty_like(ez)
    ext.reverse_magnetic_adjoint_to_ex_standard(
        actual_ex,
        actual_grad_ex,
        ex,
        hy,
        hz,
        ex_decay,
        ex_curl,
        eps_ex,
        hy,
        hz,
        hy_curl,
        hz_curl,
        inv_dy,
        inv_dz,
        low_a,
        high_a,
        low_b,
        high_b,
    )
    ext.reverse_magnetic_adjoint_to_ey_standard(
        actual_ey,
        actual_grad_ey,
        ey,
        hx,
        hz,
        ey_decay,
        ey_curl,
        eps_ey,
        hx,
        hz,
        hx_curl,
        hz_curl,
        inv_dx,
        inv_dz,
        low_a,
        high_a,
        low_b,
        high_b,
    )
    ext.reverse_magnetic_adjoint_to_ez_standard(
        actual_ez,
        actual_grad_ez,
        ez,
        hx,
        hy,
        ez_decay,
        ez_curl,
        eps_ez,
        hx,
        hy,
        hx_curl,
        hy_curl,
        inv_dx,
        inv_dy,
        low_a,
        high_a,
        low_b,
        high_b,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ex, expected_ex, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ey, expected_ey, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ez, expected_ez, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_grad_ex, expected_grad_ex, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_grad_ey, expected_grad_ey, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_grad_ez, expected_grad_ez, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_adjoint_bloch_electric_reverse_kernels_match_torch_dispatcher(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(132)
    ex_r = torch.randn((2, 4, 5), device="cuda", dtype=torch.float32)
    ex_i = torch.randn_like(ex_r)
    ey_r = torch.randn((3, 3, 5), device="cuda", dtype=torch.float32)
    ey_i = torch.randn_like(ey_r)
    ez_r = torch.randn((3, 4, 4), device="cuda", dtype=torch.float32)
    ez_i = torch.randn_like(ez_r)
    hx_r = torch.randn((3, 3, 4), device="cuda", dtype=torch.float32)
    hx_i = torch.randn_like(hx_r)
    hy_r = torch.randn((2, 4, 4), device="cuda", dtype=torch.float32)
    hy_i = torch.randn_like(hy_r)
    hz_r = torch.randn((2, 3, 5), device="cuda", dtype=torch.float32)
    hz_i = torch.randn_like(hz_r)
    ex_curl = torch.rand_like(ex_r) * 0.3 + 0.1
    ey_curl = torch.rand_like(ey_r) * 0.3 + 0.1
    ez_curl = torch.rand_like(ez_r) * 0.3 + 0.1
    phase_x = (0.37, 0.19)
    phase_y = (0.73, -0.28)
    phase_z = (0.58, 0.44)
    inv_dx, inv_dy, inv_dz = 0.7, 0.8, 0.9

    expected_hx_r = torch.empty_like(hx_r)
    expected_hx_i = torch.empty_like(hx_i)
    expected_hy_r = torch.empty_like(hy_r)
    expected_hy_i = torch.empty_like(hy_i)
    expected_hz_r = torch.empty_like(hz_r)
    expected_hz_i = torch.empty_like(hz_i)
    backend._reverse_electric_hx_bloch(
        AdjHxMidReal=expected_hx_r,
        AdjHxMidImag=expected_hx_i,
        AdjHxPostReal=hx_r,
        AdjHxPostImag=hx_i,
        AdjEyPostReal=ey_r,
        AdjEyPostImag=ey_i,
        AdjEzPostReal=ez_r,
        AdjEzPostImag=ez_i,
        EyCurl=ey_curl,
        EzCurl=ez_curl,
        phaseCosY=phase_y[0],
        phaseSinY=phase_y[1],
        phaseCosZ=phase_z[0],
        phaseSinZ=phase_z[1],
        invDy=inv_dy,
        invDz=inv_dz,
    )
    backend._reverse_electric_hy_bloch(
        AdjHyMidReal=expected_hy_r,
        AdjHyMidImag=expected_hy_i,
        AdjHyPostReal=hy_r,
        AdjHyPostImag=hy_i,
        AdjExPostReal=ex_r,
        AdjExPostImag=ex_i,
        AdjEzPostReal=ez_r,
        AdjEzPostImag=ez_i,
        ExCurl=ex_curl,
        EzCurl=ez_curl,
        phaseCosX=phase_x[0],
        phaseSinX=phase_x[1],
        phaseCosZ=phase_z[0],
        phaseSinZ=phase_z[1],
        invDx=inv_dx,
        invDz=inv_dz,
    )
    backend._reverse_electric_hz_bloch(
        AdjHzMidReal=expected_hz_r,
        AdjHzMidImag=expected_hz_i,
        AdjHzPostReal=hz_r,
        AdjHzPostImag=hz_i,
        AdjExPostReal=ex_r,
        AdjExPostImag=ex_i,
        AdjEyPostReal=ey_r,
        AdjEyPostImag=ey_i,
        ExCurl=ex_curl,
        EyCurl=ey_curl,
        phaseCosX=phase_x[0],
        phaseSinX=phase_x[1],
        phaseCosY=phase_y[0],
        phaseSinY=phase_y[1],
        invDx=inv_dx,
        invDy=inv_dy,
    )

    actual_hx_r = torch.empty_like(hx_r)
    actual_hx_i = torch.empty_like(hx_i)
    actual_hy_r = torch.empty_like(hy_r)
    actual_hy_i = torch.empty_like(hy_i)
    actual_hz_r = torch.empty_like(hz_r)
    actual_hz_i = torch.empty_like(hz_i)
    ext.reverse_electric_adjoint_to_hx_bloch(
        actual_hx_r, actual_hx_i, hx_r, hx_i, ey_r, ey_i, ez_r, ez_i,
        ey_curl, ez_curl, phase_y[0], phase_y[1], phase_z[0], phase_z[1], inv_dy, inv_dz,
    )
    ext.reverse_electric_adjoint_to_hy_bloch(
        actual_hy_r, actual_hy_i, hy_r, hy_i, ex_r, ex_i, ez_r, ez_i,
        ex_curl, ez_curl, phase_x[0], phase_x[1], phase_z[0], phase_z[1], inv_dx, inv_dz,
    )
    ext.reverse_electric_adjoint_to_hz_bloch(
        actual_hz_r, actual_hz_i, hz_r, hz_i, ex_r, ex_i, ey_r, ey_i,
        ex_curl, ey_curl, phase_x[0], phase_x[1], phase_y[0], phase_y[1], inv_dx, inv_dy,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_hx_r, expected_hx_r, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hx_i, expected_hx_i, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hy_r, expected_hy_r, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hy_i, expected_hy_i, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hz_r, expected_hz_r, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_hz_i, expected_hz_i, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_adjoint_bloch_magnetic_reverse_kernels_match_torch_dispatcher(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(134)
    ex_r = torch.randn((2, 4, 5), device="cuda", dtype=torch.float32)
    ex_i = torch.randn_like(ex_r)
    ey_r = torch.randn((3, 3, 5), device="cuda", dtype=torch.float32)
    ey_i = torch.randn_like(ey_r)
    ez_r = torch.randn((3, 4, 4), device="cuda", dtype=torch.float32)
    ez_i = torch.randn_like(ez_r)
    hx_mid_r = torch.randn((3, 3, 4), device="cuda", dtype=torch.float32)
    hx_mid_i = torch.randn_like(hx_mid_r)
    hy_mid_r = torch.randn((2, 4, 4), device="cuda", dtype=torch.float32)
    hy_mid_i = torch.randn_like(hy_mid_r)
    hz_mid_r = torch.randn((2, 3, 5), device="cuda", dtype=torch.float32)
    hz_mid_i = torch.randn_like(hz_mid_r)
    adj_hx_r = torch.randn_like(hx_mid_r)
    adj_hx_i = torch.randn_like(hx_mid_i)
    adj_hy_r = torch.randn_like(hy_mid_r)
    adj_hy_i = torch.randn_like(hy_mid_i)
    adj_hz_r = torch.randn_like(hz_mid_r)
    adj_hz_i = torch.randn_like(hz_mid_i)
    ex_decay = torch.rand_like(ex_r) * 0.2 + 0.7
    ey_decay = torch.rand_like(ey_r) * 0.2 + 0.7
    ez_decay = torch.rand_like(ez_r) * 0.2 + 0.7
    ex_curl = torch.rand_like(ex_r) * 0.3 + 0.1
    ey_curl = torch.rand_like(ey_r) * 0.3 + 0.1
    ez_curl = torch.rand_like(ez_r) * 0.3 + 0.1
    hx_curl = torch.rand_like(hx_mid_r) * 0.3 + 0.1
    hy_curl = torch.rand_like(hy_mid_r) * 0.3 + 0.1
    hz_curl = torch.rand_like(hz_mid_r) * 0.3 + 0.1
    eps_ex = torch.rand_like(ex_r) + 2.0
    eps_ey = torch.rand_like(ey_r) + 2.0
    eps_ez = torch.rand_like(ez_r) + 2.0
    phase_x = (0.37, 0.19)
    phase_y = (0.73, -0.28)
    phase_z = (0.58, 0.44)
    inv_dx, inv_dy, inv_dz = 0.7, 0.8, 0.9

    expected_ex_r = torch.empty_like(ex_r)
    expected_ex_i = torch.empty_like(ex_i)
    expected_ey_r = torch.empty_like(ey_r)
    expected_ey_i = torch.empty_like(ey_i)
    expected_ez_r = torch.empty_like(ez_r)
    expected_ez_i = torch.empty_like(ez_i)
    expected_grad_ex = torch.empty_like(ex_r)
    expected_grad_ey = torch.empty_like(ey_r)
    expected_grad_ez = torch.empty_like(ez_r)
    backend._reverse_magnetic_ex_bloch(
        AdjExPrevReal=expected_ex_r,
        AdjExPrevImag=expected_ex_i,
        GradEpsEx=expected_grad_ex,
        AdjExPostReal=ex_r,
        AdjExPostImag=ex_i,
        AdjHyMidReal=adj_hy_r,
        AdjHyMidImag=adj_hy_i,
        AdjHzMidReal=adj_hz_r,
        AdjHzMidImag=adj_hz_i,
        ExDecay=ex_decay,
        ExCurl=ex_curl,
        EpsEx=eps_ex,
        HyMidReal=hy_mid_r,
        HyMidImag=hy_mid_i,
        HzMidReal=hz_mid_r,
        HzMidImag=hz_mid_i,
        HyCurl=hy_curl,
        HzCurl=hz_curl,
        phaseCosY=phase_y[0],
        phaseSinY=phase_y[1],
        phaseCosZ=phase_z[0],
        phaseSinZ=phase_z[1],
        invDy=inv_dy,
        invDz=inv_dz,
    )
    backend._reverse_magnetic_ey_bloch(
        AdjEyPrevReal=expected_ey_r,
        AdjEyPrevImag=expected_ey_i,
        GradEpsEy=expected_grad_ey,
        AdjEyPostReal=ey_r,
        AdjEyPostImag=ey_i,
        AdjHxMidReal=adj_hx_r,
        AdjHxMidImag=adj_hx_i,
        AdjHzMidReal=adj_hz_r,
        AdjHzMidImag=adj_hz_i,
        EyDecay=ey_decay,
        EyCurl=ey_curl,
        EpsEy=eps_ey,
        HxMidReal=hx_mid_r,
        HxMidImag=hx_mid_i,
        HzMidReal=hz_mid_r,
        HzMidImag=hz_mid_i,
        HxCurl=hx_curl,
        HzCurl=hz_curl,
        phaseCosX=phase_x[0],
        phaseSinX=phase_x[1],
        phaseCosZ=phase_z[0],
        phaseSinZ=phase_z[1],
        invDx=inv_dx,
        invDz=inv_dz,
    )
    backend._reverse_magnetic_ez_bloch(
        AdjEzPrevReal=expected_ez_r,
        AdjEzPrevImag=expected_ez_i,
        GradEpsEz=expected_grad_ez,
        AdjEzPostReal=ez_r,
        AdjEzPostImag=ez_i,
        AdjHxMidReal=adj_hx_r,
        AdjHxMidImag=adj_hx_i,
        AdjHyMidReal=adj_hy_r,
        AdjHyMidImag=adj_hy_i,
        EzDecay=ez_decay,
        EzCurl=ez_curl,
        EpsEz=eps_ez,
        HxMidReal=hx_mid_r,
        HxMidImag=hx_mid_i,
        HyMidReal=hy_mid_r,
        HyMidImag=hy_mid_i,
        HxCurl=hx_curl,
        HyCurl=hy_curl,
        phaseCosX=phase_x[0],
        phaseSinX=phase_x[1],
        phaseCosY=phase_y[0],
        phaseSinY=phase_y[1],
        invDx=inv_dx,
        invDy=inv_dy,
    )

    actual_ex_r = torch.empty_like(ex_r)
    actual_ex_i = torch.empty_like(ex_i)
    actual_ey_r = torch.empty_like(ey_r)
    actual_ey_i = torch.empty_like(ey_i)
    actual_ez_r = torch.empty_like(ez_r)
    actual_ez_i = torch.empty_like(ez_i)
    actual_grad_ex = torch.empty_like(ex_r)
    actual_grad_ey = torch.empty_like(ey_r)
    actual_grad_ez = torch.empty_like(ez_r)
    ext.reverse_magnetic_adjoint_to_ex_bloch(
        actual_ex_r, actual_ex_i, actual_grad_ex, ex_r, ex_i, adj_hy_r, adj_hy_i, adj_hz_r, adj_hz_i,
        ex_decay, ex_curl, eps_ex, hy_mid_r, hy_mid_i, hz_mid_r, hz_mid_i, hy_curl, hz_curl,
        phase_y[0], phase_y[1], phase_z[0], phase_z[1], inv_dy, inv_dz,
    )
    ext.reverse_magnetic_adjoint_to_ey_bloch(
        actual_ey_r, actual_ey_i, actual_grad_ey, ey_r, ey_i, adj_hx_r, adj_hx_i, adj_hz_r, adj_hz_i,
        ey_decay, ey_curl, eps_ey, hx_mid_r, hx_mid_i, hz_mid_r, hz_mid_i, hx_curl, hz_curl,
        phase_x[0], phase_x[1], phase_z[0], phase_z[1], inv_dx, inv_dz,
    )
    ext.reverse_magnetic_adjoint_to_ez_bloch(
        actual_ez_r, actual_ez_i, actual_grad_ez, ez_r, ez_i, adj_hx_r, adj_hx_i, adj_hy_r, adj_hy_i,
        ez_decay, ez_curl, eps_ez, hx_mid_r, hx_mid_i, hy_mid_r, hy_mid_i, hx_curl, hy_curl,
        phase_x[0], phase_x[1], phase_y[0], phase_y[1], inv_dx, inv_dy,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ex_r, expected_ex_r, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ex_i, expected_ex_i, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ey_r, expected_ey_r, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ey_i, expected_ey_i, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ez_r, expected_ez_r, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_ez_i, expected_ez_i, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_grad_ex, expected_grad_ex, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_grad_ey, expected_grad_ey, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_grad_ez, expected_grad_ez, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
@pytest.mark.parametrize("axis, inv_delta", [(0, 0.7), (1, 0.8), (2, 0.9)])
def test_native_adjoint_diff_accumulators_match_torch_dispatcher(monkeypatch, axis, inv_delta):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(133 + axis)
    field = torch.randn((4, 3, 5), device="cuda", dtype=torch.float32)
    forward_shape = list(field.shape)
    forward_shape[axis] -= 1
    forward_diff = torch.randn(tuple(forward_shape), device="cuda", dtype=torch.float32)
    backward_diff = torch.randn_like(field)

    expected_forward = field.clone()
    expected_backward = field.clone()
    backend._accumulate_diff_adjoint(expected_forward, forward_diff, axis, inv_delta, forward=True)
    backend._accumulate_diff_adjoint(expected_backward, backward_diff, axis, inv_delta, forward=False)

    actual_forward = field.clone()
    actual_backward = field.clone()
    ext.accumulate_forward_diff_adjoint(actual_forward, forward_diff, axis, inv_delta)
    ext.accumulate_backward_diff_adjoint(actual_backward, backward_diff, axis, inv_delta)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_forward, expected_forward, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_backward, expected_backward, rtol=2.0e-6, atol=2.0e-7)


def _cpml_coeffs(length):
    return (
        torch.rand(length, device="cuda", dtype=torch.float32) * 0.2 + 0.7,
        torch.rand(length, device="cuda", dtype=torch.float32) * 0.05 + 0.01,
        torch.rand(length, device="cuda", dtype=torch.float32) * 0.2 + 0.8,
    )


@requires_extension_build
def test_native_adjoint_cpml_electric_reverse_kernels_match_torch_dispatcher(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(137)
    ex = torch.randn((2, 4, 5), device="cuda", dtype=torch.float32)
    ey = torch.randn((3, 3, 5), device="cuda", dtype=torch.float32)
    ez = torch.randn((3, 4, 4), device="cuda", dtype=torch.float32)
    hx = torch.randn((3, 3, 4), device="cuda", dtype=torch.float32)
    hy = torch.randn((2, 4, 4), device="cuda", dtype=torch.float32)
    hz = torch.randn((2, 3, 5), device="cuda", dtype=torch.float32)
    inv_dx, inv_dy, inv_dz = 0.7, 0.8, 0.9

    for field, method_name, backend_method, h_pos, h_neg, pos_len, neg_len, inv_pos, inv_neg, mode_keys in [
        (ex, "reverse_electric_component_ex_cpml", backend._reverse_electric_cpml_ex, hz, hy, ex.shape[1], ex.shape[2], inv_dy, inv_dz, ("y", "z")),
        (ey, "reverse_electric_component_ey_cpml", backend._reverse_electric_cpml_ey, hx, hz, ey.shape[2], ey.shape[0], inv_dz, inv_dx, ("x", "z")),
        (ez, "reverse_electric_component_ez_cpml", backend._reverse_electric_cpml_ez, hy, hx, ez.shape[0], ez.shape[1], inv_dx, inv_dy, ("x", "y")),
    ]:
        b_pos, c_pos, k_pos = _cpml_coeffs(pos_len)
        b_neg, c_neg, k_neg = _cpml_coeffs(neg_len)
        adj_post = torch.randn_like(field)
        adj_psi_pos_post = torch.randn_like(field)
        adj_psi_neg_post = torch.randn_like(field)
        decay = torch.rand_like(field) * 0.2 + 0.7
        curl = torch.rand_like(field) * 0.3 + 0.1
        eps = torch.rand_like(field) + 2.0
        psi_pos = torch.randn_like(field)
        psi_neg = torch.randn_like(field)
        expected_prev = torch.empty_like(field)
        expected_grad = torch.empty_like(field)
        expected_psi_pos = torch.empty_like(field)
        expected_psi_neg = torch.empty_like(field)
        expected_d_pos = torch.empty_like(field)
        expected_d_neg = torch.empty_like(field)
        actual_prev = torch.empty_like(field)
        actual_grad = torch.empty_like(field)
        actual_psi_pos = torch.empty_like(field)
        actual_psi_neg = torch.empty_like(field)
        actual_d_pos = torch.empty_like(field)
        actual_d_neg = torch.empty_like(field)

        if method_name.endswith("ex_cpml"):
            backend_method(
                AdjExPrev=expected_prev,
                GradEpsEx=expected_grad,
                AdjPsiPosPrev=expected_psi_pos,
                AdjPsiNegPrev=expected_psi_neg,
                AdjDPos=expected_d_pos,
                AdjDNeg=expected_d_neg,
                AdjExPost=adj_post,
                AdjPsiPosPost=adj_psi_pos_post,
                AdjPsiNegPost=adj_psi_neg_post,
                ExDecay=decay,
                ExCurl=curl,
                EpsEx=eps,
                PsiPos=psi_pos,
                PsiNeg=psi_neg,
                BPos=b_pos,
                CPos=c_pos,
                InvKappaPos=k_pos,
                BNeg=b_neg,
                CNeg=c_neg,
                InvKappaNeg=k_neg,
                HyMid=hy,
                HzMid=hz,
                invDy=inv_dy,
                invDz=inv_dz,
                yLowBoundaryMode=BOUNDARY_NONE,
                yHighBoundaryMode=BOUNDARY_NONE,
                zLowBoundaryMode=BOUNDARY_NONE,
                zHighBoundaryMode=BOUNDARY_NONE,
            )
            ext.reverse_electric_component_ex_cpml(
                actual_prev,
                actual_grad,
                actual_psi_pos,
                actual_psi_neg,
                actual_d_pos,
                actual_d_neg,
                adj_post,
                adj_psi_pos_post,
                adj_psi_neg_post,
                decay,
                curl,
                eps,
                psi_pos,
                psi_neg,
                b_pos,
                c_pos,
                k_pos,
                b_neg,
                c_neg,
                k_neg,
                hy,
                hz,
                inv_dy,
                inv_dz,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
            )
        elif method_name.endswith("ey_cpml"):
            backend_method(
                AdjEyPrev=expected_prev,
                GradEpsEy=expected_grad,
                AdjPsiPosPrev=expected_psi_pos,
                AdjPsiNegPrev=expected_psi_neg,
                AdjDPos=expected_d_pos,
                AdjDNeg=expected_d_neg,
                AdjEyPost=adj_post,
                AdjPsiPosPost=adj_psi_pos_post,
                AdjPsiNegPost=adj_psi_neg_post,
                EyDecay=decay,
                EyCurl=curl,
                EpsEy=eps,
                PsiPos=psi_pos,
                PsiNeg=psi_neg,
                BPos=b_pos,
                CPos=c_pos,
                InvKappaPos=k_pos,
                BNeg=b_neg,
                CNeg=c_neg,
                InvKappaNeg=k_neg,
                HxMid=hx,
                HzMid=hz,
                invDx=inv_dx,
                invDz=inv_dz,
                xLowBoundaryMode=BOUNDARY_NONE,
                xHighBoundaryMode=BOUNDARY_NONE,
                zLowBoundaryMode=BOUNDARY_NONE,
                zHighBoundaryMode=BOUNDARY_NONE,
            )
            ext.reverse_electric_component_ey_cpml(
                actual_prev,
                actual_grad,
                actual_psi_pos,
                actual_psi_neg,
                actual_d_pos,
                actual_d_neg,
                adj_post,
                adj_psi_pos_post,
                adj_psi_neg_post,
                decay,
                curl,
                eps,
                psi_pos,
                psi_neg,
                b_pos,
                c_pos,
                k_pos,
                b_neg,
                c_neg,
                k_neg,
                hx,
                hz,
                inv_dx,
                inv_dz,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
            )
        else:
            backend_method(
                AdjEzPrev=expected_prev,
                GradEpsEz=expected_grad,
                AdjPsiPosPrev=expected_psi_pos,
                AdjPsiNegPrev=expected_psi_neg,
                AdjDPos=expected_d_pos,
                AdjDNeg=expected_d_neg,
                AdjEzPost=adj_post,
                AdjPsiPosPost=adj_psi_pos_post,
                AdjPsiNegPost=adj_psi_neg_post,
                EzDecay=decay,
                EzCurl=curl,
                EpsEz=eps,
                PsiPos=psi_pos,
                PsiNeg=psi_neg,
                BPos=b_pos,
                CPos=c_pos,
                InvKappaPos=k_pos,
                BNeg=b_neg,
                CNeg=c_neg,
                InvKappaNeg=k_neg,
                HxMid=hx,
                HyMid=hy,
                invDx=inv_dx,
                invDy=inv_dy,
                xLowBoundaryMode=BOUNDARY_NONE,
                xHighBoundaryMode=BOUNDARY_NONE,
                yLowBoundaryMode=BOUNDARY_NONE,
                yHighBoundaryMode=BOUNDARY_NONE,
            )
            ext.reverse_electric_component_ez_cpml(
                actual_prev,
                actual_grad,
                actual_psi_pos,
                actual_psi_neg,
                actual_d_pos,
                actual_d_neg,
                adj_post,
                adj_psi_pos_post,
                adj_psi_neg_post,
                decay,
                curl,
                eps,
                psi_pos,
                psi_neg,
                b_pos,
                c_pos,
                k_pos,
                b_neg,
                c_neg,
                k_neg,
                hx,
                hy,
                inv_dx,
                inv_dy,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
                BOUNDARY_NONE,
            )
        torch.cuda.synchronize()
        for actual, expected in [
            (actual_prev, expected_prev),
            (actual_grad, expected_grad),
            (actual_psi_pos, expected_psi_pos),
            (actual_psi_neg, expected_psi_neg),
            (actual_d_pos, expected_d_pos),
            (actual_d_neg, expected_d_neg),
        ]:
            torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_adjoint_cpml_magnetic_reverse_kernels_match_torch_dispatcher(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(139)
    for field, backend_method, ext_method, pos_len, neg_len, prefix in [
        (torch.randn((3, 3, 4), device="cuda", dtype=torch.float32), backend._reverse_magnetic_cpml_hx, ext.reverse_magnetic_component_hx_cpml, 3, 4, "Hx"),
        (torch.randn((2, 4, 4), device="cuda", dtype=torch.float32), backend._reverse_magnetic_cpml_hy, ext.reverse_magnetic_component_hy_cpml, 4, 2, "Hy"),
        (torch.randn((2, 3, 5), device="cuda", dtype=torch.float32), backend._reverse_magnetic_cpml_hz, ext.reverse_magnetic_component_hz_cpml, 2, 3, "Hz"),
    ]:
        b_pos, c_pos, k_pos = _cpml_coeffs(pos_len)
        b_neg, c_neg, k_neg = _cpml_coeffs(neg_len)
        adj_post = torch.randn_like(field)
        adj_psi_pos_post = torch.randn_like(field)
        adj_psi_neg_post = torch.randn_like(field)
        decay = torch.rand_like(field) * 0.2 + 0.7
        curl = torch.rand_like(field) * 0.3 + 0.1
        expected_prev = torch.empty_like(field)
        expected_psi_pos = torch.empty_like(field)
        expected_psi_neg = torch.empty_like(field)
        expected_d_pos = torch.empty_like(field)
        expected_d_neg = torch.empty_like(field)
        actual_prev = torch.empty_like(field)
        actual_psi_pos = torch.empty_like(field)
        actual_psi_neg = torch.empty_like(field)
        actual_d_pos = torch.empty_like(field)
        actual_d_neg = torch.empty_like(field)

        backend_method(
            **{
                f"Adj{prefix}Prev": expected_prev,
                "AdjPsiPosPrev": expected_psi_pos,
                "AdjPsiNegPrev": expected_psi_neg,
                "AdjDPos": expected_d_pos,
                "AdjDNeg": expected_d_neg,
                f"Adj{prefix}Post": adj_post,
                "AdjPsiPosPost": adj_psi_pos_post,
                "AdjPsiNegPost": adj_psi_neg_post,
                f"{prefix}Decay": decay,
                f"{prefix}Curl": curl,
                "BPos": b_pos,
                "CPos": c_pos,
                "InvKappaPos": k_pos,
                "BNeg": b_neg,
                "CNeg": c_neg,
                "InvKappaNeg": k_neg,
            }
        )
        ext_method(
            actual_prev,
            actual_psi_pos,
            actual_psi_neg,
            actual_d_pos,
            actual_d_neg,
            adj_post,
            adj_psi_pos_post,
            adj_psi_neg_post,
            decay,
            curl,
            b_pos,
            c_pos,
            k_pos,
            b_neg,
            c_neg,
            k_neg,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_prev, expected_prev, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(actual_psi_pos, expected_psi_pos, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(actual_psi_neg, expected_psi_neg, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(actual_d_pos, expected_d_pos, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(actual_d_neg, expected_d_neg, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
@pytest.mark.parametrize("model_name", ["debye", "drude", "lorentz"])
def test_native_adjoint_dispersive_current_reverse_kernels_match_reference(model_name):
    ext = get_compiled_extension()
    torch.manual_seed({"debye": 109, "drude": 111, "lorentz": 113}[model_name])
    shape = (3, 4, 2)
    drive = torch.rand(shape, device="cuda", dtype=torch.float32) * 0.4 + 0.1
    decay = 0.73
    dt = 0.05
    adj_current_post = torch.randn(shape, device="cuda", dtype=torch.float32)
    adj_electric = torch.zeros(shape, device="cuda", dtype=torch.float32)

    if model_name == "debye":
        adj_polarization_post = torch.randn(shape, device="cuda", dtype=torch.float32)
        adj_polarization_prev = torch.zeros(shape, device="cuda", dtype=torch.float32)
        ext.reverse_debye_current(
            adj_electric,
            adj_polarization_prev,
            adj_polarization_post,
            adj_current_post,
            drive,
            decay,
            dt,
        )
        adj_internal = adj_polarization_post + adj_current_post / dt
        expected_electric = drive * adj_internal
        expected_polarization = decay * adj_internal - adj_current_post / dt
        torch.cuda.synchronize()
        torch.testing.assert_close(adj_electric, expected_electric, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(adj_polarization_prev, expected_polarization, rtol=2.0e-6, atol=2.0e-7)
        return

    if model_name == "drude":
        adj_current_prev = torch.zeros(shape, device="cuda", dtype=torch.float32)
        ext.reverse_drude_current(adj_electric, adj_current_prev, adj_current_post, drive, decay)
        torch.cuda.synchronize()
        torch.testing.assert_close(adj_electric, drive * adj_current_post, rtol=2.0e-6, atol=2.0e-7)
        torch.testing.assert_close(adj_current_prev, decay * adj_current_post, rtol=2.0e-6, atol=2.0e-7)
        return

    restoring = 0.19
    adj_polarization_post = torch.randn(shape, device="cuda", dtype=torch.float32)
    adj_polarization_prev = torch.zeros(shape, device="cuda", dtype=torch.float32)
    adj_current_prev = torch.zeros(shape, device="cuda", dtype=torch.float32)
    ext.reverse_lorentz_current(
        adj_electric,
        adj_polarization_prev,
        adj_current_prev,
        adj_polarization_post,
        adj_current_post,
        drive,
        decay,
        restoring,
        dt,
    )
    adj_internal = adj_current_post + dt * adj_polarization_post
    torch.cuda.synchronize()
    torch.testing.assert_close(adj_electric, drive * adj_internal, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(
        adj_polarization_prev,
        adj_polarization_post - restoring * adj_internal,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    torch.testing.assert_close(adj_current_prev, decay * adj_internal, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_adjoint_tfsf_sample_accumulators_match_torch_dispatcher(monkeypatch):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    ext = get_compiled_extension()
    torch.manual_seed(135)
    adj_field_patch = torch.randn((3, 2, 4), device="cuda", dtype=torch.float32)
    coeff_patch = torch.randn_like(adj_field_patch)
    component_scale = -0.37

    expected_scalar = torch.zeros(8, device="cuda", dtype=torch.float32)
    actual_scalar = torch.zeros_like(expected_scalar)
    backend._accumulate_tfsf_scalar_sample_adjoint(
        AdjAuxField=expected_scalar,
        AdjFieldPatch=adj_field_patch,
        CoeffPatch=coeff_patch,
        sampleIndex=3,
        componentScale=component_scale,
    )
    ext.accumulate_tfsf_scalar_sample_adjoint(actual_scalar, adj_field_patch, coeff_patch, 3, component_scale)

    sample_indices = torch.tensor([5, 2, 5, 1], device="cuda", dtype=torch.int32)
    expected_line = torch.zeros_like(expected_scalar)
    actual_line = torch.zeros_like(expected_scalar)
    backend._accumulate_tfsf_line_sample_adjoint(
        AdjAuxField=expected_line,
        AdjFieldPatch=adj_field_patch,
        CoeffPatch=coeff_patch,
        SampleIndices=sample_indices,
        sampleAxisCode=2,
        componentScale=component_scale,
    )
    ext.accumulate_tfsf_line_sample_adjoint(actual_line, adj_field_patch, coeff_patch, sample_indices, 2, component_scale)

    sample_positions = torch.linspace(-0.25, 1.75, adj_field_patch.numel(), device="cuda", dtype=torch.float32).reshape_as(adj_field_patch)
    expected_interp = torch.zeros_like(expected_scalar)
    actual_interp = torch.zeros_like(expected_scalar)
    backend._accumulate_tfsf_interpolated_sample_adjoint(
        AdjAuxField=expected_interp,
        AdjFieldPatch=adj_field_patch,
        CoeffPatch=coeff_patch,
        SamplePositions=sample_positions,
        origin=-0.1,
        ds=0.27,
        componentScale=component_scale,
    )
    ext.accumulate_tfsf_interpolated_sample_adjoint(
        actual_interp,
        adj_field_patch,
        coeff_patch,
        sample_positions,
        -0.1,
        0.27,
        component_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_scalar, expected_scalar, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_line, expected_line, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_interp, expected_interp, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_adjoint_tfsf_auxiliary_reverse_kernels_match_reference():
    ext = get_compiled_extension()
    torch.manual_seed(127)
    electric_size = 7
    magnetic_size = electric_size - 1
    source_index = 3

    adj_electric_post = torch.randn(electric_size, device="cuda", dtype=torch.float32)
    electric_decay = torch.rand(electric_size, device="cuda", dtype=torch.float32) * 0.2 + 0.8
    electric_curl = torch.rand(electric_size, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    actual_adj_electric_prev = torch.zeros_like(adj_electric_post)
    actual_adj_magnetic_after = torch.zeros(magnetic_size, device="cuda", dtype=torch.float32)

    ext.reverse_tfsf_auxiliary_electric(
        actual_adj_electric_prev,
        actual_adj_magnetic_after,
        adj_electric_post,
        electric_decay,
        electric_curl,
        source_index,
    )

    expected_adj_electric_prev = torch.zeros_like(adj_electric_post)
    expected_adj_magnetic_after = torch.zeros_like(actual_adj_magnetic_after)
    expected_adj_electric_prev[0] += adj_electric_post[0]
    for index in range(1, electric_size - 1):
        if index == source_index:
            continue
        adjoint = adj_electric_post[index]
        expected_adj_electric_prev[index] += electric_decay[index] * adjoint
        expected_adj_magnetic_after[index - 1] += electric_curl[index] * adjoint
        expected_adj_magnetic_after[index] -= electric_curl[index] * adjoint

    torch.cuda.synchronize()
    torch.testing.assert_close(actual_adj_electric_prev, expected_adj_electric_prev, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_adj_magnetic_after, expected_adj_magnetic_after, rtol=2.0e-6, atol=2.0e-7)

    adj_magnetic_after = torch.randn(magnetic_size, device="cuda", dtype=torch.float32)
    magnetic_decay = torch.rand(magnetic_size, device="cuda", dtype=torch.float32) * 0.2 + 0.8
    magnetic_curl = torch.rand(magnetic_size, device="cuda", dtype=torch.float32) * 0.1 + 0.01
    actual_adj_electric_prev = torch.zeros(electric_size, device="cuda", dtype=torch.float32)
    actual_adj_magnetic_prev = torch.empty_like(adj_magnetic_after)

    ext.reverse_tfsf_auxiliary_magnetic(
        actual_adj_electric_prev,
        actual_adj_magnetic_prev,
        adj_magnetic_after,
        magnetic_decay,
        magnetic_curl,
    )

    expected_adj_electric_prev = torch.zeros_like(actual_adj_electric_prev)
    expected_adj_magnetic_prev = magnetic_decay * adj_magnetic_after
    for index in range(magnetic_size):
        expected_adj_electric_prev[index] += magnetic_curl[index] * adj_magnetic_after[index]
        expected_adj_electric_prev[index + 1] -= magnetic_curl[index] * adj_magnetic_after[index]

    torch.cuda.synchronize()
    torch.testing.assert_close(actual_adj_magnetic_prev, expected_adj_magnetic_prev, rtol=2.0e-6, atol=2.0e-7)
    torch.testing.assert_close(actual_adj_electric_prev, expected_adj_electric_prev, rtol=2.0e-6, atol=2.0e-7)


@requires_extension_build
def test_native_module_handles_non_contiguous_views(monkeypatch):
    """The adjoint replay passes strided views (box slices of adjoint fields,
    real/imag views) through the native module surface. Pin against the pure
    Yee update formula, not against another backend."""
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    module = backend.get_native_fdtd_module()
    torch.manual_seed(7)

    # Build genuinely non-contiguous views by slicing the trailing axes.
    hx_base = torch.randn(6, 7, 8, device="cuda", dtype=torch.float32)
    ey_base = torch.randn(6, 7, 9, device="cuda", dtype=torch.float32)
    ez_base = torch.randn(6, 8, 8, device="cuda", dtype=torch.float32)
    decay_base = torch.rand(6, 7, 8, device="cuda", dtype=torch.float32) * 0.2 + 0.8
    curl_base = torch.rand(6, 7, 8, device="cuda", dtype=torch.float32) * 0.1 + 0.01

    hx = hx_base[:, :6, :7]
    ey = ey_base[:, :6, :8]
    ez = ez_base[:, :7, :7]
    decay = decay_base[:, :6, :7]
    curl = curl_base[:, :6, :7]
    assert not hx.is_contiguous() and not ey.is_contiguous()

    inv_dy = 1.7
    inv_dz = 2.3
    curl_e = (ez[:, 1:, :] - ez[:, :-1, :]) * inv_dy - (ey[:, :, 1:] - ey[:, :, :-1]) * inv_dz
    expected = hx * decay - curl * curl_e

    module.updateMagneticFieldHxStandard3D(
        Hx=hx,
        Ey=ey,
        Ez=ez,
        HxDecay=decay,
        HxCurl=curl,
        invDy=inv_dy,
        invDz=inv_dz,
    ).launchRaw(blockSize=(256, 1, 1), gridSize=(1, 1, 1))
    torch.cuda.synchronize()

    torch.testing.assert_close(hx, expected, rtol=2.0e-6, atol=2.0e-7)
    # The base tensor outside the view must be untouched by the write-back.
    torch.testing.assert_close(hx_base[:, 6, :], hx_base[:, 6, :].clone())
