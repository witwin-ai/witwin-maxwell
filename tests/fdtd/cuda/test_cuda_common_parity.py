from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FDTD backend tests.")


def test_debug_linear_indices_match_contiguous_row_major_layout():
    from witwin.maxwell.fdtd.cuda import backend

    shape = (3, 4, 5)
    linear, i_index, j_index, k_index = backend.debug_linear_indices(shape, device="cuda")
    expected = torch.arange(3 * 4 * 5, device="cuda", dtype=torch.int64).reshape(shape)

    torch.cuda.synchronize()

    torch.testing.assert_close(linear, expected)
    torch.testing.assert_close(i_index, expected // (shape[1] * shape[2]))
    torch.testing.assert_close(j_index, (expected // shape[2]) % shape[1])
    torch.testing.assert_close(k_index, expected % shape[2])


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
def test_extension_debug_linear_indices_match_contiguous_row_major_layout():
    from witwin.maxwell.fdtd.cuda import backend

    shape = (3, 4, 5)
    linear, i_index, j_index, k_index = backend.debug_linear_indices(shape, device="cuda", use_extension=True)
    expected = torch.arange(3 * 4 * 5, device="cuda", dtype=torch.int64).reshape(shape)

    torch.cuda.synchronize()

    torch.testing.assert_close(linear, expected)
    torch.testing.assert_close(i_index, expected // (shape[1] * shape[2]))
    torch.testing.assert_close(j_index, (expected // shape[2]) % shape[1])
    torch.testing.assert_close(k_index, expected % shape[2])


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("side", [0, 1])
def test_extension_clamp_field_face_matches_torch_select(axis, side):
    from witwin.maxwell.fdtd.cuda import backend

    field = torch.arange(4 * 5 * 6, device="cuda", dtype=torch.float32).reshape(4, 5, 6)
    expected = field.clone()
    expected.select(axis, 0 if side == 0 else expected.shape[axis] - 1).zero_()

    backend.get_compiled_extension().clamp_field_face(field, axis, side)
    torch.cuda.synchronize()

    torch.testing.assert_close(field, expected)


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
def test_runtime_clamp_field_face_uses_native_module(monkeypatch):
    from types import SimpleNamespace

    from witwin.maxwell.fdtd.cuda import backend
    from witwin.maxwell.fdtd.runtime.stepping import clamp_field_face

    field = torch.ones((4, 5, 6), device="cuda", dtype=torch.float32)
    real_extension = backend.get_compiled_extension()
    calls = {"count": 0}

    class CountingExtension:
        def __getattr__(self, name):
            target = getattr(real_extension, name)
            if name != "clamp_field_face":
                return target

            def wrapper(*args, **kwargs):
                calls["count"] += 1
                return target(*args, **kwargs)

            return wrapper

    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", CountingExtension())
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    solver = SimpleNamespace(fdtd_module=backend.get_native_fdtd_module())

    clamp_field_face(solver, field, 2, "high")
    torch.cuda.synchronize()

    assert calls["count"] == 1
    assert torch.count_nonzero(field[:, :, -1]).item() == 0
    assert torch.count_nonzero(field[:, :, :-1]).item() == field[:, :, :-1].numel()


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
def test_runtime_pec_enforcement_merges_native_full_faces(monkeypatch):
    from types import SimpleNamespace

    from witwin.maxwell.fdtd.boundary import BOUNDARY_PEC
    from witwin.maxwell.fdtd.cuda import backend
    from witwin.maxwell.fdtd.runtime.stepping import enforce_pec_boundaries

    real_extension = backend.get_compiled_extension()
    calls = {"face": 0, "pec": 0}

    class CountingExtension:
        def __getattr__(self, name):
            target = getattr(real_extension, name)
            if name == "clamp_field_face":
                def wrapper(*args, **kwargs):
                    calls["face"] += 1
                    return target(*args, **kwargs)

                return wrapper
            if name == "clamp_pec_boundary":
                def wrapper(*args, **kwargs):
                    calls["pec"] += 1
                    return target(*args, **kwargs)

                return wrapper
            return target

    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", CountingExtension())
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    solver = SimpleNamespace(
        fdtd_module=backend.get_native_fdtd_module(),
        has_pec_faces=True,
        boundary_x_low_code=BOUNDARY_PEC,
        boundary_x_high_code=BOUNDARY_PEC,
        boundary_y_low_code=BOUNDARY_PEC,
        boundary_y_high_code=BOUNDARY_PEC,
        boundary_z_low_code=BOUNDARY_PEC,
        boundary_z_high_code=BOUNDARY_PEC,
        Ex=torch.ones((4, 5, 6), device="cuda", dtype=torch.float32),
        Ey=torch.ones((5, 4, 6), device="cuda", dtype=torch.float32),
        Ez=torch.ones((5, 5, 5), device="cuda", dtype=torch.float32),
    )

    enforce_pec_boundaries(solver)
    torch.cuda.synchronize()

    assert calls == {"face": 0, "pec": 3}
    assert torch.count_nonzero(solver.Ex[:, 1:-1, 1:-1]).item() == solver.Ex[:, 1:-1, 1:-1].numel()
    assert torch.count_nonzero(solver.Ey[1:-1, :, 1:-1]).item() == solver.Ey[1:-1, :, 1:-1].numel()
    assert torch.count_nonzero(solver.Ez[1:-1, 1:-1, :]).item() == solver.Ez[1:-1, 1:-1, :].numel()


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run the native CUDA extension.",
)
def test_runtime_native_pec_enforcement_handles_partial_faces(monkeypatch):
    from types import SimpleNamespace

    from witwin.maxwell.fdtd.boundary import BOUNDARY_NONE, BOUNDARY_PEC
    from witwin.maxwell.fdtd.cuda import backend
    from witwin.maxwell.fdtd.runtime.stepping import enforce_pec_boundaries

    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")
    shape = (4, 5, 6)
    ex = torch.arange((shape[0] - 1) * shape[1] * shape[2], device="cuda", dtype=torch.float32).reshape(
        shape[0] - 1, shape[1], shape[2]
    )
    ey = torch.arange(shape[0] * (shape[1] - 1) * shape[2], device="cuda", dtype=torch.float32).reshape(
        shape[0], shape[1] - 1, shape[2]
    )
    ez = torch.arange(shape[0] * shape[1] * (shape[2] - 1), device="cuda", dtype=torch.float32).reshape(
        shape[0], shape[1], shape[2] - 1
    )
    expected = {"Ex": ex.clone(), "Ey": ey.clone(), "Ez": ez.clone()}
    face_specs = (
        (BOUNDARY_PEC, "low", (("Ey", 0), ("Ez", 0))),
        (BOUNDARY_NONE, "high", (("Ey", 0), ("Ez", 0))),
        (BOUNDARY_PEC, "low", (("Ex", 1), ("Ez", 1))),
        (BOUNDARY_PEC, "high", (("Ex", 1), ("Ez", 1))),
        (BOUNDARY_NONE, "low", (("Ex", 2), ("Ey", 2))),
        (BOUNDARY_PEC, "high", (("Ex", 2), ("Ey", 2))),
    )
    for code, side, targets in face_specs:
        if code != BOUNDARY_PEC:
            continue
        for field_name, axis in targets:
            field = expected[field_name]
            field.select(axis, 0 if side == "low" else field.shape[axis] - 1).zero_()

    solver = SimpleNamespace(
        fdtd_module=backend.get_native_fdtd_module(),
        has_pec_faces=True,
        boundary_x_low_code=BOUNDARY_PEC,
        boundary_x_high_code=BOUNDARY_NONE,
        boundary_y_low_code=BOUNDARY_PEC,
        boundary_y_high_code=BOUNDARY_PEC,
        boundary_z_low_code=BOUNDARY_NONE,
        boundary_z_high_code=BOUNDARY_PEC,
        Ex=ex,
        Ey=ey,
        Ez=ez,
    )

    enforce_pec_boundaries(solver)
    torch.cuda.synchronize()

    torch.testing.assert_close(solver.Ex, expected["Ex"])
    torch.testing.assert_close(solver.Ey, expected["Ey"])
    torch.testing.assert_close(solver.Ez, expected["Ez"])
