from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.boundary import expand_cpml_memory_tensor
from witwin.maxwell.fdtd.cuda import backend
from tests.fdtd.cuda import torch_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for native FDTD backend tests.",
)


_CPML_STATE_NAMES = (
    "Ex",
    "Ey",
    "Ez",
    "Hx",
    "Hy",
    "Hz",
    "psi_ex_y",
    "psi_ex_z",
    "psi_ey_x",
    "psi_ey_z",
    "psi_ez_x",
    "psi_ez_y",
    "psi_hx_y",
    "psi_hx_z",
    "psi_hy_x",
    "psi_hy_z",
    "psi_hz_x",
    "psi_hz_y",
)


_CPML_EXTENSION_METHODS = (
    "update_magnetic_hx_cpml",
    "update_magnetic_hy_cpml",
    "update_magnetic_hz_cpml",
    "update_electric_ex_cpml",
    "update_electric_ey_cpml",
    "update_electric_ez_cpml",
)


_CPML_COMPRESSED_EXTENSION_METHODS = (
    "update_magnetic_hx_cpml_compressed",
    "update_magnetic_hy_cpml_compressed",
    "update_magnetic_hz_cpml_compressed",
    "update_electric_ex_cpml_compressed",
    "update_electric_ey_cpml_compressed",
    "update_electric_ez_cpml_compressed",
)


class _CountingExtension:
    def __init__(self, extension):
        self._extension = extension
        self.calls = {name: 0 for name in _CPML_EXTENSION_METHODS}

    def __getattr__(self, name):
        target = getattr(self._extension, name)
        if name not in self.calls:
            return target

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return target(*args, **kwargs)

        return counted


# Strongly graded, per-axis-distinct node coordinates spanning (-0.48, 0.48)
# with the same 8-cell count as the uniform 0.12 grid.
_GRADED_NODES = {
    "x": np.array([-0.48, -0.40, -0.29, -0.14, 0.02, 0.15, 0.26, 0.38, 0.48], dtype=np.float64),
    "y": np.array([-0.48, -0.37, -0.28, -0.12, -0.01, 0.13, 0.22, 0.36, 0.48], dtype=np.float64),
    "z": np.array([-0.48, -0.41, -0.27, -0.16, 0.03, 0.12, 0.28, 0.35, 0.48], dtype=np.float64),
}


def _grid_spec(grid_kind: str) -> mw.GridSpec:
    if grid_kind == "graded":
        return mw.GridSpec.custom(_GRADED_NODES["x"], _GRADED_NODES["y"], _GRADED_NODES["z"])
    return mw.GridSpec.uniform(0.12)


def _dense_cpml_scene(grid_kind: str) -> mw.Scene:
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))),
        grid=_grid_spec(grid_kind),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1.0e9, amplitude=1.0),
            name="src",
        )
    )
    return scene


def _prepared_dense_cpml_solver(monkeypatch, grid_kind):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    return mw.Simulation.fdtd(
        _dense_cpml_scene(grid_kind),
        frequencies=[1.0e9],
        absorber="cpml",
        cpml_config={"memory_mode": "dense"},
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver


def _prepared_slab_cpml_solver(monkeypatch, grid_kind):
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_BACKEND", "cuda")
    return mw.Simulation.fdtd(
        _dense_cpml_scene(grid_kind),
        frequencies=[1.0e9],
        absorber="cpml",
        cpml_config={"memory_mode": "slab"},
        run_time=mw.TimeConfig(time_steps=1),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare().solver


def _seed_dense_cpml_state(solver):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(202406)
    for name in _CPML_STATE_NAMES:
        tensor = getattr(solver, name)
        tensor.copy_(torch.randn(tensor.shape, device="cuda", generator=generator))


def _copy_dense_cpml_state(source, target):
    for name in _CPML_STATE_NAMES:
        getattr(target, name).copy_(getattr(source, name))


def _advance_dense_cpml_one_step(solver):
    solver._update_magnetic_fields_cpml(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
    solver._update_electric_fields_cpml(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)
    torch.cuda.synchronize()


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA field kernels.",
)
@pytest.mark.parametrize("grid_kind", ["uniform", "graded"])
def test_compiled_cuda_extension_dense_cpml_one_step_matches_torch_cuda_reference(monkeypatch, grid_kind):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    reference_solver = _prepared_dense_cpml_solver(monkeypatch, grid_kind)
    extension_solver = _prepared_dense_cpml_solver(monkeypatch, grid_kind)
    assert reference_solver._cpml_memory_mode == "dense"
    assert extension_solver._cpml_memory_mode == "dense"

    _seed_dense_cpml_state(reference_solver)
    _copy_dense_cpml_state(reference_solver, extension_solver)

    # Step the reference solver through the frozen Torch implementations so
    # the comparison is independent of the compiled kernels.
    reference_solver.fdtd_module = torch_reference.get_native_fdtd_module()
    _advance_dense_cpml_one_step(reference_solver)

    real_extension = backend.get_compiled_extension()
    for method_name in _CPML_EXTENSION_METHODS:
        assert hasattr(real_extension, method_name)
    counted_extension = _CountingExtension(real_extension)
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", counted_extension)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")

    _advance_dense_cpml_one_step(extension_solver)

    assert counted_extension.calls == {name: 1 for name in _CPML_EXTENSION_METHODS}
    for name in _CPML_STATE_NAMES:
        torch.testing.assert_close(
            getattr(extension_solver, name),
            getattr(reference_solver, name),
            rtol=1.0e-5,
            atol=5.0e-6,
            msg=name,
        )


@pytest.mark.skipif(
    os.environ.get("WITWIN_RUN_CUDA_EXTENSION_BUILD") != "1",
    reason="Set WITWIN_RUN_CUDA_EXTENSION_BUILD=1 to compile and run native CUDA field kernels.",
)
@pytest.mark.parametrize("grid_kind", ["uniform", "graded"])
def test_compiled_cuda_extension_compressed_cpml_one_step_matches_dense_reference(monkeypatch, grid_kind):
    monkeypatch.delenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", raising=False)
    solver = _prepared_slab_cpml_solver(monkeypatch, grid_kind)
    assert solver._cpml_memory_mode == "slab"

    _seed_dense_cpml_state(solver)
    initial_state = {name: getattr(solver, name).clone() for name in _CPML_STATE_NAMES}
    dense_psi = {name: expand_cpml_memory_tensor(solver, name).clone() for name in solver._cpml_memory_layouts}
    ref_ex, ref_ey, ref_ez = solver.Ex.clone(), solver.Ey.clone(), solver.Ez.clone()
    ref_hx, ref_hy, ref_hz = solver.Hx.clone(), solver.Hy.clone(), solver.Hz.clone()
    module = solver.fdtd_module

    module.updateMagneticFieldHx3D(
        Hx=ref_hx,
        Ey=ref_ey,
        Ez=ref_ez,
        HxDecay=solver.chx_decay,
        HxCurl=solver.chx_curl,
        PsiHxY=dense_psi["psi_hx_y"],
        PsiHxZ=dense_psi["psi_hx_z"],
        InvKappaHxY=solver.cpml_inv_kappa_h_y,
        ByHxY=solver.cpml_b_h_y,
        CyHxY=solver.cpml_c_h_y,
        InvKappaHxZ=solver.cpml_inv_kappa_h_z,
        ByHxZ=solver.cpml_b_h_z,
        CyHxZ=solver.cpml_c_h_z,
        invDy=solver.inv_dy_h,
        invDz=solver.inv_dz_h,
    ).launchRaw()
    module.updateMagneticFieldHy3D(
        Hy=ref_hy,
        Ex=ref_ex,
        Ez=ref_ez,
        HyDecay=solver.chy_decay,
        HyCurl=solver.chy_curl,
        PsiHyX=dense_psi["psi_hy_x"],
        PsiHyZ=dense_psi["psi_hy_z"],
        InvKappaHyX=solver.cpml_inv_kappa_h_x,
        ByHyX=solver.cpml_b_h_x,
        CyHyX=solver.cpml_c_h_x,
        InvKappaHyZ=solver.cpml_inv_kappa_h_z,
        ByHyZ=solver.cpml_b_h_z,
        CyHyZ=solver.cpml_c_h_z,
        invDx=solver.inv_dx_h,
        invDz=solver.inv_dz_h,
    ).launchRaw()
    module.updateMagneticFieldHz3D(
        Hz=ref_hz,
        Ex=ref_ex,
        Ey=ref_ey,
        HzDecay=solver.chz_decay,
        HzCurl=solver.chz_curl,
        PsiHzX=dense_psi["psi_hz_x"],
        PsiHzY=dense_psi["psi_hz_y"],
        InvKappaHzX=solver.cpml_inv_kappa_h_x,
        ByHzX=solver.cpml_b_h_x,
        CyHzX=solver.cpml_c_h_x,
        InvKappaHzY=solver.cpml_inv_kappa_h_y,
        ByHzY=solver.cpml_b_h_y,
        CyHzY=solver.cpml_c_h_y,
        invDx=solver.inv_dx_h,
        invDy=solver.inv_dy_h,
    ).launchRaw()
    module.updateElectricFieldExCpml3D(
        Ex=ref_ex,
        Hy=ref_hy,
        Hz=ref_hz,
        ExDecay=solver.cex_decay,
        ExCurl=solver.cex_curl,
        PsiExY=dense_psi["psi_ex_y"],
        PsiExZ=dense_psi["psi_ex_z"],
        InvKappaExY=solver.cpml_inv_kappa_e_y,
        BExY=solver.cpml_b_e_y,
        CExY=solver.cpml_c_e_y,
        InvKappaExZ=solver.cpml_inv_kappa_e_z,
        BExZ=solver.cpml_b_e_z,
        CExZ=solver.cpml_c_e_z,
        invDy=solver.inv_dy_e,
        invDz=solver.inv_dz_e,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    module.updateElectricFieldEyCpml3D(
        Ey=ref_ey,
        Hx=ref_hx,
        Hz=ref_hz,
        EyDecay=solver.cey_decay,
        EyCurl=solver.cey_curl,
        PsiEyX=dense_psi["psi_ey_x"],
        PsiEyZ=dense_psi["psi_ey_z"],
        InvKappaEyX=solver.cpml_inv_kappa_e_x,
        BEyX=solver.cpml_b_e_x,
        CEyX=solver.cpml_c_e_x,
        InvKappaEyZ=solver.cpml_inv_kappa_e_z,
        BEyZ=solver.cpml_b_e_z,
        CEyZ=solver.cpml_c_e_z,
        invDx=solver.inv_dx_e,
        invDz=solver.inv_dz_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        zLowBoundaryMode=solver.boundary_z_low_code,
        zHighBoundaryMode=solver.boundary_z_high_code,
    ).launchRaw()
    module.updateElectricFieldEzCpml3D(
        Ez=ref_ez,
        Hx=ref_hx,
        Hy=ref_hy,
        EzDecay=solver.cez_decay,
        EzCurl=solver.cez_curl,
        PsiEzX=dense_psi["psi_ez_x"],
        PsiEzY=dense_psi["psi_ez_y"],
        InvKappaEzX=solver.cpml_inv_kappa_e_x,
        BEzX=solver.cpml_b_e_x,
        CEzX=solver.cpml_c_e_x,
        InvKappaEzY=solver.cpml_inv_kappa_e_y,
        BEzY=solver.cpml_b_e_y,
        CEzY=solver.cpml_c_e_y,
        invDx=solver.inv_dx_e,
        invDy=solver.inv_dy_e,
        xLowBoundaryMode=solver.boundary_x_low_code,
        xHighBoundaryMode=solver.boundary_x_high_code,
        yLowBoundaryMode=solver.boundary_y_low_code,
        yHighBoundaryMode=solver.boundary_y_high_code,
    ).launchRaw()

    for name, value in initial_state.items():
        getattr(solver, name).copy_(value)
    real_extension = backend.get_compiled_extension()
    for method_name in _CPML_COMPRESSED_EXTENSION_METHODS:
        assert hasattr(real_extension, method_name)
    counted_extension = _CountingExtension(real_extension)
    counted_extension.calls.update({name: 0 for name in _CPML_COMPRESSED_EXTENSION_METHODS})
    monkeypatch.setattr(backend, "_COMPILED_EXTENSION", counted_extension)
    monkeypatch.setenv("WITWIN_MAXWELL_FDTD_CUDA_USE_EXTENSION", "1")

    _advance_dense_cpml_one_step(solver)

    for name in _CPML_COMPRESSED_EXTENSION_METHODS:
        assert counted_extension.calls[name] == 1
    for name, reference in (("Ex", ref_ex), ("Ey", ref_ey), ("Ez", ref_ez), ("Hx", ref_hx), ("Hy", ref_hy), ("Hz", ref_hz)):
        torch.testing.assert_close(getattr(solver, name), reference, rtol=1.0e-5, atol=5.0e-6, msg=name)
    for name, reference in dense_psi.items():
        torch.testing.assert_close(expand_cpml_memory_tensor(solver, name), reference, rtol=1.0e-5, atol=5.0e-6, msg=name)
