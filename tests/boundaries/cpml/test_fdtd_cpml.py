from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.solver import calculate_required_steps
from witwin.maxwell.fdtd.boundary.cpml import _resolve_cpml_memory_mode
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("transverse,k_parallel", [("periodic", 0.), ("bloch", 0.), ("bloch", 10.), ("bloch", 25.)])
def test_cpml_fixed_k_shift_and_explicit_alpha_profiles(transverse, k_parallel):
    from witwin.maxwell.fdtd.boundary.cpml import _CPML_VECTOR_ATTRS

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(.01), boundary=mw.BoundarySpec.faces(x=transverse, y=transverse, z="pml",
            num_layers=2, bloch_wavevector=(0., k_parallel, 0.)),
    )

    def prepare(**config):
        return mw.Simulation.fdtd(scene, frequencies=(.8e9, 1.2e9), config=mw.FDTDConfig(cpml=mw.fdtd.CPMLConfig(**config), absorber="cpml",
            run_time=mw.TimeConfig(time_steps=1))).prepare().solver

    default = prepare()
    alpha = default.eps0 * (2. * np.pi * .8e9) if k_parallel else 0.
    grade = 2. if k_parallel else 3.
    q = default.c * k_parallel / (2. * np.pi * .8e9)
    cosine = np.sqrt(1. - min(q, np.sqrt(1. - (4.2 / 8.) ** 2)) ** 2)
    reflection = 10. ** (-4.2 / cosine) if k_parallel else 1e-6
    explicit = prepare(alpha_max=alpha, grading_order=grade, reflection=reflection)
    assert default.cpml_grading.alpha_max == alpha
    assert default.cpml_grading.grading_order == grade
    assert default.cpml_grading.reflection == reflection
    for name in _CPML_VECTOR_ATTRS:
        assert torch.equal(getattr(default, name), getattr(explicit, name)), name
    unshifted = prepare(alpha_max=0., grading_order=grade)
    assert unshifted.cpml_grading.alpha_max == 0.
    if k_parallel:
        assert not torch.equal(default.cpml_b_e_z, unshifted.cpml_b_e_z)
        # Here twice the light line exceeds the band minimum, so alpha is constant.
        # Check the physical DC stretch on both actual E/H sample clocks.
        eta = (default.mu0 / default.eps0) ** .5
        sigma_peak = -3 * np.log(reflection) / (2 * eta * .02) * 2
        for letter, index, depth in (("e", 1, .5), ("h", 0, .75)):
            direct = getattr(default, f"cpml_inv_kappa_{letter}_z")[index].double()
            b = getattr(default, f"cpml_b_{letter}_z")[index].double()
            c = getattr(default, f"cpml_c_{letter}_z")[index].double()
            expected = direct.new_tensor(alpha / (sigma_peak * depth ** 2 + alpha))
            torch.testing.assert_close(direct + c / (1 - b), expected, rtol=1e-5, atol=3e-7)
        custom = prepare(alpha_max=.037)
        assert custom.cpml_grading.alpha_max == .037
        assert custom.cpml_grading.grading_order == 3.
        assert custom.cpml_grading.reflection == 1e-6
        assert not torch.equal(default.cpml_b_e_z, custom.cpml_b_e_z)
        custom_grade = prepare(grading_order=4.)
        assert custom_grade.cpml_grading.alpha_max == alpha
        assert custom_grade.cpml_grading.grading_order == 4.
        custom_reflection = prepare(reflection=1e-5)
        assert custom_reflection.cpml_grading.reflection == 1e-5
    else:
        for name in _CPML_VECTOR_ATTRS:
            assert torch.equal(getattr(default, name), getattr(unshifted, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_profiles_and_boundary_attenuation():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=100.0),
                name="src",
            ),
            ),
                monitors=(
            mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ez",)),
            mw.PointMonitor("inner_x", (0.24, 0.0, 0.0), fields=("Ez",)),
            mw.PointMonitor("pml_mid_x", (0.72, 0.0, 0.0), fields=("Ez",)),
            mw.PointMonitor("pml_outer_x", (0.88, 0.0, 0.0), fields=("Ez",)),
            ),
            )

    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
            spectral_sampler=mw.fdtd.SpectralSampler(window="hanning"),
            full_field_dft=False,
        ),
    )
    prepared = sim.prepare()
    fdtd = prepared.solver
    compiled_scene = fdtd.scene

    assert fdtd.cpml_kappa_e_x.shape == (compiled_scene.Nx,)
    assert fdtd.cpml_kappa_e_y.shape == (compiled_scene.Ny,)
    assert fdtd.cpml_kappa_e_z.shape == (compiled_scene.Nz,)
    assert fdtd.cpml_kappa_h_x.shape == (compiled_scene.Nx - 1,)
    assert fdtd.cpml_kappa_h_y.shape == (compiled_scene.Ny - 1,)
    assert fdtd.cpml_kappa_h_z.shape == (compiled_scene.Nz - 1,)
    assert torch.all(fdtd.cpml_kappa_e_x == 1.0)
    assert torch.all(fdtd.cpml_kappa_h_x == 1.0)
    assert torch.any(fdtd.cpml_c_e_x != 0.0)
    assert torch.any(fdtd.cpml_c_h_x != 0.0)

    steps = calculate_required_steps(
        frequency=1e9,
        dt=fdtd.dt,
        c=fdtd.c,
        num_cycles=4,
        transient_cycles=15,
        domain_size=1.28,
    )
    prepared.simulation.config = replace(prepared.simulation.config, run_time=mw.TimeConfig(time_steps=steps))
    result = prepared.run()
    def magnitude(name):
        return abs(result.monitor(name, freq_index=0).isel(component=0).values)

    center_mag = magnitude("center")
    inner_mag = magnitude("inner_x")
    pml_mid_mag = magnitude("pml_mid_x")
    pml_outer_mag = magnitude("pml_outer_x")

    assert center_mag > 0.0
    assert inner_mag < center_mag
    assert pml_mid_mag < inner_mag
    assert pml_outer_mag < pml_mid_mag


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_simple_pml_and_cpml_both_run():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=100.0),
                name="src",
            ),
            ),
            )

    cpml_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], config=mw.FDTDConfig(absorber="cpml")).prepare().solver
    pml_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], config=mw.FDTDConfig(absorber="pml")).prepare().solver

    assert cpml_solver.cex_decay.shape == pml_solver.cex_decay.shape
    assert torch.all(cpml_solver.cpml_kappa_e_x >= 1.0)
    assert pml_solver.cpml_kappa_e_x is None
    assert pml_solver.psi_ex_y is None
    assert pml_solver.sigma_x is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_custom_cpml_config_changes_profile():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=100.0),
                name="src",
            ),
            ),
            )

    default_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], config=mw.FDTDConfig(absorber="cpml")).prepare().solver
    tuned_solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        config=mw.FDTDConfig(
            absorber="cpml",
            cpml=mw.fdtd.CPMLConfig(kappa_max=8.0, alpha_max=0.10),
        ),
    ).prepare().solver

    assert float(tuned_solver.cpml_kappa_e_x.max()) > float(default_solver.cpml_kappa_e_x.max())
    assert not torch.allclose(tuned_solver.cpml_b_e_x, default_solver.cpml_b_e_x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_psi_is_slab_allocated_but_checkpoints_expand_dense():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
                grid=mw.GridSpec.uniform(0.08),
                boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
                sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=100.0),
                name="src",
            ),
            ),
            )

    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        config=mw.FDTDConfig(
            absorber="cpml",
            cpml=mw.fdtd.CPMLConfig(memory_mode="slab"),
        ),
    ).prepare().solver
    compiled_scene = solver.scene
    thickness = int(scene.boundary.num_layers)

    assert solver.psi_ex_y.shape == (compiled_scene.Nx - 1, 2 * thickness, compiled_scene.Nz)
    assert solver.psi_ex_z.shape == (compiled_scene.Nx - 1, compiled_scene.Ny, 2 * thickness)
    assert solver.psi_hx_y.shape == (compiled_scene.Nx, 2 * thickness, compiled_scene.Nz - 1)
    assert solver.psi_ex_y.numel() < solver.Ex.numel()
    assert solver.psi_hx_y.numel() < solver.Hx.numel()

    checkpoint = capture_checkpoint_state(solver, step=0)

    assert checkpoint.tensors["psi_ex_y"].shape == solver.Ex.shape
    assert checkpoint.tensors["psi_ex_z"].shape == solver.Ex.shape
    assert checkpoint.tensors["psi_hx_y"].shape == solver.Hx.shape
    assert torch.count_nonzero(checkpoint.tensors["psi_ex_y"]).item() == 0


_MIB = 1024 * 1024


def _force_one_mib_free(monkeypatch):
    """Make the card look nearly full; the auto CPML decision must not notice."""
    calls = []

    def nearly_full(device=None):
        calls.append(device)
        return _MIB, 48 * 1024 * _MIB

    monkeypatch.setattr(torch.cuda, "mem_get_info", nearly_full)
    return calls


@pytest.mark.parametrize(
    ("dense_bytes", "expected"),
    [(2 * _MIB, "dense"), (64 * _MIB, "dense"), (64 * _MIB + 1, "slab")],
)
def test_cpml_auto_memory_mode_depends_only_on_dense_bytes(monkeypatch, dense_bytes, expected):
    calls = _force_one_mib_free(monkeypatch)
    solver = SimpleNamespace(cpml=mw.fdtd.CPMLConfig(), device=torch.device("cuda:0"))

    assert _resolve_cpml_memory_mode(solver, dense_bytes) == expected
    assert solver._cpml_memory_mode_requested == "auto"
    assert solver._cpml_dense_memory_limit_bytes == 64 * _MIB
    assert calls == []


def _cubic_cpml_scene(half_width):
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-half_width, half_width),) * 3),
                grid=mw.GridSpec.uniform(0.12),
                boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=1.0),
                name="src",
            ),
            ),
            )
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize(("half_width", "expected"), [(0.48, "dense"), (7.2, "slab")])
def test_fdtd_cpml_auto_ignores_free_memory(monkeypatch, half_width, expected):
    _force_one_mib_free(monkeypatch)
    solver = mw.Simulation.fdtd(
        _cubic_cpml_scene(half_width), frequencies=[1e9], config=mw.FDTDConfig(absorber="cpml")
    ).prepare().solver

    assert (solver._cpml_dense_memory_bytes <= 64 * _MIB) == (expected == "dense")
    assert solver._cpml_memory_mode == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_auto_prefers_dense_small_grids_and_dense_mode_is_full_shape():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))),
                grid=mw.GridSpec.uniform(0.12),
                boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
                sources=(
            mw.PointDipole(
                position=(0.0, 0.0, 0.0),
                polarization="Ez",
                width=0.05,
                source_time=mw.CW(frequency=1e9, amplitude=1.0),
                name="src",
            ),
            ),
            )

    auto_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], config=mw.FDTDConfig(absorber="cpml")).prepare().solver
    dense_solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        config=mw.FDTDConfig(
            absorber="cpml",
            cpml=mw.fdtd.CPMLConfig(memory_mode="dense"),
        ),
    ).prepare().solver

    assert auto_solver._cpml_memory_mode == "dense"
    assert dense_solver._cpml_memory_mode == "dense"
    assert dense_solver._cpml_memory_layouts == {}
    assert dense_solver.psi_ex_y.shape == dense_solver.Ex.shape
    assert dense_solver.psi_ex_z.shape == dense_solver.Ex.shape
    assert dense_solver.psi_hx_y.shape == dense_solver.Hx.shape
