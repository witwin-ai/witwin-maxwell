import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd import calculate_required_steps
from witwin.maxwell.fdtd.boundary import expand_cpml_memory_tensor
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_profiles_and_boundary_attenuation():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
            name="src",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ez",)))
    scene.add_monitor(mw.PointMonitor("inner_x", (0.24, 0.0, 0.0), fields=("Ez",)))
    scene.add_monitor(mw.PointMonitor("pml_mid_x", (0.72, 0.0, 0.0), fields=("Ez",)))
    scene.add_monitor(mw.PointMonitor("pml_outer_x", (0.88, 0.0, 0.0), fields=("Ez",)))

    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=4, transient_cycles=15),
        spectral_sampler=mw.SpectralSampler(window="hanning"),
        full_field_dft=False,
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
    prepared.simulation.config.run_time = mw.TimeConfig(time_steps=steps)
    result = prepared.run()
    observers = result.monitors
    center_mag = abs(observers["center"]["data"])
    inner_mag = abs(observers["inner_x"]["data"])
    pml_mid_mag = abs(observers["pml_mid_x"]["data"])
    pml_outer_mag = abs(observers["pml_outer_x"]["data"])

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
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
            name="src",
        )
    )

    cpml_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], absorber="cpml").prepare().solver
    pml_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], absorber="pml").prepare().solver

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
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
            name="src",
        )
    )

    default_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], absorber="cpml").prepare().solver
    tuned_solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        absorber="cpml",
        cpml_config={"kappa_max": 8.0, "alpha_max": 0.10},
    ).prepare().solver

    assert float(tuned_solver.cpml_kappa_e_x.max()) > float(default_solver.cpml_kappa_e_x.max())
    assert not torch.allclose(tuned_solver.cpml_b_e_x, default_solver.cpml_b_e_x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_psi_is_slab_allocated_but_checkpoints_expand_dense():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=100.0),
            name="src",
        )
    )

    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        absorber="cpml",
        cpml_config={"memory_mode": "slab"},
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_slab_updates_match_dense_kernels_for_one_step():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.64, 0.64), (-0.64, 0.64), (-0.64, 0.64))),
        grid=mw.GridSpec.uniform(0.08),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=1.0),
            name="src",
        )
    )

    solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        absorber="cpml",
        cpml_config={"memory_mode": "slab"},
    ).prepare().solver
    module = solver.fdtd_module

    torch.manual_seed(0)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        getattr(solver, name).copy_(torch.randn_like(getattr(solver, name)))
    for name in solver._cpml_memory_layouts:
        getattr(solver, name).copy_(torch.randn_like(getattr(solver, name)))

    initial_state = {
        name: getattr(solver, name).clone()
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", *solver._cpml_memory_layouts.keys())
    }
    dense_psi = {
        name: expand_cpml_memory_tensor(solver, name).clone()
        for name in solver._cpml_memory_layouts
    }

    ref_ex = initial_state["Ex"].clone()
    ref_ey = initial_state["Ey"].clone()
    ref_ez = initial_state["Ez"].clone()
    ref_hx = initial_state["Hx"].clone()
    ref_hy = initial_state["Hy"].clone()
    ref_hz = initial_state["Hz"].clone()

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

    solver._update_magnetic_fields_cpml(solver.Hx, solver.Hy, solver.Hz, solver.Ex, solver.Ey, solver.Ez)
    solver._update_electric_fields_cpml(solver.Ex, solver.Ey, solver.Ez, solver.Hx, solver.Hy, solver.Hz)

    for name, reference in (("Ex", ref_ex), ("Ey", ref_ey), ("Ez", ref_ez), ("Hx", ref_hx), ("Hy", ref_hy), ("Hz", ref_hz)):
        assert torch.allclose(getattr(solver, name), reference, atol=1.0e-2, rtol=1.0e-2)
    for name, reference in dense_psi.items():
        slab_dense = expand_cpml_memory_tensor(solver, name)
        assert torch.allclose(slab_dense, reference, atol=1.0e-2, rtol=1.0e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_fdtd_cpml_auto_prefers_dense_small_grids_and_dense_mode_is_full_shape():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.48, 0.48))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=2, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=mw.CW(frequency=1e9, amplitude=1.0),
            name="src",
        )
    )

    auto_solver = mw.Simulation.fdtd(scene, frequencies=[1e9], absorber="cpml").prepare().solver
    dense_solver = mw.Simulation.fdtd(
        scene,
        frequencies=[1e9],
        absorber="cpml",
        cpml_config={"memory_mode": "dense"},
    ).prepare().solver

    assert auto_solver._cpml_memory_mode == "dense"
    assert dense_solver._cpml_memory_mode == "dense"
    assert dense_solver._cpml_memory_layouts == {}
    assert dense_solver.psi_ex_y.shape == dense_solver.Ex.shape
    assert dense_solver.psi_ex_z.shape == dense_solver.Ex.shape
    assert dense_solver.psi_hx_y.shape == dense_solver.Hx.shape
