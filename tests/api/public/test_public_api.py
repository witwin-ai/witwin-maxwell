import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.evaluation import compute_steps
from witwin.maxwell.fdtd import FDTD
from witwin.maxwell.result import Result


def test_top_level_exports_are_available():
    assert mw.BoundaryKind is not None
    assert mw.ClosedSurfaceMonitor is not None
    assert mw.DebyePole is not None
    assert mw.DiagonalTensor3 is not None
    assert mw.DrudePole is not None
    assert mw.FinitePlaneMonitor is not None
    assert mw.LorentzPole is not None
    assert mw.FluxMonitor is not None
    assert mw.MaterialRegion is not None
    assert mw.ModeMonitor is not None
    assert mw.ModePort is not None
    assert mw.Scene is not None
    assert mw.SceneModule is not None
    assert mw.SimulationMethod.FDTD.value == "fdtd"
    assert mw.SpectralWindowKind.NONE.value == "none"
    assert mw.AbsorberKind.CPML.value == "cpml"
    assert mw.FDFDConfig().solver.solver_type == "gmres"
    assert isinstance(mw.FDTDConfig().spectral_sampler, mw.SpectralSampler)
    assert mw.GMRES().solver_type == "gmres"
    assert mw.TimeConfig.auto().time_steps is None
    assert mw.SpectralSampler(window="none").window == mw.SpectralWindowKind.NONE
    assert mw.CW(frequency=1e9).kind == "cw"
    assert mw.PlaneWave(source_time=mw.CW(frequency=1e9)).kind == "plane_wave"
    assert mw.GaussianBeam(source_time=mw.CW(frequency=1e9)).kind == "gaussian_beam"
    assert mw.ModeMonitor("port", size=(0.0, 0.5, 0.5), polarization="Ez", frequencies=(1e9,)).kind == "mode"
    assert mw.ModePort("port", size=(0.0, 0.5, 0.5), polarization="Ez", source_time=mw.CW(frequency=1e9)).kind == "mode_port"
    assert mw.ModeSource(size=(0.0, 0.5, 0.5), polarization="Ez", source_time=mw.CW(frequency=1e9)).kind == "mode_source"
    assert mw.TFSF(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))).kind == "tfsf"
    assert mw.Material is not None
    assert mw.Material.drude(plasma_frequency=2e15, gamma=5e13).is_dispersive
    assert not hasattr(mw, "FDFD")
    assert not hasattr(mw, "FDTD")
    assert not hasattr(mw, "FieldMonitor")
    assert not hasattr(mw, "GaussianCurrentSource")
    assert not hasattr(mw, "DFT")
    assert not hasattr(mw, "Medium")
    assert not hasattr(mw, "RunTime")


def test_geometry_with_material_builds_core_structure():
    structure = mw.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)).with_material(
        mw.Material(eps_r=3.0, sigma_e=0.25),
        name="target",
        priority=7,
    )

    assert structure.name == "target"
    assert structure.priority == 7
    assert structure.material.eps_r == 3.0
    assert structure.material.sigma_e == 0.25


def test_conductive_material_relative_permittivity_uses_sigma_e():
    material = mw.Material(eps_r=4.0, sigma_e=0.5)
    epsilon_r = material.relative_permittivity(1.0e9)

    assert np.isclose(epsilon_r.real, 4.0)
    assert epsilon_r.imag < 0.0


def test_dispersive_poles_expose_frequency_helper():
    debye = mw.DebyePole(delta_eps=2.0, tau=5e-10)
    drude = mw.DrudePole(plasma_frequency=2e15, gamma=5e13)
    lorentz = mw.LorentzPole(delta_eps=1.5, resonance_frequency=4e9, gamma=0.2e9)
    frequency = 2.5e9

    assert debye.susceptibility_at_freq(frequency) == pytest.approx(
        debye.susceptibility(2.0 * np.pi * frequency)
    )
    assert drude.susceptibility_at_freq(frequency) == pytest.approx(
        drude.susceptibility(2.0 * np.pi * frequency)
    )
    assert lorentz.susceptibility_at_freq(frequency) == pytest.approx(
        lorentz.susceptibility(2.0 * np.pi * frequency)
    )


def test_material_relative_permeability_and_nonlinear_flags():
    pole = mw.LorentzPole(delta_eps=1.5, resonance_frequency=3.0e9, gamma=0.2e9)
    material = mw.Material(mu_r=1.2, mu_lorentz_poles=(pole,))
    frequency = 2.5e9

    assert material.is_magnetic_dispersive is True
    assert material.relative_permeability(frequency) == pytest.approx(1.2 + pole.susceptibility_at_freq(frequency))
    assert mw.Material(kerr_chi3=1.0e-10).is_nonlinear is True


def test_scene_rejects_bool_subpixel_samples():
    with pytest.raises(TypeError, match="not bool"):
        mw.Scene(subpixel_samples=True, device="cpu")


def test_grid_spec_spacing_is_always_a_3_tuple():
    uniform = mw.GridSpec.uniform(0.25)
    anisotropic = mw.GridSpec.anisotropic(0.1, 0.2, 0.3)

    assert uniform.spacing == (0.25, 0.25, 0.25)
    assert uniform.is_uniform is True
    assert anisotropic.spacing == (0.1, 0.2, 0.3)
    assert anisotropic.is_uniform is False

    scene = mw.Scene(grid=anisotropic, device="cpu")
    assert scene.grid_spacing == (0.1, 0.2, 0.3)


def test_scene_keeps_compiled_grid_state_off_public_object():
    scene = mw.Scene(grid=mw.GridSpec.uniform(0.25), device="cpu")

    assert not hasattr(scene, "x")
    assert not hasattr(scene, "Nx")
    assert not hasattr(scene, "permittivity")
    assert not hasattr(scene, "compile_materials")


def test_scene_uses_add_mutators_without_with_aliases():
    scene = mw.Scene(device="cpu")
    structure = mw.Structure(
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
        material=mw.Material(eps_r=2.0),
    )
    source = mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", source_time=mw.CW(frequency=1e9))
    monitor = mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",))
    port = mw.ModePort("port", size=(0.0, 0.5, 0.5), polarization="Ez", frequencies=(1e9,))
    material_region = mw.MaterialRegion(
        name="design",
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.25, 0.25, 0.25)),
        density=torch.ones((1, 1, 1), dtype=torch.float32),
        eps_bounds=(1.0, 4.0),
    )

    returned = (
        scene.add_structure(structure)
        .add_source(source)
        .add_monitor(monitor)
        .add_port(port)
        .add_material_region(material_region)
    )

    assert returned is scene
    assert scene.structures == [structure]
    assert scene.sources == [source]
    assert scene.monitors == [monitor]
    assert scene.ports == [port]
    assert scene.material_regions == [material_region]
    assert not hasattr(scene, "with_structure")
    assert not hasattr(scene, "with_source")
    assert not hasattr(scene, "with_monitor")
    assert not hasattr(scene, "with_port")
    assert not hasattr(scene, "with_material_region")


def test_fdtd_evaluation_compute_steps_accepts_public_scene(monkeypatch):
    captured = {}

    def fake_required_steps(**kwargs):
        captured.update(kwargs)
        return 17

    monkeypatch.setattr("witwin.maxwell.fdtd.evaluation.calculate_required_steps", fake_required_steps)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-1.0, 1.0), (-0.25, 0.25))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    steps = compute_steps(scene, dt=0.1, c=2.0, frequency=1e9, num_cycles=3, transient_cycles=4)

    assert steps == 17
    assert captured["domain_size"] == pytest.approx(2.0)
    assert captured["dt"] == pytest.approx(0.1)
    assert captured["c"] == pytest.approx(2.0)
    assert captured["frequency"] == pytest.approx(1e9)


def test_simulation_fdfd_wraps_solver(monkeypatch):
    class FakeFDFD:
        def __init__(self, scene, frequency, solver_type, preconditioner, precision, ssor_omega, enable_plot, verbose):
            self.scene = scene
            self.frequency = frequency
            self.use_gpu = True
            self.solver_type = solver_type
            self.enable_plot = enable_plot
            self.verbose = verbose
            self.E_field = (
                torch.ones((3, 4, 4), dtype=torch.complex64),
                torch.full((4, 3, 4), 2.0 + 0j, dtype=torch.complex64),
                torch.full((4, 4, 3), 3.0 + 0j, dtype=torch.complex64),
            )
            self.converged = True
            self.solver_info = 0
            self.final_residual = 1e-7
            self.solve_calls = []

        def solve(self, max_iter, tol, restart):
            self.solve_calls.append((max_iter, tol, restart))

    monkeypatch.setattr("witwin.maxwell.simulation.FDFD", FakeFDFD, raising=False)
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda scene, method: None)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    sim = mw.Simulation.fdfd(
        scene,
        frequency=1e9,
        solver=mw.GMRES(max_iter=12, tol=1e-5, restart=7),
    )

    result = mw.run(sim)

    assert result.method == "fdfd"
    assert result.E.x.shape == (3, 4, 4)
    assert result.E.y.shape == (4, 3, 4)
    assert result.E.z.shape == (4, 4, 3)
    stats = result.stats()
    assert stats["solver"]["type"] == "gmres"
    assert stats["solver"]["max_iter"] == 12
    assert stats["converged"] is True


def test_simulation_accepts_scene_module_and_detects_trainable_parameters(monkeypatch):
    class FakeFDFD:
        def __init__(self, scene, frequency, solver_type, preconditioner, precision, ssor_omega, enable_plot, verbose):
            self.scene = scene
            self.E_field = (
                torch.ones((3, 4, 4), dtype=torch.complex64),
                torch.ones((4, 3, 4), dtype=torch.complex64),
                torch.ones((4, 4, 3), dtype=torch.complex64),
            )

        def solve(self, max_iter, tol, restart):
            return None

    class DensityScene(mw.SceneModule):
        def __init__(self):
            super().__init__()
            self.rho = torch.nn.Parameter(torch.full((2, 2, 2), 0.5, dtype=torch.float32))

        def to_scene(self):
            return mw.Scene(
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
                grid=mw.GridSpec.uniform(0.25),
                device="cpu",
            ).add_material_region(
                mw.MaterialRegion(
                    name="lens",
                    geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
                    density=self.rho,
                    eps_bounds=(1.0, 5.0),
                )
            )

    monkeypatch.setattr("witwin.maxwell.simulation.FDFD", FakeFDFD, raising=False)
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda scene, method: None)

    model = DensityScene()
    sim = mw.Simulation.fdfd(model, frequency=1e9)
    result = sim.run()

    assert sim.scene_input is model
    assert sim.scene_module is model
    assert sim.has_trainable_parameters is True
    assert isinstance(sim.scene, mw.Scene)
    assert len(sim.scene.material_regions) == 1
    assert result.materials.eps.scalar.device.type == "cpu"


def test_result_structured_field_accessors_expose_e_and_h_components():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    ex = torch.ones((3, 4, 4), dtype=torch.complex64)
    ey = torch.full((4, 3, 4), 2.0 + 0j, dtype=torch.complex64)
    ez = torch.full((4, 4, 3), 3.0 + 0j, dtype=torch.complex64)
    hx = torch.full((3, 4, 4), 4.0 + 0j, dtype=torch.complex64)
    hy = torch.full((4, 3, 4), 5.0 + 0j, dtype=torch.complex64)
    hz = torch.full((4, 4, 3), 6.0 + 0j, dtype=torch.complex64)
    result = Result(
        method="fdtd",
        scene=scene,
        frequency=1.0e9,
        fields={
            "EX": ex,
            "EY": ey,
            "EZ": ez,
            "HX": hx,
            "HY": hy,
            "HZ": hz,
        },
    )

    assert result.E.x.data_ptr() == ex.data_ptr()
    assert result.E.y.data_ptr() == ey.data_ptr()
    assert result.E.z.data_ptr() == ez.data_ptr()
    assert result.H.x.data_ptr() == hx.data_ptr()
    assert result.H.y.data_ptr() == hy.data_ptr()
    assert result.H.z.data_ptr() == hz.data_ptr()


def test_result_material_exposes_component_specific_tensors():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=mw.Material(
                eps_r=1.0,
                mu_r=1.0,
                epsilon_tensor=mw.DiagonalTensor3(2.0, 4.0, 8.0),
                mu_tensor=mw.DiagonalTensor3(1.5, 2.5, 3.5),
            ),
        )
    )
    result = Result(method="fdtd", scene=scene, frequency=1.0e9)
    center_index = (
        result.prepared_scene.Nx // 2,
        result.prepared_scene.Ny // 2,
        result.prepared_scene.Nz // 2,
    )

    assert result.materials.eps.x[center_index].item() == pytest.approx(2.0)
    assert result.materials.eps.y[center_index].item() == pytest.approx(4.0)
    assert result.materials.eps.z[center_index].item() == pytest.approx(8.0)
    assert result.materials.mu.x[center_index].item() == pytest.approx(1.5)
    assert result.materials.mu.y[center_index].item() == pytest.approx(2.5)
    assert result.materials.mu.z[center_index].item() == pytest.approx(3.5)
    assert result.materials.eps.scalar[center_index].item() == pytest.approx((2.0 + 4.0 + 8.0) / 3.0)
    assert result.materials.mu.scalar[center_index].item() == pytest.approx((1.5 + 2.5 + 3.5) / 3.0)
    assert result.material("eps_r")[center_index].item() == pytest.approx((2.0 + 4.0 + 8.0) / 3.0)


def test_simulation_fdtd_wraps_solver(monkeypatch):
    class FakeFDTD:
        def __init__(self, scene, frequency, absorber_type, cpml_config):
            self.scene = scene
            self.frequency = frequency
            self.absorber_type = absorber_type
            self.cpml_config = cpml_config
            self.c = 10.0
            self.dt = 0.25
            self.dft_sample_count = 8
            self.last_solve_elapsed_s = 0.042
            self.init_called = False

        def init_field(self):
            self.init_called = True

        def solve(self, time_steps, dft_frequency, enable_plot, dft_window, full_field_dft, normalize_source=False):
            self.ex = torch.ones((3, 4, 4), dtype=torch.complex64)
            self.ey = torch.ones((4, 3, 4), dtype=torch.complex64) * 2.0
            self.ez = torch.ones((4, 4, 3), dtype=torch.complex64) * 3.0
            self.solve_args = {
                "time_steps": time_steps,
                "dft_frequency": dft_frequency,
                "enable_plot": enable_plot,
                "dft_window": dft_window,
                "full_field_dft": full_field_dft,
            }
            return {"observers": {"center": {"data": np.complex64(1.0 + 2.0j)}}}

    def fake_required_steps(**kwargs):
        assert kwargs["num_cycles"] == 6
        assert kwargs["transient_cycles"] == 4
        return 21

    monkeypatch.setattr("witwin.maxwell.simulation.FDTD", FakeFDTD, raising=False)
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda scene, method: None)
    monkeypatch.setattr(
        "witwin.maxwell.simulation.calculate_required_steps",
        fake_required_steps,
        raising=False,
    )

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    sim = mw.Simulation.fdtd(
        scene,
        frequency=1e9,
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=4),
        spectral_sampler=mw.SpectralSampler(window="none"),
        absorber="cpml",
    )

    result = sim.run()

    assert result.method == "fdtd"
    assert result.E.x.dtype == torch.complex64
    assert result.E.x.data_ptr() == result.solver.ex.data_ptr()
    assert result.at().monitor("center")["data"] == np.complex64(1.0 + 2.0j)
    stats = result.stats()
    assert stats["time_steps"] == 21
    assert stats["dft_window"] == "none"
    assert stats["full_field_dft"] is False
    assert stats["elapsed_s"] == 0.042
    assert stats["ms_per_step"] == 2.0
    assert stats["steps_per_second"] == pytest.approx(500.0)
    assert result.solver.solve_args["full_field_dft"] is False


def test_simulation_fdtd_multi_frequency_wraps_solver(monkeypatch):
    class FakeFDTD:
        def __init__(self, scene, frequency, absorber_type, cpml_config):
            self.scene = scene
            self.frequency = frequency
            self.absorber_type = absorber_type
            self.cpml_config = cpml_config
            self.c = 10.0
            self.dt = 0.25
            self.dft_sample_count = 6
            self.dft_sample_counts = (6, 5)
            self.last_solve_elapsed_s = 0.084

        def init_field(self):
            pass

        def solve(self, time_steps, dft_frequency, enable_plot, dft_window, full_field_dft, normalize_source=False):
            self.ex = torch.stack(
                (
                    torch.ones((3, 4, 4), dtype=torch.complex64),
                    torch.ones((3, 4, 4), dtype=torch.complex64) * 2.0,
                ),
                dim=0,
            )
            self.ey = torch.stack(
                (
                    torch.ones((4, 3, 4), dtype=torch.complex64) * 3.0,
                    torch.ones((4, 3, 4), dtype=torch.complex64) * 4.0,
                ),
                dim=0,
            )
            self.ez = torch.stack(
                (
                    torch.ones((4, 4, 3), dtype=torch.complex64) * 5.0,
                    torch.ones((4, 4, 3), dtype=torch.complex64) * 6.0,
                ),
                dim=0,
            )
            self.solve_args = {
                "time_steps": time_steps,
                "dft_frequency": dft_frequency,
                "enable_plot": enable_plot,
                "dft_window": dft_window,
                "full_field_dft": full_field_dft,
            }
            return {
                "Ex": self.ex,
                "Ey": self.ey,
                "Ez": self.ez,
                "observers": {
                    "center": {
                        "kind": "point",
                        "fields": ("Ez",),
                        "components": {"Ez": np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)},
                        "field_indices": {"Ez": (1, 1, 1)},
                        "samples": np.array([6, 5]),
                        "frequency": 0.9e9,
                        "frequencies": (0.9e9, 1.1e9),
                        "position": (0.0, 0.0, 0.0),
                        "component": "ez",
                        "data": np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
                    }
                },
            }

    monkeypatch.setattr("witwin.maxwell.simulation.FDTD", FakeFDTD, raising=False)
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_scene", lambda scene, method: None)
    monkeypatch.setattr(
        "witwin.maxwell.simulation.calculate_required_steps",
        lambda **kwargs: 24,
        raising=False,
    )

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[0.9e9, 1.1e9],
        run_time=mw.TimeConfig.auto(steady_cycles=6, transient_cycles=4),
        spectral_sampler=mw.SpectralSampler(window="none"),
        absorber="cpml",
    )

    result = sim.run()

    assert result.frequencies == (0.9e9, 1.1e9)
    assert result.E.x.shape == (2, 3, 4, 4)
    assert result.E.x.data_ptr() == result.solver.ex.data_ptr()
    assert result.at(frequency=1.1e9).E.x.shape == (3, 4, 4)
    np.testing.assert_allclose(result.at(freq_index=0).E.y.cpu().numpy(), 3.0)
    assert result.at().monitor("center")["data"].shape == (2,)
    assert result.at(frequency=1.1e9).monitor("center")["data"] == np.complex64(3.0 + 4.0j)
    stats = result.stats()
    assert stats["num_frequencies"] == 2
    assert stats["dft_sample_counts"] == (6, 5)
    assert stats["steps_per_second"] == pytest.approx(24 / 0.084)
    assert result.solver.solve_args["dft_frequency"] == (0.9e9, 1.1e9)
    assert result.solver.solve_args["full_field_dft"] is True


def test_flux_monitor_exposes_required_tangential_fields():
    monitor = mw.FluxMonitor("port", axis="x", position=0.0, frequencies=[0.8e9, 1.0e9], normal_direction="-")

    assert monitor.fields == ("Ey", "Ez", "Hy", "Hz")
    assert monitor.frequencies == (0.8e9, 1.0e9)
    assert monitor.compute_flux is True
    assert monitor.normal_direction == "-"

    plane = mw.PlaneMonitor(
        "plane",
        axis="z",
        position=0.0,
        fields=("Ex", "Ey", "Hx", "Hy"),
        compute_flux=True,
    )
    assert plane.compute_flux is True


def test_mode_monitor_and_mode_port_resolve_first_class_modal_specs():
    monitor = mw.ModeMonitor(
        "port",
        position=(0.0, 0.0, 0.2),
        size=(0.6, 0.0, 0.6),
        polarization="Ez",
        direction="-",
        frequencies=[0.8e9, 1.0e9],
    )
    port = mw.ModePort(
        "input",
        position=(-0.2, 0.0, 0.0),
        size=(0.0, 0.6, 0.6),
        polarization="Ez",
        direction="+",
        frequencies=[1.0e9],
        source_time=mw.CW(frequency=1.0e9),
        monitor_offset=0.16,
    )

    assert monitor.axis == "y"
    assert monitor.position == pytest.approx((0.0, 0.0, 0.2))
    assert monitor.plane_position == pytest.approx(0.0)
    assert monitor.fields == ("Ex", "Ez", "Hx", "Hz")
    assert monitor.normal_direction == "-"

    scene = mw.Scene(device="cpu").add_port(port)
    resolved_source = scene.resolved_sources()[0]
    resolved_monitor = scene.resolved_monitors()[0]

    assert resolved_source.kind == "mode_source"
    assert resolved_source.name == "input::source"
    assert resolved_monitor.kind == "mode"
    assert resolved_monitor.name == "input"
    assert resolved_monitor.position == pytest.approx((-0.04, 0.0, 0.0))
    assert resolved_monitor.plane_position == pytest.approx(-0.04)


def test_finite_plane_and_closed_surface_monitors_resolve_first_class_huygens_workflow():
    finite = mw.FinitePlaneMonitor(
        "face",
        position=(0.15, 0.0, 0.0),
        size=(0.0, 0.4, 0.6),
        fields=("Ey", "Ez", "Hy", "Hz"),
        frequencies=(1.0e9,),
        compute_flux=True,
        normal_direction="+",
    )
    assert finite.kind == "finite_plane"
    assert finite.axis == "x"
    assert finite.plane_position == pytest.approx(0.15)
    assert finite.tangential_bounds == pytest.approx({"y": (-0.2, 0.2), "z": (-0.3, 0.3)})

    surface = mw.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.6, 0.8),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(device="cpu").add_monitor(surface)
    resolved = scene.resolved_monitors()

    assert surface.kind == "closed_surface"
    assert surface.bounds[0] == pytest.approx((-0.2, 0.2))
    assert surface.bounds[1] == pytest.approx((-0.3, 0.3))
    assert surface.bounds[2] == pytest.approx((-0.4, 0.4))
    assert len(resolved) == 6
    assert all(isinstance(monitor, mw.FinitePlaneMonitor) for monitor in resolved)
    assert {monitor.name for monitor in resolved} == {
        "huygens::x_neg",
        "huygens::x_pos",
        "huygens::y_neg",
        "huygens::y_pos",
        "huygens::z_neg",
        "huygens::z_pos",
    }


def test_result_reassembles_closed_surface_monitor_payloads_from_resolved_faces():
    surface = mw.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(device="cpu").add_monitor(surface)

    coords = np.linspace(-0.3, 0.3, 7)
    monitors = {}
    for face in surface.faces:
        coord_names = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[face.axis]
        payload = {
            "kind": "plane",
            "fields": face.fields,
            "components": {},
            "samples": 8,
            "frequency": 1.0e9,
            "frequencies": (1.0e9,),
            "axis": face.axis,
            "position": face.plane_position,
            "compute_flux": False,
            "normal_direction": face.normal_direction,
        }
        for component in face.fields:
            data = np.ones((coords.size, coords.size), dtype=np.complex64)
            payload["components"][component] = {
                "data": data,
                "coords": (coords, coords),
                "plane_index": 0,
                "plane_indices": (0,),
                "plane_weights": (1.0,),
                "plane_positions": (face.plane_position,),
            }
            payload[component] = data
        payload[coord_names[0]] = coords
        payload[coord_names[1]] = coords
        payload["coords"] = (coords, coords)
        monitors[face.name] = payload

    result = Result(
        method="fdtd",
        scene=scene,
        frequency=1.0e9,
        monitors=monitors,
    )

    face_payload = result.monitor("huygens::z_pos")
    np.testing.assert_allclose(face_payload["x"], np.linspace(-0.2, 0.2, 5))
    np.testing.assert_allclose(face_payload["y"], np.linspace(-0.2, 0.2, 5))
    assert face_payload["monitor_type"] == "finite_plane"
    assert face_payload["surface_name"] == "huygens"

    surface_payload = result.monitor("huygens")
    assert surface_payload["kind"] == "closed_surface"
    assert set(surface_payload["faces"]) == {"x_neg", "x_pos", "y_neg", "y_pos", "z_neg", "z_pos"}
    np.testing.assert_allclose(surface_payload["faces"]["z_pos"]["x"], np.linspace(-0.2, 0.2, 5))


def test_result_can_expand_low_face_symmetry_back_to_full_domain():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.75), (0.0, 0.5), (0.0, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        symmetry=("PMC", None, None),
        device="cpu",
    )
    result = Result(
        method="fdtd",
        scene=scene,
        frequency=1e9,
        fields={
            "EX": torch.tensor([[[1.0]], [[2.0]]]),
            "EY": torch.tensor([[[10.0]], [[20.0]], [[30.0]]]),
            "EZ": torch.tensor([[[100.0]], [[200.0]], [[300.0]]]),
        },
    )

    expanded = result.at(expand_symmetry=True)
    expanded_ex = expanded.E.x
    expanded_ey = expanded.E.y
    expanded_eps = expanded.materials.eps.scalar

    assert expanded_ex[:, 0, 0].tolist() == [-2.0, -1.0, 1.0, 2.0]
    assert expanded_ey[:, 0, 0].tolist() == [30.0, 20.0, 10.0, 20.0, 30.0]
    assert expanded_eps.shape == (5, 2, 2)


def test_fdtd_initialization_builds_dispersive_state(monkeypatch):
    class DummyModule:
        pass

    monkeypatch.setattr("witwin.maxwell.fdtd.solver._get_fdtd_module", lambda: DummyModule())
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._require_cuda_scene", lambda scene: None)

    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )
    scene.add_structure(
        mw.Structure(
            name="debye_box",
            geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
            material=mw.Material.debye(eps_inf=2.0, delta_eps=3.0, tau=1.0e-9),
        )
    )

    solver = FDTD(scene, frequency=2.0e9, absorber_type="cpml")
    solver.init_field()

    assert solver.dispersive_enabled is True
    assert len(solver._dispersive_templates["Ex"]["debye"]) == 1
    assert solver._dispersive_templates["Ex"]["debye"][0]["polarization"].shape == solver.Ex.shape
    assert solver.material_eps_r.dtype == torch.complex64


def test_result_expands_pec_symmetry_with_even_normal_component():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.75), (0.0, 0.5), (0.0, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        symmetry=("PEC", None, None),
        device="cpu",
    )
    result = Result(
        method="fdtd",
        scene=scene,
        frequency=1e9,
        fields={
            "EX": torch.tensor([[[1.0]], [[2.0]]]),
            "EZ": torch.tensor([[[10.0]], [[20.0]], [[30.0]]]),
        },
    )

    expanded_ex = result.tensor("Ex", expand_symmetry=True)
    expanded_ez = result.tensor("Ez", expand_symmetry=True)

    assert expanded_ex[:, 0, 0].tolist() == [2.0, 1.0, 1.0, 2.0]
    assert expanded_ez[:, 0, 0].tolist() == [-30.0, -20.0, 10.0, 20.0, 30.0]


def test_simulation_fdfd_rejects_non_cuda_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires scene.device to be CUDA"):
        mw.Simulation.fdfd(scene, frequency=1e9).prepare()


def test_simulation_fdtd_rejects_non_cuda_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires scene.device to be CUDA"):
        mw.Simulation.fdtd(scene, frequency=1e9).prepare()
