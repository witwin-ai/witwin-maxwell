import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.solver import FDTD
from witwin.maxwell.result import Result
from witwin.maxwell.scene import prepare_scene

from tests._fresh_interpreter import ROOT
from tests.support.public_api import public_objects


def test_top_level_exports_are_available():
    assert mw.Scene.__module__ == "witwin.maxwell.scene"
    assert mw.BoundaryKind is not None
    assert mw.monitors.ClosedSurfaceMonitor is not None
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
    assert mw.fdtd.SpectralWindowKind.NONE.value == "none"
    assert mw.fdtd.AbsorberKind.CPML.value == "cpml"
    assert mw.FDFDConfig().solver.solver_type == "gmres"
    assert isinstance(mw.FDTDConfig().spectral_sampler, mw.fdtd.SpectralSampler)
    assert mw.fdfd.FDFDSolverConfig().solver_type == "gmres"
    assert mw.TimeConfig.auto().time_steps is None
    assert mw.fdtd.SpectralSampler(window="none").window == mw.fdtd.SpectralWindowKind.NONE
    assert mw.CW(frequency=1e9).kind == "cw"
    assert mw.PlaneWave(source_time=mw.CW(frequency=1e9)).kind == "plane_wave"
    assert mw.GaussianBeam(source_time=mw.CW(frequency=1e9)).kind == "gaussian_beam"
    assert mw.ModeMonitor("port", size=(0.0, 0.5, 0.5), polarization="Ez", frequencies=(1e9,)).kind == "mode"
    assert mw.ModePort("port", size=(0.0, 0.5, 0.5), polarization="Ez", source_time=mw.CW(frequency=1e9)).kind == "mode_port"
    assert mw.ModeSource(size=(0.0, 0.5, 0.5), polarization="Ez", source_time=mw.CW(frequency=1e9)).kind == "mode_source"
    assert mw.TFSF(bounds=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))).kind == "tfsf"
    assert mw.Material is not None
    assert mw.Material.drude(plasma_frequency=2e15, gamma=5e13).is_dispersive
    assert mw.PolySlab is not None
    assert mw.geometry.ComplexPolySlab is not None
    assert not hasattr(mw, "FDFD")
    assert not hasattr(mw, "FDTD")
    assert not hasattr(mw, "FieldMonitor")
    assert not hasattr(mw, "GaussianCurrentSource")
    assert not hasattr(mw, "DFT")
    assert not hasattr(mw, "Medium")
    assert not hasattr(mw, "RunTime")

    import witwin.core as wc

    assert not hasattr(wc, "SceneBase")


def test_each_public_object_has_one_home():
    objects = public_objects()  # asserts that no name is listed twice
    assert len(mw.__all__) == len(set(mw.__all__))
    for name in ("PortData", "CPMLConfig", "LossyMetalMedium", "fit_dispersion", "fit_rational", "Cone"):
        assert name in objects and not hasattr(mw, name)
    assert mw.rf.PortData is objects["PortData"]
    assert mw.rf.rational.fit_rational is objects["fit_rational"]
    assert mw.media.fit.fit_dispersion is objects["fit_dispersion"]
    assert mw.media.library.library_material is objects["library_material"]
    assert not hasattr(mw.fdtd, "FDTDCoreFallbackWarning")  # removed in 0.7.0 (W7 row 16c)
    with pytest.raises(AttributeError):
        mw.not_a_namespace
    with pytest.raises(AttributeError):
        mw.rf.Scene  # noqa: B018  (each object has exactly one public home)


def test_removed_public_names_are_absent():
    for name in ("run", "BJT", "MOSFET"):
        assert not hasattr(mw, name)
    assert not hasattr(mw.circuits, "BJT")
    assert not hasattr(mw.fdtd, "FDTD") and not hasattr(mw.fdtd, "calculate_required_steps")
    assert not hasattr(mw.fdfd, "FDFD")


def test_namespaces_load_on_first_access():
    code = ("import sys, witwin.maxwell as mw\n"
            "assert 'witwin.maxwell.rf' not in sys.modules\n"
            "mw.rf.PortData\n"
            "assert 'witwin.maxwell.rf' in sys.modules\n")
    process = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("config, name, value", [
    (mw.FDTDConfig(), "reference_frequency", 1e9),
    (mw.FDFDConfig(), "verbose", True),
    (mw.fdtd.SpectralSampler(), "normalize_source", True),
])
def test_solver_configs_are_frozen(config, name, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, name, value)
    assert getattr(dataclasses.replace(config, **{name: value}), name) == value
    with pytest.raises(TypeError, match="run_time"):
        dataclasses.replace(mw.FDTDConfig(), run_time=None)


def test_fdtd_config_owns_reference_frequency_and_material_cache():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.25))
    cache = mw.fdtd.FDTDMaterialCache()
    simulation = mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(reference_frequency=2e9, material_cache=cache, device="cpu"))
    assert simulation.config.reference_frequency == 2e9
    assert simulation.config.material_cache is cache
    assert not hasattr(simulation, "material_cache")
    for name, value in (("reference_frequency", 2e9), ("material_cache", cache)):
        with pytest.raises(TypeError, match=name):
            mw.Simulation(scene=scene, method="fdtd", frequencies=(1e9,), **{name: value})
    with pytest.raises(TypeError, match="FDTDMaterialCache"):
        mw.FDTDConfig(material_cache=object())


def test_factories_take_one_configuration_entry():
    import inspect

    assert list(inspect.signature(mw.Simulation.fdtd).parameters) == ["scene", "frequencies", "excitations", "config"]
    assert list(inspect.signature(mw.Simulation.fdfd).parameters) == ["scene", "frequencies", "config"]
    assert not hasattr(mw, "GMRES")
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.5, 0.5),) * 3), grid=mw.GridSpec.uniform(0.25))
    for factory, config, (option, value) in (
            (mw.Simulation.fdtd, mw.FDTDConfig(full_field_dft=True), ("run_time", mw.TimeConfig())),
            (mw.Simulation.fdfd, mw.FDFDConfig(verbose=True), ("solver", mw.fdfd.FDFDSolverConfig()))):
        simulation = factory(scene, frequencies=1e9, config=config)
        assert simulation.frequencies == (1e9,) and simulation.config is config
        assert factory(scene, frequencies=(1e9, 2e9)).frequencies == (1e9, 2e9)
        assert factory(scene, frequencies=1e9).config == type(config)()
        with pytest.raises(TypeError, match="frequency"):
            factory(scene, frequency=1e9)
        with pytest.raises(TypeError, match="frequencies"):
            factory(scene)
        with pytest.raises(TypeError, match=option):  # a config field is not a factory keyword
            factory(scene, frequencies=1e9, **{option: value})
    with pytest.raises(TypeError, match="FDTDConfig"):
        mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDFDConfig(device="cpu"))
    with pytest.raises(TypeError, match="FDFDSolverConfig"):
        mw.FDFDConfig(solver=object())


def test_geometry_with_material_builds_core_structure():
    structure = mw.Structure(
        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)),
        material=mw.Material(eps_r=3.0, sigma_e=0.25),
        name='target',
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
    assert epsilon_r.imag > 0.0


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
        mw.Scene(subpixel=True)


def test_grid_spec_spacing_is_always_a_3_tuple():
    uniform = mw.GridSpec.uniform(0.25)
    anisotropic = mw.GridSpec.anisotropic(0.1, 0.2, 0.3)

    assert uniform.spacing == (0.25, 0.25, 0.25)
    assert uniform.is_uniform is True
    assert anisotropic.spacing == (0.1, 0.2, 0.3)
    assert anisotropic.is_uniform is False

    scene = mw.Scene(grid=anisotropic)
    assert scene.grid.spacing == (0.1, 0.2, 0.3)


def test_scene_keeps_compiled_grid_state_off_public_object():
    scene = mw.Scene(grid=mw.GridSpec.uniform(0.25))

    assert not hasattr(scene, "x")
    assert not hasattr(scene, "Nx")
    assert not hasattr(scene, "permittivity")
    assert not hasattr(scene, "compile_materials")


def test_scene_is_immutable_and_updated_without_with_aliases():
    scene = mw.Scene()
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

    updated = scene.updated(structures=(structure,), sources=(source,), monitors=(monitor,), ports=(port,),
                            material_regions=(material_region,))

    assert updated is not scene and scene.structures == ()
    assert updated.structures == (structure,)
    assert updated.sources == (source,)
    assert updated.monitors == (monitor,)
    assert updated.ports == (port,)
    assert updated.material_regions == (material_region,)
    assert updated.domain is scene.domain and updated.grid is scene.grid
    for name in ("add_structure", "add_source", "add_monitor", "add_port", "add_material_region", "clone", "to_tidy3d"):
        assert not hasattr(scene, name)
    with pytest.raises(dataclasses.FrozenInstanceError):
        updated.structures = ()
    assert not hasattr(scene, "with_structure")
    assert not hasattr(scene, "with_source")
    assert not hasattr(scene, "with_monitor")
    assert not hasattr(scene, "with_port")
    assert not hasattr(scene, "with_material_region")


def test_simulation_fdfd_wraps_solver(monkeypatch):
    class FakeFDFD:
        def __init__(self, scene, frequency, solver_type, preconditioner, precision, ssor_omega, enable_plot, verbose,
                     device_memory_limit_bytes=None):
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
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_device", lambda scene, method: None)

    scene = mw.Scene(
        # Three cells per axis: the Yee shapes (3, 4, 4), (4, 3, 4), (4, 4, 3) of the stand-in arrays.
        domain=mw.Domain(bounds=((-0.375, 0.375),) * 3),
        grid=mw.GridSpec.uniform(0.25),
    )
    sim = mw.Simulation.fdfd(
        scene,
        frequencies=1e9,
        config=mw.FDFDConfig(solver=mw.fdfd.FDFDSolverConfig(max_iter=12, tol=1e-5, restart=7), device="cpu"),
    )

    result = sim.run()

    assert result.method == "fdfd"
    assert result.E.x.isel(f=0).values.shape == (3, 4, 4)
    assert result.E.y.isel(f=0).values.shape == (4, 3, 4)
    assert result.E.z.isel(f=0).values.shape == (4, 4, 3)
    stats = result.stats()
    assert stats["solver"]["type"] == "gmres"
    assert stats["solver"]["max_iter"] == 12
    assert stats["converged"] is True


def test_simulation_accepts_scene_module_and_detects_trainable_parameters(monkeypatch):
    class FakeFDFD:
        def __init__(self, scene, frequency, solver_type, preconditioner, precision, ssor_omega, enable_plot, verbose,
                     device_memory_limit_bytes=None):
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
                subpixel=mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic"),
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
                grid=mw.GridSpec.uniform(0.25),
                material_regions=(
                    mw.MaterialRegion(
                        name="lens",
                        geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
                        density=self.rho,
                        eps_bounds=(1.0, 5.0),
                    ),
                ),
            )

    monkeypatch.setattr("witwin.maxwell.simulation.FDFD", FakeFDFD, raising=False)
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_device", lambda scene, method: None)

    model = DensityScene()
    sim = mw.Simulation.fdfd(model, frequencies=1e9, config=mw.FDFDConfig(device="cpu"))
    result = sim.run()

    assert sim.scene_input is model
    assert sim.scene_module is model
    assert sim.has_trainable_parameters is True
    assert isinstance(sim.scene, mw.Scene)
    assert len(sim.scene.material_regions) == 1
    assert result.materials.eps.scalar.device.type == "cpu"


def test_result_structured_field_accessors_expose_e_and_h_components():
    scene = mw.Scene(
        # Three cells per axis: the Yee shapes (3, 4, 4), (4, 3, 4), (4, 4, 3) of the stand-in arrays.
        domain=mw.Domain(bounds=((-0.375, 0.375),) * 3),
        grid=mw.GridSpec.uniform(0.25),
    )
    ex = torch.ones((3, 4, 4), dtype=torch.complex64)
    ey = torch.full((4, 3, 4), 2.0 + 0j, dtype=torch.complex64)
    ez = torch.full((4, 4, 3), 3.0 + 0j, dtype=torch.complex64)
    hx = torch.full((4, 3, 3), 4.0 + 0j, dtype=torch.complex64)
    hy = torch.full((3, 4, 3), 5.0 + 0j, dtype=torch.complex64)
    hz = torch.full((3, 3, 4), 6.0 + 0j, dtype=torch.complex64)
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

    assert result.E.x.isel(f=0).values.data_ptr() == ex.data_ptr()
    assert result.E.y.isel(f=0).values.data_ptr() == ey.data_ptr()
    assert result.E.z.isel(f=0).values.data_ptr() == ez.data_ptr()
    assert result.H.x.isel(f=0).values.data_ptr() == hx.data_ptr()
    assert result.H.y.isel(f=0).values.data_ptr() == hy.data_ptr()
    assert result.H.z.isel(f=0).values.data_ptr() == hz.data_ptr()


def test_result_material_exposes_component_specific_tensors():
    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
                grid=mw.GridSpec.uniform(0.25),
                structures=(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
                material=mw.Material(
                    eps_r=1.0,
                    mu_r=1.0,
                    epsilon_tensor=mw.DiagonalTensor3(2.0, 4.0, 8.0),
                    mu_tensor=mw.DiagonalTensor3(1.5, 2.5, 3.5),
                ),
            ),
            ),
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
        def __init__(self, scene, frequency, absorber_type, cpml):
            self.scene = scene
            self._compiled_sources = []
            self.frequency = frequency
            self.absorber_type = absorber_type
            self.cpml = cpml
            self._compiled_sources = ()
            self.Ex = torch.zeros((3, 4, 4), dtype=torch.float32)
            self.c = 10.0
            self.dt = 0.25
            self.dft_sample_count = 8
            self.last_solve_elapsed_s = 0.042
            self.init_called = False

        def init_field(self):
            self.init_called = True

        def solve(
            self,
            time_steps,
            dft_frequency,
            enable_plot,
            dft_window,
            full_field_dft,
            normalize_source=False,
            shutoff=0.0,
            shutoff_check_interval=100,
            use_cuda_graph=False,
            resume_from=None,
            stop_step=None,
        ):
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
            return {
                "Ex": self.ex,
                "Ey": self.ey,
                "Ez": self.ez,
                "observers": {"center": {
                    "kind": "point", "fields": ("Ez",), "components": {"Ez": np.complex64(1.0 + 2.0j)},
                    "field_indices": {"Ez": (1, 1, 1)}, "samples": 6, "frequency": 1e9, "frequencies": (1e9,),
                    "position": (0.0, 0.0, 0.0), "component": "ez", "Ez": np.complex64(1.0 + 2.0j),
                }},
            }

    monkeypatch.setattr("witwin.maxwell.simulation._resolve_fdtd_backend", lambda: (FakeFDTD, None))
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_device", lambda scene, method: None)
    # The stand-in solver has no fields to bind the core launches to.
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.core_path.activate_core_path", lambda solver: None)

    scene = mw.Scene(
        # Three cells per axis: the Yee shapes (3, 4, 4), (4, 3, 4), (4, 4, 3) of the stand-in arrays.
        domain=mw.Domain(bounds=((-0.375, 0.375),) * 3),
        grid=mw.GridSpec.uniform(0.25),
    )
    sim = mw.Simulation.fdtd(
        scene,
        frequencies=1e9,
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=21, steady_cycles=6, transient_cycles=4),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
            absorber="cpml", device="cpu",
        ),
    )

    result = sim.run()

    assert result.method == "fdtd"
    # A single-frequency run without the volume DFT keeps only its last-step fields.
    assert result.time_snapshot["EX"].values.dtype == torch.complex64
    assert result.time_snapshot["EX"].values.data_ptr() == result.solver.ex.data_ptr()
    assert result.at(freq_index=0).monitor("center").sel(component="Ez").values.item() == 1.0 + 2.0j
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
        def __init__(self, scene, frequency, absorber_type, cpml):
            self.scene = scene
            self._compiled_sources = []
            self.frequency = frequency
            self.absorber_type = absorber_type
            self.cpml = cpml
            self._compiled_sources = ()
            self.Ex = torch.zeros((3, 4, 4), dtype=torch.float32)
            self.c = 10.0
            self.dt = 0.25
            self.dft_sample_count = 6
            self.dft_sample_counts = (6, 5)
            self.observer_sample_count = 6
            self.observer_sample_counts = (6, 5)
            self.last_solve_elapsed_s = 0.084

        def init_field(self):
            pass

        def solve(
            self,
            time_steps,
            dft_frequency,
            enable_plot,
            dft_window,
            full_field_dft,
            normalize_source=False,
            shutoff=0.0,
            shutoff_check_interval=100,
            use_cuda_graph=False,
            resume_from=None,
            stop_step=None,
        ):
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
                        "Ez": np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64),
                    }
                },
            }

    monkeypatch.setattr("witwin.maxwell.simulation._resolve_fdtd_backend", lambda: (FakeFDTD, None))
    monkeypatch.setattr("witwin.maxwell.simulation._require_cuda_device", lambda scene, method: None)
    # The stand-in solver has no fields to bind the core launches to.
    monkeypatch.setattr("witwin.maxwell.fdtd.runtime.core_path.activate_core_path", lambda solver: None)

    scene = mw.Scene(
        # Three cells per axis: the Yee shapes (3, 4, 4), (4, 3, 4), (4, 4, 3) of the stand-in arrays.
        domain=mw.Domain(bounds=((-0.375, 0.375),) * 3),
        grid=mw.GridSpec.uniform(0.25),
    )
    sim = mw.Simulation.fdtd(
        scene,
        frequencies=[0.9e9, 1.1e9],
        config=mw.FDTDConfig(
            run_time=mw.TimeConfig(time_steps=24, steady_cycles=6, transient_cycles=4),
            spectral_sampler=mw.fdtd.SpectralSampler(window="none"),
            absorber="cpml", device="cpu",
        ),
    )

    result = sim.run()

    assert result.frequencies == (0.9e9, 1.1e9)
    assert result.E.x.values.shape == (2, 3, 4, 4)
    assert result.E.x.values.data_ptr() == result.solver.ex.data_ptr()
    assert result.at(frequency=1.1e9).E.x.values.shape == (3, 4, 4)
    np.testing.assert_allclose(result.at(freq_index=0).E.y.values.cpu().numpy(), 3.0)
    assert result.at().monitor("center").sel(component="Ez").values.shape == (2,)
    assert result.at(frequency=1.1e9).monitor("center").sel(component="Ez").values.item() == 3.0 + 4.0j
    stats = result.stats()
    assert stats["num_frequencies"] == 2
    assert stats["dft_sample_counts"] == (6, 5)
    assert stats["observer_sample_counts"] == (6, 5)
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

    scene = mw.Scene(ports=(port,))
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

    surface = mw.monitors.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.6, 0.8),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(monitors=(surface,))
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
    surface = mw.monitors.ClosedSurfaceMonitor.box(
        "huygens",
        position=(0.0, 0.0, 0.0),
        size=(0.4, 0.4, 0.4),
        frequencies=(1.0e9,),
    )
    scene = mw.Scene(monitors=(surface,))

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

    face_payload = result.raw_monitor("huygens::z_pos")
    np.testing.assert_allclose(face_payload["x"], np.linspace(-0.2, 0.2, 5))
    np.testing.assert_allclose(face_payload["y"], np.linspace(-0.2, 0.2, 5))
    assert face_payload["monitor_type"] == "finite_plane"
    assert face_payload["surface_name"] == "huygens"

    surface_payload = result.raw_monitor("huygens")
    assert surface_payload["kind"] == "closed_surface"
    assert set(surface_payload["faces"]) == {"x_neg", "x_pos", "y_neg", "y_pos", "z_neg", "z_pos"}
    np.testing.assert_allclose(surface_payload["faces"]["z_pos"]["x"], np.linspace(-0.2, 0.2, 5))


def _x_profile(values, shape):
    """A Yee component of ``shape`` that varies along x only (the symmetry axis of these tests)."""
    return torch.tensor(values)[:, None, None].expand(shape).contiguous()


def test_result_can_expand_low_face_symmetry_back_to_full_domain():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 0.75), (0.0, 0.5), (0.0, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
        symmetry=("PMC", None, None),
    )
    result = Result(
        method="fdtd",
        scene=scene,
        frequency=1e9,
        fields={
            "EX": _x_profile((1.0, 2.0, 3.0), (3, 3, 3)),
            "EY": _x_profile((10.0, 20.0, 30.0, 40.0), (4, 2, 3)),
            "EZ": _x_profile((100.0, 200.0, 300.0, 400.0), (4, 3, 2)),
        },
    )

    expanded = result.at(expand_symmetry=True)
    expanded_ex = expanded.E.x.isel(f=0).values
    expanded_ey = expanded.E.y.isel(f=0).values
    expanded_eps = expanded.materials.eps.scalar

    assert expanded_ex[:, 0, 0].tolist() == [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    assert expanded_ey[:, 0, 0].tolist() == [40.0, 30.0, 20.0, 10.0, 20.0, 30.0, 40.0]
    assert expanded.E.x.coords["x"].tolist() == [-0.625, -0.375, -0.125, 0.125, 0.375, 0.625]
    assert expanded.E.y.coords["x"].tolist() == [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    assert expanded_eps.shape == (7, 3, 3)


def test_fdtd_initialization_builds_dispersive_state(monkeypatch):
    class DummyModule:
        pass

    monkeypatch.setattr("witwin.maxwell.fdtd.solver._get_fdtd_module", lambda: DummyModule())
    monkeypatch.setattr("witwin.maxwell.fdtd.solver._require_cuda_scene", lambda scene: None)

    scene = mw.Scene(
                domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
                grid=mw.GridSpec.uniform(0.25),
                structures=(
            mw.Structure(
                name="debye_box",
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(0.5, 0.5, 0.5)),
                material=mw.Material.debye(eps_inf=2.0, delta_eps=3.0, tau=1.0e-9),
            ),
            ),
            )

    solver = FDTD(prepare_scene(scene, device="cpu"), frequency=2.0e9, absorber_type="cpml")
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
    )
    result = Result(
        method="fdtd",
        scene=scene,
        frequency=1e9,
        fields={
            "EX": _x_profile((1.0, 2.0, 3.0), (3, 3, 3)),
            "EZ": _x_profile((10.0, 20.0, 30.0, 40.0), (4, 3, 2)),
        },
    )

    expanded_ex = result.field("Ex", expand_symmetry=True).isel(f=0).values
    expanded_ez = result.field("Ez", expand_symmetry=True).isel(f=0).values

    assert expanded_ex[:, 0, 0].tolist() == [3.0, 2.0, 1.0, 1.0, 2.0, 3.0]
    assert expanded_ez[:, 0, 0].tolist() == [-40.0, -30.0, -20.0, 10.0, 20.0, 30.0, 40.0]


def test_simulation_fdfd_rejects_non_cuda_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
    )

    with pytest.raises(ValueError, match="requires config.device to be CUDA"):
        mw.Simulation.fdfd(scene, frequencies=1e9, config=mw.FDFDConfig(device="cpu")).prepare()


def test_simulation_fdtd_rejects_non_cuda_scene():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.25),
    )

    with pytest.raises(ValueError, match="requires config.device to be CUDA"):
        mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(device="cpu")).prepare()
