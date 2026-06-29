from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.temporal import apply_compiled_source_terms, apply_generic_source_terms


class _CaptureLaunch:
    def __init__(self, calls, name, kwargs):
        self._calls = calls
        self._name = name
        self._kwargs = kwargs

    def launchRaw(self, **launch_kwargs):
        self._calls.append((self._name, self._kwargs, launch_kwargs))


class _CaptureModule:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def kernel(**kwargs):
            return _CaptureLaunch(self.calls, name, kwargs)

        return kernel


def _mixed_periodic_bloch_source_solver():
    fdtd_module = _CaptureModule()
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="periodic",
        y="bloch",
        z="pml",
        bloch_wavevector=(0.0, math.pi, 0.0),
    )
    solver = SimpleNamespace(
        scene=SimpleNamespace(boundary=boundary),
        fdtd_module=fdtd_module,
        kernel_block_size=(1, 1, 1),
        Ex=object(),
        Ex_imag=object(),
        Ey=object(),
        Ey_imag=object(),
        Ez=object(),
        Ez_imag=object(),
        Hx=object(),
        Hx_imag=object(),
        Hy=object(),
        Hy_imag=object(),
        Hz=object(),
        Hz_imag=object(),
        boundary_phase_cos=(1.0, 0.25, 1.0),
        boundary_phase_sin=(0.0, 0.75, 0.0),
        _clamp_pec_boundaries=lambda: None,
    )
    return solver, fdtd_module


def _mixed_periodic_bloch_source_term():
    return {
        "field_name": "Ez",
        "offsets": (0, 0, 0),
        "patch": torch.ones((1, 1, 1), dtype=torch.float32),
        "grid": (1, 1, 1),
        "phase_real": 0.25,
        "phase_imag": -0.5,
        "delay_patch": None,
        "activation_delay_patch": None,
        "cw_cos_patch": None,
        "cw_sin_patch": None,
        "source_index": None,
        "source_time": None,
        "omega": None,
    }


def _grating_boundary(*, bloch_wavevector=(math.pi, 0.5 * math.pi, 0.0)):
    return mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector=bloch_wavevector,
    )


def _grating_scene(*, injection):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=_grating_boundary(),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=injection,
            name="grating_tfsf",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


@pytest.mark.parametrize(
    "dispatcher",
    (apply_generic_source_terms, apply_compiled_source_terms),
    ids=("generic", "compiled"),
)
def test_mixed_periodic_bloch_source_terms_prefer_bloch_dispatch(dispatcher):
    solver, fdtd_module = _mixed_periodic_bloch_source_solver()
    source_time = {"kind": "cw", "frequency": 1.0, "amplitude": 2.0, "phase": 0.0}

    dispatcher(
        solver,
        [_mixed_periodic_bloch_source_term()],
        source_time=source_time,
        omega=2.0 * math.pi,
        time_value=0.0,
    )

    assert len(fdtd_module.calls) == 1
    kernel_name, kwargs, _ = fdtd_module.calls[0]
    assert kernel_name == "addSourcePatchBloch3D"
    assert kwargs["signalReal"] == pytest.approx(0.5)
    assert kwargs["signalImag"] == pytest.approx(-1.0)
    assert kwargs["wrapAxisA"] == 1
    assert kwargs["wrapAxisB"] == 1
    assert kwargs["phaseCosA"] == pytest.approx(1.0)
    assert kwargs["phaseSinA"] == pytest.approx(0.0)
    assert kwargs["phaseCosB"] == pytest.approx(0.25)
    assert kwargs["phaseSinB"] == pytest.approx(0.75)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_mixed_bloch_pml_prepare_enables_complex_and_cpml_state():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)))
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
        absorber="cpml",
    ).prepare()
    solver = prepared.solver
    assert solver.boundary_kind == "mixed"
    assert solver.has_bloch_axes == ("x", "y")
    assert solver.has_pml_faces is True
    assert solver.complex_fields_enabled is True
    assert solver.uses_cpml is True
    assert hasattr(solver, "Ex_imag")
    assert hasattr(solver, "psi_ex_z")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_tfsf_slab_initializes_state():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)))
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
        absorber="cpml",
    ).prepare()
    state = prepared.solver._tfsf_state
    assert prepared.solver.tfsf_enabled is True
    assert state["mode"] == "slab"
    assert state["axis"] == "z"
    assert state["lower"][2] < state["upper"][2]
    assert len(state["electric_terms"]) > 0
    assert len(state["magnetic_terms"]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_tfsf_rejects_box_tfsf_with_bloch_boundaries():
    scene = _grating_scene(
        injection=mw.TFSF(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.24, 0.24)))
    )
    simulation = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1))
    with pytest.raises(NotImplementedError, match="TFSF slab"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_tfsf_slab_runtime_reports_unsupported_until_state_exists():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.0, 0.0, 1.0),
            polarization=(1.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="slab_tfsf",
        )
    )
    simulation = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1))
    with pytest.raises(NotImplementedError, match="TFSF slab"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_explicit_bloch_phases_are_resolved_on_solver():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)))

    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
    ).prepare()

    phase_x = complex(prepared.solver.boundary_phase_cos[0], prepared.solver.boundary_phase_sin[0])
    phase_y = complex(prepared.solver.boundary_phase_cos[1], prepared.solver.boundary_phase_sin[1])
    phase_z = complex(prepared.solver.boundary_phase_cos[2], prepared.solver.boundary_phase_sin[2])
    assert abs(phase_x - scene.boundary.bloch_phase_factors(scene.domain.domain_range)[0]) < 1e-6
    assert abs(phase_y - scene.boundary.bloch_phase_factors(scene.domain.domain_range)[1]) < 1e-6
    assert phase_z == 1.0 + 0.0j


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_auto_bloch_wavevector_matches_incident_direction():
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector="auto",
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=boundary,
        device="cuda",
    )
    direction = (0.2, 0.1, 0.9746794344808963)
    scene.add_source(
        mw.PlaneWave(
            direction=direction,
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="grating_tfsf",
        )
    )

    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
    ).prepare()

    kx, ky, kz = prepared.solver.resolved_bloch_wavevector
    assert kx > 0.0
    assert ky > 0.0
    assert kz == 0.0
    phase_x = complex(prepared.solver.boundary_phase_cos[0], prepared.solver.boundary_phase_sin[0])
    phase_y = complex(prepared.solver.boundary_phase_cos[1], prepared.solver.boundary_phase_sin[1])
    length_x = scene.domain.bounds[0][1] - scene.domain.bounds[0][0]
    length_y = scene.domain.bounds[1][1] - scene.domain.bounds[1][0]
    assert abs(phase_x - complex(math.cos(kx * length_x), math.sin(kx * length_x))) < 1e-6
    assert abs(phase_y - complex(math.cos(ky * length_y), math.sin(ky * length_y))) < 1e-6
    assert prepared.solver.boundary_phase_cos[2] == 1.0
    assert prepared.solver.boundary_phase_sin[2] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_auto_bloch_rejects_non_cw_source_time():
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector="auto",
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=boundary,
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.2e9),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="grating_tfsf",
        )
    )
    with pytest.raises(ValueError, match="CW"):
        mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1)).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_tfsf_rejects_gaussian_beam_slab_source():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=_grating_boundary(),
        device="cuda",
    )
    scene.add_source(
        mw.GaussianBeam(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            beam_waist=0.4,
            focus=(0.0, 0.0, 0.0),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="invalid_grating_tfsf",
        )
    )
    with pytest.raises(ValueError, match="PlaneWave"):
        mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1)).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_tfsf_allows_material_on_transverse_bloch_boundary():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)))
    scene.add_structure(
        mw.Structure(
            geometry=mw.Box(position=(-0.54, 0.0, 0.0), size=(0.12, 0.24, 0.12)),
            material=mw.Material(eps_r=2.0),
            name="transverse_edge_feature",
        )
    )

    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
        absorber="cpml",
    ).prepare()

    assert prepared.solver._tfsf_state["provider"] == "plane_wave_grating_slab_cw"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_tfsf_rejects_slab_bounds_inside_pml_margin():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.5, 0.24)))

    with pytest.raises(ValueError, match="non-PML"):
        mw.Simulation.fdtd(
            scene,
            frequencies=[1.0e9],
            run_time=mw.TimeConfig(time_steps=1),
            absorber="cpml",
        ).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_grating_explicit_bloch_rejects_non_pml_slab_normal_axis():
    boundary = mw.BoundarySpec.faces(
        default="pec",
        x="bloch",
        bloch_wavevector=(math.pi, 0.0, 0.0),
    )
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=boundary,
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=(0.2, 0.1, 0.9746794344808963),
            polarization=(1.0, 0.0, -0.20519567041703082),
            source_time=mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="invalid_grating_tfsf",
        )
    )

    simulation = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1))
    with pytest.raises(ValueError, match="normal axis to use PML"):
        simulation.prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_grating_tfsf_slab_forward_runs_after_phase_resolution():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)))
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=2),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).prepare()

    result = prepared.run()
    assert result.solver.tfsf_enabled is True
    assert result.solver._tfsf_state["provider"] == "plane_wave_grating_slab_cw"
