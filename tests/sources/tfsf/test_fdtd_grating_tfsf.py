from __future__ import annotations

import math
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.excitation.temporal import apply_compiled_source_terms, apply_generic_source_terms


_GRATING_TFSF_Z_BOUNDS = (-0.24, 0.24)


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


def _grating_scene(*, injection, boundary=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=_grating_boundary() if boundary is None else boundary,
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


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _component_volume(result, component: str):
    solver = result.solver
    field = _to_numpy(result.tensor(component))
    if component == "Ex":
        z = np.linspace(solver.scene.domain_range[4], solver.scene.domain_range[5], field.shape[2])
    elif component == "Ez":
        z = np.linspace(
            solver.scene.domain_range[4] + 0.5 * solver.scene.dz,
            solver.scene.domain_range[5] - 0.5 * solver.scene.dz,
            field.shape[2],
        )
    else:
        raise ValueError(f"Unsupported component for grating test: {component!r}.")
    return field, z


def _normal_slab_ratio(field, z_coords, bounds, *, dz):
    magnitude = np.abs(np.asarray(field))
    inside = (z_coords >= bounds[0]) & (z_coords <= bounds[1])
    outside = (z_coords < bounds[0] - dz) | (z_coords > bounds[1] + dz)
    inside_max = float(np.max(magnitude[:, :, inside]))
    outside_max = float(np.max(magnitude[:, :, outside]))
    return outside_max / max(inside_max, 1e-12), inside_max, outside_max


def _run_grating_full_field(*, scene=None, time_steps=64):
    if scene is None:
        scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=_GRATING_TFSF_Z_BOUNDS))
    return mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=time_steps),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
        absorber="cpml",
    ).run()


def _grating_flux_scene(*, with_dielectric_slab: bool):
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=_GRATING_TFSF_Z_BOUNDS))
    if with_dielectric_slab:
        scene.add_structure(
            mw.Structure(
                geometry=mw.Box(position=(0.0, 0.0, 0.0), size=(1.2, 1.2, 0.12)),
                material=mw.Material(eps_r=9.0),
                name="dielectric_slab",
            )
        )
    scene.add_monitor(mw.FluxMonitor("reflection", axis="z", position=-0.36, normal_direction="-"))
    scene.add_monitor(mw.FluxMonitor("transmission", axis="z", position=0.36, normal_direction="+"))
    return scene


@lru_cache(maxsize=None)
def _grating_flux_metrics(with_dielectric_slab: bool):
    result = mw.Simulation.fdtd(
        _grating_flux_scene(with_dielectric_slab=with_dielectric_slab),
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=256),
        spectral_sampler=mw.SpectralSampler(window="none"),
        absorber="cpml",
    ).run()
    reflection = float(torch.as_tensor(result.monitor("reflection")["flux"]).detach().cpu().real)
    transmission = float(torch.as_tensor(result.monitor("transmission")["flux"]).detach().cpu().real)
    del result
    torch.cuda.empty_cache()
    return {
        "reflection": reflection,
        "transmission": transmission,
    }


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


def _nonperiodic_slab_scene(*, direction=(0.0, 0.0, 1.0), polarization=(1.0, 0.0, 0.0),
                            source_time=None, injection=None):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=direction,
            polarization=polarization,
            source_time=source_time if source_time is not None else mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=injection if injection is not None else mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)),
            name="slab_tfsf",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_nonperiodic_tfsf_slab_initializes_axis_aligned_state():
    prepared = mw.Simulation.fdtd(
        _nonperiodic_slab_scene(),
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
        absorber="cpml",
    ).prepare()
    solver = prepared.solver
    state = solver._tfsf_state
    assert solver.tfsf_enabled is True
    assert state["provider"] == "plane_wave_axis_aligned"
    assert state["mode"] == "slab"
    assert state["axis"] == "z"
    assert state["lower"][2] < state["upper"][2]
    assert len(state["electric_terms"]) > 0
    assert len(state["magnetic_terms"]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
def test_nonperiodic_tfsf_slab_rejects_oblique_incidence():
    scene = _nonperiodic_slab_scene(
        direction=(0.2, 0.0, 0.9797958971132712),
        polarization=(0.0, 1.0, 0.0),
    )
    with pytest.raises(ValueError, match="normally incident"):
        mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1)).prepare()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_nonperiodic_tfsf_slab_forward_confines_total_field():
    bounds = (-0.24, 0.24)
    result = mw.Simulation.fdtd(
        _nonperiodic_slab_scene(injection=mw.TFSF.slab(axis="z", bounds=bounds)),
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=96),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
        absorber="cpml",
    ).run()

    field, z_coords = _component_volume(result, "Ex")
    assert np.isfinite(np.abs(field)).all()
    leakage_ratio, inside_max, outside_max = _normal_slab_ratio(
        field, z_coords, bounds, dz=result.solver.scene.dz
    )
    # The plane wave lives in the total-field slab; the scattered region outside the
    # two z-faces must not carry the incident field for an empty (no-scatterer) run.
    assert inside_max > 0.0
    assert leakage_ratio < 0.2


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_grating_tfsf_oblique_plane_wave_leakage_is_bounded():
    result = _run_grating_full_field(time_steps=64)

    ratios = []
    for component in ("Ex", "Ez"):
        field, z_coords = _component_volume(result, component)
        assert np.isfinite(np.abs(field)).all()
        leakage_ratio, inside_max, outside_max = _normal_slab_ratio(
            field,
            z_coords,
            _GRATING_TFSF_Z_BOUNDS,
            dz=result.solver.scene.dz,
        )
        assert inside_max > 0.0
        assert outside_max < inside_max * 4.0
        ratios.append(leakage_ratio)

    assert max(ratios) < 4.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_grating_tfsf_auto_bloch_phase_runs_forward():
    boundary = mw.BoundarySpec.faces(
        default="pml",
        num_layers=4,
        strength=1.0,
        x="bloch",
        y="bloch",
        z="pml",
        bloch_wavevector="auto",
    )
    scene = _grating_scene(
        injection=mw.TFSF.slab(axis="z", bounds=_GRATING_TFSF_Z_BOUNDS),
        boundary=boundary,
    )

    result = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=32),
        spectral_sampler=mw.SpectralSampler(window="none"),
        absorber="cpml",
    ).run()

    assert result.solver.resolved_bloch_wavevector[0] > 0.0
    assert result.solver.resolved_bloch_wavevector[1] > 0.0
    center = result.monitor("center")
    assert torch.isfinite(torch.as_tensor(center["Ex"]).abs()).all()
    assert torch.isfinite(torch.as_tensor(center["Ez"]).abs()).all()


_GRATING_BASE_DIRECTION = (0.2, 0.1, 0.9746794344808963)
_GRATING_BASE_POLARIZATION = (1.0, 0.0, -0.20519567041703082)
_GRATING_BASE_WAVEVECTOR = (math.pi, 0.5 * math.pi, 0.0)
_GRATING_BASE_DOMAIN = ((-0.6, 0.6), (-0.6, 0.6), (-0.8, 0.8))


def _roll_normal_from_z(triple, axis):
    """Cyclically rotate an (x, y, z)-ordered triple so the z entry (the base
    grating normal) lands on ``axis``. The permutation is a proper rotation, so a
    perpendicular (direction, polarization) pair stays perpendicular and the
    domain's deep dimension follows the normal axis."""
    x, y, z = triple
    return {"z": (x, y, z), "x": (z, x, y), "y": (y, z, x)}[axis]


def _general_axis_grating_scene(axis, *, source_time=None, bounds=(-0.24, 0.24)):
    direction = _roll_normal_from_z(_GRATING_BASE_DIRECTION, axis)
    polarization = _roll_normal_from_z(_GRATING_BASE_POLARIZATION, axis)
    wavevector = _roll_normal_from_z(_GRATING_BASE_WAVEVECTOR, axis)
    domain_bounds = _roll_normal_from_z(_GRATING_BASE_DOMAIN, axis)
    boundary_kinds = {other: "bloch" for other in "xyz"}
    boundary_kinds[axis] = "pml"
    scene = mw.Scene(
        domain=mw.Domain(bounds=domain_bounds),
        grid=mw.GridSpec.uniform(0.12),
        boundary=mw.BoundarySpec.faces(
            default="pml",
            num_layers=4,
            strength=1.0,
            bloch_wavevector=wavevector,
            **boundary_kinds,
        ),
        device="cuda",
    )
    scene.add_source(
        mw.PlaneWave(
            direction=direction,
            polarization=polarization,
            source_time=source_time if source_time is not None else mw.CW(frequency=1.0e9, amplitude=20.0),
            injection=mw.TFSF.slab(axis=axis, bounds=bounds),
            name="grating_tfsf",
        )
    )
    scene.add_monitor(mw.PointMonitor("center", (0.0, 0.0, 0.0), fields=("Ex", "Ez")))
    return scene


def _normal_axis_leakage(field, coords, bounds, axis_index, *, dcell):
    magnitude = np.abs(np.asarray(field))
    inside = (coords >= bounds[0]) & (coords <= bounds[1])
    outside = (coords < bounds[0] - dcell) | (coords > bounds[1] + dcell)
    inside_slice = [slice(None), slice(None), slice(None)]
    outside_slice = [slice(None), slice(None), slice(None)]
    inside_slice[axis_index] = inside
    outside_slice[axis_index] = outside
    inside_max = float(np.max(magnitude[tuple(inside_slice)]))
    outside_max = float(np.max(magnitude[tuple(outside_slice)]))
    return outside_max / max(inside_max, 1e-12), inside_max, outside_max


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD prepare")
@pytest.mark.parametrize("axis", ["x", "y"])
def test_grating_tfsf_slab_initializes_state_for_general_normal_axis(axis):
    prepared = mw.Simulation.fdtd(
        _general_axis_grating_scene(axis),
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
        absorber="cpml",
    ).prepare()
    state = prepared.solver._tfsf_state
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    assert prepared.solver.tfsf_enabled is True
    assert state["provider"] == "plane_wave_grating_slab_cw"
    assert state["mode"] == "slab"
    assert state["axis"] == axis
    assert state["lower"][axis_index] < state["upper"][axis_index]
    assert len(state["electric_terms"]) > 0
    assert len(state["magnetic_terms"]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("axis", ["x", "y"])
def test_grating_tfsf_general_axis_forward_confines_total_field(axis):
    # The oblique grating plane wave lives in the total-field slab spanning the two
    # faces normal to ``axis``; the scattered region beyond those faces must not
    # carry the incident field for an empty (no-scatterer) run, on any normal axis.
    bounds = (-0.24, 0.24)
    result = mw.Simulation.fdtd(
        _general_axis_grating_scene(axis, bounds=bounds),
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=64),
        spectral_sampler=mw.SpectralSampler(window="none"),
        full_field_dft=True,
        absorber="cpml",
    ).run()

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    dcell = (result.solver.scene.dx, result.solver.scene.dy, result.solver.scene.dz)[axis_index]
    lo = result.solver.scene.domain_range[2 * axis_index]
    hi = result.solver.scene.domain_range[2 * axis_index + 1]

    ratios = []
    for component in ("Ex", "Ez"):
        field = _to_numpy(result.tensor(component))
        assert np.isfinite(np.abs(field)).all()
        coords = np.linspace(lo, hi, field.shape[axis_index])
        leakage_ratio, inside_max, outside_max = _normal_axis_leakage(
            field, coords, bounds, axis_index, dcell=dcell
        )
        assert inside_max > 0.0
        assert outside_max < inside_max * 4.0
        ratios.append(leakage_ratio)
    assert max(ratios) < 4.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
@pytest.mark.parametrize("axis", ["x", "y"])
def test_grating_tfsf_general_axis_broadband_runs_finite(axis):
    # Broadband (GaussianPulse) grating injection on a non-z normal axis: the
    # delayed surface-current patches scatter into the split real/imag Bloch field
    # with the boundary wrap phase, and the pulse is not silently downgraded to CW.
    frequencies = (0.85e9, 1.0e9, 1.15e9)
    result = mw.Simulation.fdtd(
        _general_axis_grating_scene(
            axis,
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.3e9, amplitude=20.0),
            bounds=(-0.24, 0.24),
        ),
        frequencies=list(frequencies),
        run_time=mw.TimeConfig(time_steps=192),
        absorber="cpml",
    ).run()

    assert result.solver.tfsf_enabled is True
    assert result.solver.complex_fields_enabled is True
    assert result.solver._tfsf_state["provider"] == "plane_wave_grating_slab_cw"
    assert result.solver._tfsf_state["axis"] == axis
    for frequency in frequencies:
        payload = result.monitor("center", frequency=frequency)
        assert torch.isfinite(torch.as_tensor(payload["Ex"]).abs()).all()
        assert torch.isfinite(torch.as_tensor(payload["Ez"]).abs()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for FDTD")
def test_grating_tfsf_dielectric_slab_reflection_transmission_sanity():
    empty = _grating_flux_metrics(False)
    slab = _grating_flux_metrics(True)
    incident_scale = max(abs(empty["transmission"]), 1.0e-12)
    empty_reflection = abs(empty["reflection"]) / incident_scale
    reflection = abs(slab["reflection"]) / incident_scale
    transmission = abs(slab["transmission"]) / incident_scale

    assert math.isfinite(reflection)
    assert math.isfinite(transmission)
    assert 0.0 <= reflection <= 2.0
    assert 0.0 <= transmission <= 2.0
    assert reflection > empty_reflection * 2.0
