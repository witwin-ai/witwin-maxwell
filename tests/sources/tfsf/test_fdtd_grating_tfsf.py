from __future__ import annotations

import math

import pytest
import torch

import witwin.maxwell as mw


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
def test_grating_box_tfsf_still_rejected_with_bloch_boundaries():
    scene = _grating_scene(
        injection=mw.TFSF(bounds=((-0.48, 0.48), (-0.48, 0.48), (-0.24, 0.24)))
    )
    simulation = mw.Simulation.fdtd(scene, frequencies=[1.0e9], run_time=mw.TimeConfig(time_steps=1))
    with pytest.raises(NotImplementedError, match="TFSF injection"):
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
def test_grating_tfsf_slab_forward_reports_pending_runtime_after_phase_resolution():
    scene = _grating_scene(injection=mw.TFSF.slab(axis="z", bounds=(-0.24, 0.24)))
    prepared = mw.Simulation.fdtd(
        scene,
        frequencies=[1.0e9],
        run_time=mw.TimeConfig(time_steps=1),
    ).prepare()

    with pytest.raises(NotImplementedError, match="TFSF slab forward runtime"):
        prepared.run()
