import pytest
import torch

import witwin.maxwell as mw

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FDTD auto-shutoff tests require CUDA."
)


def _vacuum_scene(*, source_time):
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))),
        grid=mw.GridSpec.uniform(0.04),
        boundary=mw.BoundarySpec.pml(num_layers=6, strength=1.0),
        device="cuda",
    )
    scene.add_source(
        mw.PointDipole(
            position=(0.0, 0.0, 0.0),
            polarization="Ez",
            width=0.05,
            source_time=source_time,
            name="src",
        )
    )
    return scene


def test_ring_down_stops_early():
    # A broadband pulse radiates into the PML and E-energy collapses to a small
    # residual near-field of the point dipole. A shutoff threshold above that
    # residual floor triggers early termination well before the planned budget.
    scene = _vacuum_scene(
        source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9, amplitude=100.0)
    )
    result = mw.Simulation.fdtd(
        scene,
        frequency=1.0e9,
        run_time=mw.TimeConfig(time_steps=3000),
        full_field_dft=True,
        shutoff=5e-2,
        shutoff_check_interval=50,
    ).run()

    stats = result.stats()
    assert stats["shutoff_triggered"] is True
    assert stats["shutoff_step"] is not None
    assert stats["shutoff_step"] < stats["time_steps"]
    assert stats["steps_run"] == stats["shutoff_step"] + 1


def test_cw_parity_between_shutoff_and_disabled():
    # A continuously driven CW source keeps E-energy roughly constant, so the guard
    # must not stop it early; the resulting spectral fields must match a run with
    # shutoff disabled.
    source_time = mw.CW(frequency=1e9, amplitude=100.0)

    def _run(shutoff):
        scene = _vacuum_scene(source_time=source_time)
        return mw.Simulation.fdtd(
            scene,
            frequency=1e9,
            run_time=mw.TimeConfig(time_steps=1300),
            full_field_dft=True,
            shutoff=shutoff,
        ).run()

    result_default = _run(1e-5)
    result_disabled = _run(0.0)

    # Auto-shutoff must not have fired for the CW drive.
    assert result_default.stats()["shutoff_triggered"] is False

    fields_default = result_default.fields
    fields_disabled = result_disabled.fields
    assert set(fields_default) == {"EX", "EY", "EZ"}

    stacked_default = torch.cat([fields_default[name].reshape(-1) for name in ("EX", "EY", "EZ")])
    stacked_disabled = torch.cat([fields_disabled[name].reshape(-1) for name in ("EX", "EY", "EZ")])

    rel_error = torch.linalg.vector_norm(stacked_default - stacked_disabled) / torch.linalg.vector_norm(
        stacked_disabled
    )
    assert float(rel_error) < 1e-3


def test_broadband_early_stop_matches_full_run():
    # When auto-shutoff fires on a decaying broadband pulse, the planned-window
    # normalization restore must keep the full-field DFT spectrum equal to the
    # untruncated run. Without the restore the shortened run's 2/N scale would
    # inflate the spectrum by ~N_full/N_stop (multiple hundred percent).
    def _run(shutoff):
        scene = _vacuum_scene(
            source_time=mw.GaussianPulse(frequency=1.0e9, fwidth=0.4e9, amplitude=100.0)
        )
        return mw.Simulation.fdtd(
            scene,
            frequency=1.0e9,
            run_time=mw.TimeConfig(time_steps=3000),
            full_field_dft=True,
            shutoff=shutoff,
            shutoff_check_interval=50,
        ).run()

    result_shutoff = _run(5e-2)
    result_full = _run(0.0)

    assert result_shutoff.stats()["shutoff_triggered"] is True
    assert result_shutoff.stats()["shutoff_step"] < result_shutoff.stats()["time_steps"]

    stacked_shutoff = torch.cat(
        [result_shutoff.fields[name].reshape(-1) for name in ("EX", "EY", "EZ")]
    )
    stacked_full = torch.cat(
        [result_full.fields[name].reshape(-1) for name in ("EX", "EY", "EZ")]
    )
    rel_error = torch.linalg.vector_norm(stacked_shutoff - stacked_full) / torch.linalg.vector_norm(
        stacked_full
    )
    assert float(rel_error) < 0.05, f"early-stop spectrum drifted from full run: rel_error={float(rel_error):.4f}"
