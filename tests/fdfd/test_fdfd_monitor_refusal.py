"""FDFD must refuse scene readouts it cannot populate in Result."""

import pytest
import torch

import witwin.maxwell as mw


MONITORS = (
    mw.PointMonitor("requested", (0.0, 0.0, 0.0)),
    mw.PlaneMonitor("requested"),
    mw.FinitePlaneMonitor("requested", size=(0.1, 0.1, 0.0)),
    mw.FluxMonitor("requested"),
    mw.ClosedSurfaceMonitor.box("requested", size=(0.1,) * 3),
    mw.ModeMonitor("requested", size=(0.1, 0.1, 0.0)),
    mw.PowerLossMonitor("requested", position=(0.0,) * 3, size=(0.1,) * 3),
    mw.AngleProjectionMonitor("requested", size=(0.1,) * 3, theta=(0.1,), phi=(0.0,)),
    mw.CartesianProjectionMonitor("requested", size=(0.1,) * 3, x=(0.0,), y=(0.0,), distance=1.0),
    mw.KSpaceProjectionMonitor("requested", size=(0.1,) * 3, ux=(0.0,), uy=(0.0,)),
    mw.GaussianOverlapMonitor("requested", size=(0.1, 0.1, 0.0), beam_waist=0.02),
    mw.AstigmaticGaussianOverlapMonitor("requested", size=(0.1, 0.1, 0.0), beam_waist=(0.02, 0.03)),
    mw.SurfaceFieldTimeMonitor("requested", size=(0.1, 0.1, 0.0)),
    mw.AuxFieldTimeMonitor("requested"),
    mw.FieldTimeMonitor("requested"),
    mw.FluxTimeMonitor("requested"),
    mw.IncidentPowerDensityMonitor("requested"),
    mw.DiffractionMonitor("requested", size=(0.1, 0.1, 0.0)),
    mw.DipoleEmissionMonitor("requested", source_name="source"),
    mw.WireMonitor("requested", wire="wire", frequencies=(1e9,)),
    mw.BreakdownMonitor("requested", position=(0.0,) * 3, size=(0.1,) * 3, critical_field=1e6),
    mw.ComponentStressMonitor("requested", port="port", rating=mw.ComponentRating(voltage=1.0),
                              voltage_series="voltage", current_series="current"),
)


def _scene():
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.1, 0.1),) * 3),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.none(),
        device="cuda",
    )


@pytest.mark.parametrize("monitor", MONITORS, ids=lambda monitor: type(monitor).__name__)
def test_fdfd_refuses_scene_monitor(monitor):
    scene = _scene()
    simulation = mw.Simulation.fdfd(scene, frequency=1e9)
    scene.add_monitor(monitor)
    for prepare in (simulation.prepare, lambda: mw.Simulation.fdfd(scene, frequency=1e9).prepare()):
        with pytest.raises(NotImplementedError) as error:
            prepare()
        message = str(error.value)
        assert monitor.name in message
        assert type(monitor).__name__ in message
        assert "not supported by FDFD; use Simulation.fdtd(...) or FDFDConfig.observations" in message


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fdfd_permittivity_monitor_returns_material_payload():
    scene = _scene().add_structure(mw.Structure(
        mw.Box(size=(0.2,) * 3), mw.Material(eps_r=4.0),
    )).add_source(mw.PointDipole(
        position=(0.0, 0.0, -0.05), polarization="Ez", width=0.05,
        source_time=mw.CW(frequency=1e9), name="source",
    )).add_monitor(mw.PermittivityMonitor("permittivity"))
    result = mw.Simulation.fdfd(
        scene, frequency=1e9,
        solver=mw.GMRES(solver_type="direct", precision="double"),
    ).run()
    payload = result.monitor("permittivity")
    assert payload["frequencies"] == (1e9,)
    for axis in "xyz":
        values = payload[f"eps_{axis}"]
        torch.testing.assert_close(values, torch.full_like(values, 4.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fdfd_config_observations_return_projection_value():
    scene = _scene().add_source(mw.PointDipole(
        position=(0.0, 0.0, -0.05), polarization="Ez", width=0.05,
        source_time=mw.CW(frequency=1e9), name="source",
    ))
    config = mw.FDFDConfig(
        solver=mw.GMRES(solver_type="direct", precision="double"),
        observations=(mw.FieldProjection("probe", component="Ez", position=(0.0, 0.0, 0.025)),),
    )
    result = mw.Simulation(scene=scene, method="fdfd", frequencies=(1e9,), config=config).run()
    value = result.monitor("probe")["value"]
    assert value.abs() > 0
    torch.testing.assert_close(value, result.E.z[2, 2, 2], rtol=1e-6, atol=1e-8)
