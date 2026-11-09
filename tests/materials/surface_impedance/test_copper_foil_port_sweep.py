"""Independent excitation columns retain private surface histories and losses."""
import pytest
import torch

import witwin.maxwell as mw
from tests.materials.compiler.test_copper_foil_material_cache import _simulation


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Foil RF sweeps require CUDA.")


def _ports():
    return tuple(mw.LumpedPort(name=name, positive=(x, 0., .005), negative=(x, 0., -.005),
        voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(x, 0., -.0025), size=(.015, .015, 0.)))
        for name, x in (("left", -.01), ("right", .01)))


@pytest.mark.parametrize("graph", [False, True])
def test_foil_ade_port_sweep_matches_independent_columns_and_roundtrips_losses(tmp_path, graph):
    cache = mw.FDTDMaterialCache()
    scene = _simulation().scene.clone(sources=(), ports=_ports())
    waveform = mw.CW(frequency=1e9)
    common = dict(frequency=1e9, run_time=mw.TimeConfig(time_steps=96, dt=5e-12),
        spectral_sampler=mw.SpectralSampler(window="none"), material_cache=cache, cuda_graph=graph)
    result = mw.Simulation.fdtd(scene, excitations=mw.PortSweep(source_time=waveform), **common).run()
    assert cache.stats["misses"] == cache.stats["hits"] == 1
    assert result.conductor_losses == {}
    assert tuple(result.conductor_loss_columns) == ("left", "right")
    for column, active in enumerate(("left", "right")):
        ports = []
        for port in _ports():
            ports.append(mw.LumpedPort(name=port.name, positive=port.positive, negative=port.negative,
                voltage_path=port.voltage_path, current_surface=port.current_surface,
                termination=None if port.name == active else mw.SeriesRLC(r=50.)))
        direct = mw.Simulation.fdtd(scene.clone(ports=ports), excitations=mw.PortExcitation(active,
            source_time=waveform), **common).run()
        for name in ("left", "right"):
            for component in ("voltage", "current", "a", "b"):
                torch.testing.assert_close(getattr(result.port(name), component)[column],
                    getattr(direct.port(name), component), rtol=2e-5, atol=1e-8)
        actual, expected = result.conductor_loss_columns[active]["foil"], direct.conductor_losses["foil"]
        assert actual["dissipated_energy"] == pytest.approx(expected["dissipated_energy"], rel=2e-5)
        torch.testing.assert_close(actual["surface_power"], expected["surface_power"], rtol=2e-5, atol=1e-12)
        compact = result._array_run_data.column_results[column][0]
        torch.testing.assert_close(compact.conductor_losses["foil"]["surface_power"], actual["surface_power"])
    result.save(tmp_path / "sweep.pt")
    restored = mw.Result.load(tmp_path / "sweep.pt", scene=scene)
    torch.testing.assert_close(restored.network.s, result.network.s.cpu())
    for name in ("left", "right"):
        for component in ("voltage", "current", "a", "b"):
            torch.testing.assert_close(getattr(restored.port(name), component),
                                       getattr(result.port(name), component).cpu())
    for active in ("left", "right"):
        torch.testing.assert_close(restored.conductor_loss_columns[active]["foil"]["surface_power"],
                                   result.conductor_loss_columns[active]["foil"]["surface_power"])
    assert restored.conductor_losses == {}
