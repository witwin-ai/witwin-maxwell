"""``Simulation.prepare().run()`` of a wave-port run returns the ``Result.scene`` contract of ``Simulation.run()``.

``prepare()`` carries the PreparedScene into the WavePort sweep, the direct WavePort excitation and the
TerminalWavePort sweep; the Result holds its declarations (a ``Scene``), as the lumped network sweep does.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.result import _find_resolved_scene_monitor, _find_scene_monitor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="wave-port runs need CUDA")


def _wave_port(name, *, position, direction):
    return mw.rf.WavePort(
        name,
        position=position,
        size=(0.0, 0.60, 0.30),
        direction=direction,
        reference_plane=position[0],
        modes=(mw.rf.WaveModeSpec("te", polarization="Ez"),),
    )


def _hollow_guide_scene():
    walls = (
        ((0.0, 0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_high"),
        ((0.0, -0.40, 0.0), (1.0, 0.20, 0.70), "wall_y_low"),
        ((0.0, 0.0, 0.25), (1.0, 0.60, 0.20), "wall_z_high"),
        ((0.0, 0.0, -0.25), (1.0, 0.60, 0.20), "wall_z_low"),
    )
    return mw.Scene(
        domain=mw.Domain(bounds=((-0.6, 0.6), (-0.5, 0.5), (-0.35, 0.35))),
        grid=mw.GridSpec.uniform(0.05),
        boundary=mw.BoundarySpec.pml(num_layers=4, strength=1.0),
        structures=tuple(mw.Structure(geometry=mw.Box(position=position, size=size), material=mw.Material.pec(),
                                      name=name) for position, size, name in walls),
        ports=(_wave_port("left", position=(-0.30, 0.0, 0.0), direction="+"),
               _wave_port("right", position=(0.30, 0.0, 0.0), direction="-")),
        monitors=(mw.PointMonitor("probe", (0.0, 0.0, 0.0), fields=("Ez",)),),
    )


def _microstrip_scene():
    dx = 0.005
    x, y, z = np.arange(-4, 5) * dx, np.arange(-12, 13) * dx, np.arange(-1, 14) * dx
    return mw.Scene(
        domain=mw.Domain(bounds=((x[0], x[-1]), (y[0], y[-1]), (z[0], z[-1]))),
        grid=mw.GridSpec.custom(x, y, z),
        boundary=mw.BoundarySpec.pec(),
        structures=(
            mw.Structure(mw.geometry.PlanarPEC(((-1, -1), (1, -1), (1, 1), (-1, 1)), normal_axis="z", coordinate=0.0),
                         mw.Material.pec(), name="ground"),
            mw.Structure(mw.Box(position=(0, 0, 0.01), size=(1, 1, 0.02)), mw.Material(eps_r=3.0), name="substrate"),
            mw.Structure(mw.geometry.PlanarPEC(((-1, -0.01), (1, -0.01), (1, 0.01), (-1, 0.01)), normal_axis="z",
                                               coordinate=0.02), mw.Material.pec(), name="strip"),
        ),
        ports=(mw.rf.TerminalWavePort("P", position=(0, 0, 0.03), size=(0, 0.11, 0.06), direction="+",
                                      reference_plane=0.0, terminals=(mw.rf.TerminalRef("strip"),),
                                      reference=mw.rf.TerminalRef("ground")),),
        monitors=(mw.PointMonitor("probe", (0.0, 0.0, 0.01), fields=("Ez",)),),
    )


_CASES = {
    "waveport_sweep": (_hollow_guide_scene, 4.5e8, mw.rf.PortSweep),
    "waveport_excitation": (_hollow_guide_scene, 4.5e8, lambda: mw.rf.PortExcitation("left", mode_name="left::TE0")),
    "terminal_waveport_sweep": (_microstrip_scene, 1.0e9, mw.rf.PortSweep),
}


@pytest.mark.parametrize("case", tuple(_CASES))
def test_prepared_run_result_scene_matches_the_direct_run(case):
    build, frequency, excitation = _CASES[case]

    def simulation():
        return mw.Simulation.fdtd(
            build(), frequencies=(frequency,), excitations=excitation(),
            config=mw.FDTDConfig(run_time=mw.TimeConfig.auto(steady_cycles=2, transient_cycles=4),
                                 spectral_sampler=mw.fdtd.SpectralSampler(window="hanning"), full_field_dft=False),
        )

    direct = simulation().run()
    prepared = simulation().prepare().run()
    for result in (direct, prepared):
        assert type(result.scene) is mw.Scene
        assert result.scene.domain.bounds == direct.scene.domain.bounds
        assert tuple(s.name for s in result.scene.structures) == tuple(s.name for s in direct.scene.structures)
        # Mapped charts refuse modal and terminal wave ports, so neither entry carries a field frame.
        assert result.scene.coordinates is None
        assert "field_frame" not in result.metadata
        assert _find_scene_monitor(result.scene, "probe") is not None
        assert _find_resolved_scene_monitor(result.scene, "probe") is not None
    direct_probe, prepared_probe = direct.raw_monitor("probe"), prepared.raw_monitor("probe")
    assert direct_probe.keys() == prepared_probe.keys()
    tensors = [key for key, value in direct_probe.items() if isinstance(value, torch.Tensor)]
    assert tensors
    for key in tensors:
        assert torch.equal(prepared_probe[key], direct_probe[key]), key
