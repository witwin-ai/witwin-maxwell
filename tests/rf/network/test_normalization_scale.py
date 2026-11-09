"""Reference impedances and phase normalizations have no implicit SI unit floor."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.delay import _path_group_delay


def _network(z0):
    return mw.NetworkData(frequencies=torch.tensor([1e9], dtype=torch.float64),
                          s=torch.tensor([[[0.,.5],[.5,0.]]], dtype=torch.complex128),
                          z0=z0, port_names=("in","out"))


@pytest.mark.parametrize("length", [1., 1e-6])
def test_reference_impedance_cascade_is_relative(length):
    # The relative real-reference tolerance (_require_real_reference) is deferred to the
    # consumer-metadata rebaseline: network_math.py is a protected consumer-bridge file.
    z0 = 1e-12 * length
    result = _network(z0).cascade(_network(z0), port_map={"out":"in"})
    assert result.s[0,1,0].abs().item() == pytest.approx(.25)
    with pytest.raises(ValueError, match="same reference impedance"):
        _network(z0).cascade(_network(1.01*z0), port_map={"out":"in"})


@pytest.mark.parametrize("length", [1., 1e-6])
def test_group_delay_normalization_scales_with_frequency(length):
    frequencies = torch.linspace(1e-12, 2e-12, 8, dtype=torch.float64)/length
    delay = 1e10*length
    values = torch.exp(-2j*torch.pi*frequencies*delay)
    slope, residual = _path_group_delay(frequencies, values)
    assert slope.item() == pytest.approx(-delay, rel=1e-12)
    assert residual.item() < 1e-14


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FDTD port preparation requires CUDA")
@pytest.mark.parametrize("length", [1., 1e-6])
def test_small_dc_current_is_inconsistent_at_both_aperture_scales(length):
    circuit = mw.Circuit("dc_driven")
    node = circuit.node("input")
    circuit.add(mw.VoltageSource("V1", node, circuit.ground, 1e-8*length))
    circuit.add(mw.Resistor("R1", node, circuit.ground, 50.))
    circuit.bind_port("feed", positive=node, negative=circuit.ground)
    port = mw.LumpedPort(name="feed", positive=(0.,0.,.005*length), negative=(0.,0.,-.005*length),
                         voltage_path=mw.AxisPath("z"), current_surface=mw.Box(position=(0.,0.,-.0025*length), size=(.015*length,.015*length,0.)), reference_impedance=50.)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02*length,.02*length),)*3),
                     grid=mw.GridSpec.uniform(.005*length), boundary=mw.BoundarySpec.none(),
                     ports=[port], circuits=[circuit], device="cuda")
    with pytest.raises(ValueError, match="nonzero DC port current"):
        mw.Simulation.fdtd(scene, frequency=3e9/length).prepare()


@pytest.mark.parametrize("length", [1., 1e-6])
def test_embedded_delay_requires_relative_real_reference(length):
    from types import SimpleNamespace
    from witwin.maxwell.compiler.networks import _resolve_continuous_model
    from witwin.maxwell.rational import StateSpaceNetwork

    model = StateSpaceNetwork(A=torch.empty((0,0), dtype=torch.float64),
                              B=torch.empty((0,2), dtype=torch.float64),
                              C=torch.empty((2,0), dtype=torch.float64),
                              D=torch.zeros((2,2), dtype=torch.float64),
                              representation="S", port_order=("in","out"))
    block = SimpleNamespace(network=_network(1e-12*length*(1+1e-6j)),
                            delay_seconds=(0.,0.), max_delay_steps=1, fit=False,
                            model=model, port_order=("in","out"))
    with pytest.raises(ValueError, match="real reference impedances"):
        _resolve_continuous_model(block, dt=1e-12*length, device=torch.device("cpu"))


@pytest.mark.parametrize("length", [1., 1e-6])
def test_waveport_column_retains_small_power_wave_at_both_scales(length):
    from types import SimpleNamespace
    from witwin.maxwell.waveport_sweep import _column_scene

    port = mw.WavePort("port", position=(0.,0.,0.), size=(0.,.8*length,.6*length), direction="+", reference_plane=0.,
                       modes=(mw.WaveModeSpec("te", polarization="Ey"),))
    scene = mw.Scene(domain=mw.Domain(bounds=((-length,length),)*3),
                     grid=mw.GridSpec.uniform(.1*length), boundary=mw.BoundarySpec.none(),
                     ports=[port], device="cpu")
    mode = SimpleNamespace(mode_index=0, polarization_axis="y", family="te", conductor_model=None)
    prepared = SimpleNamespace(port=port, compiled=SimpleNamespace(modes=(mode,), electric_plane_index=10),
                               tracking=SimpleNamespace(orientation=torch.ones((1,1,1), dtype=torch.complex128)))
    manifest = SimpleNamespace(frequencies=(1e9/length,), prepared_ports=(prepared,))
    amplitude = 1e-12 * length
    column = _column_scene(scene, manifest, active_port_index=0, active_mode_index=0, frequency_index=0, amplitude=amplitude)
    assert len(column.sources) == 1
    assert column.sources[0].source_time.amplitude == pytest.approx(amplitude, rel=1e-12, abs=0)


@pytest.mark.parametrize("length", [1., 1e-6])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("lowering", ["waveport", "terminal"])
def test_modal_drive_columns_prune_roundoff_relative_to_own_vector(length, dtype, lowering, monkeypatch):
    from types import SimpleNamespace
    from witwin.maxwell import terminal_waveport, waveport_sweep

    scene = mw.Scene(domain=mw.Domain(bounds=((-length, length),) * 3),
                     grid=mw.GridSpec.uniform(.1 * length), boundary=mw.BoundarySpec.none(), device="cpu")
    port = SimpleNamespace(name="P", position=(0., 0., 0.), size=(0., length, length),
                           direction="+", terminal_count=3)
    mode = SimpleNamespace(mode_index=0, polarization_axis="y", family="te", conductor_model=None)
    eps = torch.finfo(dtype).eps
    monkeypatch.setattr(waveport_sweep, "_source_plane", lambda scene, prepared: port.position)
    monkeypatch.setattr(terminal_waveport, "_tag_terminal_mode", lambda *args: None)

    def terminal_source(port, compiled, polarization_axis, *, frequency, position, coefficient, **kwargs):
        return mw.ModeSource(position=position, size=port.size, polarization="Ey",
                             source_time=waveport_sweep._coefficient_source_time(frequency, coefficient))

    monkeypatch.setattr(terminal_waveport, "_port_mode_source", terminal_source)
    for peak in (1., 1e-10, 0.):
        coefficients = torch.tensor([1., 64 * eps, 128 * eps], dtype=dtype) * peak
        prepared = SimpleNamespace(
            port=port, compiled=SimpleNamespace(modes=(mode,) * 3), polarization_axis="y",
            modal_potentials=torch.ones((1, 3)), drive=coefficients.reshape(1, 3, 1),
            tracking=SimpleNamespace(orientation=coefficients.reshape(1, 1, 3)),
        )
        manifest = SimpleNamespace(frequencies=(1e9 / length,), prepared_ports=(prepared,))
        if lowering == "terminal":
            column = terminal_waveport._column_scene(scene, manifest, active_port=0, active_channel=0,
                                                      frequency_index=0, amplitude=1e-12 * length)
        else:
            column = waveport_sweep._column_scene(scene, manifest, active_port_index=0, active_mode_index=0,
                                                  frequency_index=0, amplitude=1e-12 * length)
        assert len(column.sources) == (2 if peak else 0)
