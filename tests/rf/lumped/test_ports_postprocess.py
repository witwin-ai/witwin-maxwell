"""Native true-time field projections, continuous limits and linear gradients."""
from itertools import product

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.scene import prepare_scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Native port hooks require CUDA")
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _port(name, x=0.0, termination=None):
    return mw.LumpedPort(
        name, positive=(x, 0., .005), negative=(x, 0., -.005), voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(position=(x, 0., -.0025), size=(.015, .015, 0.)),
        reference_impedance=50., termination=termination)


def _scene():
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),) * 3), grid=mw.GridSpec.uniform(.005),
                     boundary=mw.BoundarySpec.pml(num_layers=4), device="cuda")
    for name, x, load in (("excited", -.02, None), ("matched", -.01, mw.SeriesRLC(r=50.)),
                          ("rlc", 0., mw.SeriesRLC(r=25., l=.5e-9, c=1e-12)), ("probe", -.015, None)):
        scene.add_port(_port(name, x, load))
    for name, z in (("signal", .01), ("ground", -.01)):
        scene.add_structure(mw.Structure(mw.Box(position=(.025, 0., z), size=(.01, .01, .01)),
                                         mw.Material.pec(), name=name))
    scene.add_port(mw.TerminalPort(
        "terminal", mw.TerminalRef("signal"), mw.TerminalRef("ground"), mw.AxisPath("z"), -.0025,
        current_surface=mw.Box(position=(.025, 0., -.0025), size=(.015, .015, 0.)),
        termination=mw.SeriesRLC(r=50.)))
    return scene


def _record_h_volume(scene):
    grid = prepare_scene(scene)
    for component in COMPONENTS[3:]:
        positions = grid.z_nodes64 if component == "Hz" else grid.z_half64
        for index, position in enumerate(positions):
            scene.add_monitor(mw.PlaneMonitor(f"{component}_{index}", axis="z", position=position, fields=(component,)))


def _fields(result):
    fields = {c: result.tensor(c).movedim(0, -1) if len(result.frequencies) > 1
              else result.tensor(c)[..., None] for c in COMPONENTS[:3]}
    for component in COMPONENTS[3:]:
        positions = result.prepared_scene.z_nodes64 if component == "Hz" else result.prepared_scene.z_half64
        planes = [result.raw_monitor(f"{component}_{i}")["data"] for i in range(len(positions))]
        fields[component] = (torch.stack(planes, dim=-1).movedim(0, -1) if len(result.frequencies) > 1
                             else torch.stack(planes, dim=-1)[..., None])
    return fields


@pytest.fixture(scope="module", params=list(product(("cw", "pulse"), (False, True), ("lumped", "coax"))))
def native(request):
    signal, normalize, geometry = request.param
    if geometry == "coax":
        from tests.support.port_step_graph_bit_identity_scene import _coax_scene
        scene, excited = _coax_scene(), "feed"
    else:
        scene, excited = _scene(), "excited"
    _record_h_volume(scene)
    carrier = 40e9 if geometry == "coax" else 2e9
    frequencies = (carrier,) if signal == "cw" else (1.2 * carrier, .8 * carrier, carrier)
    # Put the pulse after the ramp, then retain a fully decayed tail. This avoids
    # window derivatives in the midpoint and reactive-branch identities.
    source = mw.CW(carrier) if signal == "cw" else mw.GaussianPulse(carrier, fwidth=.25 * carrier, delay=40 / carrier)
    prepared = mw.Simulation.fdtd(
        scene, frequencies=frequencies, excitations=mw.PortExcitation(
            excited, source_time=source, source_impedance=50. if geometry == "coax" else 200.),
        run_time=mw.TimeConfig(time_steps=12000), full_field_dft=True,
        spectral_sampler=mw.SpectralSampler(window="hanning" if signal == "cw" else "ramp",
                                           normalize_source=normalize)).prepare()
    result = prepared.run()
    yield prepared, result, _fields(result), request.param, excited
    del result, prepared
    torch.cuda.empty_cache()


def _relative(actual, expected):
    return float(((actual - expected).abs() / expected.abs()).max())


def test_native_port_phasors(native, tmp_path):
    prepared, result, fields, case, excited = native
    for port in prepared.solver.scene.ports:
        data = result.port(port.name)
        drive = result.port_drive(port.name) if port.name == excited else None
        voltage, current = mw.port_phasors(prepared, port.name, fields, frequencies=data.frequencies,
                                            dt=prepared.solver.dt, drive=drive)
        errors = _relative(voltage, data.voltage), _relative(current, data.current)
        print(f"PROVING {case} {port.name}: V={errors[0]:.9g} I={errors[1]:.9g}", flush=True)
        if data.metadata["port_role"] == "probe" and errors[1] > 1e-5:
            omega = 2 * torch.pi * data.frequencies
            ratio = current / data.current
            for sign in (-1, 1):
                residual = (ratio - torch.exp(sign * .5j * omega * prepared.solver.dt)).abs().max()
                if float(residual) < 1e-5:
                    pytest.fail(f"STOP: native probe observer-origin mismatch, exp({sign:+d} i omega dt/2)")
        assert max(errors) <= 1e-5
        if drive is None:
            with pytest.raises(ValueError, match="no recorded excitation"):
                result.port_drive(port.name)
        else:
            assert isinstance(drive, mw.PortDrive)
            assert drive.source_spectrum.shape == data.frequencies.shape
            torch.testing.assert_close(drive.source_spectrum.abs().square() / (8 * drive.source_impedance),
                                       data.available_power, rtol=0, atol=0)
    if case == ("cw", False, "lumped"):
        path = tmp_path / "drive.pt"
        result.save(path)
        restored = mw.Result.load(path, scene=result.scene, map_location="cuda:0")
        assert torch.equal(restored.port_drive(excited).source_spectrum, result.port_drive(excited).source_spectrum)
        assert torch.equal(restored.port_drive(excited).source_impedance, result.port_drive(excited).source_impedance)


def _independent_integrals(grid, port, fields):
    # This fixture's oriented voltage path is z; its loop lies on z-half.
    nodes = [np.asarray(getattr(grid, f"{axis}_nodes64")) for axis in "xyz"]
    halves = [(axis[:-1] + axis[1:]) / 2 for axis in nodes]
    ix, iy = (int(np.abs(axis - point).argmin()) for axis, point in zip(nodes[:2], port.negative[:2]))
    start, stop = (int(np.abs(nodes[2] - point[2]).argmin()) for point in (port.negative, port.positive))
    voltage = sum(fields["Ez"][ix, iy, k] * (nodes[2][k + 1] - nodes[2][k]) for k in range(start, stop))
    center = [float(x) for x in port.current_surface.position]
    size = [float(x) for x in port.current_surface.size]
    xl, xh = (int(np.abs(halves[0] - (center[0] + sign * size[0] / 2)).argmin()) for sign in (-1, 1))
    yl, yh = (int(np.abs(halves[1] - (center[1] + sign * size[1] / 2)).argmin()) for sign in (-1, 1))
    kz = int(np.abs(halves[2] - center[2]).argmin())
    bottom_top = sum((fields["Hx"][i, yl, kz] - fields["Hx"][i, yh, kz]) *
                     (nodes[0][i + 1] - nodes[0][i - 1]) / 2 for i in range(xl + 1, xh + 1))
    right_left = sum((fields["Hy"][xh, j, kz] - fields["Hy"][xl, j, kz]) *
                     (nodes[1][j + 1] - nodes[1][j - 1]) / 2 for j in range(yl + 1, yh + 1))
    # RF current enters the field: with V=+integral(E), it is the negative
    # right-hand circulation about the negative-to-positive terminal arrow.
    return voltage, -(bottom_top + right_left)


def test_continuous_phasors_on_actual_fdfd_fields():
    frequency = 2e9
    load = mw.SeriesRLC(r=25., l=.5e-9, c=1e-12)
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.02, .02),) * 3), grid=mw.GridSpec.uniform(.005),
                     boundary=mw.BoundarySpec.pec(), ports=(_port("probe"), _port("load", termination=load)),
                     sources=(mw.PointDipole(position=(-.01, -.005, 0.), polarization="Ez", width=.005,
                                             source_time=mw.CW(frequency)),), device="cuda")
    prepared = mw.Simulation.fdfd(scene, frequency=frequency,
                                 solver=mw.GMRES(solver_type="direct", precision="double", tol=1e-12)).prepare()
    result = prepared.run()
    assert result.solver_stats["converged"]
    fields = {c: result.tensor(c) for c in COMPONENTS[:3]}
    grid = prepared.solver.scene
    omega = 2 * torch.pi * frequency
    # Physical H from Faraday's law on the solved FDFD Yee E array, in vacuum.
    for own, component in enumerate(COMPONENTS[3:]):
        u, v = (own + 1) % 3, (own + 2) % 3
        widths_u = torch.as_tensor(np.diff(getattr(grid, f'{"xyz"[u]}_nodes64')), device="cuda")
        widths_v = torch.as_tensor(np.diff(getattr(grid, f'{"xyz"[v]}_nodes64')), device="cuda")
        shape_u = tuple(-1 if axis == u else 1 for axis in range(3))
        shape_v = tuple(-1 if axis == v else 1 for axis in range(3))
        fields[component] = (fields[COMPONENTS[v]].diff(dim=u) / widths_u.reshape(shape_u) -
                             fields[COMPONENTS[u]].diff(dim=v) / widths_v.reshape(shape_v)) / (1j * omega * 1.25663706212e-6)
    expected_v, expected_i = _independent_integrals(grid, scene.ports[0], fields)
    voltage, current = mw.port_phasors(prepared, "probe", fields, frequencies=(frequency,), dt=None)
    assert _relative(voltage, expected_v) <= 1e-12
    assert _relative(current, expected_i) <= 1e-12
    assert _relative(mw.port_voltage(prepared, "probe", fields), expected_v) <= 1e-12
    voltage, current = mw.port_phasors(prepared, "load", fields, frequencies=(frequency,), dt=None)
    impedance = load.impedance(torch.tensor([omega], dtype=torch.float64)).to(voltage.device)
    assert _relative(voltage, expected_v) <= 1e-12
    assert _relative(current, -expected_v / impedance.conj()) <= 1e-12
    # The override must replace the scene load, while keeping continuous V exact.
    override = mw.SeriesRLC(r=80.)
    v_override, i_override = mw.port_phasors(prepared, "load", fields, frequencies=(frequency,),
                                           dt=None, termination=override)
    assert _relative(v_override, expected_v) <= 1e-12
    assert _relative(i_override, -expected_v / 80.) <= 1e-12
    drive = mw.PortDrive(torch.tensor([1. + .2j], device="cuda", dtype=torch.complex128),
                         torch.tensor(50., device="cuda"))
    v_drive, i_drive = mw.port_phasors(prepared, "probe", fields, frequencies=(frequency,), dt=None, drive=drive)
    assert _relative(v_drive, expected_v) <= 1e-12
    assert _relative(i_drive, (drive.source_spectrum - expected_v) / drive.source_impedance) <= 1e-12
    print(f"CONTINUOUS independent probe V={_relative(mw.port_voltage(prepared, 'probe', fields), expected_v):.9g} "
          f"I={_relative(mw.port_phasors(prepared, 'probe', fields, frequencies=(frequency,), dt=None)[1], expected_i):.9g}; "
          f"RLC I={_relative(current, -expected_v / impedance.conj()):.9g}", flush=True)


def test_field_gradient_equals_analytic_line_functional():
    prepared = mw.Simulation.fdtd(_scene(), frequency=2e9, excitations=mw.PortExcitation("excited"),
                                  run_time=mw.TimeConfig(time_steps=1)).prepare()
    field = torch.randn_like(prepared.solver.Ez, dtype=torch.complex128, requires_grad=True)
    drive = mw.PortDrive(torch.tensor([1. + .2j], device="cuda"), torch.tensor(50., device="cuda"))
    voltage, current = mw.port_phasors(prepared, "excited", {"Ez": field}, frequencies=(2e9,),
                                      dt=prepared.solver.dt, drive=drive)
    gradient, = torch.autograd.grad(voltage.abs().square().sum(), field, retain_graph=True)
    grid = prepared.solver.scene
    nodes = [np.asarray(getattr(grid, f'{axis}_nodes64')) for axis in 'xyz']
    ix, iy = (int(np.abs(axis - point).argmin()) for axis, point in zip(nodes[:2], (-.02, 0.)))
    lo, hi = (int(np.abs(nodes[2] - point).argmin()) for point in (-.005, .005))
    lengths = torch.as_tensor(np.diff(nodes[2])[lo:hi], device=field.device)
    factor = torch.cos(torch.tensor(torch.pi * 2e9 * prepared.solver.dt, dtype=torch.float64, device=field.device))
    analytic_v = (field[ix, iy, lo:hi] * lengths).sum() * factor
    analytic = torch.zeros_like(field)
    analytic[ix, iy, lo:hi] = 2 * analytic_v * factor * lengths
    torch.testing.assert_close(gradient, analytic, rtol=1e-12, atol=0)
    current_gradient, = torch.autograd.grad(current.real.sum(), field)
    expected_current = torch.zeros_like(field)
    expected_current[ix, iy, lo:hi] = -factor * lengths / 50
    torch.testing.assert_close(current_gradient, expected_current, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="requires a drive"):
        mw.port_phasors(prepared, "excited", {"Ez": field}, frequencies=(2e9,), dt=None)


def _unsupported(kind):
    from tests.support.port_step_graph_bit_identity_scene import _wire_gap_scene, _wire_node_scene

    if kind == "sheet":
        from tests.materials.thin_sheet.test_thin_sheet_feeds import _simulation
        return _simulation("pec", device="cuda", steps=1).prepare(), "feed"
    if kind in ("wire", "wire_terminal"):
        scene = _wire_gap_scene() if kind == "wire" else _wire_node_scene()
    else:
        scene = mw.Scene(domain=mw.Domain(bounds=((-0.04, .04),) * 3), grid=mw.GridSpec.uniform(.005),
                         boundary=mw.BoundarySpec.none(), ports=(_port("feed"),), device="cuda")
        if kind == "circuit":
            circuit = mw.Circuit("load")
            node = circuit.node("input")
            circuit.add(mw.Resistor("R", node, circuit.ground, 50.))
            circuit.bind_port("feed", positive=node, negative=circuit.ground)
            scene = scene.clone(circuits=(circuit,))
        elif kind == "network":
            band = torch.tensor([1e9, 2e9, 3e9], dtype=torch.float64)
            model = mw.StateSpaceNetwork(A=torch.tensor([[-1e9]], dtype=torch.float64),
                                         B=torch.zeros((1, 1), dtype=torch.float64),
                                         C=torch.zeros((1, 1), dtype=torch.float64),
                                         D=torch.tensor([[.02]], dtype=torch.float64),
                                         representation="Y", port_order=("load",))
            network = mw.NetworkData.from_y(frequencies=band, y=model.evaluate(band), z0=50., port_names=("load",))
            scene = scene.clone(networks=(mw.NetworkBlock("load", network=network, model=model, fit=False,
                                                           connections={"load": "feed"}),))
        elif kind == "mode":
            scene = scene.clone(ports=(mw.ModePort("feed", size=(.02, .02, 0.)),))
        elif kind == "wave":
            scene = scene.clone(ports=(mw.WavePort("feed", position=(0., 0., 0.), size=(.02, .02, 0.),
                                                  direction="+", reference_plane=0., modes=(mw.WaveModeSpec("te"),)),))
        elif kind == "terminal_wave":
            from tests.rf.terminal_wave.test_terminal_wave_port_api import _microstrip_scene
            scene = _microstrip_scene().clone(device="cuda")
            return mw.Simulation.fdfd(scene, frequency=2e9).prepare(), "P"
    if kind in ("mode", "wave"):
        return mw.Simulation.fdfd(scene, frequency=2e9).prepare(), "feed"
    return mw.Simulation.fdtd(scene, frequency=2e9, run_time=mw.TimeConfig(time_steps=1)).prepare(), "feed"


@pytest.mark.parametrize("kind, message", [("wire", "Wire-bound"), ("wire_terminal", "Wire-bound"),
                                           ("circuit", "Circuit-bound"), ("network", "Network-bound"),
                                           ("sheet", "Sheet-gap"), ("mode", "modal projections"),
                                           ("wave", "modal projections"), ("terminal_wave", "modal projections")])
def test_unsupported_ports_require_more_than_yee_fields(kind, message):
    prepared, name = _unsupported(kind)
    with pytest.raises(NotImplementedError, match=message):
        mw.port_voltage(prepared, name, {})
    with pytest.raises(NotImplementedError, match=message):
        mw.port_phasors(prepared, name, {}, frequencies=(2e9,), dt=None)
