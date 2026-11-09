"""Small split-bank contracts; full physics runs live in the acceptance scripts."""
from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state, restore_checkpoint_state
from witwin.maxwell.fdtd.adjoint import _FDTDGradientBridge
from witwin.maxwell.fdtd.lumped import (
    LUMPED_STATE_NAMES, apply_lumped_runtime, prepare_imaginary_lumped_state,
    prepare_lumped_runtime,
)
from witwin.maxwell.fdtd.ports import (
    PortDFTAccumulator, _validate_bloch_lumped_object, apply_port_runtimes,
    finalize_port_data, prepare_port_runtimes, pullback_port_runtimes, replay_port_runtimes,
)


@pytest.mark.parametrize("model", [
    mw.Resistor("r", (1., 0., 0.), (0., 0., 0.), 2.),
    mw.Capacitor("c", (1., 0., 0.), (0., 0., 0.), .3),
    mw.Inductor("l", (1., 0., 0.), (0., 0., 0.), .4),
    mw.SeriesRLC(r=2., l=.4, c=.3), mw.ParallelRLC(r=2., l=.4, c=.3),
])
def test_split_banks_match_independent_lti_updates_and_checkpoint(model):
    geometry = SimpleNamespace(port_name="load", voltage_indices=torch.tensor([[0, 0, 0]]),
                               voltage_weights=torch.tensor([1.], dtype=torch.float64))
    def prepare():
        return prepare_lumped_runtime(geometry, dt=.1, eps_edge=torch.ones((2, 2, 2), dtype=torch.float64),
                                      yee_control_volume=1., termination=model, diagnostics=True)
    real, imag, split = prepare(), prepare(), prepare()
    prepare_imaginary_lumped_state(split, ())
    solver = SimpleNamespace(complex_fields_enabled=True, _lumped_element_runtimes=((split, "Ex"),))
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(solver, name, torch.zeros((2, 2, 2), dtype=torch.float64))
        setattr(solver, name + "_imag", torch.zeros((2, 2, 2), dtype=torch.float64))
    e_real, e_imag = solver.Ex.clone(), solver.Ex_imag.clone()
    for step in range(6):
        for field in (e_real, solver.Ex):
            field.add_(step * .2)
        for field in (e_imag, solver.Ex_imag):
            field.add_(-step * .3)
        apply_lumped_runtime(real, e_real)
        apply_lumped_runtime(imag, e_imag)
        apply_port_runtimes(solver)
        assert torch.equal(e_real, solver.Ex)
        assert torch.equal(e_imag, solver.Ex_imag)
        for name in LUMPED_STATE_NAMES:
            assert torch.equal(getattr(split, name), getattr(real, name))
            assert torch.equal(getattr(split, name + "_imag"), getattr(imag, name))
    checkpoint = capture_checkpoint_state(solver, 6)
    for name in LUMPED_STATE_NAMES:
        getattr(split, name).fill_(99.)
        getattr(split, name + "_imag").fill_(99.)
    restore_checkpoint_state(solver, checkpoint)
    for name in LUMPED_STATE_NAMES:
        assert torch.equal(getattr(split, name), getattr(real, name))
        assert torch.equal(getattr(split, name + "_imag"), getattr(imag, name))
    with pytest.raises(NotImplementedError, match="adjoint/pullback.*discrete transpose.*imaginary state bank"):
        replay_port_runtimes(solver, {}, {}, time_value=0.)

    # A quadrature sample makes the auxiliary DFT's sign observable in both V/I.
    frequency = torch.tensor([1.], dtype=torch.float64)
    accumulators = [PortDFTAccumulator(frequency) for _ in range(2)]
    for accumulator, v, i in zip(accumulators, (2., 3.), (-4., -5.)):
        accumulator.accumulate_precomputed(torch.tensor(v), torch.tensor(i),
            voltage_kernel=torch.tensor([1. + 2.j]), current_kernel=torch.tensor([1. + 2.j]),
            window_weight=torch.ones(1))
    port = SimpleNamespace(name="load", reference_impedance=50., reference_plane=0.)
    runtime = SimpleNamespace(accumulator=accumulators[0], accumulator_imag=accumulators[1],
        drive_accumulator=None, port=port, frequencies=frequency, lumped=split,
        wire_provider=None, sheet_feed=None, geometry=SimpleNamespace(direction=1, axis="x"),
        circuit_port=None, sample_index=1, window_type="none")
    solver._port_runtimes = (runtime,)
    solver._time_plan = {"spectral_phasor_factor": 2.}
    solver._spectral_integration_dt = None
    solver._spectral_reference_frequency = 1.
    solver._normalize_source = False
    data = finalize_port_data(solver)["load"]
    assert torch.equal(data.voltage, torch.tensor([-8. + 14.j], dtype=torch.complex128))
    assert torch.equal(data.current, torch.tensor([12. - 26.j], dtype=torch.complex128))


def test_bloch_passive_support_guards():
    solver = SimpleNamespace(scene=SimpleNamespace(
        boundary=mw.BoundarySpec.bloch((1., 0., 0.)),
        domain=mw.Domain(bounds=((-1., 1.),) * 3), structures=()))
    def port(**kwargs):
        options = dict(name="load", negative=(-.2, 0., 0.), positive=(.2, 0., 0.),
            voltage_path=mw.AxisPath("x"),
            current_surface=mw.Box(position=(0., 0., 0.), size=(0., .4, .4)),
            termination=mw.SeriesRLC(r=50.))
        options.update(kwargs)
        return mw.LumpedPort(**options)
    interior = port()
    _validate_bloch_lumped_object(solver, interior, interior.termination)
    with pytest.raises(NotImplementedError, match="Driven lumped port.*v2"):
        _validate_bloch_lumped_object(solver, interior, None, mw.PortExcitation("load"))
    seam_path = port(negative=(-1., 0., 0.))
    with pytest.raises(NotImplementedError, match="'load' voltage path.*x Bloch seam"):
        _validate_bloch_lumped_object(solver, seam_path, seam_path.termination)
    seam_surface = port(current_surface=mw.Box(position=(0., 0., 0.), size=(0., 2., .4)))
    with pytest.raises(NotImplementedError, match="'load' current surface.*y Bloch seam"):
        _validate_bloch_lumped_object(solver, seam_surface, seam_surface.termination)


@pytest.mark.parametrize("runtime_name", ["_port_runtimes", "_lumped_element_runtimes"])
def test_bloch_lumped_gradient_bridge_rejects_complex_adjoint(runtime_name):
    solver = SimpleNamespace(complex_fields_enabled=True, **{runtime_name: (object(),)})
    bridge = object.__new__(_FDTDGradientBridge)
    with pytest.raises(NotImplementedError, match="Complex-field lumped FDTD.*adjoint/pullback.*v1"):
        bridge._validate_supported_configuration(solver)


def test_bloch_lumped_pullback_rejects_complex_adjoint():
    solver = SimpleNamespace(complex_fields_enabled=True)
    with pytest.raises(NotImplementedError, match="Complex-field lumped adjoint/pullback.*discrete transpose.*imaginary state bank"):
        pullback_port_runtimes(solver, (), {}, port_sample_adjoints={},
                               eps_by_field={}, time_value=0.)


@pytest.mark.parametrize("unsupported", ["object", "termination"])
def test_bloch_lumped_rejects_non_lti_coupling(unsupported):
    solver = SimpleNamespace(scene=SimpleNamespace(structures=()))
    load = mw.LumpedPort("load", negative=(-.2, 0., 0.), positive=(.2, 0., 0.),
                         voltage_path=mw.AxisPath("x"), termination=mw.SeriesRLC(r=50.),
                         current_surface=mw.Box(position=(0., 0., 0.), size=(0., .4, .4), device="cpu"))
    diode = mw.Diode("diode", mw.CircuitNode("a"), mw.CircuitNode("0"), saturation_current=1e-12)
    # Nonlinear devices cannot use the independent passive LTI state banks,
    # either directly or through a runtime termination override.
    obj, termination = ((diode, None) if unsupported == "object" else (load, diode))
    with pytest.raises(NotImplementedError, match="Complex lumped coupling supports only passive LTI R/L/C"):
        _validate_bloch_lumped_object(solver, obj, termination)


@pytest.mark.parametrize("coupling", ["circuits", "networks"])
def test_bloch_lumped_rejects_circuits_and_embedded_networks(coupling):
    scene = SimpleNamespace(ports=(), lumped_elements=(), structures=(), circuits=(), networks=())
    block = SimpleNamespace(bindings=(), connected_port_names=(), name="block")
    setattr(scene, coupling, (block,))
    solver = SimpleNamespace(device="cpu", complex_fields_enabled=True, scene=scene)
    with pytest.raises(NotImplementedError, match="Circuits/MNA and embedded networks under Bloch boundaries"):
        prepare_port_runtimes(solver, (1e9,))


def test_bloch_lumped_rejects_copper_terminal_on_cpu():
    foil = mw.Structure(mw.Box(position=(0., 0., 0.), size=(0., .4, .4), device="cpu"),
                       mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9)), name="foil")
    solver = SimpleNamespace(scene=SimpleNamespace(structures=(foil,)))
    load = mw.LumpedPort("load", negative=(0., 0., 0.), positive=(.2, 0., 0.),
                         voltage_path=mw.AxisPath("x"), termination=mw.SeriesRLC(r=50.),
                         current_surface=mw.Box(position=(.1, 0., 0.), size=(0., .4, .4), device="cpu"))
    with pytest.raises(NotImplementedError, match="'load'.*terminal on CopperFoil 'foil'.*imaginary bank.*two-sided"):
        _validate_bloch_lumped_object(solver, load, load.termination)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Copper terminal preparation requires CUDA.")
def test_bloch_copper_port_rejection_and_passive_support_guards():
    # The voltage row is normal to these two sheets, so it never overlaps a
    # tangential split trace. Terminal contact itself must reject this coupling.
    foil = mw.CopperFoil(thickness=2e-6, frequency_range=(.5e9, 2e9))
    structures = tuple(mw.Structure(
        mw.Box(position=torch.tensor((0., 0., z), dtype=torch.float64),
               size=torch.tensor((.02, .02, 0.), dtype=torch.float64)),
        foil, name=name) for name, z in (("lower", -.005), ("upper", .005)))
    load = mw.LumpedPort("foil_load", negative=(0., 0., -.005), positive=(0., 0., .005),
        voltage_path=mw.AxisPath("z"), termination=mw.SeriesRLC(r=50.),
        current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)))
    scene = mw.Scene(domain=mw.Domain(bounds=((- .03, .03),) * 3),
        grid=mw.GridSpec.uniform(.005), device="cuda:0", structures=structures, ports=(load,),
        boundary=mw.BoundarySpec.faces(x="bloch", y="bloch", z="pml", num_layers=2,
                                      bloch_wavevector=(0., 5., 0.)),
        sources=(mw.PointDipole(name="drive", position=(0., 0., .015), polarization="Ex",
                               source_time=mw.GaussianPulse(frequency=1e9, fwidth=.5e9)),))
    with pytest.raises(NotImplementedError, match="'foil_load'.*terminal on CopperFoil.*imaginary bank.*two-sided"):
        mw.Simulation.fdtd(scene, frequency=1e9, run_time=mw.TimeConfig(time_steps=20)).prepare()
