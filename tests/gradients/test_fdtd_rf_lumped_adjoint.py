from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.ports import CompiledPortGeometry
from witwin.maxwell.fdtd.adjoint.core import _prepare_forward_pack
from witwin.maxwell.fdtd.adjoint.bridge import _FDTDGradientBridge
from witwin.maxwell.fdtd.adjoint.seeds import (
    _build_output_seeds,
    _schedule_to_tensor_pack,
)
from witwin.maxwell.fdtd.checkpoint import capture_checkpoint_state, lumped_state_name
from witwin.maxwell.fdtd.boundary import BOUNDARY_NONE
from witwin.maxwell.fdtd.lumped import (
    apply_lumped_runtime,
    prepare_lumped_runtime,
    pullback_lumped_runtime,
    replay_lumped_runtime,
)
from witwin.maxwell.fdtd.ports import PortDFTAccumulator
from witwin.maxwell.lumped import SeriesRLC
from witwin.maxwell.network import PortData


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires one CUDA device",
)


def _geometry() -> CompiledPortGeometry:
    device = torch.device("cuda")
    return CompiledPortGeometry(
        port_name="rf",
        axis="x",
        direction=1,
        voltage_component="Ex",
        voltage_indices=torch.tensor(
            ((0, 0, 0), (1, 0, 0)),
            device=device,
            dtype=torch.int64,
        ),
        voltage_weights=torch.tensor(
            (0.4, 0.6),
            device=device,
            dtype=torch.float64,
        ),
        current_components=(),
        current_indices=(),
        current_weights=(),
        reference_impedance=50.0,
    )


def _runtime(*, eps, r=None, l=None, c=None):  # noqa: E741 - circuit notation
    volume = torch.tensor(
        (
            ((0.5, 0.7), (0.6, 0.8)),
            ((0.25, 0.9), (0.55, 0.65)),
            ((0.4, 0.75), (0.45, 0.85)),
        ),
        device="cuda",
        dtype=torch.float64,
    )
    return prepare_lumped_runtime(
        _geometry(),
        dt=torch.tensor(0.2, device="cuda", dtype=torch.float64),
        eps_edge=eps,
        yee_control_volume=volume,
        termination=SeriesRLC(r=r, l=l, c=c),
    )


@pytest.mark.parametrize(
    ("component", "value"),
    (("r", 0.8), ("l", 0.35), ("c", 0.7)),
)
def test_exact_local_series_pullback_matches_cuda_autograd(component, value):
    eps = torch.tensor(
        (
            ((2.0, 1.2), (1.5, 1.1)),
            ((3.0, 1.4), (1.6, 1.3)),
            ((2.5, 1.8), (1.7, 1.9)),
        ),
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    parameter = torch.tensor(
        value,
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    kwargs = {"r": None, "l": None, "c": None}
    kwargs[component] = parameter
    runtime = _runtime(eps=eps, **kwargs)
    field = torch.tensor(
        (
            ((0.8, 0.1), (-0.2, 0.05)),
            ((-0.35, 0.02), (0.15, -0.1)),
            ((0.4, -0.25), (0.3, 0.12)),
        ),
        device="cuda",
        dtype=torch.float64,
        requires_grad=True,
    )
    old_i = torch.tensor(0.13, device="cuda", dtype=torch.float64, requires_grad=True)
    old_vc = torch.tensor(-0.17, device="cuda", dtype=torch.float64, requires_grad=True)
    drive = torch.tensor(0.21, device="cuda", dtype=torch.float64, requires_grad=True)
    corrected, next_i, next_vc, trace = replay_lumped_runtime(
        runtime,
        field,
        inductor_current=old_i,
        capacitor_voltage=old_vc,
        drive=drive,
        field_name="Ex",
        kind="element",
        index=0,
    )
    bar_field = torch.linspace(
        -0.4,
        0.5,
        field.numel(),
        device="cuda",
        dtype=torch.float64,
    ).reshape_as(field)
    bar_i = torch.tensor(0.31, device="cuda", dtype=torch.float64)
    bar_vc = torch.tensor(-0.27, device="cuda", dtype=torch.float64)
    bar_voltage = torch.tensor(0.19, device="cuda", dtype=torch.float64)
    bar_network_current = torch.tensor(-0.23, device="cuda", dtype=torch.float64)
    objective = (
        torch.sum(corrected * bar_field)
        + next_i * bar_i
        + next_vc * bar_vc
        + trace.voltage_midpoint * bar_voltage
        - trace.branch_current * bar_network_current
    )
    expected = torch.autograd.grad(
        objective,
        (field, old_i, old_vc, eps, parameter, drive),
    )

    actual = pullback_lumped_runtime(
        trace,
        bar_field,
        inductor_current_adjoint=bar_i,
        capacitor_voltage_adjoint=bar_vc,
        voltage_sample_adjoint=bar_voltage,
        network_current_sample_adjoint=bar_network_current,
        eps_edge=eps,
    )
    parameter_gradient = {
        "r": actual.grad_resistance,
        "l": actual.grad_inductance,
        "c": actual.grad_capacitance,
    }[component]
    for found, reference in zip(
        (
            actual.field_adjoint,
            actual.inductor_current_adjoint,
            actual.capacitor_voltage_adjoint,
            actual.grad_eps,
            parameter_gradient,
            actual.grad_drive,
        ),
        expected,
    ):
        torch.testing.assert_close(found, reference, rtol=2.0e-13, atol=2.0e-13)


def test_lumped_checkpoint_and_three_step_replay_match_forward_state():
    eps = torch.ones((3, 2, 2), device="cuda", dtype=torch.float64)
    runtime = _runtime(
        eps=eps,
        r=torch.tensor(0.8, device="cuda", dtype=torch.float64),
        l=torch.tensor(0.35, device="cuda", dtype=torch.float64),
        c=torch.tensor(0.7, device="cuda", dtype=torch.float64),
    )
    initial = torch.linspace(
        -0.3,
        0.4,
        eps.numel(),
        device="cuda",
        dtype=torch.float64,
    ).reshape_as(eps)
    port_runtime = SimpleNamespace(lumped=runtime, field_name="Ex")
    solver = SimpleNamespace(
        Ex=initial.clone(),
        Ey=torch.zeros_like(initial),
        Ez=torch.zeros_like(initial),
        Hx=torch.zeros_like(initial),
        Hy=torch.zeros_like(initial),
        Hz=torch.zeros_like(initial),
        complex_fields_enabled=False,
        uses_cpml=False,
        tfsf_enabled=False,
        dispersive_enabled=False,
        magnetic_dispersive_enabled=False,
        _port_runtimes=(port_runtime,),
        _lumped_element_runtimes=(),
    )
    checkpoint = capture_checkpoint_state(solver, step=0)
    inductor_name = lumped_state_name("port", 0, "inductor_current")
    capacitor_name = lumped_state_name("port", 0, "capacitor_voltage")
    assert checkpoint.schema.lumped_state_names == (inductor_name, capacitor_name)

    forward_field = initial.clone()
    drives = (
        torch.tensor(0.1, device="cuda", dtype=torch.float64),
        torch.tensor(-0.2, device="cuda", dtype=torch.float64),
        torch.tensor(0.05, device="cuda", dtype=torch.float64),
    )
    replay_field = checkpoint.tensors["Ex"]
    replay_i = checkpoint.tensors[inductor_name]
    replay_vc = checkpoint.tensors[capacitor_name]
    for drive in drives:
        apply_lumped_runtime(runtime, forward_field, thevenin_voltage=drive)
        replay_field, replay_i, replay_vc, _trace = replay_lumped_runtime(
            runtime,
            replay_field,
            inductor_current=replay_i,
            capacitor_voltage=replay_vc,
            drive=drive,
            field_name="Ex",
            kind="port",
            index=0,
        )
    torch.testing.assert_close(replay_field, forward_field, rtol=0.0, atol=0.0)
    torch.testing.assert_close(replay_i, runtime.inductor_current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(replay_vc, runtime.capacitor_voltage, rtol=0.0, atol=0.0)


def test_zero_internal_lc_flags_are_mirrored_and_open_resistance_is_rejected():
    eps = torch.ones((3, 2, 2), device="cuda", dtype=torch.float64)
    zero_runtime = prepare_lumped_runtime(
        _geometry(),
        dt=torch.tensor(0.2, device="cuda", dtype=torch.float64),
        eps_edge=eps,
        yee_control_volume=torch.ones_like(eps),
        termination=SimpleNamespace(
            kind="series_rlc",
            r=torch.tensor(0.8, device="cuda", dtype=torch.float64),
            l=torch.tensor(0.0, device="cuda", dtype=torch.float64),
            c=torch.tensor(0.0, device="cuda", dtype=torch.float64),
        ),
    )
    assert not zero_runtime.inductance_active
    assert not zero_runtime.capacitance_active

    field = torch.ones_like(eps)
    corrected, next_i, next_vc, _trace = replay_lumped_runtime(
        zero_runtime,
        field,
        inductor_current=torch.zeros((), device="cuda", dtype=torch.float64),
        capacitor_voltage=torch.zeros((), device="cuda", dtype=torch.float64),
        drive=torch.zeros((), device="cuda", dtype=torch.float64),
        field_name="Ex",
        kind="port",
        index=0,
    )
    expected = field.clone()
    apply_lumped_runtime(zero_runtime, expected)
    torch.testing.assert_close(corrected, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(next_i, zero_runtime.inductor_current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(next_vc, zero_runtime.capacitor_voltage, rtol=0.0, atol=0.0)

    open_runtime = prepare_lumped_runtime(
        _geometry(),
        dt=torch.tensor(0.2, device="cuda", dtype=torch.float64),
        eps_edge=eps,
        yee_control_volume=torch.ones_like(eps),
        resistance=torch.tensor(torch.inf, device="cuda", dtype=torch.float64),
    )
    port_runtime = SimpleNamespace(
        port=SimpleNamespace(name="open_port"),
        lumped=open_runtime,
    )
    solver = SimpleNamespace(
        scene=SimpleNamespace(structures=()),
        _full_aniso_cpml_overlap=False,
        boundary_x_low_code=BOUNDARY_NONE,
        boundary_x_high_code=BOUNDARY_NONE,
        boundary_y_low_code=BOUNDARY_NONE,
        boundary_y_high_code=BOUNDARY_NONE,
        boundary_z_low_code=BOUNDARY_NONE,
        boundary_z_high_code=BOUNDARY_NONE,
        complex_fields_enabled=False,
        conductive_enabled=False,
        _port_runtimes=(port_runtime,),
        _lumped_element_runtimes=(),
    )
    bridge = object.__new__(_FDTDGradientBridge)
    with pytest.raises(NotImplementedError, match="open internal series resistance"):
        bridge._validate_supported_configuration(solver)


def test_port_voltage_current_and_available_power_dft_seeds_are_exact():
    frequencies = torch.tensor((0.3, 0.55), device="cuda", dtype=torch.float64)
    voltage_samples = tuple(
        torch.tensor(value, device="cuda", dtype=torch.float64, requires_grad=True)
        for value in (0.2, -0.1, 0.35, 0.15)
    )
    current_samples = tuple(
        torch.tensor(value, device="cuda", dtype=torch.float64, requires_grad=True)
        for value in (-0.05, 0.12, 0.08, -0.03)
    )
    drive_samples = tuple(
        torch.tensor(value, device="cuda", dtype=torch.float64, requires_grad=True)
        for value in (0.7, 0.1, -0.4, 0.2)
    )
    accumulator = PortDFTAccumulator(frequencies)
    drive_accumulator = PortDFTAccumulator(frequencies)
    dt = torch.tensor(0.2, device="cuda", dtype=torch.float64)
    weights = torch.ones((4, 2), device="cuda", dtype=torch.float64)
    for index, (voltage, current, drive) in enumerate(
        zip(voltage_samples, current_samples, drive_samples)
    ):
        sample_time = (index + 0.5) * dt
        accumulator.accumulate(
            voltage,
            current,
            electric_sample_time=sample_time,
            magnetic_sample_time=sample_time,
        )
        drive_accumulator.accumulate(
            drive,
            torch.zeros_like(drive),
            electric_sample_time=sample_time,
            magnetic_sample_time=sample_time,
        )
    voltage, current = accumulator.phasors(normalization="peak")
    source_voltage, _ = drive_accumulator.phasors(normalization="peak")
    resistance = torch.tensor(2.5, device="cuda", dtype=torch.float64)
    available_power = source_voltage.abs().square() / (8.0 * resistance)
    data = PortData(
        port_name="rf",
        frequencies=frequencies,
        voltage=voltage,
        current=current,
        z0=50.0,
        available_power=available_power,
    )
    pack = _prepare_forward_pack({"ports": {"rf": data}})
    grad_voltage = torch.tensor(
        (0.4 - 0.2j, -0.3 + 0.1j),
        device="cuda",
        dtype=torch.complex128,
    )
    grad_current = torch.tensor(
        (-0.1 + 0.25j, 0.2 - 0.35j),
        device="cuda",
        dtype=torch.complex128,
    )
    grad_power = torch.tensor((0.7, -0.45), device="cuda", dtype=torch.float64)
    objective = (
        torch.real(torch.sum(torch.conj(grad_voltage) * voltage))
        + torch.real(torch.sum(torch.conj(grad_current) * current))
        + torch.sum(grad_power * available_power)
    )
    expected = torch.autograd.grad(
        objective,
        voltage_samples + current_samples + drive_samples,
    )

    port_runtime = SimpleNamespace(
        port=SimpleNamespace(name="rf"),
        frequencies=frequencies,
        window_weights=weights,
        accumulator=accumulator,
        drive_accumulator=drive_accumulator,
        lumped=SimpleNamespace(dt=dt, resistance=resistance),
    )
    solver = SimpleNamespace(
        scene=None,
        complex_fields_enabled=False,
        _normalize_source=False,
        _source_time=None,
        observers_enabled=False,
        observers=(),
        _dft_entries=(),
        _observer_spectral_entries=(),
        _point_observer_groups={},
        _plane_observer_groups={},
        _port_runtimes=(port_runtime,),
    )
    empty_schedule = _schedule_to_tensor_pack((), device="cuda", dtype=torch.float64)
    seed_runtime = _build_output_seeds(
        solver,
        pack,
        (grad_voltage, grad_current, grad_power),
        dft_schedule=empty_schedule,
        observer_schedule=empty_schedule,
    )
    batch = seed_runtime.port_batches[0]
    actual = tuple(batch.voltage_samples) + tuple(batch.current_samples) + tuple(batch.drive_samples)
    for found, reference in zip(actual, expected):
        torch.testing.assert_close(found, reference, rtol=2.0e-13, atol=2.0e-13)


def _physical_port(*, termination=None, reference_impedance=50.0):
    return mw.LumpedPort(
        name="feed",
        positive=(0.0, 0.0, 0.005),
        negative=(0.0, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(0.0, 0.0, -0.0025),
            size=(0.015, 0.015, 0.0),
        ),
        reference_impedance=reference_impedance,
        termination=termination,
    )


def _physical_scene(*, termination=None, reference_impedance=50.0):
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(
            _physical_port(
                termination=termination,
                reference_impedance=reference_impedance,
            ),
        ),
        device="cuda",
    )
    return scene


def _passive_port_objective(scene):
    result = mw.Simulation.fdtd(
        scene,
        frequency=3.0e9,
        run_time=mw.TimeConfig(time_steps=32),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    voltage = result.port("feed").voltage
    return voltage.abs().square().sum(), result


def _assert_three_step_central_difference(parameter, objective, steps):
    loss, result = objective()
    loss.backward()
    adjoint = float(parameter.grad.detach())
    assert abs(adjoint) > 1.0e-12
    center = float(parameter.detach())
    finite_differences = []
    for step in steps:
        with torch.no_grad():
            parameter.fill_(center + step)
        plus = float(objective()[0].detach())
        with torch.no_grad():
            parameter.fill_(center - step)
        minus = float(objective()[0].detach())
        finite_differences.append((plus - minus) / (2.0 * step))
    with torch.no_grad():
        parameter.fill_(center)
    relative_errors = [
        abs(value - adjoint) / max(abs(value), abs(adjoint), 1.0e-30)
        for value in finite_differences
    ]
    assert min(relative_errors) < 0.02, (
        f"adjoint={adjoint}, finite_differences={finite_differences}, "
        f"relative_errors={relative_errors}"
    )
    return result


@pytest.mark.parametrize(
    ("component", "center", "steps"),
    (
        ("r", 50.0, (5.0, 2.5, 1.25)),
        ("l", 1.0e-9, (1.0e-10, 5.0e-11, 2.5e-11)),
        ("c", 1.0e-12, (1.0e-13, 5.0e-14, 2.5e-14)),
    ),
)
def test_port_series_rlc_adjoint_matches_three_step_central_difference(
    component,
    center,
    steps,
):
    parameter = torch.tensor(center, device="cuda", requires_grad=True)
    termination = mw.SeriesRLC(**{component: parameter})
    scene = _physical_scene(termination=termination)
    scene.add_source(
        mw.PointDipole(
            position=(0.005, 0.0, 0.0),
            polarization="Ez",
            width=0.002,
            source_time=mw.CW(frequency=3.0e9, amplitude=1000.0),
        )
    )
    result = _assert_three_step_central_difference(
        parameter,
        lambda: _passive_port_objective(scene),
        steps,
    )
    assert result.port("feed").voltage.grad_fn is not None


@pytest.mark.parametrize(
    ("element_type", "center", "steps"),
    (
        (mw.Resistor, 50.0, (5.0, 2.5, 1.25)),
        (mw.Inductor, 1.0e-9, (1.0e-10, 5.0e-11, 2.5e-11)),
        (mw.Capacitor, 1.0e-12, (1.0e-13, 5.0e-14, 2.5e-14)),
    ),
)
def test_standalone_rlc_adjoint_matches_three_step_central_difference(
    element_type,
    center,
    steps,
):
    parameter = torch.tensor(center, device="cuda", requires_grad=True)
    scene = _physical_scene()
    value_name = {
        mw.Resistor: "resistance",
        mw.Inductor: "inductance",
        mw.Capacitor: "capacitance",
    }[element_type]
    scene.add_lumped_element(
        element_type(
            "standalone",
            positive=(0.01, 0.0, 0.005),
            negative=(0.01, 0.0, -0.005),
            **{value_name: parameter},
        )
    )

    def objective():
        result = mw.Simulation.fdtd(
            scene,
            frequency=3.0e9,
            excitations=mw.PortExcitation(
                "feed",
                source_time=mw.CW(3.0e9),
            ),
            run_time=mw.TimeConfig(time_steps=32),
            spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()
        voltage = result.port("feed").voltage
        return voltage.abs().square().sum(), result

    result = _assert_three_step_central_difference(
        parameter,
        objective,
        steps,
    )
    assert result.port("feed").voltage.grad_fn is not None


def test_port_amplitude_and_available_power_match_three_step_central_difference():
    amplitude = torch.tensor(1.0, device="cuda", requires_grad=True)
    scene = _physical_scene()

    def objective():
        result = mw.Simulation.fdtd(
            scene,
            frequency=3.0e9,
            excitations=mw.PortExcitation(
                "feed",
                amplitude=amplitude,
                source_time=mw.CW(3.0e9),
            ),
            run_time=mw.TimeConfig(time_steps=24),
            spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()
        port = result.port("feed")
        return (
            port.voltage.abs().square().sum()
            + port.available_power.sum(),
            result,
        )

    result = _assert_three_step_central_difference(
        amplitude,
        objective,
        (0.1, 0.05, 0.025),
    )
    port = result.port("feed")
    assert port.voltage.grad_fn is not None
    assert port.current.grad_fn is not None
    assert port.available_power.grad_fn is not None


class _DensityPortScene(mw.SceneModule):
    def __init__(self, value=0.0):
        super().__init__()
        self.logit = torch.nn.Parameter(
            torch.tensor(float(value), device="cuda")
        )

    def to_scene(self):
        scene = _physical_scene()
        density = torch.sigmoid(self.logit).expand(2, 2, 2)
        scene.add_material_region(
            mw.MaterialRegion(
                name="feed_design",
                geometry=mw.Box(
                    position=(0.0, 0.0, 0.0),
                    size=(0.015, 0.015, 0.015),
                ),
                density=density,
                eps_bounds=(1.0, 4.0),
            )
        )
        return scene


def test_material_region_density_and_lumped_port_adjoint_coexist():
    model = _DensityPortScene().cuda()

    def objective():
        result = mw.Simulation.fdtd(
            model,
            frequency=3.0e9,
            excitations=mw.PortExcitation(
                "feed",
                source_time=mw.CW(3.0e9),
            ),
            run_time=mw.TimeConfig(time_steps=24),
            spectral_sampler=mw.SpectralSampler(window="none"),
        ).run()
        voltage = result.port("feed").voltage
        return voltage.abs().square().sum(), result

    result = _assert_three_step_central_difference(
        model.logit,
        objective,
        (0.08, 0.04, 0.02),
    )
    assert result.port("feed").voltage.grad_fn is not None


def test_unsupported_trainable_rf_combinations_fail_explicitly():
    source_impedance = torch.tensor(50.0, device="cuda", requires_grad=True)
    with pytest.raises(NotImplementedError, match="trainable source_impedance"):
        mw.Simulation.fdtd(
            _physical_scene(),
            frequency=3.0e9,
            excitations=mw.PortExcitation(
                "feed",
                source_impedance=source_impedance,
            ),
            run_time=mw.TimeConfig(time_steps=8),
        ).run()

    reference_impedance = torch.tensor(
        50.0,
        device="cuda",
        requires_grad=True,
    )
    with pytest.raises(NotImplementedError, match="trainable reference_impedance"):
        mw.Simulation.fdtd(
            _physical_scene(reference_impedance=reference_impedance),
            frequency=3.0e9,
            excitations=mw.PortExcitation("feed"),
            run_time=mw.TimeConfig(time_steps=8),
        ).run()

    parallel_r = torch.tensor(50.0, device="cuda", requires_grad=True)
    with pytest.raises(NotImplementedError, match="ParallelRLC"):
        mw.Simulation.fdtd(
            _physical_scene(termination=mw.ParallelRLC(r=parallel_r)),
            frequency=3.0e9,
            run_time=mw.TimeConfig(time_steps=8),
        ).run()


def test_trainable_observer_only_port_fails_explicitly():
    model = _DensityPortScene().cuda()
    with pytest.raises(NotImplementedError, match="observer-only contour ports"):
        mw.Simulation.fdtd(
            model,
            frequency=3.0e9,
            run_time=mw.TimeConfig(time_steps=8),
        ).run()


def test_terminal_port_uses_the_same_amplitude_adjoint_runtime():
    amplitude = torch.tensor(1.0, device="cuda", requires_grad=True)
    structures = (
        mw.Structure(
            name="signal",
            geometry=mw.Box(
                position=(0.0, 0.0, 0.0125),
                size=(0.015, 0.015, 0.005),
            ),
            material=mw.Material.pec(),
        ),
        mw.Structure(
            name="ground",
            geometry=mw.Box(
                position=(0.0, 0.0, -0.0125),
                size=(0.015, 0.015, 0.005),
            ),
            material=mw.Material.pec(),
        ),
    )
    port = mw.TerminalPort(
        name="terminal_feed",
        positive_terminal=mw.TerminalRef("signal"),
        negative_terminal=mw.TerminalRef("ground"),
        integration_path=mw.AxisPath("z"),
        reference_plane=0.0025,
        reference_impedance=50.0,
    )
    scene = mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        structures=structures,
        ports=(port,),
        device="cuda",
    )
    result = mw.Simulation.fdtd(
        scene,
        frequency=3.0e9,
        excitations=mw.PortExcitation(
            "terminal_feed",
            amplitude=amplitude,
            source_time=mw.CW(3.0e9),
        ),
        run_time=mw.TimeConfig(time_steps=16),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()
    loss = (
        result.port("terminal_feed").voltage.abs().square().sum()
        + result.port("terminal_feed").available_power.sum()
    )
    loss.backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad)
    assert abs(float(amplitude.grad)) > 0.0
