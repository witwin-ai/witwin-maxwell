from dataclasses import replace

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.lumped import PortSweep
from witwin.maxwell.network import PortData
from witwin.maxwell.network_sweep import (
    aggregate_network_columns,
    resolve_network_run_manifest,
)
from witwin.maxwell.result import Result


def _port(
    name,
    x,
    *,
    reference_impedance=50.0,
    reference_plane=None,
    termination=None,
):
    return mw.LumpedPort(
        name=name,
        positive=(x, 0.0, 0.005),
        negative=(x, 0.0, -0.005),
        voltage_path=mw.AxisPath("z"),
        current_surface=mw.Box(
            position=(x, 0.0, -0.0025),
            size=(0.005, 0.015, 0.0),
        ),
        reference_impedance=reference_impedance,
        reference_plane=reference_plane,
        termination=termination,
    )


def _scene(*, device="cpu", ports=None):
    return mw.Scene(
        domain=mw.Domain(
            bounds=((-0.02, 0.02), (-0.02, 0.02), (-0.02, 0.02))
        ),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=(
            (_port("p1", -0.005), _port("p2", 0.005))
            if ports is None
            else ports
        ),
        device=device,
    )


def _known_columns():
    frequencies = torch.tensor((1.0e9, 2.0e9), dtype=torch.float64)
    expected = torch.tensor(
        (
            ((0.1 + 0.2j, 0.7 - 0.1j), (0.7 - 0.1j, -0.2 + 0.1j)),
            ((0.2 + 0.0j, 0.6 - 0.2j), (0.6 - 0.2j, -0.1 + 0.3j)),
        ),
        dtype=torch.complex128,
    )
    columns = []
    for input_index, input_name in enumerate(("p1", "p2")):
        incident = torch.full((2,), 2.0 + input_index, dtype=torch.complex128)
        column = {}
        for output_index, output_name in enumerate(("p1", "p2")):
            a = incident if output_index == input_index else torch.zeros_like(incident)
            b = expected[:, output_index, input_index] * incident
            column[output_name] = PortData.from_power_waves(
                port_name=output_name,
                frequencies=frequencies,
                a=a,
                b=b,
                z0=50.0,
            )
        columns.append(column)
    return frequencies, expected, tuple(columns)


def _replace_entry(columns, column_index, port_name, **changes):
    updated = [dict(column) for column in columns]
    updated[column_index][port_name] = replace(
        updated[column_index][port_name],
        **changes,
    )
    return tuple(updated)


def test_manifest_uses_scene_order_and_a_deterministic_broadband_source():
    sweep = PortSweep()
    manifest = resolve_network_run_manifest(
        _scene(),
        sweep,
        (1.0e9, 2.0e9),
    )

    assert manifest.port_names == ("p1", "p2")
    assert manifest.reference_impedances == (50.0, 50.0)
    assert manifest.source_time.kind == "gaussian_pulse"
    assert manifest.metadata()["execution"] == "single_device_sequential"


def test_manifest_rejects_trainable_sweep_amplitude_before_execution():
    sweep = PortSweep(amplitude=torch.tensor(1.0, requires_grad=True))

    with pytest.raises(NotImplementedError, match="amplitude is trainable"):
        resolve_network_run_manifest(_scene(), sweep, (1.0e9,))


def test_simulation_prepare_rejects_trainable_sweep_parameters_before_solver_build():
    simulation = mw.Simulation.fdtd(
        _scene(),
        frequency=1.0e9,
        excitations=PortSweep(amplitude=torch.tensor(1.0, requires_grad=True)),
    )

    with pytest.raises(NotImplementedError, match="amplitude is trainable"):
        simulation.prepare()


def test_manifest_rejects_trainable_z0_on_an_unselected_matched_port():
    scene = _scene(
        ports=(
            _port("p1", -0.005),
            _port(
                "p2",
                0.005,
                reference_impedance=torch.tensor(50.0, requires_grad=True),
            ),
        )
    )

    with pytest.raises(NotImplementedError, match="trainable reference impedance"):
        resolve_network_run_manifest(scene, PortSweep(ports=("p1",)), (1.0e9,))


def test_manifest_rejects_trainable_declared_termination_before_standard_sweep_error():
    scene = _scene(
        ports=(
            _port("p1", -0.005),
            _port(
                "p2",
                0.005,
                termination=mw.SeriesRLC(r=torch.tensor(50.0, requires_grad=True)),
            ),
        )
    )

    with pytest.raises(NotImplementedError, match="trainable termination"):
        resolve_network_run_manifest(scene, PortSweep(), (1.0e9,))


def test_manifest_strictly_rejects_a_nonzero_imaginary_reference_impedance():
    scene = _scene(
        ports=(
            _port("p1", -0.005),
            _port("p2", 0.005, reference_impedance=50.0 + 1.0e-15j),
        )
    )

    with pytest.raises(ValueError, match="must be real"):
        resolve_network_run_manifest(scene, PortSweep(), (1.0e9,))


@pytest.mark.parametrize("frequency", (0.0, -1.0, float("nan"), float("inf")))
def test_manifest_rejects_nonpositive_or_nonfinite_frequencies(frequency):
    with pytest.raises(ValueError, match="positive and finite"):
        resolve_network_run_manifest(_scene(), PortSweep(), (frequency,))


@pytest.mark.parametrize("frequencies", ((2.0e9, 1.0e9), (1.0e9, 1.0e9)))
def test_manifest_requires_strictly_increasing_unique_frequencies(frequencies):
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_network_run_manifest(_scene(), PortSweep(), frequencies)


def test_known_port_columns_assemble_exact_network_and_stacked_port_data():
    frequencies, expected, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))

    ports, network = aggregate_network_columns(manifest, columns)

    torch.testing.assert_close(network.s, expected)
    assert network.port_names == ("p1", "p2")
    assert torch.all(network.valid_columns)
    assert ports["p1"].voltage.shape == (2, 2)
    assert ports["p2"].current.shape == (2, 2)
    assert ports["p1"].metadata["excitation_port_names"] == ("p1", "p2")


def test_aggregate_rejects_a_missing_excitation_column():
    frequencies, _, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))

    with pytest.raises(ValueError, match="column count"):
        aggregate_network_columns(manifest, columns[:1])


def test_aggregate_rejects_nonfinite_port_data_before_column_normalization():
    frequencies, _, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))
    bad_voltage = columns[1]["p1"].voltage.clone()
    bad_voltage[0] = torch.nan
    columns = _replace_entry(columns, 1, "p1", voltage=bad_voltage)

    with pytest.raises(RuntimeError, match="non-finite values"):
        aggregate_network_columns(manifest, columns)


@pytest.mark.parametrize(
    "case, match",
    (
        ("frequency_values", "identical frequencies"),
        ("frequency_dtype", "frequencies must use the same dtype"),
        ("signal_dtype", "same signal dtype"),
        ("z0", "reference impedance does not match"),
        ("direction", "direction, reference plane, or wave convention"),
        ("reference_plane", "direction, reference plane, or wave convention"),
        ("phasor_convention", "direction, reference plane, or wave convention"),
        ("power_wave_convention", "direction, reference plane, or wave convention"),
        ("current_convention", "direction, reference plane, or wave convention"),
    ),
)
def test_aggregate_rejects_inconsistent_column_contracts(case, match):
    frequencies, _, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))
    entry = columns[1]["p1"]
    if case == "frequency_values":
        changes = {"frequencies": torch.tensor((1.0e9, 2.5e9), dtype=torch.float64)}
    elif case == "frequency_dtype":
        changes = {"frequencies": entry.frequencies.to(dtype=torch.float32)}
    elif case == "signal_dtype":
        changes = {
            "voltage": entry.voltage.to(dtype=torch.complex64),
            "current": entry.current.to(dtype=torch.complex64),
        }
    elif case == "z0":
        changes = {"z0": 75.0}
    elif case == "direction":
        changes = {"direction": "-"}
    elif case == "reference_plane":
        changes = {"reference_plane": 0.25}
    elif case == "phasor_convention":
        changes = {"phasor_convention": "different phasor convention"}
    elif case == "power_wave_convention":
        changes = {"power_wave_convention": "different wave convention"}
    else:
        changes = {"metadata": {"current_convention": "different current convention"}}
    columns = _replace_entry(columns, 1, "p1", **changes)

    with pytest.raises((TypeError, ValueError), match=match):
        aggregate_network_columns(manifest, columns)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_aggregate_rejects_columns_split_across_cpu_and_cuda():
    frequencies, _, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))
    entry = columns[1]["p1"]
    columns = _replace_entry(
        columns,
        1,
        "p1",
        frequencies=entry.frequencies.cuda(),
        voltage=entry.voltage.cuda(),
        current=entry.current.cuda(),
        z0=entry.z0.cuda(),
    )

    with pytest.raises(ValueError, match="same device"):
        aggregate_network_columns(manifest, columns)


def test_result_snapshot_embeds_detached_network_payload(tmp_path):
    frequencies, expected, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))
    ports, network = aggregate_network_columns(manifest, columns)
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequencies=tuple(frequencies),
        ports=ports,
        network=network,
    )
    path = tmp_path / "network_result.pt"

    result.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert result.network is network
    assert result.stats()["has_network"] is True
    assert payload["network"]["data_type"] == "NetworkData"
    assert payload["network"]["port_names"] == ("p1", "p2")
    torch.testing.assert_close(payload["network"]["s"], expected)
    assert payload["network"]["s"].device.type == "cpu"
    assert not payload["network"]["s"].requires_grad


def test_result_snapshot_rejects_unsafe_embedded_network_metadata(tmp_path):
    frequencies, _, columns = _known_columns()
    manifest = resolve_network_run_manifest(_scene(), PortSweep(), tuple(frequencies))
    ports, network = aggregate_network_columns(manifest, columns)
    unsafe = mw.NetworkData(
        frequencies=network.frequencies,
        s=network.s,
        z0=network.z0,
        port_names=network.port_names,
        valid_columns=network.valid_columns,
        metadata={"unsafe": object()},
    )
    result = Result(
        method="fdtd",
        scene=_scene(),
        frequencies=tuple(frequencies),
        ports=ports,
        network=unsafe,
    )

    with pytest.raises(TypeError, match="network.metadata"):
        result.save(tmp_path / "unsafe-network-result.pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_two_port_single_device_sweep_returns_network_data():
    # Gate class (S0.3): symmetric. The reciprocity (|S12-S21|<0.02) and passivity
    # asserts below run on a mirror-symmetric fixture, so S12==S21 follows from
    # symmetry, not physics. This is a runtime/contract smoke test, NOT a
    # wave-level reciprocity gate (that lives in tests/rf/wave_validation on a
    # physically asymmetric two-port).
    result = mw.Simulation.fdtd(
        _scene(device="cuda"),
        frequency=2.5e9,
        excitations=PortSweep(
            source_time=mw.GaussianPulse(frequency=2.5e9, fwidth=1.0e9)
        ),
        run_time=mw.TimeConfig(time_steps=64),
        spectral_sampler=mw.SpectralSampler(window="none"),
    ).run()

    assert result.network is not None
    assert result.network.s.shape == (1, 2, 2)
    assert result.network.s.device.type == "cuda"
    assert torch.all(torch.isfinite(result.network.s))
    assert torch.max(torch.abs(result.network.s[:, 0, 1] - result.network.s[:, 1, 0])) < 0.02
    assert torch.max(torch.linalg.svdvals(result.network.s)) <= 1.02
    assert result.port("p1").voltage.shape == (2, 1)
    assert result.port("p2").current.shape == (2, 1)
    ports = {name: result.port(name) for name in result.network.port_names}
    for input_index, input_name in enumerate(result.network.port_names):
        incident = ports[input_name].a[input_index]
        assert torch.all(torch.abs(incident) > 0.0)
        for output_index, output_name in enumerate(result.network.port_names):
            output = ports[output_name]
            expected_column_entry = output.b[input_index] / incident
            torch.testing.assert_close(
                result.network.s[:, output_index, input_index],
                expected_column_entry,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            if output_index == input_index:
                continue
            assert torch.max(torch.abs(output.a[input_index])) <= (
                1.0e-5 * torch.max(torch.abs(output.b[input_index]))
            )
            torch.testing.assert_close(
                output.voltage[input_index],
                -output.z0[input_index] * output.current[input_index],
                rtol=1.0e-5,
                atol=1.0e-12,
            )
    assert result.fields == {}
