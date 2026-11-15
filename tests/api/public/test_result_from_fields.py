"""External field producers share native Result access and persistence."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.postprocess import FieldProjector


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.to(expected.device), expected)
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for name in expected:
            _assert_equal(actual[name], expected[name])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    else:
        assert actual == expected


def _phasors(result):
    """The single-frequency phasor tensors, as an external producer supplies them."""
    return {name: data.isel(f=0).values for name, data in result.fields.items()}


def _payloads(result):
    # The recorded payload dicts, before Result.monitor types them and raw_monitor crops finite planes.
    return dict(result._monitors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_from_fields_matches_native_payloads_and_snapshot(tmp_path):
    frequency = 2.75e9
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.none(),
        ports=[mw.LumpedPort(
            "feed", positive=(0., 0., .005), negative=(0., 0., -.005),
            voltage_path=mw.AxisPath("z"),
            current_surface=mw.Box(position=(0., 0., -.0025), size=(.015, .015, 0.)),
        )],
        monitors=[
            mw.PlaneMonitor("plane", axis="z", position=.005, fields=("Ex", "Ey", "Ez")),
            mw.FinitePlaneMonitor("sampled", position=(0., 0., .005),
                                  size=(.015, .015, 0.), sample_spacing=.004),
            mw.FluxMonitor("flux", axis="z", position=.005),
            mw.monitors.ClosedSurfaceMonitor.box("surface", size=(.021,) * 3, sample_spacing=.004),
            mw.monitors.AngleProjectionMonitor("projection", size=(.021,) * 3, sample_spacing=.004,
                                      theta=(.4, 1.2), phi=(0.,), radius=1.),
        ],
    )
    native = mw.Simulation.fdtd(
        scene, frequencies=frequency, excitations=mw.rf.PortExcitation(
            "feed", source_time=mw.GaussianPulse(frequency=frequency, fwidth=1e9),
        ), config=mw.FDTDConfig(full_field_dft=True, cuda_graph=False,
        run_time=mw.TimeConfig(time_steps=96, transient_cycles=0, steady_cycles=1),
        spectral_sampler=mw.fdtd.SpectralSampler(window="none", normalize_source=False)),
    ).run()
    port = native.ports["feed"]
    assert torch.any(port.voltage != 0)
    # Single-run FDTD retains fields and ports; assemble its one-port network
    # through the public payload constructor without running a second solve.
    network = mw.rf.NetworkData(
        frequencies=port.frequencies, s=(port.b / port.a).reshape(-1, 1, 1),
        z0=port.z0, port_names=("feed",),
    )
    native = mw.Result(
        method=native.method, scene=native.scene, prepared_scene=native.prepared_scene,
        frequencies=native.frequencies, fields=_phasors(native), monitors=_payloads(native),
        ports=native.ports, network=network, solver=native.solver,
        metadata=native.metadata, solver_stats=native._solver_stats,
    )
    projected = FieldProjector.from_monitor(native, "surface").angles((.4, 1.2), (0.,), radius=1.)
    _assert_equal(native.monitor("projection"), projected)

    metadata = {"origin": "test", "kernel": "test-kernel"}
    rebuilt = mw.Result.from_fields(
        native.scene, frequencies=native.frequencies, kernel="test-kernel",
        fields=_phasors(native), monitors=_payloads(native), ports=native.ports,
        network=native.network, metadata=metadata, prepared_scene=native.prepared_scene,
    )
    assert native.fields and rebuilt.fields.keys() == native.fields.keys()
    for name, data in native.fields.items():
        # Only a solver record states the Yee sample-time label of its phasors (E: 0.0 steps);
        # an external producer supplies none. Everything else matches the native record.
        external = rebuilt.fields[name]
        assert type(external) is type(data)
        assert data.attrs["field_time_offset_steps"] == 0.0
        assert "field_time_offset_steps" not in external.attrs
        for item in fields(data):
            expected = getattr(data, item.name)
            if item.name == "attrs":
                expected = {key: value for key, value in expected.items() if key != "field_time_offset_steps"}
            _assert_equal(getattr(external, item.name), expected)
    for component in ("x", "y", "z"):
        assert torch.equal(getattr(rebuilt.E, component).values, getattr(native.E, component).values)
        assert torch.equal(rebuilt.field(f"E{component}").isel(f=0).values, native.field(f"E{component}").isel(f=0).values)
    for name in {*native.monitors, *(monitor.name for monitor in scene.monitors)}:
        _assert_equal(rebuilt.monitor(name), native.monitor(name))
    _assert_equal(rebuilt.ports, native.ports)
    _assert_equal(rebuilt.network, native.network)
    assert rebuilt.ports["feed"] is native.ports["feed"]
    assert rebuilt.network is native.network
    assert rebuilt.metadata["kernel"] == "test-kernel"
    assert rebuilt.stats()["method"] == "external"
    rebuilt.metadata["origin"] = "changed"
    assert rebuilt.metadata["origin"] == "changed"
    assert metadata["origin"] == "test"

    path = tmp_path / "external.pt"
    rebuilt.save(path)
    restored = mw.Result.load(path, scene=native.scene, map_location="cuda")
    assert restored.method == "external"
    assert restored.metadata == rebuilt.metadata
    _assert_equal(restored.fields, rebuilt.fields)
    for name in {*native.monitors, *(monitor.name for monitor in scene.monitors)}:
        _assert_equal(restored.monitor(name), rebuilt.monitor(name))
    _assert_equal(restored.ports, rebuilt.ports)
    _assert_equal(restored.network, rebuilt.network)


def test_from_fields_rejects_invalid_kernel_metadata_on_cpu():
    scene = mw.Scene(domain=mw.Domain(bounds=((-1., 1.),) * 3),
                     grid=mw.GridSpec.uniform(.5))
    for kernel in ("", None, 1):
        with pytest.raises(ValueError, match="non-empty string"):
            mw.Result.from_fields(scene, frequencies=(1e9,), kernel=kernel)
    with pytest.raises(ValueError, match="metadata kernel"):
        mw.Result.from_fields(scene, frequencies=(1e9,), kernel="test-kernel",
                              metadata={"kernel": "other-kernel"})
    result = mw.Result.from_fields(scene, frequencies=(1e9,), kernel="test-kernel")
    assert result.metadata == {"kernel": "test-kernel"}
