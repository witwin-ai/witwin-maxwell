"""The slimmed 0.7.0 Result: kept members, namespace accessors, typed volume fields and snapshot schema 3."""

from dataclasses import dataclass

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.errors import SchemaError
from witwin.maxwell.result import RESULT_SNAPSHOT_SCHEMA_VERSION, Result
from witwin.maxwell.results import TensorData
from witwin.maxwell.scene import prepare_scene

KEPT = {
    "method", "scene", "prepared_scene", "frequencies", "metadata", "stats",
    "fields", "E", "H", "time_snapshot", "field", "monitor", "monitors",
    "port", "ports", "port_drive", "network",
    "material", "materials", "at",
    "save", "load", "from_fields", "plot", "release_solver", "raw_monitor",
    "query_fields",
}
INSTANCE = {"method", "scene", "frequencies", "solver"}
REMOVED = ("frequency", "select", "compare_harmonic_cutoff", "power_loss", "solver_stats", "tensor", "raw_output")
MOVED = {
    "antenna": ("antenna", "array_basis"),
    "breakdown": ("breakdown", "breakdown_names", "breakdown_data", "breakdown_events"),
    "esd": ("component_stress", "component_stress_names", "esd_waveform", "esd_waveform_names",
            "esd_generator", "esd_generator_names"),
    "electrostatic": ("capacitance", "electrostatic", "electrostatic_prebias"),
    "coordinates": ("chart_fields",),
    "sar": ("sar", "incident_power_density"),
    "circuits": ("circuit", "circuits"),
    "rf": ("embedded_network", "embedded_networks", "conductor_losses", "conductor_loss_columns"),
    "execution": ("save_sharded", "load_sharded", "is_sharded", "sharded_manifest", "shard_paths"),
    "postprocess": ("power_loss",),
    "mom": ("far_field", "harmonic_state", "refine_harmonic_power", "harmonic_power_balance", "query_dc",
            "query_dc_magnetic", "query_dc_dual", "surface_currents", "rayleigh_orders", "cylindrical_orders",
            "bloch_spectrum"),
}
FREQS = (1.0e9, 1.5e9)


def _scene(symmetry=None):
    return mw.Scene(domain=mw.Domain(bounds=((0.0, 0.75), (0.0, 0.5), (0.0, 0.5))), grid=mw.GridSpec.uniform(0.25),
                    symmetry=symmetry)


def _phasors(scene, frequencies, *, seed=0):
    prepared = prepare_scene(scene, device="cpu")
    generator = torch.Generator().manual_seed(seed)
    shape = (len(frequencies),) if len(frequencies) > 1 else ()
    return prepared, {f"E{axis.upper()}": torch.randn(
        (*shape, *(getattr(prepared, f"N{b}_e{axis}") for b in "xyz")), dtype=torch.complex64, generator=generator)
        for axis in "xyz"}


def test_result_keeps_only_its_core_members():
    public = {name for name in dir(Result) if not name.startswith("_")}
    assert public == KEPT - INSTANCE
    result = Result.from_fields(_scene(), frequencies=FREQS[:1], kernel="test")
    instance = {name for name in vars(result) if not name.startswith("_")}
    assert instance == INSTANCE
    for removed in REMOVED:
        assert not hasattr(result, removed)
    for namespace, names in MOVED.items():
        module = getattr(mw, namespace)
        for name in names:
            assert name in module.__all__ and callable(getattr(module, name))
            assert not hasattr(result, name)


def test_stats_carries_the_solver_statistics():
    result = Result(method="fdfd", scene=_scene(), frequency=FREQS[0], solver_stats={"converged": True, "iterations": 7})
    stats = result.stats()
    assert stats["converged"] is True and stats["iterations"] == 7
    assert stats["method"] == "fdfd" and stats["frequencies"] == FREQS[:1]
    assert "solver_method" not in stats


def test_stats_keeps_the_solver_method_apart_from_the_result_method():
    result = Result(method="fdfd", scene=_scene(), frequency=FREQS[0], solver_stats={"method": "gmres"})
    stats = result.stats()
    assert stats["method"] == "fdfd" and stats["solver_method"] == "gmres"
    assert result._solver_stats == {"method": "gmres"}


@pytest.mark.parametrize("frequencies", [FREQS[:1], FREQS], ids=["single", "multi"])
def test_field_returns_the_phasor_field_data(frequencies):
    scene = _scene()
    _, phasors = _phasors(scene, frequencies)
    result = Result.from_fields(scene, frequencies=frequencies, kernel="test", fields=phasors)
    ex = result.field("Ex")
    assert isinstance(ex, mw.FieldData) and ex.dims == ("f", "x", "y", "z")
    assert torch.equal(ex.values, result.fields["EX"].values)
    assert ex.values.data_ptr() == phasors["EX"].data_ptr()
    family = result.field("E")
    assert set(family) == {"EX", "EY", "EZ"} and all(isinstance(data, mw.FieldData) for data in family.values())
    for index, frequency in enumerate(frequencies):
        stored = phasors["EY"][index] if len(frequencies) > 1 else phasors["EY"]
        for selected in (result.field("Ey", freq_index=index), result.field("Ey", frequency=frequency),
                         result.at(freq_index=index).E.y, result.at(frequency=frequency).field("Ey")):
            assert selected.dims == ("x", "y", "z") and torch.equal(selected.values, stored)
    assert torch.equal(result.E.z.values, result.fields["EZ"].values)
    assert set(result.E.as_dict()) == {"x", "y", "z"}
    with pytest.raises(KeyError):
        result.field("Hx")


def test_field_expands_symmetry_with_mirrored_coordinates():
    scene = _scene(symmetry=("PEC", None, None))
    prepared, phasors = _phasors(scene, FREQS[:1])
    result = Result.from_fields(scene, frequencies=FREQS[:1], kernel="test", fields=phasors)
    half = result.field("Ex")
    full = result.field("Ex", expand_symmetry=True)
    count = half.sizes["x"]
    assert full.sizes["x"] == 2 * count  # Ex is staggered off the symmetry plane: mirrored in full
    assert torch.equal(full.values[:, count:], half.values)
    # PEC: the plane-normal Ex mirrors even, the tangential Ez odd.
    assert torch.equal(full.values[:, :count], torch.flip(half.values, dims=(1,)))
    plane = float(prepared.x_nodes64[0])
    assert torch.allclose(full.coords["x"][:count], 2 * plane - torch.flip(half.coords["x"], dims=(0,)), rtol=0, atol=0)
    ez = result.field("Ez", expand_symmetry=True)  # on the plane: the shared sample is kept once
    assert ez.sizes["x"] == 2 * result.field("Ez").sizes["x"] - 1
    assert ez.coords["x"].shape == (ez.sizes["x"],)
    ez_half = result.field("Ez").values
    assert torch.equal(ez.values[:, :ez_half.shape[1] - 1], -torch.flip(ez_half[:, 1:], dims=(1,)))
    assert torch.equal(result.at(expand_symmetry=True).E.x.values, full.values)


def test_field_refuses_a_time_snapshot():
    scene = _scene()
    _, snapshot = _phasors(scene, FREQS[:1])
    result = Result(method="fdtd", scene=scene, frequency=FREQS[0], fields={k: v.real for k, v in snapshot.items()},
                    solver_stats={"full_field_dft": False, "steps_run": 10, "dt": 1e-12})
    for read in (lambda: result.field("Ex"), lambda: result.E.x, lambda: result.field("E")):
        with pytest.raises(ValueError, match="time_snapshot"):
            read()
    assert torch.equal(result.time_snapshot["EX"].values, snapshot["EX"].real)


def test_power_loss_explicit_inputs_need_a_declared_monitor():
    result = Result(method="fdtd", scene=_scene(), frequency=FREQS[0])
    with pytest.raises(KeyError, match="PowerLossMonitor"):
        mw.postprocess.power_loss(result, "missing", electric_fields={})


def test_namespace_accessors_read_the_result():
    breakdown = mw.breakdown.BreakdownResultData(events=(), final_state=torch.zeros(1), dissipated_energy=torch.zeros(1),
                                                 total_dissipated_energy=0.0, grid_shape=(1, 1, 1))
    result = Result(method="fdtd", scene=_scene(), frequency=FREQS[0], breakdown=breakdown,
                    metadata={"electrostatic_prebias": {"source": "electrostatic"}})
    assert mw.breakdown.breakdown_data(result) is breakdown and mw.breakdown.breakdown_events(result) == ()
    assert mw.breakdown.breakdown_names(result) == ()
    assert mw.circuits.circuits(result) == {} and mw.rf.embedded_networks(result) == {}
    assert mw.rf.conductor_losses(result) == {} and mw.rf.conductor_loss_columns(result) == {}
    assert mw.electrostatic.electrostatic_prebias(result) == {"source": "electrostatic"}
    assert mw.esd.esd_waveform_names(result) == () and mw.esd.component_stress_names(result) == ()
    assert not mw.execution.is_sharded(result) and mw.execution.shard_paths(result) == ()
    assert mw.execution.sharded_manifest(result) is None
    with pytest.raises(KeyError):
        mw.circuits.circuit(result, "c")
    with pytest.raises(AttributeError, match="Simulation.capacitance"):
        mw.electrostatic.capacitance(result)
    with pytest.raises(ValueError, match="coordinate chart"):
        mw.coordinates.chart_fields(result)


# --- snapshot schema 3 ----------------------------------------------------------------
def _typed_monitors(device="cpu"):
    f = torch.tensor(FREQS, dtype=torch.float64)
    generator = torch.Generator().manual_seed(3)
    values = lambda *shape: torch.randn(*shape, dtype=torch.complex64, generator=generator).to(device)
    return {
        "point": mw.FieldData(values(2, 2), ("component", "f"), {"component": ("Ex", "Hy"), "f": f}, "mixed",
                              {"component_units": {"Ex": "V/m", "Hy": "A/m"}, "normalization": "raw"}),
        "flux": mw.FluxData(values(2).real, ("f",), {"f": f}),
        "mode": mw.ModeAmplitudeData(values(1, 2, 2), ("mode", "propagation", "f"),
                                     {"propagation": ("+", "-"), "f": f},
                                     attrs={"power": values(1, 2, 2).real, "effective_index": (1.5, 1.6)}),
        "s": mw.SParameterData(values(2, 2, 2), ("f", "out_port", "in_port"), {"f": f},
                               attrs={"port_names": ("a", "b"), "z0": 50.0}),
        "far": mw.FarFieldData.from_directions(values(2, 1), [[0.0, 0.0, 1.0]], f, None, "V"),
        "far_grid": mw.FarFieldData(values(2, 3, 1, 2), ("f", "theta", "phi", "component"),
                                    {"f": f, "component": ("E_theta", "E_phi")}),
        "generic": TensorData(values(2), ("f",), {"f": f}, attrs={"tag": "plain"}),
        "faces": {"x_low": {"Ez": mw.FieldData(values(2, 3), ("f", "x"), {"f": f}, "V/m")}},
    }


def _assert_same(left, right):
    if isinstance(right, TensorData):
        assert type(left) is type(right) and left.dims == right.dims and left.units == right.units
        assert torch.equal(left.values.cpu(), right.values.cpu())
        assert left.coords.keys() == right.coords.keys()
        for key, value in right.coords.items():
            _assert_same(left.coords[key], value)
        assert left.attrs.keys() == right.attrs.keys()
        for key, value in right.attrs.items():
            _assert_same(left.attrs[key], value)
    elif isinstance(right, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in right:
            _assert_same(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize("method", ["external", "fdfd", "fdtd"])
def test_snapshot_schema3_round_trips_every_typed_payload(tmp_path, method):
    scene = _scene()
    _, phasors = _phasors(scene, FREQS)
    monitors = _typed_monitors()
    kwargs = dict(kernel="test") if method == "external" else {}
    if method == "external":
        result = Result.from_fields(scene, frequencies=FREQS, fields=phasors, monitors=monitors, **kwargs)
    else:
        result = Result(method=method, scene=scene, frequencies=FREQS, fields=phasors, monitors=monitors,
                        solver_stats={"converged": True}, metadata={"tag": method})
    result.save(tmp_path / "result.pt")
    payload = torch.load(tmp_path / "result.pt", weights_only=False)
    assert payload["schema_version"] == RESULT_SNAPSHOT_SCHEMA_VERSION == 3
    assert "format_version" not in payload
    assert payload["monitors"]["mode"]["__witwin_tensor_data__"] == "witwin.maxwell.results.typed:ModeAmplitudeData"

    restored = Result.load(tmp_path / "result.pt", scene=scene)
    assert restored.method == method and restored.frequencies == FREQS
    assert restored.stats() == result.stats()
    _assert_same(restored._monitors, monitors)
    for name, data in result.fields.items():
        _assert_same(restored.fields[name], data)


def test_snapshot_schema3_round_trips_electrostatic_and_capacitance(tmp_path):
    scene = _scene()
    for method in ("electrostatic", "capacitance"):
        result = Result(method=method, scene=scene, frequency=0.0, solver_stats={"iterations": 3, "residual": 1e-9})
        result.save(tmp_path / f"{method}.pt")
        restored = Result.load(tmp_path / f"{method}.pt", scene=scene)
        assert restored.method == method and restored.stats() == result.stats()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_snapshot_schema3_keeps_device_payloads_detached(tmp_path):
    scene = _scene()
    monitors = _typed_monitors("cuda")
    monitors["flux"] = mw.FluxData(monitors["flux"].values.requires_grad_(), ("f",), monitors["flux"].coords)
    result = Result.from_fields(scene, frequencies=FREQS, kernel="test", monitors=monitors)
    result.save(tmp_path / "result.pt")
    restored = Result.load(tmp_path / "result.pt", scene=scene)
    assert restored._monitors["flux"].values.device.type == "cpu"
    assert not restored._monitors["flux"].values.requires_grad
    _assert_same(restored._monitors["mode"], monitors["mode"])


def test_schema2_snapshot_is_refused(tmp_path):
    scene = _scene()
    result = Result.from_fields(scene, frequencies=FREQS[:1], kernel="test")
    result.save(tmp_path / "result.pt")
    payload = torch.load(tmp_path / "result.pt", weights_only=False)
    payload.update(schema_version=2, format_version=1)
    torch.save(payload, tmp_path / "schema2.pt")
    with pytest.raises(SchemaError, match="schema_version=2"):
        Result.load(tmp_path / "schema2.pt", scene=scene)


@dataclass(frozen=True, eq=False, repr=False)
class _ForeignData(TensorData):
    pass


def test_snapshot_refuses_typed_payloads_outside_the_package(tmp_path):
    scene = _scene()
    f = torch.tensor(FREQS[:1], dtype=torch.float64)
    result = Result.from_fields(scene, frequencies=FREQS[:1], kernel="test",
                                monitors={"foreign": _ForeignData(torch.ones(1), ("f",), {"f": f})})
    result.save(tmp_path / "result.pt")
    with pytest.raises(SchemaError, match="unknown typed payload"):
        Result.load(tmp_path / "result.pt", scene=scene)
