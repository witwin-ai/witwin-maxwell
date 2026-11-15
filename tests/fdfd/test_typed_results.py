"""FDFD results are physics-typed: one Yee FieldData per component and typed observations."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.fdfd.results import field_data, observation_data
from witwin.maxwell.fdfd.sweep import combine_results
from witwin.maxwell.scene import prepare_scene

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FREQ = 1.0e9
FREQS = (0.9e9, 1.1e9)
PHASOR = {"phasor_convention": "exp(-i omega t), peak", "normalization": "raw"}
OBSERVATIONS = (
    mw.fdfd.FieldProjection("probe", component="Ez", position=(0.0, 0.0, 0.06)),
    mw.fdfd.FieldProjection("hprobe", component="Hy", position=(0.06, 0.0, 0.0)),
    mw.fdfd.FieldProjection.voltage("gap", (0.0, 0.0, -0.04), (0.0, 0.0, 0.04)),
    mw.fdfd.FarFieldProjection("far", bounds=((-0.16, 0.16),) * 3, direction=(1.0, 0.0, 0.0), polarization=(0.0, 0.0, 1.0)),
)
EXPECTED = {"probe": (mw.FieldData, "V/m"), "hprobe": (mw.FieldData, "A/m"), "gap": (mw.FieldData, "V"),
            "far": (mw.FarFieldData, "V")}
# Fixed-iteration GMRES is bitwise repeatable; the direct solve is not (about 1e-16).
REPEATABLE = mw.fdfd.FDFDSolverConfig(solver_type="gmres", precision="double", max_iter=200, restart=50)


def _scene(eps_r=None):
    scene = mw.Scene(domain=mw.Domain(bounds=((-0.2, 0.2),) * 3), grid=mw.GridSpec.uniform(0.04),
                     boundary=mw.BoundarySpec.pml(num_layers=2),
                     sources=(
                         mw.PointDipole(position=(0.0, 0.0, 0.0), polarization="Ez", width=0.01, source_time=None),
                     ))
    if eps_r is not None:
        scene = scene.updated(structures=(
            *scene.structures,
            mw.Structure(geometry=mw.Box(position=(0.0, 0.0, 0.12), size=(0.08,) * 3),
                         material=mw.Material(eps_r=eps_r)),
        ))
    return scene


def _yee_fields(prepared, *, seed):
    generator = torch.Generator().manual_seed(seed)
    return {f"E{axis.upper()}": torch.randn(tuple(getattr(prepared, f"N{b}_e{axis}") for b in "xyz"),
                                            dtype=torch.complex64, generator=generator) for axis in "xyz"}


def _assert_yee_field_data(typed, stored, prepared, frequencies):
    assert set(typed) == {"EX", "EY", "EZ"}
    for name, data in typed.items():
        assert isinstance(data, mw.FieldData)
        assert data.dims == ("f", "x", "y", "z") and data.units == "V/m"
        component = name[0] + name[1].lower()
        assert dict(data.attrs) == {**PHASOR, "component": component, "component_units": {component: "V/m"},
                                    "yee_location": f"edge_{name[1].lower()}"}
        assert data.values.data_ptr() == stored[name].data_ptr()
        assert data.values.shape == (len(frequencies), *stored[name].shape[-3:])
        assert torch.equal(data.coords["f"].cpu(), torch.tensor(frequencies, dtype=torch.float64))
        for axis in "xyz":
            grid = "half" if axis == name[1].lower() else "nodes"
            assert torch.equal(data.coords[axis].cpu(), torch.from_numpy(getattr(prepared, f"{axis}_{grid}64")))


def _assert_observations(monitors, values, frequencies):
    for index, spec in enumerate(OBSERVATIONS):
        data = monitors[spec.name]
        kind, units = EXPECTED[spec.name]
        far = isinstance(spec, mw.fdfd.FarFieldProjection)
        # A far-field projection is one direction (D=1) of FarFieldData.from_directions.
        assert type(data) is kind and data.dims == (("f", "direction") if far else ("f",)) and data.units == units
        assert {key: data.attrs[key] for key in PHASOR} == PHASOR
        assert torch.equal(data.coords["f"].cpu(), torch.tensor(frequencies, dtype=torch.float64))
        expected = values[..., index].reshape(len(frequencies), *((1,) if far else ()))
        assert torch.equal(data.values, expected)
    assert monitors["probe"].attrs["position"] == (0.0, 0.0, 0.06)
    assert monitors["probe"].attrs["component_units"] == {"Ez": "V/m"}
    assert monitors["hprobe"].attrs["component_units"] == {"Hy": "A/m"}
    assert monitors["gap"].attrs["projection"] == "voltage"
    far = monitors["far"]
    assert "direction" not in far.attrs and far.attrs["polarization"] == (0.0, 0.0, 1.0)
    assert [float(far.attrs[key]) for key in ("ux", "uy", "uz")] == [1.0, 0.0, 0.0]
    assert torch.equal(far.coords["direction"].cpu(), torch.arange(1))


def _same_attrs(left, right):
    assert left.keys() == right.keys()
    for key, value in right.items():
        assert torch.equal(left[key], value) if isinstance(value, torch.Tensor) else left[key] == value


def test_typed_fields_observations_sweep_and_snapshot_on_cpu(tmp_path):
    scene = _scene("cpu")
    prepared = prepare_scene(scene, device="cpu")
    runs = []
    for seed, frequency in enumerate(FREQS):
        fields = _yee_fields(prepared, seed=seed)
        values = torch.randn(len(OBSERVATIONS), dtype=torch.complex64, generator=torch.Generator().manual_seed(9 + seed))
        monitors = observation_data(OBSERVATIONS, values, frequency)
        _assert_observations(monitors, values, (frequency,))
        runs.append((fields, values, mw.Result(method="fdfd", scene=scene, prepared_scene=prepared, frequency=frequency,
                                               fields=fields, monitors=monitors, metadata={"tag": "p6c"})))
    single = runs[0][2]
    _assert_yee_field_data(single.fields, runs[0][0], prepared, FREQS[:1])
    assert single.monitor("probe", freq_index=0).dims == ()

    sweep = combine_results([result for _, _, result in runs])
    assert sweep.metadata == {"tag": "p6c"}
    _assert_yee_field_data(sweep.fields, sweep._fields, prepared, FREQS)
    for index, (fields, values, _) in enumerate(runs):
        for name, field in fields.items():
            assert torch.equal(sweep.fields[name].isel(f=index).values, field)
            assert torch.equal(sweep.fields[name].sel(f=FREQS[index]).values, field)
        assert torch.equal(sweep.at(freq_index=index).monitor("gap").values, values[2])
        assert torch.equal(sweep.monitor("far", frequency=FREQS[index]).isel(direction=0).values, values[3])
    _assert_observations(sweep.monitors, torch.stack([values for _, values, _ in runs]), FREQS)

    sweep.save(tmp_path / "sweep.pt")
    restored = mw.Result.load(tmp_path / "sweep.pt", scene=scene)
    for name, data in sweep.monitors.items():
        loaded = restored.monitor(name)
        assert type(loaded) is type(data) and loaded.dims == data.dims and loaded.units == data.units
        _same_attrs(loaded.attrs, data.attrs)
        assert torch.equal(loaded.values, data.values) and torch.equal(loaded.coords["f"], data.coords["f"])
    for name, data in sweep.fields.items():
        assert torch.equal(restored.fields[name].values, data.values)


@requires_cuda
def test_single_run_returns_yee_field_data_and_typed_observations():
    result = mw.Simulation.fdfd(_scene(), frequencies=FREQ, config=mw.FDFDConfig(solver=REPEATABLE, observations=OBSERVATIONS)).run()
    _assert_yee_field_data(result.fields, result._fields, result.prepared_scene, (FREQ,))
    assert result.fields["EZ"].values.data_ptr() == result.E.z.values.data_ptr()
    for spec in OBSERVATIONS:
        data = result.monitor(spec.name)
        far = isinstance(spec, mw.fdfd.FarFieldProjection)
        assert isinstance(data, EXPECTED[spec.name][0]) and data.values.shape == ((1, 1) if far else (1,))
        assert bool(data.values.abs().gt(0).all())


@requires_cuda
def test_sweep_equals_single_runs_and_keeps_metadata():
    def simulation(frequencies):
        return mw.Simulation(scene=_scene(), method="fdfd", frequencies=frequencies, metadata={"tag": "p6c"},
                             config=mw.FDFDConfig(solver=REPEATABLE, observations=OBSERVATIONS))

    sweep = simulation(FREQS).run()
    assert sweep.metadata["tag"] == "p6c"
    for index, frequency in enumerate(FREQS):
        single = simulation((frequency,)).run()
        for name in ("EX", "EY", "EZ"):
            assert torch.equal(sweep.fields[name].isel(f=index).values, single.fields[name].isel(f=0).values)
        for spec in OBSERVATIONS:
            swept = sweep.monitor(spec.name, freq_index=index).values
            alone = single.monitor(spec.name, frequency=frequency).values
            if isinstance(spec, mw.fdfd.FarFieldProjection):
                # The Huygens functional is a long sparse row; its CUDA SpMV sum is
                # not bitwise repeatable between identical solves (also on master).
                torch.testing.assert_close(swept, alone, rtol=1e-12, atol=0.0)
            else:
                assert torch.equal(swept, alone)


@requires_cuda
@pytest.mark.parametrize("frequencies", [(FREQ,), FREQS], ids=["single", "multi"])
def test_gradient_flows_through_typed_observation_and_fields(frequencies):
    # Observe exactly on one Ez sample: the projection stencil interpolates on the
    # prepared (material-dtype) Yee coordinates, so take the position from them.
    prepared = prepare_scene(_scene(eps_r=3.0))
    index = {axis: int((getattr(prepared, name).double() - target).abs().argmin())
             for axis, name, target in (("x", "x", 0.0), ("y", "y", 0.0), ("z", "z_half", 0.06))}
    position = tuple(float(getattr(prepared, name)[index[axis]]) for axis, name in (("x", "x"), ("y", "y"), ("z", "z_half")))
    probe = mw.fdfd.FieldProjection("probe", component="Ez", position=position)

    def gradient(read):
        eps_r = torch.tensor(3.0, device="cuda", requires_grad=True)
        result = mw.Simulation.fdfd(_scene(eps_r=eps_r), frequencies=list(frequencies),
                                    config=mw.FDFDConfig(
                                        solver=mw.fdfd.FDFDSolverConfig(solver_type="direct", precision="double"),
                                        observations=(probe,),
                                    )).run()
        values = read(result)
        assert values.requires_grad
        values.abs().square().sum().backward()
        return eps_r.grad

    observed = gradient(lambda result: result.monitor("probe").values)
    sampled = gradient(lambda result: result.fields["EZ"].isel(**index).values)
    assert bool(observed.abs() > 0)
    # Separate direct solves agree to roundoff, not bitwise.
    torch.testing.assert_close(observed, sampled, rtol=1e-9, atol=0.0)
