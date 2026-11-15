"""FDTD monitors and volume fields return physics-typed data wrapping the recorded tensors.

Every typed value is compared with ``torch.equal`` to the solver payload that
``Result.raw_monitor`` returns, so the types change containers and metadata but
never a number; gradients through ``.values`` equal gradients through the payload.
"""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.results import TensorData

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

FREQUENCIES = (2.0e9, 2.3e9)
_SUBPIXEL = mw.SubpixelSpec(samples=(1, 1, 1), averaging="arithmetic", pec="staircase")


def _scene(*, frequencies=FREQUENCIES):
    pulse = mw.GaussianPulse(frequency=2.0e9, fwidth=0.5e9)
    scene = mw.Scene(
                subpixel=_SUBPIXEL, domain=mw.Domain(bounds=((-0.32, 0.32),) * 3), grid=mw.GridSpec.uniform(0.04),
                boundary=mw.BoundarySpec.pml(num_layers=6),
                structures=(
            mw.Structure(geometry=mw.Box(position=(0, 0, 0), size=(1.28, 0.20, 0.30)),
                         material=mw.Material(eps_r=4.0), name="waveguide"),
            ),
                sources=(
            mw.ModeSource(position=(-0.20, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", source_time=pulse,
                          name="mode"),
            mw.PointDipole(position=(0.0, 0.1, 0.2), polarization="Ey", width=0.04, source_time=pulse,
                           name="dipole"),
            ),
            )
    scene = scene.updated(monitors=(
        *scene.monitors,
        *(
        mw.PointMonitor("point", (0.05, 0.02, 0.03), fields=("Ez", "Hy")),
        mw.PlaneMonitor("plane", axis="x", position=0.1, fields=("Ez",)),
        mw.PlaneMonitor("aligned", axis="x", position=0.1, fields=("Ey", "Ez", "Hy", "Hz")),
        mw.PlaneMonitor("native", axis="x", position=0.1, fields=("Ey", "Hx")),
        mw.FluxMonitor("flux", axis="x", position=0.12),
        mw.ModeMonitor("mode_out", position=(0.15, 0, 0), size=(0, 0.5, 0.5), polarization="Ez", direction="+"),
        mw.monitors.ClosedSurfaceMonitor.box("box", position=(0.0, 0.1, 0.2), size=(0.16, 0.16, 0.16)),
        mw.monitors.DipoleEmissionMonitor("emission", "dipole", frequencies=frequencies),
        mw.monitors.GaussianOverlapMonitor("overlap", position=(0.1, 0, 0), size=(0, 0.4, 0.4), beam_waist=0.1,
                                           polarization=(0, 0, 1)),
        mw.monitors.FieldTimeMonitor("trace", components=("Ez", "Hy"), position=(0.1, 0, 0), interval=3),
        mw.monitors.FieldTimeMonitor("sheet", components=("Ez", "Hy"), position=(0.1, 0, 0), size=(0, 0.2, 0.2),
                                     interval=5),
        mw.monitors.FluxTimeMonitor("time_flux", axis="x", position=0.12, interval=2),
        mw.monitors.PermittivityMonitor("perm", position=(0.0, 0.0, 0.0), size=(0.2, 0.2, 0.2)),
    ),
    ))
    return scene


@pytest.fixture(scope="module")
def result():
    return mw.Simulation.fdtd(_scene(), frequencies=FREQUENCIES,
                              config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=160), full_field_dft=False)).run()


def _f(data):
    return tuple(data.coords["f"].tolist())


def test_point_monitor_is_field_data_stacking_the_recorded_components(result):
    raw = result.raw_monitor("point")
    data = result.monitor("point")
    assert isinstance(data, mw.FieldData)
    assert data.dims == ("component", "f") and data.coords["component"] == ("Ez", "Hy") and _f(data) == FREQUENCIES
    assert data.units == "mixed" and data.attrs["component_units"] == {"Ez": "V/m", "Hy": "A/m"}
    assert data.attrs["phasor_convention"] == "exp(-i omega t), peak" and data.attrs["normalization"] == "raw"
    for label in ("Ez", "Hy"):
        component = data.sel(component=label)
        assert torch.equal(component.values, raw["components"][label])
        assert component.attrs["component"] == label and component.units == raw_units(label)
    enum_selected = data.sel(component=mw.monitors.FieldComponent.HY)
    assert torch.equal(enum_selected.values, raw["components"]["Hy"]) and enum_selected.attrs["component"] == "Hy"
    selected = result.monitor("point", freq_index=1)
    assert selected.dims == ("component",)
    assert torch.equal(selected.values, torch.stack([raw["components"][label][1] for label in ("Ez", "Hy")]))
    assert "data" not in raw


def raw_units(label):
    return "V/m" if label[0] == "E" else "A/m"


def test_plane_monitors_keep_native_or_aligned_samples(result):
    raw = result.raw_monitor("plane")
    data = result.monitor("plane")
    assert data.dims == ("component", "f", "y", "z") and data.attrs["yee_location"] == "edge_z"
    assert torch.equal(data.values[0], raw["components"]["Ez"]["data"])
    assert torch.equal(data.coords["y"].cpu(), torch.as_tensor(raw["components"]["Ez"]["coords"][0]))

    raw = result.raw_monitor("aligned")
    data = result.monitor("aligned")
    assert data.attrs["yee_location"] == "cell_center" and data.coords["component"] == ("Ey", "Ez", "Hy", "Hz")
    for index, label in enumerate(("Ey", "Ez", "Hy", "Hz")):
        assert torch.equal(data.values[index], raw[label])

    # Hx is normal to an x plane: the alignment does not cover it, so components stay on their own grids.
    raw = result.raw_monitor("native")
    data = result.monitor("native")
    assert isinstance(data, dict) and set(data) == {"Ey", "Hx"}
    for label, item in data.items():
        assert item.dims == ("f", "y", "z") and item.attrs["component"] == label
        assert torch.equal(item.values, raw["components"][label]["data"])


def test_flux_monitors_are_flux_data(result):
    for name in ("flux", "box"):
        raw = result.raw_monitor(name)
        data = result.monitor(name)
        assert isinstance(data, mw.FluxData) and data.dims == ("f",) and data.units == "W"
        assert torch.equal(data.values, raw["flux"])
        assert torch.equal(result.monitor(name, frequency=FREQUENCIES[1]).values,
                           result.raw_monitor(name, frequency=FREQUENCIES[1])["flux"])


def test_mode_and_gaussian_overlaps_are_mode_amplitudes(result):
    data = result.monitor("mode_out")
    assert isinstance(data, mw.ModeAmplitudeData) and data.dims == ("mode", "propagation", "f") and _f(data) == FREQUENCIES
    for index in range(len(FREQUENCIES)):
        modal = result._mode_overlap("mode_out", result.raw_monitor("mode_out", freq_index=index))
        at = data.isel(mode=0, f=index)
        assert torch.equal(at.sel(propagation="+").values, modal["amplitude_forward"])
        assert torch.equal(at.sel(propagation=mw.monitors.Direction.MINUS).values, modal["amplitude_backward"])
        assert torch.equal(at.sel(propagation="+").attrs["power"], modal["power_forward"])
        assert at.attrs["effective_index"] == modal["effective_index"]
        selected = result.monitor("mode_out", freq_index=index)
        assert selected.dims == ("mode", "propagation") and torch.equal(selected.values, at.values[None])

    data = result.monitor("overlap")
    assert data.units == "sqrt(W)" and _f(data) == FREQUENCIES
    fractions = data.sel(propagation="+").isel(mode=0).attrs["power_fraction"]
    assert fractions.shape == (len(FREQUENCIES),)


def test_dipole_emission_is_delivered_power(result):
    raw = result.raw_monitor("emission")
    data = result.monitor("emission")
    assert isinstance(data, mw.FluxData) and data.dims == ("f",)
    assert torch.equal(data.values, raw["power_delivered"])
    assert torch.equal(data.attrs["current_spectrum"], torch.as_tensor(raw["current_spectrum"]))
    purcell = mw.postprocess.purcell_factor(result, result, "emission")
    assert torch.equal(purcell["purcell_factor"], torch.ones_like(purcell["purcell_factor"]))


def test_time_traces_have_a_time_dimension(result):
    raw = result.raw_monitor("trace")
    data = result.monitor("trace")
    assert data.dims == ("component", "t") and torch.equal(data.coords["t"], raw["t"])
    assert torch.equal(data.sel(component="Hy").values, raw["components"]["Hy"])
    assert data.attrs["field_time_offsets_steps"] == raw["field_time_offsets_steps"]

    raw = result.raw_monitor("sheet")
    data = result.monitor("sheet")
    assert set(data) == {"Ez", "Hy"} and "data" not in raw and "field" not in raw
    for label, item in data.items():
        assert item.dims == ("t", "y", "z") and torch.equal(item.values, raw["components"][label])
        assert item.sizes["y"] == len(raw["component_coords"][label][0])

    raw = result.raw_monitor("time_flux")
    data = result.monitor("time_flux")
    assert isinstance(data, mw.FluxData) and data.dims == ("t",) and torch.equal(data.values, raw["flux"])


def test_material_monitor_stacks_the_permittivity_components(result):
    raw = result.raw_monitor("perm")
    data = result.monitor("perm")
    assert set(data) == {"eps"} and isinstance(data["eps"], TensorData)
    assert data["eps"].dims == ("component", "f", "x", "y", "z")
    for index, axis in enumerate("xyz"):
        assert torch.equal(data["eps"].values[index], raw[f"eps_{axis}"])


def test_monitors_maps_every_recorded_monitor_to_its_typed_data(result):
    monitors = result.monitors
    assert isinstance(monitors["point"], mw.FieldData) and isinstance(monitors["flux"], mw.FluxData)
    assert torch.equal(monitors["point"].values, result.monitor("point").values)


def test_single_frequency_fields_move_to_time_snapshot():
    scene = _scene(frequencies=FREQUENCIES[:1])
    snapshot_run = mw.Simulation.fdtd(scene, frequencies=FREQUENCIES[:1],
                                      config=mw.FDTDConfig(run_time=mw.TimeConfig(time_steps=60))).run()
    with pytest.raises(ValueError, match="time_snapshot"):
        snapshot_run.fields
    snapshot = snapshot_run.time_snapshot
    assert set(snapshot) == {"EX", "EY", "EZ"}
    ez = snapshot["EZ"]
    assert ez.dims == ("x", "y", "z") and ez.attrs["kind"] == "time_snapshot" and ez.attrs["step"] == 60
    assert torch.equal(ez.values, snapshot_run.time_snapshot["EZ"].values)

    phasor_run = mw.Simulation.fdtd(scene, frequencies=FREQUENCIES[:1], config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=60), full_field_dft=True)).run()
    fields = phasor_run.fields
    assert fields["EZ"].dims == ("f", "x", "y", "z") and fields["EZ"].attrs["yee_location"] == "edge_z"
    assert torch.equal(fields["EZ"].values[0], phasor_run.E.z.isel(f=0).values)
    with pytest.raises(ValueError, match="no time snapshot"):
        phasor_run.time_snapshot


def test_gradients_through_typed_values_equal_gradients_through_the_payload():
    density = torch.full((1, 1, 1), 0.5, device="cuda", requires_grad=True)
    scene = mw.Scene(subpixel=_SUBPIXEL, domain=mw.Domain(bounds=((-0.4, 0.4),) * 3),
                     grid=mw.GridSpec.uniform(0.1), boundary=mw.BoundarySpec.mur(),
                     material_regions=(
                         mw.MaterialRegion(name="density", geometry=mw.Box(position=(0., 0., 0.), size=(0.2,) * 3),
                                           density=density, eps_bounds=(1., 4.)),
                     ),
                     sources=(
                         mw.PointDipole(position=(-0.2, 0., 0.), polarization="Ez", width=0.1,
                                        source_time=mw.GaussianPulse(1e9, 0.5e9)),
                     ),
                     monitors=(
                         mw.PointMonitor("probe", (0.1, 0., 0.), fields=("Ez", "Hy")),
                         mw.PlaneMonitor("plane", axis="z", position=0.0, fields=("Ez",)),
                     ))
    result = mw.Simulation.fdtd(scene, frequencies=1e9, config=mw.FDTDConfig(
        run_time=mw.TimeConfig(time_steps=120), cuda_graph=False)).run()
    typed = result.monitor("probe").sel(component="Ez").values.abs().square().sum()
    typed = typed + result.monitor("plane").values.abs().square().sum()
    raw = result.raw_monitor("probe")["components"]["Ez"].abs().square().sum()
    raw = raw + result.raw_monitor("plane")["components"]["Ez"]["data"].abs().square().sum()
    assert torch.equal(typed.detach(), raw.detach())
    (typed_grad,) = torch.autograd.grad(typed, density, retain_graph=True)
    (raw_grad,) = torch.autograd.grad(raw, density)
    assert torch.equal(typed_grad, raw_grad) and bool(typed_grad.abs().sum() > 0)
