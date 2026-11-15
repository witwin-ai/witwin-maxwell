"""MoM results are physics-typed: PointMonitors as FieldData, far fields as FarFieldData, layered S as SParameterData."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.results import far_field_data, point_monitor_data
from witwin.maxwell.result import _monitor_mapping_from_snapshot, _monitor_mapping_snapshot

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

PHASOR = {"phasor_convention": "exp(-i omega t), peak", "normalization": "raw"}
FREQ = 3e8


def _samples(count, *, seed=0, requires_grad=False):
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randn(count, 6, dtype=torch.complex128, generator=generator)
    return samples.requires_grad_(requires_grad)


def test_point_monitor_units_follow_the_shared_component_rule():
    samples = _samples(1)
    for fields, units in ((("Ex", "Ez"), "V/m"), (("Hy",), "A/m"), (("Ex", "Hy"), "mixed")):
        data = point_monitor_data(mw.PointMonitor("p", (0., 0., 0.), fields=fields), samples, (FREQ,))
        assert (data.units, dict(data.attrs["component_units"])) == mw.results.field_component_units(fields)


def test_far_field_data_is_a_view_with_directions_in_attrs():
    generator = torch.Generator().manual_seed(3)
    directions = torch.nn.functional.normalize(torch.randn(4, 3, dtype=torch.float64, generator=generator), dim=-1)
    electric, magnetic = (torch.randn(4, 3, dtype=torch.complex128, generator=generator).requires_grad_()
                          for _ in range(2))
    far_e, far_h = far_field_data(electric, magnetic, directions, FREQ)
    for data, raw, components, units in ((far_e, electric, ("Ex", "Ey", "Ez"), "V"),
                                         (far_h, magnetic, ("Hx", "Hy", "Hz"), "A")):
        assert type(data) is mw.FarFieldData and data.dims == ("f", "direction", "component") and data.units == units
        assert data.values.data_ptr() == raw.data_ptr() and torch.equal(data.values[0], raw)
        assert data.coords["component"] == components and torch.equal(data.coords["direction"], torch.arange(4))
        assert torch.equal(data.coords["f"], torch.tensor((FREQ,), dtype=torch.float64))
        assert {key: data.attrs[key] for key in PHASOR} == PHASOR and "direction" not in data.attrs
        for axis, key in enumerate(("ux", "uy", "uz")):
            assert torch.equal(data.attrs[key], directions[:, axis])
        picked = data.isel(f=0, direction=2)
        assert torch.equal(picked.values, raw[2]) and float(picked.attrs["uz"]) == float(directions[2, 2])
    assert far_e.attrs["quantity"] == "r*exp(-ikr)*E" and far_h.attrs["quantity"] == "r*exp(-ikr)*H"
    far_e.sel(component="Ez").values.abs().square().sum().backward()
    expected = torch.zeros_like(electric)
    expected[:, 2] = 2 * electric[:, 2].detach()
    torch.testing.assert_close(electric.grad, expected, rtol=1e-15, atol=0)


def test_point_monitor_data_layout_selection_and_autograd():
    monitor = mw.PointMonitor("probe", (.3, .2, .1), fields=("Hy", "Ex"))
    samples = _samples(2, requires_grad=True)
    data = point_monitor_data(monitor, samples, (FREQ, 2 * FREQ), region_id=0, distance_to_surface_m=.1)
    assert type(data) is mw.FieldData and data.dims == ("component", "f") and data.units == "mixed"
    assert data.coords["component"] == ("Hy", "Ex")
    assert torch.equal(data.coords["f"], torch.tensor((FREQ, 2 * FREQ), dtype=torch.float64))
    assert dict(data.attrs) == {**PHASOR, "component_units": {"Hy": "A/m", "Ex": "V/m"}, "position": (.3, .2, .1),
                                "field_kind": "total", "sampling": "physical_position", "region_id": 0,
                                "distance_to_surface_m": .1}
    # Declaration order, Ex..Hz sample columns: Hy is column 4, Ex column 0.
    assert torch.equal(data.values, samples[:, [4, 0]].T)
    assert torch.equal(data.sel(component="Ex", f=2 * FREQ).values, samples[1, 0])
    data.sel(component="Hy").values.abs().square().sum().backward()
    expected = torch.zeros_like(samples)
    expected[:, 4] = 2 * samples[:, 4].detach()
    torch.testing.assert_close(samples.grad, expected, rtol=1e-15, atol=0)


def test_point_monitor_snapshot_round_trips_type_and_dc_record():
    monitor = mw.PointMonitor("inside", (0., 0., 0.), fields=("Ex", "Hx"))
    samples = _samples(2, seed=1)
    dc = {"frequency": 0., "components": {"Ex": torch.tensor(1.5, dtype=torch.float64)},
          "potential_V": torch.tensor(.25, dtype=torch.float64), "magnetic_available": False}
    data = point_monitor_data(monitor, samples, (FREQ, 2 * FREQ), region_id=1, distance_to_surface_m=.2,
                              dc=dc, physical_qualification=False)
    restored = _monitor_mapping_from_snapshot(_monitor_mapping_snapshot({"inside": data}))["inside"]
    assert type(restored) is mw.FieldData and restored.dims == data.dims and restored.units == "mixed"
    assert restored.coords["component"] == data.coords["component"]
    assert torch.equal(restored.coords["f"], data.coords["f"]) and torch.equal(restored.values, data.values)
    assert restored.attrs["dc"]["components"]["Ex"].item() == 1.5
    assert restored.attrs["dc"]["magnetic_available"] is False
    assert {key: restored.attrs[key] for key in ("region_id", "position", "component_units")} == {
        "region_id": 1, "position": (0., 0., 0.), "component_units": {"Ex": "V/m", "Hx": "A/m"}}


@requires_cuda
def test_linear_point_monitors_are_field_data_equal_to_queries(tmp_path):
    from witwin.maxwell.mom.operator import IntegrationConfig
    scene = mw.Scene(structures=[mw.Structure(geometry=mw.Sphere(radius=.1), material=mw.Material(eps_r=2.25))],
                     sources=[mw.PlaneWave()],
                     monitors=[mw.PointMonitor("probe", (.3, .2, .1), fields=("Ex", "Hy")),
                               mw.PointMonitor("magnetic", (.3, .2, .1), fields=("Hz",))])
    result = mw.Simulation.mom(scene, frequencies=FREQ, config=mw.mom.MoMConfig(
        subdivisions=0, integration=IntegrationConfig(order=3, singular_order=6))).run()
    point = torch.tensor([[.3, .2, .1]], device="cuda", dtype=torch.float64)
    electric, magnetic = result.query_fields(point, region_id="exterior", field_kind="total")
    probe = result.monitor("probe")
    assert type(probe) is mw.FieldData and probe.dims == ("component", "f") and probe.units == "mixed"
    assert probe.values.shape == (2, 1) and probe.values.device.type == "cuda"
    assert {key: probe.attrs[key] for key in PHASOR} == PHASOR
    assert probe.attrs["region_id"] == 0 and probe.attrs["structure_name"] is None
    torch.testing.assert_close(probe.values[:, 0], torch.stack((electric[0, 0], magnetic[0, 1])))
    assert result.monitor("magnetic").units == "A/m"
    assert torch.equal(result.monitor("probe", frequency=FREQ).values, probe.values[:, 0])
    assert torch.equal(result.at(freq_index=0).monitor("probe").sel(component="Hy").values, probe.values[1, 0])
    directions = torch.tensor([[0., 0., 1.], [.6, 0., .8]], device="cuda", dtype=torch.float64)
    raw = mw.mom.surface_currents(result).far_field(directions)
    typed = mw.mom.far_field(result, directions)
    for data, values, units in zip(typed, raw, ("V", "A")):
        assert type(data) is mw.FarFieldData and data.dims == ("f", "direction", "component") and data.units == units
        assert torch.equal(data.values[0], values) and torch.equal(data.coords["f"].cpu(), torch.tensor((FREQ,), dtype=torch.float64))
        assert torch.equal(data.attrs["ux"], directions[:, 0])
    result.save(tmp_path / "mom.pt")
    restored = mw.Result.load(tmp_path / "mom.pt", scene=scene).monitor("probe")
    assert type(restored) is mw.FieldData and restored.coords["component"] == ("Ex", "Hy")
    assert torch.equal(restored.values, probe.values.cpu()) and dict(restored.attrs) == dict(probe.attrs)


@requires_cuda
def test_layered_s_parameters_are_a_view_with_live_gradients():
    from tests.support.layered_gradient_fixtures import FREQUENCIES, SIGMA, loop_config, loop_scene

    def run():
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        network = mw.Simulation.mom(loop_scene(sheet_sigma=sigma), frequencies=FREQUENCIES,
                                    config=loop_config("sheet")).run().network
        return sigma, network

    sigma, network = run()
    s = network.s_parameters
    assert type(s) is mw.SParameterData and s.dims == ("f", "out_port", "in_port")
    assert s.values.data_ptr() == network.s.data_ptr() and s.values is network.s
    assert s.coords["out_port"] == s.coords["in_port"] == ("p1", "p2")
    assert torch.equal(s.coords["f"], network.frequencies.to(s.coords["f"].device))
    assert torch.equal(s.attrs["z0"], network.z0)
    typed, = torch.autograd.grad(s.sel(out_port="p2", in_port="p1").values.abs().sum(), sigma)
    sigma, network = run()
    raw, = torch.autograd.grad(network.s[:, 1, 0].abs().sum(), sigma)
    assert typed.item() != 0 and torch.equal(typed, raw)


@requires_cuda
@pytest.mark.parametrize("backend", ["DENSE", "MATRIX_FREE"])
def test_layered_network_s_stays_the_tensor_under_typed_s_parameters(backend):
    """R5-3 (a): ``NetworkData.s`` stays the [f, out, in] tensor; ``s_parameters`` is its SParameterData view.

    Covers the dense and the streamed (matrix_free) layered backends, configured with the option Enums;
    the recorded layered payload and network metadata keep the value strings.
    """
    import dataclasses

    from tests.support.layered_gradient_fixtures import FREQUENCIES, loop_config, loop_scene

    member = mw.mom.LayeredOperatorBackend[backend]
    config = loop_config("sheet")
    config = dataclasses.replace(config, layered=dataclasses.replace(
        config.layered, operator_backend=member, conductor_model=mw.mom.LayeredConductorModel.SHEET))
    assert config.layered.operator_backend is member
    result = mw.Simulation.mom(loop_scene(), frequencies=FREQUENCIES, config=config).run()
    network = result.network
    assert type(network.s) is torch.Tensor and network.s.shape == (len(FREQUENCIES), 2, 2)
    s = network.s_parameters
    assert type(s) is mw.SParameterData and s.dims == ("f", "out_port", "in_port")
    assert s.values is network.s and s.coords["out_port"] == s.coords["in_port"] == network.port_names
    assert torch.equal(s.coords["f"], network.frequencies.to(s.coords["f"].device))
    assert torch.equal(s.attrs["z0"], network.z0)
    assert torch.equal(s.sel(out_port="p2", in_port="p1").values, network.s[:, 1, 0])
    layered = result.metadata["layered"]
    assert layered.get("operator_backend", "dense") == member.value
    assert type(layered.get("operator_backend", "")) is str
    assert type(network.metadata["conductor_model"]) is str and network.metadata["conductor_model"] == "sheet"
