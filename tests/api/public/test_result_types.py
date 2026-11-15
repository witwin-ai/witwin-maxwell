from __future__ import annotations

import sys

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.results import (
    FarFieldData,
    FieldData,
    FluxData,
    ModeAmplitudeData,
    SParameterData,
    TensorData,
    field_component_units,
)

FREQUENCIES = torch.tensor([1.0e9, 1.5e9, 2.0e9], dtype=torch.float64)
X = torch.tensor([-1.0e-3, 0.0, 1.0e-3, 2.0e-3], dtype=torch.float64)


def _field(values=None, device="cpu"):
    if values is None:
        values = torch.arange(3 * 3 * 4, dtype=torch.float64, device=device).reshape(3, 3, 4)
    return FieldData(
        values,
        ("component", "f", "x"),
        coords={"component": ("Ex", "Ey", "Ez"), "f": FREQUENCIES, "x": X},
        units="V/m",
        attrs={"normalization": "raw", "yee_location": "edge"},
    )


def test_types_are_exported_at_the_top_level():
    for name in ("FieldData", "FluxData", "ModeAmplitudeData", "SParameterData", "FarFieldData"):
        assert getattr(mw, name) is getattr(mw.results, name)
        assert name in mw.__all__
        assert issubclass(getattr(mw, name), TensorData)


def test_construction_keeps_the_tensor_and_freezes_metadata():
    values = torch.zeros(3, 3, 4, dtype=torch.complex64)
    data = _field(values)
    assert data.values is values
    assert data.sizes == {"component": 3, "f": 3, "x": 4}
    assert data.coords["component"] == ("Ex", "Ey", "Ez")
    with pytest.raises(TypeError):
        data.attrs["units"] = "A/m"
    with pytest.raises(AttributeError):
        data.units = "A/m"
    assert repr(data) == "FieldData(component: 3, f: 3, x: 4; units='V/m', dtype=torch.complex64, device=cpu)"


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (dict(dims=("component", "f")), ValueError, "name 2 dimensions"),
        (dict(dims=("f", "f", "x")), ValueError, "unique"),
        (dict(dims=("f", "component", "x")), ValueError, "keep the order"),
        (dict(coords={"y": X}), ValueError, "not in dims"),
        (dict(coords={"x": X[:3]}), ValueError, "length 4"),
        (dict(coords={"component": ("Ex", "Ex", "Ez")}), ValueError, "unique"),
        (dict(coords={"component": ("Ex", "Ey")}), ValueError, "2 labels"),
        (dict(coords={"x": torch.ones(4, dtype=torch.complex64)}), TypeError, "real numeric"),
        (dict(attrs={"units": "V/m"}), ValueError, "units"),
        (dict(attrs={"normalization": "unit_power"}), ValueError, "normalization"),
    ],
)
def test_construction_refuses_inconsistent_metadata(kwargs, error, match):
    arguments = dict(values=torch.zeros(3, 3, 4), dims=("component", "f", "x"))
    arguments.update(kwargs)
    with pytest.raises(error, match=match):
        FieldData(**arguments)


def test_layouts_admit_ordered_subsets():
    FieldData(torch.zeros(3, 2), ("component", "f"))
    FieldData(torch.zeros(3, 2, 4, 5), ("component", "f", "x", "z"))
    FarFieldData(torch.zeros(2, 5, 7, 2), ("f", "theta", "phi", "component"))
    FarFieldData(torch.zeros(2, 5, 7, 2), ("f", "ux", "uy", "component"))
    ModeAmplitudeData(torch.zeros(2, 2, 3), ("mode", "propagation", "f"), coords={"propagation": ("+", "-")})
    assert FluxData(torch.zeros(3), ("f",), coords={"f": FREQUENCIES}).units == "W"
    assert SParameterData(torch.zeros(3, 2, 2), ("f", "out_port", "in_port")).units == "1"
    with pytest.raises(ValueError, match="keep the order"):
        FarFieldData(torch.zeros(5, 7), ("theta", "uy"))
    TensorData(torch.zeros(2, 3), ("a", "b"))


def test_isel_drops_integer_dims_and_keeps_slices_and_sequences():
    data = _field()
    point = data.isel(component=2, x=-1)
    assert type(point) is FieldData
    assert point.dims == ("f",)
    assert torch.equal(point.values, data.values[2, :, -1])
    assert set(point.coords) == {"f"}

    window = data.isel(x=slice(1, 3), component=[2, 0])
    assert window.dims == ("component", "f", "x")
    assert torch.equal(window.values, data.values[[2, 0]][:, :, 1:3])
    assert window.coords["component"] == ("Ez", "Ex")
    assert torch.equal(window.coords["x"], X[1:3])
    assert window.units == "V/m" and window.attrs == data.attrs

    view = data.isel(component=1, x=slice(0, 2))
    assert view.values.untyped_storage().data_ptr() == data.values.untyped_storage().data_ptr()


def test_isel_refuses_unknown_dims_and_bad_positions():
    data = _field()
    with pytest.raises(ValueError, match="no dimension 'y'"):
        data.isel(y=0)
    with pytest.raises(IndexError, match="out of range"):
        data.isel(x=[0, 4])
    with pytest.raises(TypeError, match="integer sequence"):
        data.isel(x=[0.5])
    with pytest.raises(TypeError, match="integer sequence"):
        data.isel(x=True)


def test_sel_by_label_and_coordinate_value():
    data = _field()
    ez = data.sel(component="Ez", f=1.5e9)
    assert ez.dims == ("x",)
    assert torch.equal(ez.values, data.values[2, 1])

    subset = data.sel(f=[2.0e9, 1.0e9], component=["Ey"])
    assert subset.dims == ("component", "f", "x")
    assert torch.equal(subset.values, data.values[[1]][:, [2, 0]])
    assert torch.equal(subset.coords["f"], FREQUENCIES[[2, 0]])

    near = data.sel(x=0.9e-3, method="nearest")
    assert torch.equal(near.values, data.values[:, :, 2])
    assert torch.equal(data.sel(f=torch.tensor(2.0e9)).values, data.values[:, 2])


def test_sel_matches_after_rounding_to_the_coordinate_dtype():
    data = FluxData(torch.arange(3.0), ("f",), coords={"f": torch.tensor([1.1e9, 1.2e9, 1.3e9], dtype=torch.float32)})
    assert float(data.sel(f=1.2e9).values) == 1.0


def test_sel_refuses_missing_values_and_uncoordinated_dims():
    data = _field()
    with pytest.raises(KeyError, match="not found"):
        data.sel(f=1.25e9)
    with pytest.raises(KeyError, match="'Hx' not found"):
        data.sel(component="Hx")
    with pytest.raises(ValueError, match="labelled"):
        data.sel(component="Ex", method="nearest")
    with pytest.raises(ValueError, match="method"):
        data.sel(f=1.0e9, method="pad")
    with pytest.raises(TypeError, match="slices"):
        data.sel(f=slice(1.0e9, 2.0e9))
    bare = TensorData(torch.zeros(2, 3), ("a", "b"))
    with pytest.raises(ValueError, match="no coordinates"):
        bare.sel(a=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_gradients_flow_through_sel_and_isel(dtype):
    values = torch.randn(3, 3, 4, dtype=dtype, requires_grad=True)

    def select(tensor):
        data = _field(tensor)
        return (
            data.sel(component="Ey", f=[2.0e9, 1.0e9]).values,
            data.isel(x=slice(1, None), component=[2, 0]).values,
            data.sel(x=0.0).isel(f=-1).values,
        )

    assert torch.autograd.gradcheck(select, (values,))


def test_to_xarray_returns_a_detached_host_copy():
    pytest.importorskip("xarray")
    values = torch.randn(3, 3, 4, dtype=torch.complex128, requires_grad=True)
    array = _field(values).to_xarray()
    assert array.dims == ("component", "f", "x")
    assert list(array.coords["component"].values) == ["Ex", "Ey", "Ez"]
    assert (array.coords["f"].values == FREQUENCIES.numpy()).all()
    assert array.attrs == {"normalization": "raw", "yee_location": "edge", "units": "V/m"}
    assert (array.values == values.detach().numpy()).all()
    assert array.sel(component="Ez", f=1.5e9).values.tolist() == values[2, 1].detach().tolist()


def test_to_xarray_without_xarray_names_the_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError, match=r"FieldData.to_xarray\(\) needs the optional dependency xarray"):
        _field().to_xarray()


def _directions(dtype=torch.float64):
    theta = torch.tensor([0.0, 0.4, 1.1, 2.0, 3.0], dtype=dtype)
    phi = torch.tensor([0.0, 0.9, -1.3, 2.5, 4.0], dtype=dtype)
    return torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()), dim=1)


def _far_field(values=None, device="cpu"):
    if values is None:
        values = torch.arange(3 * 5 * 2, dtype=torch.float64, device=device).reshape(3, 5, 2)
    return FarFieldData.from_directions(
        values, _directions(), FREQUENCIES, ("Etheta", "Ephi"), "V", attrs={"normalization": "raw"}
    )


def test_far_field_over_directions_holds_unit_vectors_in_attrs():
    directions = _directions()
    data = _far_field()
    assert data.dims == ("f", "direction", "component")
    assert torch.equal(data.coords["direction"], torch.arange(5))
    assert data.coords["component"] == ("Etheta", "Ephi")
    assert data.units == "V" and data.attrs["normalization"] == "raw"
    for axis, key in enumerate(("ux", "uy", "uz")):
        assert torch.equal(data.attrs[key], directions[:, axis])
    FarFieldData(torch.zeros(3, 2), ("f", "component"))  # the angular layouts need no direction attrs


def test_direction_selection_keeps_each_selected_vector():
    directions = _directions()
    data = _far_field()
    for index in ([3, 0, 3], slice(1, 4), torch.tensor([4, 1])):
        view = data.isel(direction=index)
        assert type(view) is FarFieldData
        expected = torch.arange(5)[index]
        assert torch.equal(view.coords["direction"], expected)
        assert torch.equal(view.values, data.values[:, expected])
        assert torch.equal(torch.stack([view.attrs[k] for k in ("ux", "uy", "uz")], 1), directions[expected])

    by_value = data.sel(direction=[4, 2], component="Ephi")
    assert by_value.dims == ("f", "direction")
    assert torch.equal(by_value.attrs["uz"], directions[[4, 2], 2])

    single = data.sel(direction=2).isel(f=0)
    assert single.dims == ("component",) and "direction" not in single.coords
    assert torch.equal(torch.stack([single.attrs[k] for k in ("ux", "uy", "uz")]), directions[2])
    assert single.attrs["ux"].ndim == 0
    assert data.isel(f=1).attrs["ux"] is data.attrs["ux"]


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: FarFieldData.from_directions(torch.zeros(3, 5, 2), _directions() * 1.001, FREQUENCIES, ("a", "b"), "V"),
         ValueError, "unit vectors"),
        (lambda: FarFieldData.from_directions(torch.zeros(3, 5, 2), _directions()[:, :2], FREQUENCIES, ("a", "b"), "V"),
         ValueError, r"shape \[D, 3\]"),
        (lambda: FarFieldData.from_directions(torch.zeros(3, 4, 2), _directions(), FREQUENCIES, ("a", "b"), "V"),
         ValueError, "length 4"),
        (lambda: FarFieldData.from_directions(torch.zeros(3, 5, 2), _directions(), FREQUENCIES[:2], ("a", "b"), "V"),
         ValueError, "length 3"),
        (lambda: FarFieldData.from_directions(torch.zeros(3, 5, 2), _directions(), FREQUENCIES, ("a", "b"), "V",
                                              attrs={"uz": torch.ones(5)}),
         ValueError, "do not pass"),
        (lambda: FarFieldData(torch.zeros(3, 5), ("f", "direction")), ValueError, "needs attrs"),
        (lambda: FarFieldData(torch.zeros(3, 5), ("f", "direction"),
                              attrs={"ux": torch.ones(4), "uy": torch.zeros(4), "uz": torch.zeros(4)}),
         ValueError, r"shape \(5,\)"),
        (lambda: FarFieldData(torch.zeros(3, 1), ("f", "direction"),
                              attrs={"ux": torch.ones(1, dtype=torch.long), "uy": torch.zeros(1), "uz": torch.zeros(1)}),
         TypeError, "real floating"),
        (lambda: FarFieldData(torch.zeros(3, 2), ("f", "theta"),
                              attrs={"ux": torch.tensor(1.0), "uy": torch.tensor(0.0), "uz": torch.tensor(0.0)}),
         ValueError, "layout"),
        (lambda: FarFieldData(torch.zeros(3, 1), ("f", "direction"),
                              attrs={"ux": torch.tensor([float("nan")]), "uy": torch.zeros(1), "uz": torch.zeros(1)}),
         ValueError, "unit vectors"),
    ],
)
def test_far_field_directions_refuse_inconsistent_vectors(call, error, match):
    with pytest.raises(error, match=match):
        call()


def test_unit_vectors_are_checked_at_construction_only(monkeypatch):
    from witwin.maxwell.results import typed

    calls = []
    check = typed._check_unit_vectors
    monkeypatch.setattr(typed, "_check_unit_vectors", lambda *vectors: calls.append(1) or check(*vectors))
    data = _far_field()
    assert len(calls) == 1
    data.isel(direction=[4, 0]).isel(direction=1)
    data.isel(direction=slice(1, 3), f=0).sel(direction=2, component="Ephi")
    data.to("cpu")
    assert len(calls) == 1
    FarFieldData(data.values, data.dims, coords=dict(data.coords), units="V", attrs=dict(data.attrs))
    assert len(calls) == 2


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_gradients_flow_through_direction_selection(dtype):
    values = torch.randn(3, 5, 2, dtype=dtype, requires_grad=True)
    directions = _directions().requires_grad_()

    def select(tensor, vectors):
        data = FarFieldData.from_directions(tensor, vectors, FREQUENCIES, ("Etheta", "Ephi"), "V")
        picked = data.sel(direction=[4, 1], f=1.5e9)
        return picked.values, data.isel(direction=slice(2, None)).values, picked.attrs["uz"], data.isel(direction=3).attrs["uy"]

    assert torch.autograd.gradcheck(select, (values, directions))


def test_direction_attrs_export_as_non_dimension_coordinates():
    pytest.importorskip("xarray")
    array = _far_field().to_xarray()
    assert array.dims == ("f", "direction", "component")
    assert array.coords["uz"].dims == ("direction",)
    assert (array.coords["uz"].values == _directions()[:, 2].numpy()).all()
    assert set(array.attrs) == {"normalization", "units"}
    assert float(_far_field().isel(direction=1).to_xarray().coords["ux"]) == float(_directions()[1, 0])


def test_field_component_units_labels_kinds():
    assert field_component_units(("Ex", "Ey")) == ("V/m", {"Ex": "V/m", "Ey": "V/m"})
    assert field_component_units(["Hz"]) == ("A/m", {"Hz": "A/m"})
    assert field_component_units(("Ex", "Hy")) == ("mixed", {"Ex": "V/m", "Hy": "A/m"})
    for bad in ((), ("Ex", "Jz"), ("",)):
        with pytest.raises(ValueError, match="E\\*/H\\* labels"):
            field_component_units(bad)
    units, component_units = field_component_units(("Ex", "Hy"))
    mixed = FieldData(torch.zeros(2, 3), ("component", "f"), coords={"component": ("Ex", "Hy")}, units=units,
                      attrs={"component_units": component_units})
    assert mixed.units == "mixed" and dict(mixed.attrs["component_units"]) == {"Ex": "V/m", "Hy": "A/m"}
    with pytest.raises(ValueError, match="component_units"):
        FieldData(torch.zeros(2, 3), ("component", "f"), units="mixed")


def _network(device="cpu"):
    s = torch.tensor(
        [[[0.1, 0.9], [0.9, 0.2]], [[0.2, 0.8], [0.8, 0.3]], [[0.3, 0.7], [0.7, 0.4]]],
        dtype=torch.complex128,
        device=device,
    )
    return mw.rf.NetworkData(frequencies=FREQUENCIES.to(device), s=s, z0=50.0, port_names=("a", "b"))


def test_network_s_parameters_share_the_s_matrix():
    network = _network()
    data = network.s_parameters
    assert type(data) is SParameterData
    assert data.values is network.s
    assert data.dims == ("f", "out_port", "in_port")
    assert data.coords["out_port"] == data.coords["in_port"] == ("a", "b")
    assert torch.equal(data.coords["f"], network.frequencies)
    assert data.attrs["z0"] is network.z0
    assert data.attrs["port_names"] == ("a", "b")
    assert data.attrs["phasor_convention"] == network.phasor_convention
    assert torch.equal(data.sel(out_port="b", in_port="a").values, network.s[:, 1, 0])


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
def test_device_placement_follows_the_values():
    values = torch.randn(3, 3, 4, dtype=torch.complex64, device="cuda")
    data = _field(values)
    assert data.device.type == "cuda"
    assert all(coord.device.type == "cuda" for coord in (data.coords["f"], data.coords["x"]))
    for view in (data.sel(component="Ez", f=[1.0e9, 2.0e9]), data.isel(x=[3, 0]), data.sel(x=1.1e-3, method="nearest")):
        assert view.values.device == values.device
        assert all(coord.device == values.device for coord in view.coords.values() if isinstance(coord, torch.Tensor))
    host = data.to("cpu")
    assert host.device.type == "cpu" and host.coords["f"].device.type == "cpu"
    assert torch.equal(host.values, values.cpu())
    assert host.to("cuda").device.type == "cuda"


@cuda
def test_device_transfer_moves_tensor_attrs_and_keeps_gradients():
    network = _network("cuda")
    data = network.s_parameters
    assert data.device.type == "cuda" and data.attrs["z0"].device.type == "cuda"
    host = data.to("cpu")
    assert host.attrs["z0"].device.type == "cpu" and host.coords["f"].device.type == "cpu"

    values = torch.randn(3, 3, 4, dtype=torch.complex128, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(lambda tensor: _field(tensor).sel(component="Ex").to("cpu").values, (values,))
    assert _field(values).to_xarray().shape == (3, 3, 4)


@cuda
def test_direction_far_field_moves_its_vectors_with_the_values():
    values = torch.randn(3, 5, 2, dtype=torch.complex64, device="cuda", requires_grad=True)
    data = FarFieldData.from_directions(values, _directions(), FREQUENCIES, ("Etheta", "Ephi"), "V")
    assert all(data.attrs[key].device == values.device for key in ("ux", "uy", "uz"))
    view = data.sel(direction=[3, 1])
    assert view.attrs["ux"].device == values.device and view.coords["direction"].device == values.device
    host = data.to("cpu")
    assert all(host.attrs[key].device.type == "cpu" for key in ("ux", "uy", "uz"))
    assert torch.equal(host.attrs["uy"], _directions()[:, 1])
    assert host.to("cuda").attrs["uz"].device.type == "cuda"
    data.isel(direction=4).values.abs().sum().backward()
    assert values.grad[:, 4].abs().sum() > 0 and values.grad[:, :4].abs().sum() == 0


@cuda
def test_direction_selection_on_cuda_does_not_synchronise():
    data = FarFieldData.from_directions(
        torch.randn(3, 5, 2, device="cuda"), _directions(), FREQUENCIES, ("Etheta", "Ephi"), "V"
    )
    torch.cuda.set_sync_debug_mode("error")
    try:
        views = (data.isel(direction=3), data.isel(direction=slice(1, 4)), data.isel(f=1, direction=0), data.to("cuda"))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(views[1].attrs["uz"].cpu(), _directions()[1:4, 2])
