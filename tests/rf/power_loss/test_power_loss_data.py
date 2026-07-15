import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.compiler.monitors import compile_fdtd_observers
from witwin.maxwell.monitors import POWER_LOSS_CHANNELS, PowerLossMonitor
from witwin.maxwell.postprocess.power_loss import compute_power_loss_data
from witwin.maxwell.postprocess.antenna import compute_antenna_data
from witwin.maxwell.power_loss import PowerLossData
from witwin.maxwell.scene import prepare_scene


def _compiled_monitor(
    *,
    device="cpu",
    channels=("conduction",),
    frequencies=(1.0e9, 2.0e9),
    material=None,
    grid=None,
):
    material = material or mw.Material(sigma_e_tensor=mw.DiagonalTensor3(2.0, 3.0, 4.0))
    grid = grid or mw.GridSpec.uniform(0.5)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=grid,
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(4.0, 4.0, 4.0)),
                material=material,
            ),
        ),
        device=device,
    )
    monitor = PowerLossMonitor(
        "loss",
        position=(0.5, 0.5, 0.5),
        size=(1.0, 1.0, 1.0),
        frequencies=frequencies,
        channels=channels,
    )
    return compile_power_loss_monitor(prepare_scene(scene), monitor)


def _constant_fields(compiled, *, values=(1.0, 2.0, 3.0), requires_grad=False):
    frequency_count = len(compiled.monitor.frequencies)
    fields = {}
    for component, value in zip(("Ex", "Ey", "Ez"), values):
        fields[component] = torch.full(
            (frequency_count, *compiled.full_component_shapes[component]),
            complex(value),
            device=compiled.device,
            dtype=torch.complex64,
            requires_grad=requires_grad,
        )
    return fields


def _frequency_tensor(compiled):
    return torch.tensor(
        compiled.monitor.frequencies,
        device=compiled.device,
        dtype=torch.float32,
    )


def test_power_loss_monitor_has_explicit_region_frequency_and_channel_contract():
    monitor = PowerLossMonitor(
        "loss",
        position=(1.0, 2.0, 3.0),
        size=(2.0, 4.0, 6.0),
        frequencies=[1.0e9],
        channels=("conduction", "surface"),
    )

    assert monitor.kind == "power_loss"
    assert monitor.bounds == ((0.0, 2.0), (0.0, 4.0), (0.0, 6.0))
    assert monitor.frequencies == (1.0e9,)
    assert monitor.channels == ("conduction", "surface")
    assert set(POWER_LOSS_CHANNELS) == {
        "conduction",
        "electric_dispersion",
        "magnetic_dispersion",
        "nonlinear",
        "circuit",
        "surface",
        "wire",
    }

    with pytest.raises(ValueError, match="positive lengths"):
        PowerLossMonitor("bad", position=(0, 0, 0), size=(1, 0, 1))
    with pytest.raises(ValueError, match="unique"):
        PowerLossMonitor(
            "bad",
            position=(0, 0, 0),
            size=(1, 1, 1),
            channels=("conduction", "conduction"),
        )
    with pytest.raises(ValueError, match="Unsupported"):
        PowerLossMonitor(
            "bad",
            position=(0, 0, 0),
            size=(1, 1, 1),
            channels=("invented",),
        )


def test_compiler_uses_yee_edge_control_volumes_and_global_component_ids():
    nodes = (0.0, 0.2, 0.55, 1.0)
    compiled = _compiled_monitor(grid=mw.GridSpec.custom(nodes, nodes, nodes))

    all_ids = []
    expected_sigma = {"Ex": 2.0, "Ey": 3.0, "Ez": 4.0}
    for component in ("Ex", "Ey", "Ez"):
        assert (
            compiled.component_masks[component].shape
            == compiled.full_component_shapes[component]
        )
        assert compiled.component_volumes[component].ndim == 1
        assert torch.all(compiled.component_volumes[component] > 0.0)
        torch.testing.assert_close(
            compiled.conductivity[component],
            torch.full_like(
                compiled.conductivity[component], expected_sigma[component]
            ),
        )
        all_ids.append(compiled.global_ids[component])
    concatenated = torch.cat(all_ids)
    assert torch.unique(concatenated).numel() == concatenated.numel()


def test_static_electric_conduction_matches_peak_phasor_identity_without_squeeze():
    compiled = _compiled_monitor()
    frequencies = _frequency_tensor(compiled)
    fields = _constant_fields(compiled)

    data = compute_power_loss_data(
        compiled,
        frequencies,
        electric_fields=fields,
    )

    assert isinstance(data, PowerLossData)
    assert data.frequencies.shape == (2,)
    assert data.total.shape == (2,)
    assert data.conduction.shape == (2,)
    expected = torch.zeros(2, device=compiled.device)
    for component in ("Ex", "Ey", "Ez"):
        density = (
            0.5
            * compiled.conductivity[component][None, :]
            * torch.abs(
                fields[component][:, compiled.component_masks[component]]
            ).square()
        )
        torch.testing.assert_close(data.density("conduction", component), density)
        expected = expected + torch.sum(
            density * compiled.component_volumes[component][None, :], dim=1
        )
    torch.testing.assert_close(data.conduction, expected)
    torch.testing.assert_close(data.total, expected)
    assert data.channels == ("conduction",)
    assert data.power_unit == "W"
    assert data.volume_density_unit == "W/m^3"
    assert data.volume_unit == "m^3"
    assert data.material_ids is None
    assert data.geometry_ids is None


def test_explicit_volume_and_integrated_channels_sum_without_fabricating_others():
    compiled = _compiled_monitor(
        channels=("conduction", "electric_dispersion", "surface")
    )
    frequencies = _frequency_tensor(compiled)
    fields = _constant_fields(compiled)
    ex_shape = compiled.full_component_shapes["Ex"]
    dispersive_density = torch.stack(
        (
            torch.full(ex_shape, 0.25, device=compiled.device),
            torch.full(ex_shape, 0.5, device=compiled.device),
        )
    )
    surface_power = torch.tensor([1.0, 2.0], device=compiled.device)
    occupancy = {
        component: torch.full(shape, 0.75, device=compiled.device)
        for component, shape in compiled.full_component_shapes.items()
    }
    material_ids = {
        "Ex": torch.full(ex_shape, 7, device=compiled.device, dtype=torch.int64)
    }
    geometry_ids = {
        "Ex": torch.full(ex_shape, 11, device=compiled.device, dtype=torch.int64)
    }

    data = compute_power_loss_data(
        compiled,
        frequencies,
        electric_fields=fields,
        volume_channels={"electric_dispersion": {"Ex": dispersive_density}},
        integrated_channels={"surface": surface_power},
        occupancy=occupancy,
        material_ids=material_ids,
        geometry_ids=geometry_ids,
    )

    expected_dispersion = torch.sum(
        dispersive_density[:, compiled.component_masks["Ex"]]
        * compiled.component_volumes["Ex"][None, :],
        dim=1,
    )
    torch.testing.assert_close(data.channel("electric_dispersion"), expected_dispersion)
    torch.testing.assert_close(data.channel("surface"), surface_power)
    torch.testing.assert_close(
        data.total,
        data.conduction + expected_dispersion + surface_power,
    )
    assert data.channels == ("conduction", "electric_dispersion", "surface")
    assert "magnetic_dispersion" not in data.channel_power
    assert "surface" not in data.volume_density
    assert torch.all(data.occupancy["Ey"] == 0.75)
    assert torch.all(data.material_ids["Ex"] == 7)
    assert torch.all(data.geometry_ids["Ex"] == 11)


def test_power_loss_preserves_field_density_and_integrated_channel_gradients():
    compiled = _compiled_monitor(
        channels=("conduction", "electric_dispersion", "surface")
    )
    frequencies = _frequency_tensor(compiled)
    fields = _constant_fields(compiled, requires_grad=True)
    selected_count = compiled.component_volumes["Ex"].numel()
    density = torch.full(
        (2, selected_count),
        0.25,
        device=compiled.device,
        requires_grad=True,
    )
    surface = torch.tensor(
        [1.0, 2.0],
        device=compiled.device,
        requires_grad=True,
    )

    data = compute_power_loss_data(
        compiled,
        frequencies,
        electric_fields=fields,
        volume_channels={"electric_dispersion": {"Ex": density}},
        integrated_channels={"surface": surface},
    )
    data.total.sum().backward()

    for field in fields.values():
        assert field.grad is not None
        assert torch.all(torch.isfinite(field.grad))
    assert density.grad is not None
    assert torch.all(torch.isfinite(density.grad))
    torch.testing.assert_close(surface.grad, torch.ones_like(surface))


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_power_loss_data_remains_on_the_compiled_single_device(device):
    compiled = _compiled_monitor(device=device)
    data = compute_power_loss_data(
        compiled,
        _frequency_tensor(compiled),
        electric_fields=_constant_fields(compiled),
    )
    resolved_device = torch.empty((), device=device).device

    assert data.device == resolved_device
    for value in data.channel_power.values():
        assert value.device == resolved_device
    for component_map in data.volume_density.values():
        for value in component_map.values():
            assert value.device == resolved_device
    for mapping in (data.component_volumes, data.global_ids):
        for value in mapping.values():
            assert value.device == resolved_device


def test_missing_shape_and_frequency_contracts_fail_early():
    compiled = _compiled_monitor(channels=("conduction", "surface"))
    frequencies = _frequency_tensor(compiled)
    fields = _constant_fields(compiled)

    with pytest.raises(ValueError, match="No physical input"):
        compute_power_loss_data(compiled, frequencies, electric_fields=fields)
    bad_fields = dict(fields)
    bad_fields["Ex"] = bad_fields["Ex"][0]
    with pytest.raises(ValueError, match=r"\[F, \.\.\.\]"):
        compute_power_loss_data(
            compiled,
            frequencies,
            electric_fields=bad_fields,
            integrated_channels={"surface": torch.ones(2)},
        )
    with pytest.raises(ValueError, match="match the compiled"):
        compute_power_loss_data(
            compiled,
            torch.tensor([1.0e9, 2.1e9]),
            electric_fields=fields,
            integrated_channels={"surface": torch.ones(2)},
        )
    with pytest.raises(ValueError, match="unrequested"):
        compute_power_loss_data(
            compiled,
            frequencies,
            electric_fields=fields,
            integrated_channels={
                "surface": torch.ones(2),
                "wire": torch.ones(2),
            },
        )


def test_automatic_conduction_rejects_loss_mechanisms_it_cannot_classify():
    sheet = mw.Medium2D(sigma_s=1.0e-3)
    with pytest.raises(NotImplementedError, match="2D sheet"):
        _compiled_monitor(material=sheet)

    magnetic = mw.Material(sigma_e=1.0, sigma_m=2.0)
    with pytest.raises(NotImplementedError, match="magnetic conductivity"):
        _compiled_monitor(material=magnetic)


def test_surface_and_line_densities_use_explicit_area_and_length_measures():
    compiled = _compiled_monitor(channels=("surface", "wire"))
    frequencies = _frequency_tensor(compiled)
    face_areas = {"radiator": torch.tensor([0.2, 0.3], device=compiled.device)}
    line_lengths = {"feed": torch.tensor([0.1, 0.4, 0.5], device=compiled.device)}
    surface_density = torch.tensor(
        [[2.0, 4.0], [3.0, 5.0]], device=compiled.device, requires_grad=True
    )
    line_density = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
        device=compiled.device,
        requires_grad=True,
    )

    data = compute_power_loss_data(
        compiled,
        frequencies,
        surface_channels={"surface": {"radiator": surface_density}},
        face_areas=face_areas,
        line_channels={"wire": {"feed": line_density}},
        line_lengths=line_lengths,
        source_result_fingerprint="fixture:surface-line",
    )

    expected_surface = torch.sum(surface_density * face_areas["radiator"], dim=1)
    expected_wire = torch.sum(line_density * line_lengths["feed"], dim=1)
    torch.testing.assert_close(data.surface, expected_surface)
    torch.testing.assert_close(data.wire, expected_wire)
    torch.testing.assert_close(data.total, expected_surface + expected_wire)
    assert data.surface_density_unit == "W/m^2"
    assert data.line_density_unit == "W/m"
    assert data.area_unit == "m^2"
    assert data.length_unit == "m"
    assert data.source_result_fingerprint == "fixture:surface-line"
    data.total.sum().backward()
    assert surface_density.grad is not None
    assert line_density.grad is not None


def test_result_power_loss_is_typed_and_loss_monitor_adds_no_runtime_observer():
    frequencies = (1.0e9, 2.0e9)
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=mw.GridSpec.uniform(0.5),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(4.0, 4.0, 4.0)),
                material=mw.Material(sigma_e=2.0),
            ),
        ),
        device="cpu",
    )
    monitor = mw.PowerLossMonitor(
        "loss",
        position=(0.5, 0.5, 0.5),
        size=(1.0, 1.0, 1.0),
        frequencies=frequencies,
    )
    scene.add_monitor(monitor)
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    fields = _constant_fields(compiled)
    result = mw.Result(
        method="fdtd",
        scene=scene,
        prepared_scene=prepared,
        frequencies=frequencies,
        fields={name.upper(): value for name, value in fields.items()},
    )

    assert compile_fdtd_observers(scene) == []
    data = result.monitor("loss")
    assert isinstance(data, mw.PowerLossData)
    assert data.source_result_fingerprint.startswith("runtime-result:")
    torch.testing.assert_close(data.total, result.power_loss("loss").total)
    with pytest.raises(ValueError, match="explicit frequency axis"):
        result.monitor("loss", freq_index=0)


def test_radiated_plus_loss_power_balance_exit_gate_is_below_three_percent():
    frequencies = torch.tensor([1.0e9], dtype=torch.float64)
    theta = torch.linspace(0.0, math.pi, 361, dtype=torch.float64)
    phi = torch.linspace(0.0, 2.0 * math.pi, 361, dtype=torch.float64)
    intensity = torch.full(
        (1, theta.numel(), phi.numel()),
        0.7 / (4.0 * math.pi),
        dtype=torch.float64,
    )
    port = mw.PortData.from_power_waves(
        port_name="feed",
        frequencies=frequencies,
        a=torch.tensor([math.sqrt(1.25)], dtype=torch.complex128),
        b=torch.tensor([math.sqrt(0.25)], dtype=torch.complex128),
        z0=50.0,
    )
    antenna = compute_antenna_data(
        frequencies=frequencies,
        theta=theta,
        phi=phi,
        radiation_intensity=intensity,
        driven_port=port,
    )
    compiled = _compiled_monitor(channels=("circuit",), frequencies=(1.0e9,))
    loss = compute_power_loss_data(
        compiled,
        frequencies,
        integrated_channels={"circuit": torch.tensor([0.3], dtype=torch.float64)},
    )

    relative_error = torch.abs(antenna.p_rad + loss.total - antenna.p_accepted) / antenna.p_accepted
    assert float(relative_error[0]) < 0.03
