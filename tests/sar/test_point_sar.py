import dataclasses

import numpy as np
import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.mass_density import (
    BACKGROUND_TISSUE_ID,
    OCCUPANCY_EPSILON,
    compile_mass_density,
)
from witwin.maxwell.compiler.power_loss import compile_power_loss_monitor
from witwin.maxwell.postprocess.sar import compute_sar
from witwin.maxwell.sar import PowerNormalization
from witwin.maxwell.scene import prepare_scene

_USE_RHO = object()


def _uniform_cube_result(
    *,
    device="cpu",
    sigma=0.5,
    rho=1000.0,
    e0=2.0,
    frequencies=(1.0e9, 2.0e9),
    channels=("conduction",),
    dtype=torch.complex64,
    requires_grad=False,
    mass_density=_USE_RHO,
    scale=1.0,
):
    """Uniform lossy cube filling the domain, illuminated by a constant field.

    Constant Yee-edge fields make the conduction density exactly
    ``0.5*sigma*|E|^2`` at every edge, so point SAR has a closed form and the
    analytic cross-check is machine-tight rather than FDTD-accuracy-limited.

    ``scale`` multiplies the domain extent, grid spacing, and monitor/box geometry
    together, so the node count (hence field shape) is invariant while the physical
    grid spacing changes -- used to build two same-shaped, differently-spaced runs.
    """
    mass_density = rho if mass_density is _USE_RHO else mass_density
    material = mw.Material(sigma_e=sigma, mass_density=mass_density, name="tissue")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 2.0 * scale),) * 3),
        grid=mw.GridSpec.uniform(0.1 * scale),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(
                    position=(1.0 * scale,) * 3, size=(8.0 * scale,) * 3
                ),
                material=material,
            ),
        ),
        device=device,
    )
    monitor = mw.PowerLossMonitor(
        "loss",
        position=(1.0 * scale,) * 3,
        size=(1.0 * scale,) * 3,
        frequencies=frequencies,
        channels=channels,
    )
    scene.add_monitor(monitor)
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    fields = {}
    for component in ("Ex", "Ey", "Ez"):
        # Result stores per-frequency grids; single-frequency runs carry no
        # leading frequency axis (Result.tensor restacks per requested frequency).
        shape = compiled.full_component_shapes[component]
        if len(frequencies) > 1:
            shape = (len(frequencies), *shape)
        fields[component.upper()] = torch.full(
            shape,
            complex(e0),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    result = mw.Result(
        method="fdtd",
        scene=scene,
        prepared_scene=prepared,
        frequencies=frequencies,
        fields=fields,
    )
    return result, dict(sigma=sigma, rho=rho, e0=e0, frequencies=frequencies)


def _nanmax(tensor):
    flat = tensor.reshape(tensor.shape[0], -1)
    return torch.where(torch.isnan(flat), torch.full_like(flat, -float("inf")), flat).max(dim=1).values


def test_point_sar_matches_analytic_conduction():
    result, params = _uniform_cube_result()
    sar = result.sar("loss")

    analytic = 0.5 * params["sigma"] * (3.0 * params["e0"] ** 2) / params["rho"]
    peak = _nanmax(sar.point_sar("total"))
    torch.testing.assert_close(
        peak, torch.full_like(peak, analytic), rtol=1e-5, atol=1e-9
    )
    # Single channel: conduction == total everywhere valid.
    conduction = sar.point_sar("conduction")
    total = sar.point_sar("total")
    valid = sar.valid[None].expand_as(total)
    torch.testing.assert_close(conduction[valid], total[valid])
    assert sar.sar_unit == "W/kg"
    assert sar.channels == ("conduction", "total")


def test_volume_integrated_power_closes_against_power_loss_total():
    result, _ = _uniform_cube_result()
    sar = result.sar("loss")

    q_total = sar.absorbed_power_density["total"]
    region_power = (q_total * sar.cell_volume[None]).sum(dim=(1, 2, 3))
    electric_total = sar.provenance["electric_channel_power"]
    torch.testing.assert_close(region_power, electric_total, rtol=1e-5, atol=1e-9)

    # And the electric channel powers match the PowerLossData directly.
    power_loss = result.power_loss("loss")
    torch.testing.assert_close(region_power, power_loss.total, rtol=1e-5, atol=1e-9)


def test_per_channel_decomposition_sums_to_total():
    result, _ = _uniform_cube_result(channels=("conduction", "electric_dispersion"))
    prepared = result.prepared_scene
    monitor = result.scene.monitors[0]
    compiled = compile_power_loss_monitor(prepared, monitor)
    # Supply an explicit dispersive volume density on the Ex edges.
    ex_shape = compiled.full_component_shapes["Ex"]
    dispersive = torch.stack(
        (
            torch.full(ex_shape, 0.1, device=prepared.device),
            torch.full(ex_shape, 0.2, device=prepared.device),
        )
    )
    sar = result.sar(
        "loss",
        volume_channels={"electric_dispersion": {"Ex": dispersive}},
    )
    total = sar.point_sar("total")
    summed = sar.point_sar("conduction") + sar.point_sar("electric_dispersion")
    valid = sar.valid[None].expand_as(total)
    torch.testing.assert_close(summed[valid], total[valid], rtol=1e-5, atol=1e-9)
    assert set(sar.channels) == {"conduction", "electric_dispersion", "total"}


def test_missing_mass_density_raises():
    result, _ = _uniform_cube_result(mass_density=None)
    with pytest.raises(ValueError, match="without a mass_density"):
        result.sar("loss")


def test_no_electric_volumetric_channel_fails_closed():
    # Build a PowerLossData carrying only an integrated (non-volumetric) channel.
    from witwin.maxwell.postprocess.power_loss import compute_power_loss_data
    from witwin.maxwell.sar import PowerNormalization

    material = mw.Material(sigma_e=0.5, mass_density=1000.0, name="tissue")
    scene = mw.Scene(
        domain=mw.Domain(bounds=((0.0, 1.0),) * 3),
        grid=mw.GridSpec.uniform(0.25),
        boundary=mw.BoundarySpec.none(),
        structures=(
            mw.Structure(
                name="bulk",
                geometry=mw.Box(position=(0.5, 0.5, 0.5), size=(4.0, 4.0, 4.0)),
                material=material,
            ),
        ),
        device="cpu",
    )
    monitor = mw.PowerLossMonitor(
        "loss", position=(0.5, 0.5, 0.5), size=(0.5, 0.5, 0.5),
        frequencies=(1.0e9,), channels=("circuit",),
    )
    prepared = prepare_scene(scene)
    compiled = compile_power_loss_monitor(prepared, monitor)
    power_loss = compute_power_loss_data(
        compiled,
        torch.tensor([1.0e9], dtype=torch.float64),
        integrated_channels={"circuit": torch.tensor([0.3], dtype=torch.float64)},
    )
    mass = compile_mass_density(prepared)
    with pytest.raises(ValueError, match="volumetric electric-loss channel"):
        compute_sar(
            prepared_scene=prepared,
            monitor=monitor,
            power_loss=power_loss,
            mass=mass,
            compiled_loss=compiled,
            normalization=PowerNormalization.none(),
        )


def test_cpu_float64_oracle_parity():
    """Independent float64 numpy collocation reproduces the float32 reducer."""
    result, params = _uniform_cube_result(
        frequencies=(1.0e9,), e0=1.5, sigma=0.8, rho=1100.0
    )
    sar = result.sar("loss")

    prepared = result.prepared_scene
    monitor = result.scene.monitors[0]
    compiled = compile_power_loss_monitor(prepared, monitor)
    sigma = params["sigma"]
    e0 = params["e0"]
    rho = params["rho"]

    # Oracle: scatter each selected edge's power (0.5*sigma*|E|^2 * edge_volume)
    # to its two end nodes with weight 0.5, in float64, entirely independent of
    # the reducer's torch code path.
    nx, ny, nz = prepared.Nx, prepared.Ny, prepared.Nz
    node_power = np.zeros((nx, ny, nz), dtype=np.float64)
    axis_of = {"Ex": 0, "Ey": 1, "Ez": 2}
    for component in ("Ex", "Ey", "Ez"):
        mask = compiled.component_masks[component].cpu().numpy()
        volumes = np.zeros(mask.shape, dtype=np.float64)
        volumes[mask] = compiled.component_volumes[component].cpu().numpy().astype(np.float64)
        density = 0.5 * sigma * (e0 ** 2)
        edge_power = density * volumes
        axis = axis_of[component]
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        n = mask.shape[axis]
        lo[axis] = slice(0, n)
        node_lo = [slice(None)] * 3
        node_lo[axis] = slice(0, n)
        node_hi = [slice(None)] * 3
        node_hi[axis] = slice(1, n + 1)
        node_power[tuple(node_lo)] += 0.5 * edge_power
        node_power[tuple(node_hi)] += 0.5 * edge_power

    dual = (
        np.asarray(prepared.dx_dual64)[:, None, None]
        * np.asarray(prepared.dy_dual64)[None, :, None]
        * np.asarray(prepared.dz_dual64)[None, None, :]
    )
    oracle_sar = node_power / dual / rho  # W/kg on full node grid

    region = sar.provenance["region_index_bounds"]
    sl = tuple(slice(start, stop) for start, stop in region)
    oracle_region = oracle_sar[sl]
    reducer = sar.point_sar("total")[0].cpu().numpy()
    valid = sar.valid.cpu().numpy()
    np.testing.assert_allclose(
        reducer[valid], oracle_region[valid], rtol=1e-4, atol=1e-9
    )


def _compute_sar_with_mass(result, mass, *, normalization=None):
    """Run the point-SAR reducer with an explicitly supplied mass model."""
    prepared = result.prepared_scene
    monitor = result.scene.monitors[0]
    compiled = compile_power_loss_monitor(prepared, monitor)
    power_loss = result.power_loss("loss")
    return compute_sar(
        prepared_scene=prepared,
        monitor=monitor,
        power_loss=power_loss,
        mass=mass,
        compiled_loss=compiled,
        normalization=normalization or PowerNormalization.none(),
    )


def test_occupancy_below_epsilon_is_excluded_from_point_sar_and_statistics():
    """A cell whose tissue fill fraction is below the occupancy epsilon must be
    masked out (NaN point SAR, dropped from per-tissue statistics), even when it
    still carries a positive effective density and absorbed power.

    This isolates the occupancy-validity clause of ``valid``: only the cell's
    occupancy is pushed below the epsilon; its ``rho_cell`` stays positive, so the
    ``rho_cell > 0`` clause alone would keep it valid. Deleting the occupancy clause
    (``valid = rho_cell > 0``) makes this test go red — the recorded falsification.
    """
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    prepared = result.prepared_scene
    mass = compile_mass_density(prepared)

    baseline = _compute_sar_with_mass(result, mass)
    region = baseline.provenance["region_index_bounds"]
    # Full-grid physical-center node; it sits inside the monitor region and is a
    # fully occupied, valid tissue cell in the baseline.
    center = (prepared.Nx // 2, prepared.Ny // 2, prepared.Nz // 2)
    local = tuple(center[axis] - region[axis][0] for axis in range(3))
    assert all(0 <= local[axis] < baseline.valid.shape[axis] for axis in range(3))
    assert bool(baseline.valid[local])
    assert torch.isfinite(baseline.point_sar("total")[0][local])
    baseline_valid = int(baseline.valid.sum())
    baseline_count = baseline.statistics[0]["cell_count"]

    occupancy = mass.occupancy.clone()
    occupancy[center] = 0.1 * OCCUPANCY_EPSILON  # below epsilon, still > 0
    starved = dataclasses.replace(mass, occupancy=occupancy)
    sar = _compute_sar_with_mass(result, starved)

    # The starved cell is masked: NaN point SAR, dropped from the valid count and
    # from the per-tissue cell tally; every other cell is unchanged.
    assert not bool(sar.valid[local])
    assert torch.isnan(sar.point_sar("total")[0][local])
    assert int(sar.valid.sum()) == baseline_valid - 1
    assert sar.statistics[0]["cell_count"] == baseline_count - 1


def test_region_statistics_report_power_and_sar_per_tissue():
    result, params = _uniform_cube_result(frequencies=(1.0e9,))
    sar = result.sar("loss")

    assert set(sar.statistics) == {0}
    stats = sar.statistics[0]
    assert stats["name"] == "tissue"
    torch.testing.assert_close(
        stats["total_absorbed_power"],
        sar.provenance["electric_channel_power"],
        rtol=1e-5,
        atol=1e-9,
    )
    analytic = 0.5 * params["sigma"] * (3.0 * params["e0"] ** 2) / params["rho"]
    assert float(stats["max_sar"][0]) == pytest.approx(analytic, rel=1e-5)
    assert float(stats["mass_kg"]) > 0.0
    assert stats["cell_count"] > 0
    assert BACKGROUND_TISSUE_ID not in sar.statistics


def test_normalization_source_amplitude_scales_sar_by_square():
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    base = result.sar("loss")
    scaled = result.sar(
        "loss", normalization=mw.PowerNormalization.source(amplitude=2.0)
    )
    ratio = scaled.point_sar("total")[base.valid[None].expand_as(scaled.point_sar("total"))] / base.point_sar(
        "total"
    )[base.valid[None].expand_as(base.point_sar("total"))]
    torch.testing.assert_close(ratio, torch.full_like(ratio, 4.0), rtol=1e-5, atol=1e-6)


def test_accepted_power_normalization_fails_closed_without_port():
    """accepted_power normalization must fail closed when the named port is absent."""
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    with pytest.raises(KeyError, match="feed"):
        result.sar(
            "loss",
            normalization=mw.PowerNormalization.accepted_power(port="feed", watts=1.0),
        )


def test_input_power_normalization_fails_closed():
    """input_power scaling has no source-power diagnostic in this build; fail closed."""
    result, _ = _uniform_cube_result(frequencies=(1.0e9,))
    with pytest.raises(NotImplementedError, match="input_power"):
        result.sar(
            "loss",
            normalization=mw.PowerNormalization.input_power(watts=1.0),
        )


def test_point_sar_preserves_field_autograd():
    result, _ = _uniform_cube_result(
        frequencies=(1.0e9,), dtype=torch.complex64, requires_grad=True
    )
    sar = result.sar("loss")
    total = sar.point_sar("total")
    objective = torch.nan_to_num(total, nan=0.0).sum()
    objective.backward()
    for field in result.fields.values():
        assert field.grad is not None
        assert torch.all(torch.isfinite(field.grad))


def test_sar_result_serialization_roundtrip(tmp_path):
    result, _ = _uniform_cube_result(frequencies=(1.0e9, 2.0e9))
    sar = result.sar("loss")
    path = tmp_path / "sar.pt"
    sar.save(path)
    loaded = mw.SARResult.load(path)

    torch.testing.assert_close(
        torch.nan_to_num(loaded.point_sar("total"), nan=0.0),
        torch.nan_to_num(sar.point_sar("total").cpu(), nan=0.0),
    )
    assert loaded.provenance["grid_hash"] == sar.provenance["grid_hash"]
    torch.testing.assert_close(loaded.coordinates["x"], sar.coordinates["x"].cpu())
    assert set(loaded.statistics) == set(sar.statistics)
    torch.testing.assert_close(
        loaded.statistics[0]["total_absorbed_power"],
        sar.statistics[0]["total_absorbed_power"].cpu(),
    )
    assert loaded.normalization.kind == "source"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sar_stays_on_cuda_device():
    result, _ = _uniform_cube_result(device="cuda", frequencies=(1.0e9,))
    sar = result.sar("loss")
    assert sar.point_sar("total").device.type == "cuda"
    assert sar.rho_cell.device.type == "cuda"
    assert sar.valid.device.type == "cuda"
