"""Test owned diagnostic algebra using CPU geometry, without advancing fields."""

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.compiler.copper_foil import compile_copper_faces
from witwin.maxwell.fdtd.distributed.copper import (
    _global_trace,
    _merge_owned_copper_reports,
)
from witwin.maxwell.fdtd_parallel import FDTDPartitionPlan
from witwin.maxwell.scene import prepare_scene


def fixture():
    scene = mw.Scene(
        domain=mw.Domain(bounds=((-0.02, 0.02),) * 3),
        grid=mw.GridSpec.uniform(0.005),
        boundary=mw.BoundarySpec.pml(num_layers=3),
        structures=(
            mw.Structure(
                name="foil",
                geometry=mw.Box(
                    position=torch.zeros(3, dtype=torch.float64),
                    size=torch.tensor((0.02, 0.02, 0.0), dtype=torch.float64),
                ),
                material=mw.CopperFoil(
                    thickness=2e-6, frequency_range=(0.5e9, 2e9)
                ),
            ),
        ),
        device="cpu",
    )
    prepared = prepare_scene(scene)
    plan = FDTDPartitionPlan(
        (prepared.Nx, prepared.Ny, prepared.Nz),
        ("cuda:0", "cuda:1"),
        low_pml_cells=3,
        high_pml_cells=3,
    )
    allparts = []
    frequency_count = 2
    for rank, layout in enumerate(plan.shard_layouts):
        bulk = []
        surface = []
        for face in compile_copper_faces(prepared, device="cpu"):
            name = "E" + "xyz"[face.component]
            comp = layout.component(name)
            index = face.indices
            select = torch.ones(len(index), dtype=torch.bool)
            for d, sl in enumerate(comp.owned_global_slice):
                select &= (index[:, d] >= sl.start) & (index[:, d] < sl.stop)
            index = index[select]
            edge_count = len(index)
            value = (index.double() * torch.tensor((1.0, 0.01, 0.0001))).sum(-1)
            fields = (
                torch.stack((value + 1j * value, value * 3 - 2j * value))
                .unsqueeze(0)
                .repeat(frequency_count, 1, 1)
                .to(torch.complex128)
                .requires_grad_()
            )
            common = dict(
                component=name,
                indices=index - comp.global_origin[0] * torch.tensor((1, 0, 0)),
                normal_axis="z",
                normal_coordinate=0.0,
            )
            bulk.append(
                _global_trace(
                    dict(
                        common,
                        normal_half_widths=torch.full(
                            (2, 1), 0.0025, dtype=torch.float64
                        ),
                        electric_fields=fields,
                        conduction_coefficient=torch.ones(
                            (frequency_count, 2, edge_count), dtype=torch.float64
                        ),
                        dispersion_coefficient=torch.full(
                            (frequency_count, 2, edge_count),
                            2.0,
                            dtype=torch.float64,
                        ),
                    ),
                    comp.global_origin,
                )
            )
            lower = index.double() / 100
            upper = lower + 0.01
            lower[:, 2] = upper[:, 2] = 0.0
            surface.append(
                _global_trace(
                    dict(
                        common,
                        power=value.repeat(frequency_count, 1),
                        area=torch.full((edge_count,), 0.0001, dtype=torch.float64),
                        # Distinct per-edge free-edge factors expose any
                        # metadata (first-chunk) treatment of this edge array.
                        edge_impedance_factor=1 + index[:, 1].double() / 7 + rank,
                        lower_bounds=lower,
                        upper_bounds=upper,
                        autograd_provenance="CPU controlled diagnostic algebra only",
                    ),
                    comp.global_origin,
                )
            )
        r = dict(
            dissipated_energy=float(rank + 1),
            stored_energy=float(rank + 2),
            frequency_range=(0.5e9, 2e9),
            frequencies=torch.tensor((1e9, 1.2e9), dtype=torch.float64),
            energy_representation="real",
            coordinate_frame="cartesian",
            periodic_axes=(),
            spectral_context=dict(
                dt=5e-12,
                frequencies=(1e9, 1.2e9),
                starts=(0, 0),
                total_steps=96,
                window="none",
                observed_steps=96.0,
                normalization=torch.tensor((96.0, 96.0), dtype=torch.float64),
            ),
            surface_power=torch.tensor((rank + 1.0, rank + 2.0), dtype=torch.float64),
            spectral_valid=torch.ones(frequency_count, dtype=torch.bool),
            pec_contact_edge_count=0,
            active_tangential_edge_count=sum(len(t["indices"]) for t in bulk),
            free_edge_trace_count=rank + 3,
            max_relative_model_error=1e-4,
            certified_relative_model_error_bound=2e-4,
            bulk_electric_traces=bulk,
            surface_power_traces=surface,
        )
        allparts.append(("foil", r))
    return allparts


def test_actual_partition_geometry_owned_merge_has_one_complete_trace_per_component():
    parts = fixture()
    merged = _merge_owned_copper_reports(parts)["foil"]
    assert len(merged["bulk_electric_traces"]) == len(merged["surface_power_traces"]) == 2
    for key in ("bulk_electric_traces", "surface_power_traces"):
        for trace in merged[key]:
            source = [
                t for _, r in parts for t in r[key]
                if t["component"] == trace["component"]
            ]
            expected = {tuple(index.tolist()) for t in source for index in t["indices"]}
            assert len(trace["indices"]) == len(expected) and {
                tuple(v.tolist()) for v in trace["indices"]
            } == expected
            for row, index in enumerate(trace["indices"]):
                for t in source:
                    match = torch.nonzero((t["indices"] == index).all(-1)).flatten()
                    if len(match):
                        j = int(match[0])
                        assert len(match) == 1
                        for field in (
                            (
                                "electric_fields", "normal_half_widths",
                                "conduction_coefficient", "dispersion_coefficient",
                            )
                            if key == "bulk_electric_traces"
                            else ("power", "area", "edge_impedance_factor", "lower_bounds", "upper_bounds")
                        ):
                            dimension = (
                                0 if field in ("area", "edge_impedance_factor", "lower_bounds", "upper_bounds")
                                else -1
                            )
                            reference = t[field]
                            if field == "normal_half_widths" and reference.shape == (2, 1):
                                reference = reference.expand(2, len(t["indices"]))
                            torch.testing.assert_close(
                                trace[field].select(dimension, row),
                                reference.select(dimension, j),
                                rtol=0,
                                atol=0,
                            )
    assert merged["dissipated_energy"] == 3 and merged["stored_energy"] == 5
    assert merged["free_edge_trace_count"] == 7
    # The shards own different trace counts of at least one component.
    counts = {(t["component"], rank): len(t["indices"]) for rank, (_, r) in enumerate(parts)
              for t in r["surface_power_traces"]}
    assert any(counts[(name, 0)] != counts[(name, 1)] for name in ("Ex", "Ey"))
    torch.testing.assert_close(
        merged["surface_power"], torch.tensor((3.0, 5.0), dtype=torch.float64)
    )


def test_merge_is_order_independent_without_mutating_original_report_and_preserves_gradient():
    parts = fixture()
    before = [r["surface_power"].clone() for _, r in parts]
    first = _merge_owned_copper_reports(parts)["foil"]
    reverse = _merge_owned_copper_reports(parts[::-1])["foil"]
    for a, _ in zip(first["bulk_electric_traces"], reverse["bulk_electric_traces"]):
        # Match by name because shard support can change component insertion order.
        b = next(
            t for t in reverse["bulk_electric_traces"]
            if t["component"] == a["component"]
        )
        torch.testing.assert_close(a["electric_fields"], b["electric_fields"], rtol=0, atol=0)
    loss = sum(
        t["electric_fields"].real.square().sum()
        + t["electric_fields"].imag.square().sum()
        for t in first["bulk_electric_traces"]
    )
    loss.backward()
    for (_, r), value in zip(parts, before):
        assert torch.equal(r["surface_power"], value)
        for t in r["bulk_electric_traces"]:
            torch.testing.assert_close(
                t["electric_fields"].grad,
                2 * t["electric_fields"].detach(),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "key",
    ("dt", "frequencies", "starts", "total_steps", "window", "observed_steps", "normalization"),
)
def test_actual_sampling_context_mismatch_is_not_renormalized(key):
    p = fixture()
    r = p[1][1]["spectral_context"]
    values = dict(
        dt=6e-12,
        frequencies=(1e9, 1.3e9),
        starts=(1, 0),
        total_steps=97,
        window="hanning",
        observed_steps=95.0,
        normalization=torch.tensor((95.0, 96.0), dtype=torch.float64),
    )
    r[key] = values[key]
    with pytest.raises(ValueError, match="sampling|frequency/clock/window/normalization"):
        _merge_owned_copper_reports(p)


@pytest.mark.parametrize(
    "kind",
    (
        "duplicate", "normal", "shape", "frequency_shape", "dtype",
        "indices_dtype", "negative_index", "surface_geometry",
    ),
)
def test_malformed_actual_owned_trace_context_raises(kind):
    p = fixture()
    first = p[0][1]["bulk_electric_traces"][0]
    last = p[1][1]["bulk_electric_traces"][0]
    if kind == "duplicate":
        last["indices"][0] = first["indices"][0]
    elif kind == "normal":
        last["normal_coordinate"] = 0.001
    elif kind == "shape":
        last["electric_fields"] = last["electric_fields"][..., :-1]
    elif kind == "frequency_shape":
        last["electric_fields"] = last["electric_fields"][:1]
    elif kind == "dtype":
        last["electric_fields"] = last["electric_fields"].to(torch.complex64)
    elif kind == "indices_dtype":
        last["indices"] = last["indices"].int()
    elif kind == "negative_index":
        last["indices"][0, 0] = -1
    else:
        p[1][1]["surface_power_traces"][0]["upper_bounds"] = (
            p[1][1]["surface_power_traces"][0]["upper_bounds"][:, :2]
        )
    with pytest.raises(ValueError, match="Copper"):
        _merge_owned_copper_reports(p)


def test_globalize_owned_indices_does_not_change_local_input():
    t = dict(indices=torch.tensor([[0, 2, 3], [1, 2, 3]], dtype=torch.int64))
    saved = t["indices"].clone()
    actual = _global_trace(t, (24, 0, 0))
    assert torch.equal(t["indices"], saved)
    assert torch.equal(actual["indices"], saved + torch.tensor((24, 0, 0)))


def test_duplicate_owner_guard_has_actual_raising_witness(record_property):
    import hashlib
    import json
    from pathlib import Path

    parts = fixture()
    parts[1][1]["bulk_electric_traces"][0]["indices"][0] = (
        parts[0][1]["bulk_electric_traces"][0]["indices"][0]
    )
    try:
        _merge_owned_copper_reports(parts)
    except ValueError as error:
        trace = error.__traceback__
        frames = []
        while trace is not None:
            f = trace.tb_frame
            path = Path(f.f_code.co_filename).resolve()
            if f.f_code.co_name == "_report_require":
                frames.append(
                    dict(
                        file=str(path),
                        line=trace.tb_lineno,
                        function=f.f_code.co_name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        message=str(error),
                    )
                )
            trace = trace.tb_next
        assert len(frames) == 1
        record_property("actual_added_raise_witness", json.dumps(frames))
    else:
        raise AssertionError("Duplicate owner did not exercise the data contract guard.")


def test_actual_runtime_broadcast_half_width_contract_retains_gradient():
    parts = fixture()
    for _, r in parts:
        for t in r["bulk_electric_traces"]:
            assert t["normal_half_widths"].shape == (2, 1)
            t["normal_half_widths"].requires_grad_()
    result = _merge_owned_copper_reports(parts)["foil"]
    sum(t["normal_half_widths"].sum() for t in result["bulk_electric_traces"]).backward()
    for _, r in parts:
        for t in r["bulk_electric_traces"]:
            torch.testing.assert_close(
                t["normal_half_widths"].grad,
                torch.full((2, 1), float(len(t["indices"])), dtype=torch.float64),
                rtol=0,
                atol=0,
            )


def test_explicit_per_edge_half_widths_remain_supported():
    parts = fixture()
    for _, r in parts:
        for t in r["bulk_electric_traces"]:
            t["normal_half_widths"] = (
                t["normal_half_widths"].expand(2, len(t["indices"])).clone()
            )
    result = _merge_owned_copper_reports(parts)["foil"]
    assert all(
        t["normal_half_widths"].shape == (2, len(t["indices"]))
        for t in result["bulk_electric_traces"]
    )


@pytest.mark.parametrize("count", (0, 1))
def test_empty_and_single_owned_rows_keep_frequency_and_two_side_axes(count):
    parts = fixture()
    for _, r in parts:
        r["active_tangential_edge_count"] = 0
        for key in ("bulk_electric_traces", "surface_power_traces"):
            for t in r[key]:
                t["indices"] = t["indices"][:count]
                r["active_tangential_edge_count"] += (
                    count if key == "bulk_electric_traces" else 0
                )
                names = (
                    ("electric_fields", "conduction_coefficient", "dispersion_coefficient")
                    if key == "bulk_electric_traces" else ("power",)
                )
                for name in names:
                    t[name] = t[name][..., :count]
                if key == "surface_power_traces":
                    for name in ("area", "edge_impedance_factor", "lower_bounds", "upper_bounds"):
                        t[name] = t[name][:count]
    result = _merge_owned_copper_reports(parts)["foil"]
    assert result["active_tangential_edge_count"] == 4 * count
    for t in result["bulk_electric_traces"]:
        assert t["electric_fields"].shape == (2, 2, 2 * count) and (
            t["normal_half_widths"].shape == (2, 2 * count)
        )


@pytest.mark.parametrize(
    "kind", ("bad_rows", "bad_columns", "negative", "zero", "nonfinite", "wrong_type")
)
def test_half_width_schema_rejects_malformed_physical_geometry(kind):
    parts = fixture()
    t = parts[1][1]["bulk_electric_traces"][0]
    if kind == "bad_rows":
        t["normal_half_widths"] = torch.ones(3, 1, dtype=torch.float64)
    elif kind == "bad_columns":
        t["normal_half_widths"] = torch.ones(2, len(t["indices"]) + 1, dtype=torch.float64)
    elif kind == "negative":
        t["normal_half_widths"][0, 0] = -0.0025
    elif kind == "zero":
        t["normal_half_widths"][0, 0] = 0.0
    elif kind == "nonfinite":
        t["normal_half_widths"][0, 0] = float("nan")
    else:
        t["normal_half_widths"] = [[0.0025], [0.0025]]
    with pytest.raises(ValueError, match="half widths"):
        _merge_owned_copper_reports(parts)


@pytest.mark.parametrize("kind", ("missing", "negative_norm", "normshape", "normdtype"))
def test_single_or_all_shards_malformed_clock_context_is_not_accepted(kind):
    parts = fixture()
    for _, r in parts:
        if kind == "missing":
            del r["spectral_context"]
        elif kind == "negative_norm":
            r["spectral_context"]["normalization"][0] = -1.0
        elif kind == "normshape":
            r["spectral_context"]["normalization"] = torch.ones(1, dtype=torch.float64)
        else:
            r["spectral_context"]["normalization"] = (
                r["spectral_context"]["normalization"].float()
            )
    with pytest.raises(ValueError, match="Copper"):
        _merge_owned_copper_reports(parts)


def test_time_only_energy_reports_do_not_invent_missing_spectral_trace_data():
    parts = fixture()
    for _, r in parts:
        for key in (
            "bulk_electric_traces", "surface_power_traces", "surface_power",
            "spectral_valid", "frequencies", "spectral_context", "coordinate_frame",
            "periodic_axes",
        ):
            r.pop(key)
    result = _merge_owned_copper_reports(parts)["foil"]
    assert result["dissipated_energy"] == 3.0 and result["stored_energy"] == 5.0
    assert "surface_power_traces" not in result and (
        "bulk_electric_traces" not in result and "frequencies" not in result
    )
