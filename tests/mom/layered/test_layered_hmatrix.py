"""D1 gates: H-matrix/ACA far field for the layered MoM (operator_backend="hmatrix").

- The cluster partition covers every unordered face pair exactly once, and
  every near pair of the assembly predicate lies in an inadmissible leaf block
  (exact integer counts), on a two-plane layout with vias (cross-plane blocks).
- The index-list sampler returns tiles bitwise equal to the dense assembly tiles.
- The compressed far action agrees with the exact matrix-free far action within
  the a-posteriori block bound sum_b e_b (|x_S| + |x_T|), and is complex
  symmetric to rounding.
- Compression and action are torch.equal on repeat (three builds), and full
  solves are torch.equal on repeat (three runs).
- F-A test mesh (two metals + vias, substrate): |dY|_F/|Y|_F <= 1e-4 against
  dense, the compression bound <= 1e-5 and the GMRES bound <= 1e-5.
- GR1 gradients through the hmatrix currents agree with dense.
"""

import dataclasses
import gc
import json
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered import hmatrix as hm
from witwin.maxwell.mom.layered.assembly import _frequency_view, assemble_layered, assembly_operands
from witwin.maxwell.mom.layered.matrix_free import LayeredMatrixFreeOperator, NearFieldLU, build_near_structure
from witwin.maxwell.mom.layered.network import solve_prepared_network
from tests.mom.layered.test_layered_lowfreq import loop
from tests.support.layered_gradient_fixtures import FREQUENCIES, SIGMA, loop_config, loop_scene
from tests.support.rfic_fixtures import fixture_fa


pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")]


@pytest.fixture(autouse=True)
def budget(record_property):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    record_property("seconds", seconds)
    record_property("peak_allocated_bytes", torch.cuda.max_memory_allocated())
    assert torch.cuda.max_memory_allocated() < 6 * 2**30


def hmatrix(config):
    return dataclasses.replace(config, layered=dataclasses.replace(config.layered, operator_backend="hmatrix"))


def relative(a, b):
    return (torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b)).tolist()


def fa_prepared(frequencies):
    fixture = fixture_fa()
    return mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=hmatrix(fixture.config)).prepare().solver.prepared


def test_backend_config():
    config = mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="hmatrix")
    assert config.payload()["operator_backend"] == "hmatrix"
    with pytest.raises(ValueError, match="operator_backend must be one of .dense., .matrix_free., .hmatrix."):
        mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="aca")


def test_partition_covers_pairs_and_keeps_near_pairs_in_leaves(record_property):
    prepared = fa_prepared((1e9,))
    structure = build_near_structure(prepared)
    clusters = hm.build_cluster_blocks(prepared, structure)
    stats = clusters.stats
    record_property("clusters", json.dumps(stats))
    assert stats["unordered_pairs_covered"] == stats["unordered_pairs"]
    assert stats["near_pairs_in_leaves"] == stats["near_pairs_unordered"]
    # Two metal planes and the via walls, with compressed cross-plane blocks.
    assert stats["z_classes"] >= 3 and stats["cross_class_blocks"] > 0
    assert stats["admissible_face_pairs"] > stats["leaf_far_pairs"]
    faces = len(prepared.layout.mesh.faces)
    assert torch.equal(torch.sort(clusters.faces).values, torch.arange(faces, device="cuda"))


def test_sampler_tiles_are_dense_tiles():
    prepared = loop((1e9,), copper=True)
    structure = build_near_structure(prepared)
    clusters = hm.build_cluster_blocks(prepared, structure)
    table = _frequency_view(prepared.table, 0, 1)
    green, _ = assembly_operands(prepared.layout, table)
    sampler = hm.TileSampler(green, structure, table)
    dense = assemble_layered(prepared.layout, prepared.table, frequency_index=0, far_dtype=torch.float32,
                             _tiles=True, _network_accuracy=True)
    faces = len(prepared.layout.mesh.faces)
    dense = dense.reshape(faces, faces, 10)
    # All admissible block pairs, the leaf far pairs and the near pairs.
    row_start, m, col_start, n = clusters.blocks.unbind(1)
    i, j = hm._pair_positions(row_start, m, col_start, n, torch.zeros_like(m, dtype=torch.bool))
    rows = torch.cat((clusters.faces[i], clusters.leaf_pairs[:, 0], structure.pairs[:, 0]))
    cols = torch.cat((clusters.faces[j], clusters.leaf_pairs[:, 1], structure.pairs[:, 1]))
    assert torch.equal(sampler(rows, cols), dense[rows, cols])


def compressed(prepared, structure, clusters, index):
    solver = NearFieldLU(structure, prepared.excitation.device)
    operator = hm.LayeredHMatrixOperator(prepared, structure, clusters, index, solver)
    return operator, solver


def test_compressed_action_matches_exact_far_action(record_property):
    frequencies = (1e3, 1e9, 1e11)
    prepared = fa_prepared(frequencies)
    structure = build_near_structure(prepared)
    clusters = hm.build_cluster_blocks(prepared, structure)
    c = clusters
    row_start, m, col_start, n = c.blocks.unbind(1)
    records = []
    for index, frequency in enumerate(frequencies):
        operator, solver = compressed(prepared, structure, clusters, index)
        x = operator.face_coefficients(operator.drive())
        approx = operator.far_moments(x)
        exact = LayeredMatrixFreeOperator.far_moments(operator, x)
        record = dict(frequency=frequency, **{k: v for k, v in operator.hmatrix.stats().items()
                                              if "rank_" in k and "histogram" not in k})
        for ch in operator.hmatrix.channels:
            comp = slice(ch["component"], ch["component"] + ch["width"])
            norms = x[c.faces][:, comp].abs().square().sum((1, 2))
            prefix = torch.cat((norms.new_zeros(1), torch.cumsum(norms, 0)))
            xt = (prefix[row_start + m] - prefix[row_start]).clamp_min(0).sqrt()
            xs = (prefix[col_start + n] - prefix[col_start]).clamp_min(0).sqrt()
            bound = float((ch["errors"] * (xt + xs)).sum())
            error = float(torch.linalg.vector_norm(approx[:, comp] - exact[:, comp]))
            reference = float(torch.linalg.vector_norm(exact[:, comp]))
            record[f"channel{ch['component']}"] = dict(error=error / reference, bound=bound / reference)
            assert error <= bound
            assert error <= 1e-5 * reference
        # Complex symmetry of the compressed far operator: y^T F x = x^T F y.
        generator = torch.Generator(device="cuda").manual_seed(5 + index)
        y = torch.randn(x.shape, generator=generator, device="cuda", dtype=torch.complex128)
        left = (y * operator.far_moments(x)).sum((0, 1))
        right = (x * operator.far_moments(y)).sum((0, 1))
        record["symmetry"] = float(((left - right).abs() / left.abs()).max())
        assert record["symmetry"] <= 1e-12
        records.append(record)
        solver.free()
        del operator
    record_property("action", json.dumps(records))


def test_compression_and_action_repeat_bitwise():
    prepared = loop((1e10,), copper=True)
    structure = build_near_structure(prepared)
    first = None
    for _ in range(3):
        clusters = hm.build_cluster_blocks(prepared, structure)
        operator, solver = compressed(prepared, structure, clusters, 0)
        x = operator.face_coefficients(operator.drive())
        values = [clusters.blocks, clusters.leaf_pairs, clusters.entries, operator.hmatrix.leaf_tiles,
                  *(ch[key] for ch in operator.hmatrix.channels for key in ("factors", "ranks", "errors")),
                  operator.far_moments(x)]
        if first is None:
            first = values
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, values))
        solver.free()
        del operator


def test_repeat_determinism():
    first = None
    for _ in range(3):
        prepared = loop((1e3, 1e10), copper=True)
        prepared.metadata["operator_backend"] = "hmatrix"
        network, currents = solve_prepared_network(prepared)
        values = (network.to_y(), currents)
        if first is None:
            first = values
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, values))
        del prepared, network, currents
        gc.collect()


def test_fa_agreement_with_dense(record_property):
    frequencies = (1e3, 1e9, 1e11)
    fixture = fixture_fa()
    dense = mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=fixture.config).run()
    result = mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=hmatrix(fixture.config)).run()
    errors = relative(result.network.to_y(), dense.network.to_y())
    metadata = result.network.metadata
    record_property("agreement", json.dumps(dict(
        frequencies=frequencies, relative_error=errors, steps=metadata["block_gmres_steps"],
        bound=metadata["admittance_bound"], worst_case=metadata["compression_bound_worst_case"],
        quadrature=metadata["compression_estimate_quadrature"],
        hmatrix=metadata["hmatrix"], structure=metadata["near_field"]), default=str))
    assert metadata["operator_backend"] == "hmatrix"
    assert max(errors) <= 1e-4
    assert max(metadata["admittance_bound"]) <= 1e-5


def test_exact_check_opt_in(monkeypatch, record_property):
    config = mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="hmatrix",
                                     hmatrix_exact_check=True)
    assert config.payload()["hmatrix_exact_check"] is True
    assert "hmatrix_exact_check" not in dataclasses.replace(config, hmatrix_exact_check=False).payload()
    with pytest.raises(ValueError, match="requires operator_backend='hmatrix'"):
        mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="matrix_free",
                                hmatrix_exact_check=True)
    prepared = loop((1e9, 1e11), copper=True)
    prepared.metadata.update(operator_backend="hmatrix", hmatrix_exact_check=True)
    network, _ = solve_prepared_network(prepared)
    checks = network.metadata["compression_exact_check"]
    record_property("exact_check", json.dumps(dict(
        exact=checks, worst_case=network.metadata["compression_bound_worst_case"],
        quadrature=network.metadata["compression_estimate_quadrature"])))
    assert max(checks) <= hm.COMPRESSION_TOLERANCE
    # The worst-case diagnostic dominates the exact first-order error.
    assert all(a <= b for a, b in zip(checks, network.metadata["compression_bound_worst_case"]))
    monkeypatch.setattr(hm, "COMPRESSION_TOLERANCE", 1e-300)
    with pytest.raises(RuntimeError, match="exact far-action check"):
        solve_prepared_network(prepared)


def test_gradients_match_dense(record_property):
    records = {}
    for backend in ("dense", "hmatrix"):
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        rough = torch.tensor(.5e-6, dtype=torch.float64, requires_grad=True)
        config = loop_config("sheet")
        if backend == "hmatrix":
            config = hmatrix(config)
        result = mw.Simulation.mom(loop_scene(sheet_sigma=sigma, via_sigma=sigma * .5, rms_height=rough),
                                   frequencies=FREQUENCIES, config=config).run()
        y = result.network.to_y()
        rows = []
        for value in y.reshape(-1):
            for part in (value.real, value.imag):
                rows.append(torch.stack(torch.autograd.grad(part, (sigma, rough), retain_graph=True)))
        records[backend] = (y.detach(), torch.stack(rows))
        del result
    (y_dense, g_dense), (y, g) = records["dense"], records["hmatrix"]
    errors = ((g - g_dense).abs().amax(0) / g_dense.abs().amax(0)).tolist()
    record_property("gradient_relative_error", json.dumps(errors))
    assert max(relative(y, y_dense)) <= 1e-4
    assert max(errors) <= 1e-4
