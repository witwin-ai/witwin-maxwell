"""PH1 gates: layered dense operator protocol, Krylov drive and the 24 N^2 dense footprint."""

import gc
import json

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.linalg import StoppingRule, solve_reference
from witwin.maxwell.mom.operator_protocol import KrylovView, SurfaceOperatorProtocol
from witwin.maxwell.mom.layered.lowfreq import scaled_system
from witwin.maxwell.mom.layered.network import _wall_nodes, frequency_chunk_capacity, solve_prepared_network
from witwin.maxwell.mom.layered.operator import (
    dense_chunk_bytes, dense_frequency_bytes, workspace_elements)
from tests.mom.layered.test_layered_lowfreq import loop
from tests.support.rfic_fixtures import fixture_fa, SWEEP


pytestmark = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")


@pytest.fixture(scope="module")
def factored():
    prepared = loop((1e3, 1e9), copper=True)
    system = scaled_system(prepared)
    system.factor()
    solution, info = system.solve()
    return prepared, system, solution, info


@requires_cuda
def test_operator_protocol_and_krylov_match_direct(factored, record_property):
    prepared, system, solution, info = factored
    records = []
    for k in range(len(system.frequencies)):
        operator = system.operator(k)
        assert isinstance(operator, SurfaceOperatorProtocol)
        n = operator.shape[0]
        assert operator.shape == (n, n) and operator.dtype == torch.complex128
        b = system.rhs[k].mT.contiguous()
        # The direct backend action reproduces the chunk solve bitwise.
        assert torch.equal(operator.solve(b), solution[k].mT)
        generator = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn((3, n), generator=generator, device="cuda", dtype=torch.complex128)
        y = torch.randn((3, n), generator=generator, device="cuda", dtype=torch.complex128)
        left = (operator.apply(x) * y.conj()).sum(-1)
        right = (x * operator.adjoint(y).conj()).sum(-1)
        assert ((left - right).abs() / left.abs()).max() < 1e-12
        for kind in ("none", "row_norm", "row_column_norm"):
            rule = StoppingRule(method="gmres", preconditioner=kind, rtol=1e-12, atol=0.,
                                maxiter=4 * n, restart=n)
            krylov, log = solve_reference(KrylovView(operator), b, rule)
            error = (torch.linalg.vector_norm(krylov - solution[k].mT) /
                     torch.linalg.vector_norm(solution[k])).item()
            records.append(dict(frequency=k, preconditioner=kind, converged=log["converged"],
                                relative_to_direct=error))
            assert log["converged"]
            # Krylov 1e-12 residual times the bounded scaled condition (< 1e6).
            assert error < 1e-6
        receipt = operator.persistence_receipt()
        assert receipt["operator_backend"] == "dense" and receipt["space"] == "equilibrated_loop_tree"
        stats = operator.operator_stats()
        assert stats["storage_bytes"] == 16 * n * n + 8 * workspace_elements(n, len(prepared.layout.mesh.faces))
    record_property("krylov", json.dumps(records))
    assert set(info[0]) == {"complex64"}


@requires_cuda
def test_operator_refuses_changed_storage(factored):
    _, system, _, _ = factored
    operator = system.operator(0)
    workspace = system.workspace
    system.workspace = workspace.clone()
    with pytest.raises(ValueError, match="storage changed"):
        operator.apply(torch.zeros(operator.shape[0], device="cuda", dtype=torch.complex128))
    system.workspace = workspace


@requires_cuda
def test_dense_footprint_estimate_matches_measured_peak(record_property):
    prepared = fixture_fa().simulation(SWEEP[:4]).prepare().solver.prepared
    n, faces, ports = prepared.layout.basis.n_edges, len(prepared.layout.mesh.faces), prepared.excitation.shape[1]
    two = 16 * n * n + 8 * workspace_elements(n, faces)
    assert faces * (faces + 1) <= n * n and two == 24 * n * n + 8 * (n * n % 2)
    records = {}
    for chunk in (1, 2):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        network, currents = solve_prepared_network(prepared, frequency_chunk_size=chunk)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        outputs = 16 * 4 * n * ports + 16 * 4 * ports * ports
        estimate = (chunk * dense_frequency_bytes(n, faces, ports)
                    + dense_chunk_bytes(faces, _wall_nodes(prepared.table)) + outputs)
        records[chunk] = dict(measured_peak_bytes=peak, estimate_bytes=estimate, two_matrices_bytes=chunk * two,
                              ratio=peak / estimate)
        assert network.metadata["frequency_chunk_size"] == chunk
        assert chunk * two <= peak <= estimate
        assert peak >= .9 * estimate
        del network, currents
    record_property("footprint", json.dumps(dict(unknowns=n, faces=faces, chunks=records)))


@pytest.mark.parametrize("name,unknowns,faces,ports,wall_nodes", [
    ("F-B fine sheet", 14411, 10400, 4, 12), ("F-A closed fine", 18040, 12028, 2, 12)])
def test_fine_fixture_admission(name, unknowns, faces, ports, wall_nodes, record_property):
    one = dense_frequency_bytes(unknowns, faces, ports)
    # Workspace rounds N^2 up to an even element count (8 bytes for odd N).
    assert one <= 24 * unknowns * unknowns + 8 + 160 * faces * 64 + 256 * unknowns * ports
    local = 46 * 2**30
    assert frequency_chunk_capacity(unknowns, faces, ports, 25, local, wall_nodes=wall_nodes) >= 4
    with pytest.raises(RuntimeError, match="One dense layered frequency"):
        frequency_chunk_capacity(unknowns, faces, ports, 25, 6 * 2**30, wall_nodes=wall_nodes)
    record_property(name, json.dumps(dict(frequency_GiB=one / 2**30,
                                          chunk_scratch_GiB=dense_chunk_bytes(faces, wall_nodes) / 2**30)))


def test_operator_backend_config():
    config = mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5)
    assert config.operator_backend == "dense"
    assert "operator_backend" not in config.payload()
    with pytest.raises(ValueError, match="operator_backend must be one of .dense., .matrix_free., .hmatrix."):
        mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="gmres")
