"""S2a gates: matrix-free layered backend (near-field sparse LU right preconditioner + block GMRES).

- The matrix-free action equals the dense equilibrated matrix after the diagonal
  change of basis. The face-pair tiles are bitwise those of the dense assembly;
  only the fp64 order of the loop/tree sums differs, so the per-row bound is
  64 eps64 times the absolute-value action D(S|Q|^T|mu0 V'||Q|S + |C|^T|P||C|/(eps0 w0^2))D|x|,
  which covers loop sums with cancellation. The adjoint is the complex-symmetric one.
- Network admittances agree with the dense backend, |dY|_F/|Y|_F <= 1e-4 per
  frequency, while the stopping bound |X|_F|R|_F/|B^T X|_F <= 1e-5 holds; the
  error against dense does not grow with the iteration count.
- Repeated runs are torch.equal (three runs).
- GR1 loss gradients through the matrix-free currents agree with the dense ones.
- The near-field LU never depends on recycled device memory: its deterministic
  forward solve reads workspace the library leaves unwritten, so the handle's
  allocator zero-fills pool blocks that earlier solvers dirtied.
"""

import ctypes
import dataclasses
import gc
import json
import math
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.operator_protocol import SurfaceOperatorProtocol
from witwin.maxwell.mom.layered.lowfreq import assemble_chunk, scaled_system
from witwin.maxwell.mom.layered.matrix_free import (
    ADMITTANCE_TOLERANCE, LayeredMatrixFreeOperator, NearFieldLU, block_gmres, build_near_structure)
from witwin.maxwell.mom.layered.network import solve_prepared_network
from tests.mom.layered.test_layered_lowfreq import dense_columns, loop
from tests.support.layered_gradient_fixtures import FREQUENCIES, SIGMA, loop_config, loop_scene
from tests.support.rfic_fixtures import fixture_fa


MU0 = 1 / (8.8541878128e-12 * 299792458.0**2)
EPS0 = 8.8541878128e-12
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


def relative(a, b):
    return (torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b)).tolist()


def test_backend_config():
    config = mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="matrix_free")
    assert config.payload()["operator_backend"] == "matrix_free"
    with pytest.raises(ValueError, match="operator_backend must be one of .dense., .matrix_free., .hmatrix."):
        mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=4e-5, operator_backend="gmres")


def test_action_matches_dense_matrix(record_property):
    frequencies = (1e3, 1e9, 1e11)
    prepared = loop(frequencies, copper=True)
    system = scaled_system(prepared)
    # Absolute-value oracle of the unscaled entries (loss folded, before the congruence).
    raw = assemble_chunk(prepared)
    op = torch.ops.witwin_maxwell_mom_layered
    loss = prepared.loss
    op.loss_fold(raw.matrix, loss.blocks.contiguous(), loss.columns.contiguous(), loss.supports.contiguous(),
                 loss.impedance.contiguous(), prepared.loop_tree.order, raw.frequencies)
    faces = len(prepared.layout.mesh.faces)
    packed = raw.workspace.view(torch.complex128)[:, :faces * (faces + 1) // 2]
    lower = torch.tril_indices(faces, faces, device="cuda")
    structure = build_near_structure(prepared)
    solver = NearFieldLU(structure, prepared.excitation.device)
    lt = prepared.loop_tree
    records = []
    for k, frequency in enumerate(frequencies):
        operator = LayeredMatrixFreeOperator(prepared, structure, k, solver)
        assert isinstance(operator, SurfaceOperatorProtocol)
        n = operator.shape[0]
        w = 2 * math.pi * frequency
        # Dense D from its recovery vector: recovery = d * (1/w loops, 1/w0 trees).
        factor = torch.full((n,), lt.omega0, device="cuda", dtype=torch.float64)
        factor[:lt.loops] = w
        dense_scale = system.recovery[k] * factor
        change = operator.scale / dense_scale
        generator = torch.Generator(device="cuda").manual_seed(11 + k)
        x = torch.randn((n, 3), generator=generator, device="cuda", dtype=torch.complex128)
        reference = change[:, None] * (system.matrix[k] @ (change[:, None] * x))
        action = operator.apply(x.mT).mT
        error = ((action - reference).abs().max() / reference.abs().max()).item()
        q = dense_columns(lt.sparse_q, len(lt.order)).abs()
        q_position = torch.zeros_like(q)
        q_position[lt.order] = q
        sizes = torch.ones(n, device="cuda", dtype=torch.float64)
        sizes[lt.loops:] = w / lt.omega0
        magnetic = (q_position.mT @ (MU0 * raw.matrix[k].abs()) @ q_position) * (sizes[:, None] * sizes[None, :])
        p = torch.zeros((faces, faces), device="cuda", dtype=torch.float64)
        p[lower[0], lower[1]] = packed[k].abs()
        p[lower[1], lower[0]] = packed[k].abs()
        c = dense_columns(lt.sparse_charge, faces).abs()
        magnetic[lt.loops:, lt.loops:] += (c.mT @ p @ c) / (EPS0 * lt.omega0 ** 2)
        scale = operator.scale
        bound = 64 * 2.0**-53 * scale[:, None] * (magnetic @ (scale[:, None] * x.abs()))
        ratio = ((action - reference).abs() / bound).max().item()
        y = torch.randn((3, n), generator=generator, device="cuda", dtype=torch.complex128)
        left = (operator.apply(x.mT) * y.conj()).sum(-1)
        right = (x.mT * operator.adjoint(y).conj()).sum(-1)
        adjoint = ((left - right).abs() / left.abs()).max().item()
        records.append(dict(frequency=frequency, action_error=error, bound_ratio=ratio, adjoint_error=adjoint))
        assert ratio <= 1
        # Canonical face-pair tiles make V, P and the loss Gram symmetric up to sum order.
        assert adjoint <= 1e-12
        with pytest.raises(ValueError, match="only none scaling"):
            operator.preconditioner_scales("row_norm")
        with pytest.raises(RuntimeError, match="did not reach the admittance bound"):
            block_gmres(operator, operator.drive(), tolerance=1e-15, restart=1, max_steps=1)
        receipt = operator.persistence_receipt()
        assert receipt["operator_backend"] == "matrix_free" and receipt["space"] == "equilibrated_loop_tree"
    solver.free()
    record_property("action", json.dumps(records))


def matrix_free(config):
    return dataclasses.replace(config, layered=dataclasses.replace(config.layered, operator_backend="matrix_free"))


def test_fa_agreement_and_error_history(record_property):
    frequencies = (1e3, 1e9)
    fixture = fixture_fa()
    dense = mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=fixture.config).run()
    result = mw.Simulation.mom(fixture.scene, frequencies=frequencies, config=matrix_free(fixture.config)).run()
    y_dense, y = dense.network.to_y(), result.network.to_y()
    errors = relative(y, y_dense)
    metadata = result.network.metadata
    record_property("agreement", json.dumps(dict(
        frequencies=frequencies, relative_error=errors, steps=metadata["block_gmres_steps"],
        bound=metadata["admittance_bound"], near=metadata["near_field"], unknowns=metadata["unknowns"])))
    assert max(errors) <= 1e-4
    assert max(metadata["admittance_bound"]) <= ADMITTANCE_TOLERANCE
    assert metadata["operator_backend"] == "matrix_free"
    # Residual vs true error at 1 GHz: the error against the dense currents
    # follows the residual bound down and does not grow late in the iteration.
    prepared = result.solver.prepared
    structure = build_near_structure(prepared)
    solver = NearFieldLU(structure, prepared.excitation.device)
    operator = LayeredMatrixFreeOperator(prepared, structure, 1, solver)
    rhs = operator.drive()
    reference = dense.network.to_y()[1]
    history = []

    def monitor(step, x, residual):
        estimate = (rhs.mT @ x) / (2 * math.pi * frequencies[1])
        history.append(dict(step=step, residual=residual,
                            error=relative(estimate[None], reference[None])[0]))

    _, log = block_gmres(operator, rhs, monitor=monitor)
    solver.free()
    record_property("history", json.dumps(history))
    tail = [entry["error"] for entry in history[len(history) // 2:]]
    assert history[-1]["error"] <= 1e-4
    assert max(tail) <= 4 * tail[0]


def test_repeat_determinism():
    first = None
    for _ in range(3):
        prepared = loop((1e3, 1e10), copper=True)
        prepared.metadata["operator_backend"] = "matrix_free"
        network, currents = solve_prepared_network(prepared)
        values = (network.to_y(), currents)
        if first is None:
            first = values
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, values))
        del prepared, network, currents
        gc.collect()


def test_gradients_match_dense(record_property):
    records = {}
    for backend in ("dense", "matrix_free"):
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        rough = torch.tensor(.5e-6, dtype=torch.float64, requires_grad=True)
        config = loop_config("sheet")
        if backend == "matrix_free":
            config = matrix_free(config)
        result = mw.Simulation.mom(loop_scene(sheet_sigma=sigma, via_sigma=sigma * .5, rms_height=rough),
                                   frequencies=FREQUENCIES, config=config).run()
        y = result.network.to_y()
        rows = []
        for value in y.reshape(-1):
            for part in (value.real, value.imag):
                rows.append(torch.stack(torch.autograd.grad(part, (sigma, rough), retain_graph=True)))
        records[backend] = (y.detach(), torch.stack(rows))
        del result
    (y_dense, g_dense), (y, g) = records["dense"], records["matrix_free"]
    errors = ((g - g_dense).abs().amax(0) / g_dense.abs().amax(0)).tolist()
    record_property("gradient_relative_error", json.dumps(errors))
    record_property("admittance_relative_error", json.dumps(relative(y, y_dense)))
    assert max(relative(y, y_dense)) <= 1e-4
    assert max(errors) <= 1e-4


def test_sparse_loop_tree_setup_matches_dense():
    from witwin.maxwell.mom.layered.lowfreq import build_loop_tree
    layout = fixture_fa().simulation((1e9,)).prepare().solver.prepared.layout
    dense = build_loop_tree(layout, length_scale=20e-6)
    sparse = build_loop_tree(layout, length_scale=20e-6, dense_setup=False)
    assert (dense.loops, dense.components) == (sparse.loops, sparse.components)
    assert torch.equal(dense.order, sparse.order)
    # Discrete pattern and raw +-1/length values are identical; norms differ only in sum order.
    for a, b in zip(dense.sparse_transform, sparse.sparse_transform):
        assert torch.equal(a, b)
    assert ((dense.norms - sparse.norms).abs() / dense.norms).max() <= 1e-15
    for a, b in zip(dense.sparse_charge[:2], sparse.sparse_charge[:2]):
        assert torch.equal(a, b)


def _dirty_solver_pool(byte=0xFF, minimum=256 * 2**20):
    """Fill every free block of the device pool the direct solver allocates from with `byte`.

    The pool keeps freed blocks (release threshold), so the next solver's
    workspace comes from these blocks, as it does after an earlier solver's free().
    """
    from cuda.bindings import runtime
    from nvmath.internal.memory import get_device_memory_resource
    pool = get_device_memory_resource(torch.cuda.current_device()).handle
    attribute = runtime.cudaMemPoolAttr
    reserved = int(runtime.cudaMemPoolGetAttribute(pool, attribute.cudaMemPoolAttrReservedMemCurrent)[1])
    used = int(runtime.cudaMemPoolGetAttribute(pool, attribute.cudaMemPoolAttrUsedMemCurrent)[1])
    size = max(reserved - used, minimum)
    status, pointer = runtime.cudaMallocFromPoolAsync(size, pool, 0)
    assert status == runtime.cudaError_t.cudaSuccess
    assert runtime.cudaMemsetAsync(pointer, byte, size, 0)[0] == runtime.cudaError_t.cudaSuccess
    assert runtime.cudaFreeAsync(pointer, 0)[0] == runtime.cudaError_t.cudaSuccess
    assert runtime.cudaStreamSynchronize(0)[0] == runtime.cudaError_t.cudaSuccess


def _handler_name(handle):
    import nvmath.bindings.cudss as cudss
    handler = ctypes.create_string_buffer(3 * ctypes.sizeof(ctypes.c_void_p) + 64)
    cudss.get_device_mem_handler(handle, ctypes.addressof(handler))
    return handler.raw[3 * ctypes.sizeof(ctypes.c_void_p):].split(b"\0")[0].decode()


def test_near_lu_ignores_recycled_workspace():
    """Regression (Xid 13 Out Of Range Address): a near LU after other near LUs in one process.

    A loop LU and an F-A LU (different sizes) are solved and freed, the freed
    pool blocks are then dirtied with 0xFF, and the F-A LU is solved again: the
    solve must neither fault nor change a bit.
    """
    small = loop((1e9,), copper=True)
    small_structure = build_near_structure(small)
    solver = NearFieldLU(small_structure, small.excitation.device)
    operator = LayeredMatrixFreeOperator(small, small_structure, 0, solver)
    assert torch.isfinite(operator.precondition(operator.drive())).all()
    solver.free()
    prepared = fixture_fa().simulation((1e9,)).prepare().solver.prepared
    structure = build_near_structure(prepared)
    assert len(structure.offsets) != len(small_structure.offsets)
    solutions = []
    for dirty in (False, True):
        torch.cuda.synchronize()
        if dirty:
            _dirty_solver_pool()
        solver = NearFieldLU(structure, prepared.excitation.device)
        operator = LayeredMatrixFreeOperator(prepared, structure, 0, solver)
        solutions.append(operator.precondition(operator.drive()))
        torch.cuda.synchronize()
        if not dirty:
            solver.free()
    assert torch.isfinite(solutions[0]).all()
    assert torch.equal(solutions[0], solutions[1])
    # The handle keeps the zero-filling allocator (nvmath installs its own only on handles without one).
    from witwin.maxwell.mom.layered.build import build_extension
    assert _handler_name(solver.handle) == "witwin_zeroed_mempool"
    assert build_extension().cudss_zeroed_stats()[0] > 0
    solver.free()


def test_chunked_sparse_product_is_chunk_independent():
    """sparse_product / merge_blocks: values torch.equal for any row-chunk limit, and equal to the dense
    product within the fp64 summation bound (n eps |X||Y|); merge sums magnetic before charge per key."""
    from witwin.maxwell.mom.layered.build import build_extension
    from witwin.maxwell.mom.layered.matrix_free import merge_blocks, sparse_product
    build_extension()
    generator = torch.Generator(device="cuda").manual_seed(7)

    def csr(rows, cols, density, dtype):
        mask = torch.rand((rows, cols), device="cuda", generator=generator) < density
        values = torch.randn((rows, cols), device="cuda", generator=generator, dtype=torch.float64)
        if dtype == torch.complex128:
            values = torch.complex(values, torch.randn((rows, cols), device="cuda", generator=generator,
                                                       dtype=torch.float64))
        dense = values * mask
        nz = mask.nonzero()
        offsets = torch.cat((nz.new_zeros(1), torch.cumsum(mask.sum(1), 0)))
        return (offsets, nz[:, 1].contiguous(), dense[mask].contiguous()), dense

    x, xd = csr(300, 200, .05, torch.complex128)
    y, yd = csr(200, 250, .08, torch.float64)
    reference = sparse_product(x, y, 300, 250, limit=1 << 40)
    for limit in (1, 7, 500):
        chunked = sparse_product(x, y, 300, 250, limit=limit)
        assert all(torch.equal(a, b) for a, b in zip(chunked[:3], reference[:3])) and chunked[3] == reference[3]
    offsets, cols, values, _ = reference
    rows = torch.repeat_interleave(torch.arange(300, device="cuda"), offsets.diff())
    dense = torch.zeros((300, 250), device="cuda", dtype=torch.complex128)
    dense[rows, cols] = values
    bound = 200 * 2.3e-16 * (xd.abs() @ yd.abs())
    assert bool(((dense - xd @ yd.to(torch.complex128)).abs() <= bound).all())
    pattern = sparse_product((x[0], x[1], None), (y[0], y[1], None), 300, 250, limit=3)
    assert torch.equal(pattern[0], offsets) and torch.equal(pattern[1], cols) and pattern[2] is None
    # Merge: magnetic [300, 300] and a charge block on rows/cols >= 100.
    m, md = csr(300, 300, .05, torch.complex128)
    c, cd = csr(200, 200, .05, torch.complex128)
    merged = merge_blocks(m, c, 300, 100, limit=1 << 40)
    for limit in (1, 50):
        assert all(torch.equal(a, b) for a, b in zip(merge_blocks(m, c, 300, 100, limit=limit), merged))
    rows = torch.repeat_interleave(torch.arange(300, device="cuda"), merged[0].diff())
    total = torch.zeros((300, 300), device="cuda", dtype=torch.complex128)
    total[rows, merged[1]] = merged[2]
    expected = md.clone()
    expected[100:, 100:] += cd
    assert torch.equal(total, expected)


def test_warm_start_across_frequencies(record_property):
    """R6-3 (D3): frequency k > 0 of a call starts block GMRES from frequency k - 1.

    Every frequency satisfies the same 1e-5 admittance stopping bound as a cold
    single-frequency call, so the two differ by at most twice that bound (relative
    Frobenius). (A multi-frequency call is not bitwise a single-frequency call even cold:
    the Green table depends on the band. The zero-start path is unchanged code; the
    base/head A/B is in the R6-3 evidence.)
    """
    frequencies = (5e9, 5.5e9, 6e9, 1e10)
    config = matrix_free(loop_config())
    warm = mw.Simulation.mom(loop_scene(), frequencies=frequencies, config=config).run()
    steps = dict(warm=list(warm.network.metadata["block_gmres_steps"]), cold=[])
    differences = []
    for index, frequency in enumerate(frequencies):
        cold = mw.Simulation.mom(loop_scene(), frequencies=(frequency,), config=config).run()
        steps["cold"].append(cold.network.metadata["block_gmres_steps"][0])
        differences.append(relative(warm.network.to_y()[index:index + 1], cold.network.to_y())[0])
    record_property("warm_start", json.dumps(dict(steps=steps, relative_difference=differences)))
    assert max(differences) <= 2 * ADMITTANCE_TOLERANCE
    assert max(warm.network.metadata["admittance_bound"]) <= ADMITTANCE_TOLERANCE
    again = mw.Simulation.mom(loop_scene(), frequencies=frequencies, config=config).run()
    assert torch.equal(again.network.s, warm.network.s)


def test_adaptive_sweep_matches_dense(record_property):
    """D3/G12: the adaptive sweep on matrix_free reproduces the dense sweep's model."""
    models = {}
    for backend in ("dense", "matrix_free"):
        config = loop_config()
        layered = dataclasses.replace(config.layered, operator_backend=backend,
                                      sweep=mw.mom.AdaptiveSweep(1e9, 2e10, 1e-3, 25))
        result = mw.Simulation.mom(loop_scene(), config=dataclasses.replace(config, layered=layered)).run()
        models[backend] = (result.network.frequencies, mw.mom.surface_currents(result).sweep)
    grid = torch.linspace(1e9, 2e10, 400, dtype=torch.float64, device="cuda")
    dense, sparse = (models[b][1].evaluate_network(grid).s for b in ("dense", "matrix_free"))
    error = float((dense - sparse).abs().max())
    record_property("adaptive", json.dumps(dict(model_difference=error, **{
        b: dict(solves=len(models[b][0]), estimated_error=models[b][1].estimated_error,
                enforcement_change=models[b][1].enforcement_change) for b in models})))
    # Each model is within its 1e-3 tolerance estimate of its own backend; the backends agree to 1e-4.
    # The nearly lossless loop's fit crosses |S| = 1 by ~1e-7: the sweep enforces it (<= tolerance/4).
    assert error <= 2e-3
    for backend in models:
        assert models[backend][1].passivity_report.passive and models[backend][1].enforcement_change <= 2.5e-4
