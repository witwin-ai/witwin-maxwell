"""G2 gates: surface_admittance (thick metal) on the matrix_free and hmatrix backends.

The per-body admittance blocks are compressed H-matrices of W_b in the action
(body_compress.py) and enter the sparse near LU only on the pattern keys
(body_block_lu.py) of both backends.

- Action: A = M + Far equals the dense equilibrated system with the W blocks
  folded, inside the per-row bound 64 eps64 times the absolute-value action
  (S2a's bound with |W| added to the magnetic entries before the congruence)
  plus the compression term |d Q^T|_2 eps_b |Q d|_2 |x|_2 per column, eps_b
  the dropped Frobenius mass of c W_b.
  The adjoint is the true A^H (DSA blocks are not symmetric).
- Network: matrix_free and hmatrix agree with dense to |dY|_F/|Y|_F <= 1e-4 per
  frequency in all three regimes (low, window, DSA), with the GMRES stopping
  bound <= 1e-5 and bounded block GMRES steps.
- Compression: every admissible block is a cross approximation with explicit
  residual |R|_F <= TOLERANCE |W_b|_F; the reported dropped mass equals
  |c W_b - W~_b|_F, the compressed apply matches the dense block within it,
  low-regime blocks have rank 0, repeats (ranks and factors) are torch.equal.
- Preconditioner: PatternBodyLU applies M^-1 (dense oracle) of the pattern plus
  the body blocks on the pattern keys, and is torch.equal on repeat.
- Repeated runs are torch.equal.
"""

import dataclasses
import gc
import json
import math
import time

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.mom.layered.lowfreq import assemble_chunk, transform_chunk
from witwin.maxwell.mom.layered.admittance_near import body_values
from witwin.maxwell.mom.layered.body_block_lu import PatternBodyLU, near_preconditioner
from witwin.maxwell.mom.layered.body_compress import TOLERANCE, CompressedBodyBlock, fold_constant
from witwin.maxwell.mom.layered.body_preconditioner import body_preconditioner_structure
from witwin.maxwell.mom.layered.matrix_free import (
    ADMITTANCE_TOLERANCE, LayeredMatrixFreeOperator, build_near_structure)
from tests.mom.layered.test_layered_lowfreq import dense_columns
from tests.support.rfic_closed_references import strip_fixture


MU0 = 1 / (8.8541878128e-12 * 299792458.0**2)
EPS0 = 8.8541878128e-12
# Low regime, crossover window and DSA for the 2-um strip (t/delta = 0.3 at 9.8e7 Hz, 0.6 at 3.9e8 Hz).
FREQUENCIES = (1e6, 2e8, 1e10)
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


def backend(config, name):
    return dataclasses.replace(config, layered=dataclasses.replace(config.layered, operator_backend=name))


def relative(a, b):
    return (torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b)).tolist()


def strip(name, frequencies=FREQUENCIES):
    # Coarse 40/4 um strip mesh: the per-body DSA operators dominate the run time.
    scene, config = strip_fixture(model="surface_admittance", maximum_um=40., edge_um=4.)
    return mw.Simulation.mom(scene, frequencies=frequencies, config=backend(config, name))


def test_config_admits_iterative_backends():
    for name in ("matrix_free", "hmatrix"):
        config = mw.mom.LayeredSurfaceConfig(bridge_width_m=1e-5, max_edge_length_m=1e-5,
                                         conductor_model="surface_admittance", operator_backend=name)
        assert config.payload()["operator_backend"] == name


def test_action_with_body_blocks_matches_dense_matrix(record_property):
    prepared = strip("matrix_free").prepare().solver.prepared
    blocks = prepared.admittance.blocks(FREQUENCIES)
    system = transform_chunk(prepared, assemble_chunk(prepared), admittance_blocks=blocks)
    # Absolute-value oracle of the unscaled entries (W folded, before the congruence).
    raw = assemble_chunk(prepared)
    lt = prepared.loop_tree
    prepared.admittance.fold(raw, lt.order, blocks)
    faces = len(prepared.layout.mesh.faces)
    packed = raw.workspace.view(torch.complex128)[:, :faces * (faces + 1) // 2]
    lower = torch.tril_indices(faces, faces, device="cuda")
    structure = build_near_structure(prepared)
    assert structure.stats["admittance_bodies"] == len(prepared.admittance.bodies) == 1
    solver = near_preconditioner(structure, None, prepared.excitation.device)
    records = []
    for k, frequency in enumerate(FREQUENCIES):
        operator = LayeredMatrixFreeOperator(prepared, structure, k, solver)
        n = operator.shape[0]
        w = 2 * math.pi * frequency
        factor = torch.full((n,), lt.omega0, device="cuda", dtype=torch.float64)
        factor[:lt.loops] = w
        change = operator.scale / (system.recovery[k] * factor)
        generator = torch.Generator(device="cuda").manual_seed(31 + k)
        x = torch.randn((n, 3), generator=generator, device="cuda", dtype=torch.complex128)
        reference = change[:, None] * (system.matrix[k] @ (change[:, None] * x))
        action = operator.apply(x.mT).mT
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
        # Compression: |d Q^T E Q d x|_2 <= |Q d|_2^2 eps_b |x|_2 per column (E = c (W - W~)).
        compression = torch.zeros(3, device="cuda", dtype=torch.float64)
        for b, block, d in zip(structure.bodies, operator.bodies, operator.body_scales):
            qd = torch.zeros((len(b.ids), len(b.columns)), device="cuda", dtype=torch.float64)
            qd[b.q_rows, b.q_cols] = b.q_values
            qd = qd * d[None]
            compression += (torch.linalg.matrix_norm(qd, 2) ** 2 * block.stats["dropped_frobenius"]
                            * torch.linalg.vector_norm(x, dim=0))
        # |err_i| <= bound_i + |compression error_i|, so the excess over the rounding bound is <= the compression term.
        excess = torch.linalg.vector_norm(((action - reference).abs() - bound).clamp(min=0), dim=0)
        assert (excess <= compression).all()
        ratio = ((action - reference).abs() / bound).max().item()
        # True adjoint: <A x, y> = <x, A^H y>; the DSA block makes A nonsymmetric.
        y = torch.randn((3, n), generator=generator, device="cuda", dtype=torch.complex128)
        left = (operator.apply(x.mT) * y.conj()).sum(-1)
        right = (x.mT * operator.adjoint(y).conj()).sum(-1)
        adjoint = ((left - right).abs() / left.abs()).max().item()
        dense = system.matrix[k]
        asymmetry = ((dense - dense.mT).abs().max() / dense.abs().max()).item()
        records.append(dict(frequency=frequency, bound_ratio=ratio, excess=excess.tolist(),
                            compression=compression.tolist(), adjoint_error=adjoint, asymmetry=asymmetry,
                            error=((action - reference).abs().max() / reference.abs().max()).item()))
        assert adjoint <= 1e-12
        del operator
    solver.free()
    record_property("action", json.dumps(records))
    # The DSA frequency carries a nonsymmetric block, so the adjoint test is not vacuous.
    assert records[-1]["asymmetry"] > 1e-6


@pytest.mark.parametrize("name", ("matrix_free", "hmatrix"))
def test_strip_agreement_with_dense(name, record_property):
    dense = strip("dense").run().network.to_y()
    result = strip(name).run()
    errors = relative(result.network.to_y(), dense)
    metadata = result.network.metadata
    record_property("agreement", json.dumps(dict(
        frequencies=FREQUENCIES, relative_error=errors, steps=metadata["block_gmres_steps"],
        bound=metadata["admittance_bound"], unknowns=metadata["unknowns"],
        columns=metadata["near_field"]["admittance_basis_columns"],
        block_nnz=metadata["near_field"]["admittance_block_nnz"])))
    assert metadata["operator_backend"] == name and metadata["conductor_model"] == "surface_admittance"
    assert max(errors) <= 1e-4
    assert max(metadata["admittance_bound"]) <= ADMITTANCE_TOLERANCE
    assert max(metadata["block_gmres_steps"]) <= 30


def test_repeat_determinism():
    first = None
    for _ in range(2):
        result = strip("matrix_free", (1e6, 1e10)).run()
        values = (result.network.to_y(), mw.mom.surface_currents(result).currents)
        if first is None:
            first = values
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, values))
        del result
        gc.collect()


def test_interior_operator_launch_split_is_bitwise():
    """A1 interior kernels launch in bounded chunks (admittance.LAUNCH); every pair keeps one owner thread or
    block (far: one warp), so any split, including tiny chunks, is torch.equal to a single launch per kernel."""
    from types import SimpleNamespace
    from witwin.maxwell.mom.layered.admittance import interior_operators
    from tests.mom.layered.test_layered_admittance import SIGMA, UM, _box
    vertices, faces, basis = _box(4, (8 * UM, 4 * UM, 2 * UM))
    body = SimpleNamespace(vertices=vertices, local_faces=faces, basis=basis, trial=basis)
    omega = 2 * math.pi * 1e9
    eps0 = 8.8541878128e-12
    mu0 = 4e-7 * math.pi
    kb = complex(omega * math.sqrt(mu0 * eps0))
    kc = complex(omega * (mu0 * (eps0 + 1j * SIGMA / omega)) ** .5)
    single = interior_operators(body, kc, kb, launch=(1 << 62, 1 << 62))
    for launch in ((128, 3), (1000, 17)):
        split = interior_operators(body, kc, kb, launch=launch)
        assert all(torch.equal(split[key], single[key]) for key in single)


def test_pattern_body_lu_is_pattern_inverse(record_property):
    """PatternBodyLU applies M^-1, M = D-scaled wide pattern + D B_b D on the pattern keys only (dense oracle)."""
    prepared = strip("matrix_free").prepare().solver.prepared
    structure = build_near_structure(prepared)
    wide = body_preconditioner_structure(prepared, structure)
    solver = near_preconditioner(structure, wide, prepared.excitation.device)
    assert isinstance(solver, PatternBodyLU)
    n = len(structure.offsets) - 1
    rows = torch.repeat_interleave(torch.arange(n, device="cuda"), wide.offsets.diff())
    records = []
    for k, frequency in enumerate(FREQUENCIES):
        operator = LayeredMatrixFreeOperator(prepared, structure, k, solver, wide)
        dense = torch.zeros((n, n), device="cuda", dtype=torch.complex128)
        dense[rows, wide.cols] = operator._scaled(wide, operator.near_values(wide))
        full = dense.clone()
        blocks = body_values(structure.bodies, prepared.admittance.body_blocks(frequency), frequency,
                             operator.basis_scale)
        for b, block in zip(structure.bodies, blocks):
            scale = operator.scale[b.columns]
            block = block * (scale[:, None] * scale[None])
            full[b.columns[:, None], b.columns[None]] += block
        on_pattern = torch.zeros_like(dense)
        on_pattern[rows, wide.cols] = (full - dense)[rows, wide.cols]
        dense += on_pattern
        generator = torch.Generator(device="cuda").manual_seed(7 + k)
        r = torch.randn((n, 2), generator=generator, device="cuda", dtype=torch.complex128)
        z = operator.precondition(r)
        residual = ((dense @ z - r).abs().max() / r.abs().max()).item()
        outside = (torch.linalg.matrix_norm(full - dense) / torch.linalg.matrix_norm(full - dense + on_pattern)).item()
        reference = torch.linalg.solve(dense, r)
        error = ((z - reference).abs().max() / reference.abs().max()).item()
        # Normwise backward error of the applied M^-1 (independent of the conditioning of M).
        norm = dense.abs().sum(dim=1).max()
        backward = ((dense @ z - r).abs().max() / (norm * z.abs().max() + r.abs().max())).item()
        condition = (norm * torch.linalg.inv(dense).abs().sum(dim=1).max()).item()
        records.append(dict(frequency=frequency, error=error, residual=residual, backward=backward, condition=condition,
                            outside_mass=outside, factor_nnz=solver.stats["factor_nnz"]))
        # Derived bounds (numerical_equivalence_standard.md, fp64 setup form 64 eps): the cuDSS LU of the
        # pattern matrix M (no body fill beyond the body-free pattern's) is backward stable, so its normwise
        # backward error is <= 64 eps64, and the forward error against the dense solve is <= cond_inf(M) * 64 eps64.
        # The |r|-relative residual only carries cond(M) and is recorded, not gated.
        bound = 64 * torch.finfo(torch.float64).eps
        assert backward <= bound
        assert error <= condition * bound
        assert torch.equal(operator.precondition(r), z)
        del operator
    solver.free()
    record_property("pattern_lu", json.dumps(records))


@pytest.mark.parametrize("frequency", (1e6, 1e10))
def test_compressed_body_block_matches_dense(frequency, record_property):
    """H-matrix W_b: eps_b = |c W - W~|_F exactly, |(W~ - W) u| <= eps_b |u|, every block residual <= TOLERANCE |W|_F;
    low-regime admissible blocks are exactly rank 0; transposed apply and repeats are exact."""
    prepared = strip("matrix_free", (frequency,)).prepare().solver.prepared
    structure = build_near_structure(prepared)
    (b,), (plan,) = structure.bodies, structure.body_plans
    ids, w = prepared.admittance.body_blocks(frequency)[b.owner]
    constant = fold_constant(frequency)
    block = CompressedBodyBlock(plan, w, constant)
    again = CompressedBodyBlock(plan, w, constant)
    # cuSOLVER SVD on repeat: identical ranks, factors and CSR layout.
    assert block.stats == again.stats and block.rank == again.rank
    assert all(torch.equal(x, y) for x, y in zip(block.left + block.right, again.left + again.right))
    q = torch.zeros((len(b.ids), len(b.columns)), device="cuda", dtype=torch.complex128)
    q[b.q_rows, b.q_cols] = b.q_values.to(torch.complex128)
    generator = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn((len(b.columns), 2), generator=generator, device="cuda", dtype=torch.complex128)
    d = torch.rand(len(b.columns), generator=generator, device="cuda", dtype=torch.float64) + .5
    reference = d[:, None] * (q.mT @ ((constant * w) @ (q @ (d[:, None] * x))))
    out = block.apply(x, d)
    dropped = block.stats["dropped_frobenius"]
    qd = torch.linalg.matrix_norm(q * d[None], 2).item()
    error = torch.linalg.vector_norm(out - reference, dim=0)
    rounding = 64 * 2.0**-53 * qd ** 2 * torch.linalg.matrix_norm(constant * w).item() * torch.linalg.vector_norm(x, dim=0)
    assert (error <= qd ** 2 * dropped * torch.linalg.vector_norm(x, dim=0) + rounding).all()
    far_blocks = sum(len(r) for r, _ in plan.far)
    assert dropped <= math.sqrt(far_blocks) * TOLERANCE * block.stats["w_frobenius"]
    # The reported dropped mass is the exact Frobenius error of W~ (reconstructed through its CSR factors).
    op = torch.ops.witwin_maxwell_mom_layered
    eye = torch.eye(len(ids), device="cuda", dtype=torch.complex128)
    stacked = torch.cat((eye, op.csr_matmul(*block.right, eye))) if block.rank else eye
    exact = torch.linalg.matrix_norm(op.csr_matmul(*block.left, stacked) - constant * w).item()
    assert abs(exact - dropped) <= 1e-6 * dropped + 64 * 2.0**-53 * block.stats["w_frobenius"]
    transposed = block.apply(x, d, transpose=True)
    reference_t = d[:, None] * (q.mT @ ((constant * w).mT @ (q @ (d[:, None] * x))))
    assert (torch.linalg.vector_norm(transposed - reference_t, dim=0)
            <= qd ** 2 * dropped * torch.linalg.vector_norm(x, dim=0) + rounding).all()
    assert torch.equal(block.apply(x, d), out)
    if frequency < 1e7:
        assert block.rank == 0
    record_property("compression", json.dumps(dict(block.stats, error=error.tolist(), rounding=rounding.tolist(),
                                                   exact_dropped=exact)))


@pytest.mark.parametrize("frequency", (1e6, 1e9, 1e11))
def test_interior_far_kernel_matches_serial_oracle_bitwise(frequency):
    """R5-2: the warp-per-pair far kernel sums every output slot in the serial (i,j,k,l) node order with the
    same IEEE operations, so all seven operators are torch.equal to the one-thread-per-pair oracle. The
    frequencies cover the Taylor branch of F (low), the base far orders and the |kc|-resolved order 10 (DSA)."""
    from types import SimpleNamespace
    from witwin.maxwell.mom.layered.admittance import interior_operators
    from tests.mom.layered.test_layered_admittance import SIGMA, UM, _box
    from tests.support.layered_admittance_oracle import serial_far_interior_operators
    vertices, faces, basis = _box(4, (8 * UM, 4 * UM, 2 * UM))
    body = SimpleNamespace(vertices=vertices, local_faces=faces, basis=basis, trial=basis)
    omega = 2 * math.pi * frequency
    eps0 = 8.8541878128e-12
    mu0 = 4e-7 * math.pi
    kb = complex(omega * math.sqrt(mu0 * eps0))
    kc = complex(omega * (mu0 * (eps0 + 1j * SIGMA / omega)) ** .5)
    warp = interior_operators(body, kc, kb, launch=(1000, 17))
    serial = serial_far_interior_operators(body, kc, kb, launch=(1000, 17))
    assert all(torch.equal(warp[key], serial[key]) for key in serial)
