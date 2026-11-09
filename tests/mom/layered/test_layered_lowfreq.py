"""R5 GPU gates: complete device cycles, stable MPIE, and static port limits."""

import json
import math

import pytest
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.assembly import assemble_layered, assemble_potentials
from witwin.maxwell.mom.layered.lowfreq import build_loop_tree, scaled_system
from witwin.maxwell.mom.layered.mesh import mesh_rectangles
from witwin.maxwell.mom.layered.network import (
    prepare_layered_network, solve_dense, solve_prepared_network)
from tests.mom.layered.test_layered_network import (
    coplanar_strips, gap_port, rfic_stack, square_loop)


pytestmark = pytest.mark.cuda
EPS0 = 8.8541878128e-12
MU0 = 1 / (EPS0 * 299792458.0**2)


def vacuum():
    return LayerStack((Layer("air", 1e-3, 1.),), HalfSpace(), HalfSpace())


def loop(frequencies, *, copper=False):
    side, width, gap, z = 100e-6, 5e-6, 5e-6, 0.
    rectangles = square_loop(side, width, gap, z)
    port = gap_port("loop", (gap / 2, -side / 2, z), (-gap / 2, -side / 2, z), "x", width)
    return prepare_layered_network(
        vacuum(), rectangles, [port], frequencies, bridge_width=width, max_edge=5e-6,
        sheet_materials=[(5.8e7, 1e-6, None)] * len(rectangles) if copper else None)


def measure(record_property, **values):
    for key, value in values.items():
        record_property(key, json.dumps(value))
        print(f"{key}={value}")


def graph_reference(plus, minus, faces):
    # Independent CPU connected-components traversal, only a small test oracle.
    adjacency = [[] for _ in range(faces)]
    for a, b in zip(plus, minus):
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen, components = set(), 0
    for start in range(faces):
        if start in seen:
            continue
        components += 1
        pending = [start]
        seen.add(start)
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
    return components


@pytest.mark.parametrize("kind", ["loop", "strips", "rfic", "disconnected"])
def test_basis_complete_and_deterministic(kind, record_property):
    if kind == "loop":
        rectangles = square_loop(100e-6, 5e-6, 5e-6, 0.)
        rectangles.append((-2.5e-6, 2.5e-6, -52.5e-6, -47.5e-6, 0.))
    else:
        rectangles, _ = coplanar_strips(40e-6, 5e-6, 5e-6, 2e-6, 0.)
        if kind != "disconnected":
            rectangles += [(0., 2e-6, -2.5e-6, 2.5e-6, 0.),
                           (38e-6, 40e-6, -2.5e-6, 2.5e-6, 0.)]
        if kind == "rfic":
            z = rfic_stack().interfaces()[5]
            rectangles = [(*r[:4], z) for r in rectangles]
    planes = torch.tensor(sorted({r[4] for r in rectangles}), device="cuda", dtype=torch.float64)
    layout = mesh_rectangles(rectangles, planes, 2.5e-6)
    lt = build_loop_tree(layout, length_scale=10e-6)
    basis = layout.basis
    n, f = basis.n_edges, len(layout.mesh.faces)
    components = graph_reference(basis.plus_faces.tolist(), basis.minus_faces.tolist(), f)
    assert lt.components == components
    assert lt.trees == f - components
    assert lt.loops == n - (f - components)
    assert torch.linalg.matrix_rank(lt.transform) == n
    d = layout.mesh.areas[:, None] * basis.divergence
    divergence = d @ lt.lam
    # Unit-integral P0 convention: D is integrated face flux, in metres.
    defect = divergence.abs().max().item() if lt.loops else 0.
    bound = 1e-12 * lt.lam.abs().max().item() if lt.loops else 0.
    assert defect <= bound
    for _ in range(2):
        again = build_loop_tree(layout, length_scale=10e-6)
        for name in ("order", "parents", "transform", "q", "r", "charge"):
            assert torch.equal(getattr(lt, name), getattr(again, name))
    measure(record_property, unknowns=n, faces=f, components=components,
            loops=lt.loops, trees=lt.trees, divergence_defect=defect, divergence_bound=bound)


def test_separate_potentials_match_summed_assembly(record_property):
    prepared = loop((1e9,))
    vector, scalar = assemble_potentials(prepared.layout, prepared.table)
    d = (prepared.layout.mesh.areas[:, None] * prepared.layout.basis.divergence).to(torch.complex128)
    w = 2 * math.pi * 1e9
    reconstructed = -1j * w * MU0 * vector + 1j / (w * EPS0) * (d.T @ scalar @ d)
    reference = assemble_layered(prepared.layout, prepared.table)
    relative = ((reconstructed - reference).abs().max() / reference.abs().max()).item()
    assert relative <= 64 * torch.finfo(torch.float64).eps
    for _ in range(2):
        v, p = assemble_potentials(prepared.layout, prepared.table)
        assert torch.equal(v, vector)
        assert torch.equal(p, scalar)
        assert torch.equal(assemble_layered(prepared.layout, prepared.table), reference)
    measure(record_property, reexpression_relative=relative)


@pytest.fixture(scope="module")
def pec_loop():
    prepared = loop((1., 1e3, 1e6, 100e6, 1e9, 10e9))
    return prepared, solve_prepared_network(prepared)[0]


def test_loop_inductance_flat_to_dc(pec_loop, record_property):
    prepared, network = pec_loop
    z = network.to_z()[:, 0, 0]
    omega = 2 * math.pi * prepared.table.frequencies
    inductance = -z.imag / omega
    flatness = (inductance[:5] / inductance[1] - 1).abs().max().item()
    assert flatness < .002
    old = assemble_layered(prepared.layout, prepared.table, frequency_index=4)
    x, _ = solve_dense(old, prepared.excitation)
    old_z = (1 / (prepared.excitation.T @ x))[0, 0]
    relative = (z[4] / old_z - 1).abs().item()
    assert relative < 1e-6
    measure(record_property, frequencies_Hz=prepared.table.frequencies.tolist(),
            inductance_H=inductance.tolist(), flatness=flatness, GHz_legacy_relative=relative)


def test_scaled_condition_is_frequency_bounded(pec_loop, record_property):
    prepared, _ = pec_loop
    assert prepared.layout.basis.n_edges <= 2000
    matrix, _, _ = scaled_system(prepared)
    stable = torch.linalg.cond(matrix).tolist()
    # Direct assembly is diagnostic only: it becomes numerically singular
    # before DC, so its computed condition eventually saturates at 1/eps.
    original = [torch.linalg.cond(assemble_layered(prepared.layout, prepared.table,
                                                  frequency_index=i)).item()
                for i in (1, 2, 3, 4, 5)]
    variation = max(stable[1:]) / min(stable[1:])
    assert variation < 10
    assert original[0] / original[-1] > 1e6
    measure(record_property, unknowns=prepared.layout.basis.n_edges, scaled_condition=stable, unscaled_condition=original,
            scaled_variation=variation)


def test_copper_dc_and_continuity(record_property):
    frequencies = tuple(10.**(3 + i * .5) for i in range(15))
    prepared = loop(frequencies, copper=True)
    network, _ = solve_prepared_network(prepared)
    resistance = network.to_z()[:, 0, 0].real
    # R4 reference: same sheet Gram term at 100 MHz with Z_sheet=1/(sigma*t).
    reference = assemble_layered(prepared.layout, prepared.table, frequency_index=10)
    from witwin.maxwell.mom.layered.loss import SheetLoss
    sheet = prepared.loss
    dc_impedance = torch.where(sheet.impedance.real > 0,
                               1 / (5.8e7 * 1e-6), 0).to(torch.complex128)
    SheetLoss(sheet.blocks, sheet.columns, sheet.supports, dc_impedance).add_to(reference, 10)
    x, _ = solve_dense(reference, prepared.excitation)
    dc = (1 / (prepared.excitation.T @ x))[0, 0].real.item()
    relative = abs(resistance[0].item() / dc - 1)
    assert relative < .005
    assert (resistance > 0).all()
    # Bound local changes and second differences on a half-decade log sweep.
    ratio = resistance[1:] / resistance[:-1]
    assert ratio.max() < 2
    assert ratio.min() > .99
    measure(record_property, frequencies_Hz=frequencies, resistance_ohm=resistance.tolist(),
            dc_reference_ohm=dc, dc_relative=relative, max_adjacent_ratio=ratio.max().item())


def test_capacitive_static_limit(record_property):
    rectangles = [(-110e-6, -10e-6, -50e-6, 50e-6, 0.),
                  (0., 100e-6, -50e-6, 50e-6, 0.)]
    port = gap_port("capacitor", (0., 0., 0.), (-10e-6, 0., 0.), "x", 5e-6)
    prepared = prepare_layered_network(vacuum(), rectangles, [port],
                                       (1., 1e3, 1e6, 100e6), bridge_width=5e-6, max_edge=20e-6)
    network, currents = solve_prepared_network(prepared)
    z = network.to_z()[:, 0, 0]
    capacitance = 1 / (2 * math.pi * prepared.table.frequencies * z.imag)
    assert (z.imag > 0).all()
    variation = (capacitance / capacitance[1] - 1).abs().max().item()
    assert variation < .005
    # A gradient gap source has exactly zero cycle work before QR.
    work = torch.ops.witwin_maxwell_mom_layered.port_work(
        prepared.loop_tree.transform, prepared.layout.basis.edge_lengths.contiguous(),
        prepared.excitation)
    assert (work[:prepared.loop_tree.loops] == 0).all()
    current_y = prepared.excitation.T @ currents
    mismatch = ((current_y - network.to_y()).abs() / network.to_y().abs()).max().item()
    # NetworkData stores S: its round trip can contaminate a tiny ReY with
    # rounding. Reactive charge/current and C must retain the static limit.
    reactive_error = (current_y.imag / network.to_y().imag - 1).abs().max().item()
    assert reactive_error < 1e-12
    normalized = currents / prepared.table.frequencies[:, None, None]
    current_flatness = (torch.linalg.vector_norm(normalized - normalized[0], dim=1) /
                        torch.linalg.vector_norm(normalized[0], dim=0)).max().item()
    assert current_flatness < .005
    measure(record_property, capacitance_F=capacitance.tolist(), flatness=variation,
            S_roundtrip_current_relative=mismatch, reactive_current_relative=reactive_error,
            current_over_frequency_flatness=current_flatness)


def test_rfic_low_frequency_reciprocity_passivity_determinism(record_property):
    stack = rfic_stack()
    strips, ports = coplanar_strips(100e-6, 5e-6, 5e-6, 2e-6, stack.interfaces()[5])
    prepared = prepare_layered_network(stack, strips, ports, (1e6, 1e9),
                                       bridge_width=2e-6, max_edge=5e-6, edge_cell=1.25e-6)
    network, currents = solve_prepared_network(prepared)
    # Test dissipation on native current-derived Y, before the ill-conditioned
    # S->Y round trip at DC. Also keep the public S passivity gate unchanged.
    y = prepared.excitation.T @ currents
    roundtrip = network.to_y()
    symmetry = (torch.linalg.matrix_norm(y - y.transpose(-1, -2)) /
                torch.linalg.matrix_norm(y)).max().item()
    passive = torch.linalg.matrix_norm(network.s, ord=2).max().item()
    conductance = torch.linalg.eigvalsh((y + y.mH) / 2).min().item()
    assert symmetry < 1e-10
    assert passive <= 1 + 1e-9
    assert conductance > 0
    for chunk_size in (1, 2):
        again, x = solve_prepared_network(prepared, frequency_chunk_size=chunk_size)
        assert torch.equal(again.s, network.s)
        assert torch.equal(again.to_y(), roundtrip)
        assert torch.equal(x, currents)
    measure(record_property, relative_reciprocity=symmetry, max_S_singular_value=passive,
            min_conductance_S=conductance,
            S_roundtrip_min_conductance_S=torch.linalg.eigvalsh((roundtrip + roundtrip.mH) / 2).min().item())


@pytest.mark.parametrize("frequency", [0., .5])
def test_dc_is_a_limit_not_a_zero_frequency_solve(frequency):
    with pytest.raises(ValueError, match="DC is a limit"):
        loop((frequency,))


@pytest.mark.parametrize("copper", [False, True])
def test_frequency_chunks_are_bitwise_identical(copper, record_property):
    prepared = loop((1e3, 1e6, 100e6, 1e9), copper=copper)
    reference, currents = solve_prepared_network(prepared, frequency_chunk_size=4)
    assert reference.metadata["frequency_chunk_size"] == 4
    for size in (1, 2, 1):
        network, x = solve_prepared_network(prepared, frequency_chunk_size=size)
        assert network.metadata["frequency_chunk_size"] == size
        assert torch.equal(x, currents)
        assert torch.equal(network.s, reference.s)
        assert torch.equal(network.to_y(), reference.to_y())
        for key in ("factor_dtype", "refinement_steps", "backward_error"):
            assert network.metadata[key] == reference.metadata[key]
    assert set(reference.metadata["factor_dtype"]) == {"complex64"}
    assert max(reference.metadata["backward_error"]) <= 64 * 2.**-53
    measure(record_property, factor_dtype=reference.metadata["factor_dtype"],
            refinement_steps=reference.metadata["refinement_steps"],
            backward_error=reference.metadata["backward_error"])


def test_refinement_escalates_only_failed_frequencies():
    from witwin.maxwell.mom.layered.build import build_extension
    from witwin.maxwell.mom.layered.network import solve_scaled_chunk
    build_extension()
    # Second matrix is nonsingular in fp64 but exactly singular after fp32
    # conversion. The well-conditioned first frequency must retain fp32 LU.
    matrix = torch.tensor([[[2., .25], [.25, 1.]],
                           [[1., 1.], [1., 1. + 1e-9]]],
                          device="cuda", dtype=torch.complex128)
    rhs = torch.tensor([[[1.], [2.]], [[1.], [2.]]], device="cuda", dtype=torch.complex128)
    x, info = solve_scaled_chunk(matrix, rhs)
    assert info[0] == ("complex64", "complex128")
    assert max(info[2]) <= 64 * 2.**-53
    assert torch.allclose(matrix @ x, rhs, atol=2e-7, rtol=0.)
    for _ in range(2):
        again, metadata = solve_scaled_chunk(matrix, rhs)
        assert torch.equal(again, x)
        assert metadata == info


def test_memory_admission_depends_on_one_frequency():
    from witwin.maxwell.mom.layered.network import frequency_chunk_capacity
    n, faces = 8000, 5500
    one = 16 * (12 * n * n + faces * faces)
    available = 2 * one
    assert frequency_chunk_capacity(n, faces, 1, available) == 1
    assert frequency_chunk_capacity(n, faces, 50, available) == 1
    assert frequency_chunk_capacity(n, faces, 50, 4 * one) == 3
    with pytest.raises(RuntimeError, match="One dense layered frequency"):
        frequency_chunk_capacity(n, faces, 50, one)
