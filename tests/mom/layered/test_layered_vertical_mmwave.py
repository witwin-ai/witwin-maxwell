"""G1 gates: mm-wave z admission of vertical Green tables (graded z segments).

The static-image remainder keeps a k^2 d log d singularity at every dielectric
interface; ungraded Chebyshev z tables converge like f^2 K^-4 and fail the 1e-6
z gate above ~20-30 GHz on a 254 um eps 9.8 grounded microstrip. A failing
table is rebuilt with ratio-4 geometric pieces toward the dielectric
interfaces (vertical_refine.py); an admitted ungraded table is unchanged.

Acceptance (CUDA): admission at 60/100/150/200 GHz with the predicted level
on the first graded build, bitwise repeat, a network-level check against a
converged reference (one more ratio-4 level: a 16x smaller corner piece) <= 1e-6, and the
fail-closed ConvergenceError. Host: piece geometry and mesh breakpoints.
"""

import functools
import math

import pytest
import torch

import witwin.maxwell as mw
from witwin.maxwell.errors import ConvergenceError
from witwin.maxwell.mom.layered import LayeredGreenTable
from witwin.maxwell.mom.layered import network, vertical_refine
from witwin.maxwell.mom.layered.build import build_extension
from witwin.maxwell.mom.layered.vertical_refine import dielectric_interfaces, graded_segments, mesh_interfaces

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
H, WIDTH = 254e-6, 254e-6


def grounded_stack():
    return mw.mom.LayerStack((mw.mom.Layer("substrate", H, 9.8),), "pec", mw.mom.HalfSpace())


def microstrip_table(frequency):
    return LayeredGreenTable.build(grounded_stack(), (0.0, H), (frequency,), 1.01 * math.hypot(4e-3, H),
                                   device="cuda", via_spans=((0.0, H),), z_nodes=12,
                                   extract_distance=max(math.sqrt(2) * H, H))


def test_graded_pieces_and_mesh_breakpoints():
    stack = grounded_stack()
    assert dielectric_interfaces(stack) == (H,)
    assert graded_segments([(0.0, H)], stack, 2, 4) == [(0.0, H - H / 4), (H - H / 4, H - H / 16), (H - H / 16, H)]
    two = mw.mom.LayerStack((mw.mom.Layer("a", 100e-6, 12.9), mw.mom.Layer("b", 10e-6, 3.0)), "pec", mw.mom.HalfSpace())
    pieces = graded_segments([(0.0, 100e-6), (100e-6, 110e-6)], two, 1, 4)
    # The PEC ground is never graded; the interior interface is graded from both sides.
    expected = [(0.0, 75e-6), (75e-6, 100e-6), (100e-6, 101.25e-6), (101.25e-6, 108.75e-6), (108.75e-6, 110e-6)]
    assert [b for _, b in pieces[:-1]] == [a for a, _ in pieces[1:]]
    assert pieces == [(pytest.approx(a, abs=1e-18), pytest.approx(b, abs=1e-18)) for a, b in expected]

    class Table:
        vertical = None
    assert mesh_interfaces(stack, Table()) == stack.interfaces()


@requires_cuda
@pytest.mark.cuda
def test_ungraded_table_kept_below_the_gate(record_property):
    # 3 GHz admits ungraded: the table is the pre-G1 table (no grading, no record).
    table = microstrip_table(3e9)
    assert table.vertical.grading is None and table.vertical.grading_record is None
    assert table.vertical.z_nodes == 12 and table.vertical.segments.shape[0] == 1
    record_property("z_admission", table.vertical.z_interpolation_error.item())


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("frequency", (60e9, 100e9, 150e9, 200e9))
def test_mmwave_grounded_table_admits(frequency, record_property):
    table = microstrip_table(frequency)
    vertical, record = table.vertical, table.vertical.grading_record
    record_property("grading_record", record)
    assert record["ungraded_z_error"] > 1e-6
    assert vertical.z_interpolation_error.item() <= 1e-6 and vertical.rho_interpolation_error.item() <= 1e-6
    # The delta^2 K^-4 prediction picks the admitted level on the first graded build.
    assert record["levels"] == record["predicted_levels"]
    assert vertical.grading == (record["levels"], 4) and vertical.z_nodes == 12
    if frequency == 60e9:
        # Bitwise repeat at the cheapest graded level that fits the 6 GiB test cap twice.
        names = ("z", "segments", "remainder", "derivatives", "horizontal_remainder", "horizontal_derivatives")
        first = {name: getattr(vertical, name).cpu() for name in names}
        del table, vertical
        torch.cuda.empty_cache()
        again = microstrip_table(frequency).vertical
        for name in names:
            assert torch.equal(first[name], getattr(again, name).cpu())


@requires_cuda
@pytest.mark.cuda
def test_graded_admission_fails_closed(monkeypatch):
    stack = mw.mom.LayerStack((mw.mom.Layer("substrate", 1e-3, 4.1),), mw.mom.HalfSpace(), mw.mom.HalfSpace())
    monkeypatch.setattr(build_extension(), "vertical_error", lambda actual, expected: torch.tensor(2e-6, device="cuda"))
    monkeypatch.setattr(vertical_refine, "MAX_GRADING_LEVELS", 1)
    with pytest.raises(ConvergenceError, match="Vertical z interpolation admission failed.*after 1 grading levels"):
        LayeredGreenTable.build(stack, (.2e-3, .4e-3), (1e9,), 2e-3, device="cuda", via_spans=((.2e-3, .4e-3),))


def _microstrip_scene():
    stack = grounded_stack()
    scene = mw.Scene(background=stack, structures=(
                mw.Structure(mw.geometry.PlanarPEC(
            [(0., -WIDTH / 2), (2e-3, -WIDTH / 2), (2e-3, WIDTH / 2), (0., WIDTH / 2)], normal_axis="z", coordinate=H),
            mw.Material.pec()),
            ))
    scene = scene.updated(ports=(
        *scene.ports,
        *(mw.LumpedPort(name, positive=(x, 0., H), negative=(x, 0., 0.), voltage_path=mw.AxisPath("z")) for name, x in (("p1", 0.), ("p2", 2e-3))),
    ))
    return scene


@requires_cuda
@pytest.mark.cuda
def test_mmwave_grounded_network_against_refined_table(monkeypatch, record_property):
    config = mw.mom.MoMConfig(layered=mw.mom.LayeredSurfaceConfig(bridge_width_m=WIDTH, max_edge_length_m=H / 2,
                                                          edge_cell_m=WIDTH / 8))
    def solve():
        # 30 GHz (graded level 1, reference level 2) fits the 6 GiB test cap; the probe covers 60-200 GHz.
        return mw.Simulation.mom(_microstrip_scene(), frequencies=(30e9,), config=config).run()
    # Graded piece cuts are wall-mesh breakpoints. Mesh head and reference with the
    # reference's (finer, superset) cuts so the comparison isolates the table.
    def common_mesh(stack, table):
        pieces = graded_segments([(0.0, H)], stack, 2, vertical_refine.GRADING_RATIO)
        return tuple(sorted(set(stack.interfaces()) | {z for piece in pieces for z in piece}))
    monkeypatch.setattr(network, "mesh_interfaces", common_mesh)
    head = solve()
    assert head.network.metadata["vertical_z_grading"]["levels"] == head.solver.prepared.table.vertical.grading[0]
    again = solve()
    assert torch.equal(head.network.s, again.network.s)
    assert torch.equal(mw.mom.surface_currents(head).currents, mw.mom.surface_currents(again).currents)
    y = head.network.to_y()
    assert head.network.metadata["vertical_z_grading"]["levels"] == 1
    del head, again
    torch.cuda.empty_cache()
    monkeypatch.setattr(vertical_refine, "admitted_vertical_table",
                        functools.partial(vertical_refine.admitted_vertical_table, extra_levels=1))
    reference = solve()
    yr = reference.network.to_y()
    error = (torch.linalg.matrix_norm(y - yr) / torch.linalg.matrix_norm(yr)).max().item()
    record_property("relative_y_vs_refined", error)
    assert error <= 1e-6


def _process_cap_gib():
    total = torch.cuda.get_device_properties(0).total_memory
    return torch.cuda.get_per_process_memory_fraction(0) * total / 2**30


@requires_cuda
@pytest.mark.cuda
@pytest.mark.slow
def test_mmwave_via_stack_table_admits(record_property):
    # Two dielectric interfaces (GaAs-like/polymer/air) over PEC: the 10 um span is graded from both ends.
    # The level-2 graded build peaks at ~6 GiB, so this gate needs a >= 7 GiB process cap: the gpu0.py runner,
    # or a 7.5 GiB gpu0 slot at a 7.1 GiB torch cap (a 7.5 GiB torch cap let the caching allocator reserve
    # 7.37 GiB, 7.69 GiB on the device with the context; R4-2).
    if _process_cap_gib() < 7:
        pytest.skip("needs a >= 7 GiB process cap (level-2 graded via-stack table peaks at ~6 GiB)")
    stack = mw.mom.LayerStack((mw.mom.Layer("gaas", 100e-6, 12.9), mw.mom.Layer("polymer", 10e-6, 3.0)), "pec", mw.mom.HalfSpace())
    table = LayeredGreenTable.build(stack, (100e-6, 110e-6), (200e9,), 1.01 * math.hypot(1.1e-3, 175e-6), device="cuda",
                                    via_spans=((100e-6, 110e-6),), z_nodes=8, extract_distance=2 * math.sqrt(2) * 50e-6)
    record = table.vertical.grading_record
    record_property("grading_record", record)
    assert record["ungraded_z_error"] > 1e-6 and table.vertical.z_interpolation_error.item() <= 1e-6
    assert record["levels"] >= record["predicted_levels"] and record["pieces"] == 2 * record["levels"] + 1
