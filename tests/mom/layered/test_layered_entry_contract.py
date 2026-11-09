"""P1 mixed-precision face-entry gates against independent fixed fp64 rules."""

import pytest
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.assembly import assemble_layered
from witwin.maxwell.mom.layered.green import LayeredGreenTable
from witwin.maxwell.mom.layered.mesh import mesh_rectangles
from witwin.maxwell.mom.layered.network import prepare_layered_network
from tests.mom.layered.test_layered_lowfreq import loop
from tests.mom.layered.test_layered_network import coplanar_strips, rfic_stack

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("kind", ("rfic", "rfic_low", "loop", "halfspace"))
def test_p1_mixed_entries_against_fixed_fp64_rules(kind, record_property):
    if kind in ("rfic", "rfic_low"):
        stack = rfic_stack()
        rectangles, ports = coplanar_strips(
            100e-6, 5e-6, 5e-6, 2e-6, stack.interfaces()[5])
        prepared = prepare_layered_network(
            stack, rectangles, ports, (1e6, 1e9, 50e9), bridge_width=2e-6,
            max_edge=5e-6, edge_cell=1.25e-6)
        layout, table = prepared.layout, prepared.table
    elif kind == "loop":
        prepared = loop((1., 1e3, 1e6, 100e6, 1e9, 10e9))
        layout, table = prepared.layout, prepared.table
    else:
        stack = LayerStack((Layer("substrate", 100e-6, 11.9),),
                           HalfSpace(11.9), HalfSpace())
        table = LayeredGreenTable.build(stack, (100e-6,), (10e9,),
                                         2.2e-3, device="cuda")
        layout = mesh_rectangles(
            [(0., 80e-6, 0., 10e-6, 100e-6),
             (5e-6, 85e-6, 20e-6, 30e-6, 100e-6)],
            table.planes, 10e-6, h_edge=2e-6)
    index = 0 if kind == "rfic_low" else len(table.frequencies)-1
    options = dict(frequency_index=index, _tiles=True)
    reference = assemble_layered(layout, table, far_order=8, near_order=32, **options)
    actual = assemble_layered(layout, table, far_dtype=torch.float32,
                              _network_accuracy=True, **options)
    scale = reference.abs().clone()
    scale[..., :9] = scale[..., :9].amax(-1, keepdim=True)
    scale.clamp_min_(torch.finfo(torch.float64).tiny)
    error = ((actual-reference).abs()/scale).max().item()
    record_property("max_relative_face_entry_error", error)
    assert error <= 1e-6
