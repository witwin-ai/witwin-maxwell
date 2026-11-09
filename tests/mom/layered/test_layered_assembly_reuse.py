"""Static geometry reuse across complex image weights, chunks and tail tiles."""
from dataclasses import replace

import pytest
import torch

from witwin.maxwell.layers import HalfSpace, Layer, LayerStack
from witwin.maxwell.mom.layered.assembly import assemble_potentials
from witwin.maxwell.mom.layered.green import LayeredGreenTable
from witwin.maxwell.mom.layered.mesh import mesh_rectangles

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_static_geometry_frequency_chunks_and_three_repeats(dtype):
    stack = LayerStack((Layer("air", 1e-3),), HalfSpace(), HalfSpace())
    table = LayeredGreenTable.build(
        stack, (0.,), tuple(1e6 + i * 1e7 for i in range(35)), 1e-3, device="cuda")
    layout = mesh_rectangles(
        [(0., 20e-6, 0., 20e-6, 0.), (200e-6, 220e-6, 0., 20e-6, 0.)],
        table.planes, 10e-6)
    coefficients = table.image_coefficients.clone()
    index = torch.arange(35, device="cuda", dtype=torch.float64)
    coefficients[:, 0] *= torch.complex(torch.cos(index), torch.sin(index))[:, None]
    coefficients[:, 1] *= torch.complex(-torch.sin(index), torch.cos(index))[:, None]
    table = replace(table, image_coefficients=coefficients)
    options = dict(far_dtype=dtype, _network_accuracy=True, source_tile=7)
    first = assemble_potentials(layout, table, frequency_index=-1, **options)
    for _ in range(2):
        again = assemble_potentials(layout, table, frequency_index=-1, **options)
        assert all(torch.equal(a, b) for a, b in zip(first, again))
    for start, stop in ((0, 1), (1, 7), (7, 35)):
        chunk = assemble_potentials(layout, table, frequency_index=-1,
                                    frequency_start=start, frequency_stop=stop, **options)
        assert all(torch.equal(a[start:stop], b) for a, b in zip(first, chunk))
